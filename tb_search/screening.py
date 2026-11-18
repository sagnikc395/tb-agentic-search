"""Build the biomni-backed LLM or agent, and run agent-mode screening.

Everything talks to a model we serve ourselves. On Unity that is vLLM, started
in its own job with a reasoning parser, so this process — biomni included —
knows the inference engine only by URL:

    vllm serve Qwen/Qwen3-32B --reasoning-parser qwen3 \
        --enable-auto-tool-choice --tool-call-parser hermes --port 8000

biomni calls that the "Custom" source, and both the plain per-call screener and
the full A1 tool loop take it, which is what lets the agent do its tool calling
against our own engine. Ollama is biomni's other local backend and works the
same way, minus the separated reasoning traces.
"""

import os
import time
from pathlib import Path

from tb_search.config import ConfigError, Settings
from tb_search.documents import extract_text
from tb_search.prompts import BATCH_SCREEN_PROMPT
from tb_search.report import verdict_line
from tb_search.verdicts import (
    as_bool,
    as_confidence,
    parse_verdict_list,
)

# A self-served model has no rate limit, but the server it runs on can still be
# briefly unavailable — a request queue that filled up, or a worker restarting.
RETRY_WAIT_SECONDS = (5, 20)
# Ollama's own default port, where its model list lives.
OLLAMA_HOST = "http://localhost:11434"


def _check_custom(settings: Settings) -> None:
    """Ping the endpoint, so a cluster job that is not up yet says so.

    The model list is checked first because it names what the server is actually
    holding, which is the mistake this catches: a vLLM server serves exactly the
    one model it was started with, under the id it was started with.
    """
    from openai import OpenAI

    client = OpenAI(base_url=settings.base_url, api_key=settings.api_key, timeout=30)
    try:
        served = [model.id for model in client.models.list().data]
    except Exception as e:
        raise ConfigError(
            f"No OpenAI-compatible server answered at {settings.base_url}: {e}\n"
            "On Unity, check the vLLM job is running and the port is reachable "
            "from here (an ssh tunnel, or the compute node's hostname)."
        ) from e
    if served and settings.model not in served:
        raise ConfigError(
            f"{settings.base_url} does not serve {settings.model!r}. "
            f"It serves: {', '.join(served)}"
        )
    try:
        client.chat.completions.create(
            model=settings.model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
            **({"reasoning_effort": "none"} if settings.reasoning_effort else {}),
        )
    except Exception as e:
        raise ConfigError(
            f"{settings.base_url} rejected a request for {settings.model!r}: {e}"
        ) from e


def _check_ollama(settings: Settings) -> None:
    """Check Ollama is running and has the model pulled."""
    import requests

    host = os.getenv("OLLAMA_HOST") or OLLAMA_HOST
    try:
        response = requests.get(f"{host.rstrip('/')}/api/tags", timeout=10)
        response.raise_for_status()
        pulled = [model["name"] for model in response.json().get("models", [])]
    except Exception as e:
        raise ConfigError(
            f"Ollama did not answer at {host}: {e}\nStart it with `ollama serve`."
        ) from e
    # Ollama tags carry a version suffix: "qwen3:32b" answers to "qwen3:32b"
    # and, for the default tag, to "qwen3".
    if pulled and not any(
        name == settings.model or name.split(":")[0] == settings.model
        for name in pulled
    ):
        raise ConfigError(
            f"Ollama has not pulled {settings.model!r}. It holds: "
            f"{', '.join(pulled)}\nPull it with `ollama pull {settings.model}`."
        )


def check_connection(settings: Settings) -> None:
    """One cheap direct call, so a server that is down fails here, not mid-run."""
    if settings.is_custom:
        _check_custom(settings)
    else:
        _check_ollama(settings)


def apply_reasoning_effort(llm, effort: str | None):
    """Send `reasoning_effort` with every request this chat model makes.

    It goes in `extra_body` rather than as a typed parameter so it reaches vLLM
    as a top-level request field whatever the OpenAI client happens to know
    about, and it is set on the model itself rather than bound, because A1 also
    calls `with_structured_output` on it and a bound runnable has no such method.
    """
    if not effort or llm is None:
        return llm
    try:
        llm.extra_body = {**(llm.extra_body or {}), "reasoning_effort": effort}
    except (AttributeError, ValueError):
        # An older langchain-openai has no extra_body field; model_kwargs is
        # forwarded to the client the same way.
        kwargs = dict(getattr(llm, "model_kwargs", None) or {})
        body = {**(kwargs.get("extra_body") or {}), "reasoning_effort": effort}
        llm.model_kwargs = {**kwargs, "extra_body": body}
    return llm


def reasoning_chat_model(settings: Settings):
    """A ChatOpenAI that keeps the reasoning field the base class throws away.

    langchain's ChatOpenAI tracks the official OpenAI schema and drops anything
    else a provider adds, vLLM's separated `reasoning` (older servers:
    `reasoning_content`) among it — so the traces would never reach us. This
    subclass copies it back onto the message's additional_kwargs, where
    tb_search.reasoning looks for it.

    It stays a ChatOpenAI subclass on purpose: biomni's A1 calls
    `with_structured_output` on whatever model it is given.
    """
    from langchain_openai import ChatOpenAI

    class ChatOpenAIKeepingReasoning(ChatOpenAI):
        def _create_chat_result(self, response, generation_info=None):
            result = super()._create_chat_result(response, generation_info)
            for generation, choice in zip(result.generations, _choices(response)):
                message = (choice or {}).get("message") or {}
                for key in ("reasoning", "reasoning_content"):
                    text = message.get(key)
                    if isinstance(text, str) and text.strip():
                        generation.message.additional_kwargs[key] = text
            return result

    return ChatOpenAIKeepingReasoning(
        model=settings.model,
        temperature=settings.temperature,
        max_tokens=8192,
        base_url=settings.base_url,
        api_key=settings.api_key,
    )


def _choices(response) -> list[dict]:
    """The response's choices as plain dicts, whatever shape it arrived in."""
    if not isinstance(response, dict):
        try:
            response = response.model_dump(
                exclude={"choices": {"__all__": {"message": {"parsed"}}}},
                warnings=False,
            )
        except Exception:  # noqa: BLE001 - no trace is better than a failed call
            return []
    return [
        choice for choice in (response.get("choices") or []) if isinstance(choice, dict)
    ]


def _prime_environment(settings: Settings) -> None:
    """Put the endpoint where biomni's own code looks for it, not just ours."""
    if settings.is_custom:
        os.environ["BIOMNI_CUSTOM_BASE_URL"] = settings.base_url or ""
        os.environ["BIOMNI_CUSTOM_API_KEY"] = settings.api_key
        # The OpenAI client reads this when nothing is passed explicitly.
        os.environ.setdefault("OPENAI_API_KEY", settings.api_key)


def build_screener(settings: Settings):
    """A plain chat model — no A1 prompt around it.

    The cheap path: ~1k tokens per call instead of the A1 agent's ~35k-token
    system prompt, so a run over thousands of isolates is one short request
    each, and a smaller served model can still handle it.
    """
    from biomni.config import default_config

    default_config.temperature = settings.temperature
    _prime_environment(settings)
    return apply_reasoning_effort(_chat_model(settings), settings.reasoning_effort)


def _chat_model(settings: Settings):
    """The chat model for this backend: ours when it can carry reasoning.

    A server reached over the OpenAI-compatible API is the case the
    reasoning-keeping subclass covers. Ollama goes through biomni, which builds
    a ChatOllama for it; its thinking arrives inside the answer instead, where
    tb_search.reasoning picks it out of the <think> tags.
    """
    if settings.is_custom:
        return reasoning_chat_model(settings)

    from biomni.llm import get_llm

    return get_llm(
        model=settings.model,
        source=settings.source,
        temperature=settings.temperature,
        base_url=settings.base_url,
        api_key=settings.api_key if settings.is_custom else None,
    )


def build_agent(settings: Settings):
    """Construct a Biomni A1 agent wired to our own model server.

    The source matters: biomni's auto-detection sees the slash in a HuggingFace
    model id like "Qwen/Qwen3-32B" and would route it to Ollama on its own.
    "Custom" is the branch that points langchain at base_url, which is how the
    agent's tool calling runs against the inference engine on the cluster.
    """
    from biomni.agent import A1
    from biomni.config import default_config

    # A1 forwards llm/source/base_url to get_llm but not temperature, so that
    # one goes through default_config.
    default_config.llm = settings.model
    default_config.source = settings.source
    default_config.temperature = settings.temperature
    default_config.base_url = settings.base_url
    _prime_environment(settings)

    agent = A1(
        path=settings.data_path,
        llm=settings.model,
        source=settings.source,
        use_tool_retriever=settings.use_tool_retriever,
        timeout_seconds=settings.timeout_seconds,
        base_url=settings.base_url,
        api_key=settings.api_key if settings.is_custom else None,
        # An empty list (not None) is biomni's signal to skip its download.
        expected_data_lake_files=None if settings.load_data_lake else [],
    )
    # A1 built its own plain ChatOpenAI, which drops the reasoning field; swap
    # in the one that keeps it, so agent runs are traceable too. The agent uses
    # self.llm for every step, so replacing it after init is enough.
    if settings.is_custom:
        agent.llm = _chat_model(settings)
    apply_reasoning_effort(getattr(agent, "llm", None), settings.reasoning_effort)
    return agent


def is_transient_error(exc: Exception) -> bool:
    """Whether this looks like the server being busy rather than a bad request.

    A model server under load, or one whose worker is restarting, refuses a
    request in a way that clears by itself; a prompt too long for the context
    window does not, and retrying it only wastes the time twice.
    """
    text = str(exc).lower()
    return any(
        tag in text
        for tag in (
            "429",
            "500",
            "502",
            "503",
            "504",
            "timeout",
            "timed out",
            "connection",
            "overloaded",
            "temporarily",
        )
    )


def retry_transient(action, *args):
    """Run `action`, waiting out a busy or restarting server up to twice."""
    for attempt, wait in enumerate((*RETRY_WAIT_SECONDS, None)):
        try:
            return action(*args)
        except Exception as e:
            if wait is None or not is_transient_error(e):
                raise
            print(
                f"    server busy ({type(e).__name__}); retrying in {wait}s"
                f" [attempt {attempt + 1}]",
                flush=True,
            )
            time.sleep(wait)


def _row(path: Path, verdict: dict, fallback_reason: str) -> dict:
    """The one result-row shape written to JSON and the summary line."""
    return {
        "file": path.name,
        "path": str(path),
        "tb_related": as_bool(verdict.get("tb_related")),
        "confidence": as_confidence(verdict.get("confidence")),
        "reason": str(verdict.get("reason", "")).strip() or fallback_reason,
    }


def classify_documents_agent(
    agent, docs: list[Path], batch_size: int, max_chars: int
) -> list[dict]:
    """Ask the full A1 agent to judge the docs in batches; one row per doc."""
    rows: list[dict] = []
    batches = [docs[i : i + batch_size] for i in range(0, len(docs), batch_size)]

    for batch_number, batch in enumerate(batches, start=1):
        listing = "\n\n".join(
            f"Doc {index} — {path.name}\n{extract_text(path, max_chars)}"
            for index, path in enumerate(batch)
        )
        prompt = BATCH_SCREEN_PROMPT.replace("{DOCS}", listing)

        try:
            _log, answer = agent.go(prompt)
        except Exception as e:  # noqa: BLE001 - a failed batch must not stop the run
            print(
                f"  ! agent call {batch_number}/{len(batches)} failed, marking batch unknown: {e}"
            )
            if "context" in str(e).lower() or "length" in str(e).lower():
                print(
                    "    hint: A1's system prompt is ~35k tokens; serve the model "
                    "with a larger --max-model-len, try --batch-size 1, or drop --agent"
                )
            answer = ""

        verdicts = parse_verdict_list(answer or "", len(batch))
        if len(verdicts) < len(batch):  # one retry, with a sterner reminder
            try:
                _log, answer = agent.go(
                    prompt
                    + "\n\nREMINDER: reply with ONLY the JSON list, no other text."
                )
                verdicts = parse_verdict_list(answer or "", len(batch))
            except Exception as e:  # noqa: BLE001 - leave unparsed docs as unknown
                print(f"  ! retry failed, leaving unparsed docs as unknown: {e}")

        print(f"Batch {batch_number}/{len(batches)}:")
        for index, path in enumerate(batch):
            verdict = verdicts.get(index, {})
            row = _row(
                path,
                verdict,
                "agent verdict could not be parsed" if not verdicts else "",
            )
            rows.append(row)
            print(f"  {verdict_line(row)}")
        print()

    return rows
