"""Build the Biomni A1 agent, run it, and screen documents with it.

Every mode in this pipeline is one A1 run. The agent talks to Groq's hosted
endpoint — an OpenAI-compatible API with tool calling, which is what lets A1 do
its tool calling, and with the model's thinking returned in its own `reasoning`
field, which is what every prediction stores beside its variants:

    POST https://api.groq.com/openai/v1/chat/completions
    { "model": "openai/gpt-oss-120b", "reasoning_effort": "high",
      "include_reasoning": true, ... }

biomni calls that the "Groq" source. Nothing here holds weights or needs a GPU;
what it does need is a key (GROQ_API_KEY) and patience with the rate limit.

A1's system prompt is ~35k tokens and rides on every step of every run, so the
model has to have a context window well past that — 131k on the GPT-OSS models,
which is why one of those is the default — and every step of every run is
charged for and counted against the per-minute token limit.
"""

import os
import random
import re
import time
from pathlib import Path

from tb_search.config import DEFAULT_MAX_RETRIES, ConfigError, Settings
from tb_search.documents import read_text, section_excerpt
from tb_search.prompts import BATCH_SCREEN_PROMPT
from tb_search.reasoning import REASONING_KEYS, reasoning_body, split_thinking
from tb_search.report import verdict_line
from tb_search.usage import record_completion, record_wait
from tb_search.verdicts import (
    as_bool,
    as_confidence,
    parse_verdict_list,
    quote_in_text,
)

# How long to wait before retrying a rate-limited or briefly failing request,
# when Groq did not say. Doubles per attempt from here, with jitter, so a sweep
# that meets the per-minute limit backs off instead of hammering it.
BASE_WAIT_SECONDS = 5
MAX_WAIT_SECONDS = 90
# Groq answers a 429 with how long to wait, either in the Retry-After header or
# in the message itself ("Please try again in 7.481s"). Obeying that is faster
# and politer than guessing.
RETRY_AFTER_IN_MESSAGE = re.compile(
    r"try again in\s+([0-9.]+)\s*(ms|s|m)?", re.IGNORECASE
)
# What a run that ran out of context should be told to do about it.
CONTEXT_HINT = (
    "    hint: A1's system prompt is ~35k tokens; cut --batch-size / --max-chars, "
    "or use a model with a larger context window"
)
# The failure that is specific to a hosted endpoint, and the one most likely to
# stop this pipeline dead: not the context window, but the account's per-minute
# token budget being smaller than a single A1 request. Groq answers that with a
# 413, and no amount of waiting or retrying fixes it — one request simply cannot
# be afforded, so the message has to say so rather than suggest a shorter
# excerpt. Groq's free tier is 8,000 TPM, which A1's system prompt alone clears.
TOKEN_BUDGET_HINT = (
    "    hint: this is the account's tokens-per-minute budget, not the model's "
    "context window. Almost all of an A1 request is its system prompt, so a "
    "smaller --batch-size or --max-chars barely moves it:\n"
    "      ~36k tokens per request carrying the whole tool catalogue\n"
    "      ~11k tokens per request with --tools (a retrieved tool subset)\n"
    "    --tools is therefore the one real lever. Past it the tier has to go up: "
    "Groq's free tier is 8,000 TPM, which neither of those fits "
    "(https://console.groq.com/settings/billing)."
)


def check_connection(settings: Settings) -> None:
    """One cheap direct call, so a bad key or model id fails here, not mid-run.

    This is the one request in the pipeline that is not an A1 run, and
    deliberately so: every task goes through the agent, but a preflight has to
    fail with a readable reason, and an A1 run would wrap a dead key in a tool
    loop and ~36k tokens of system prompt before reporting anything. One
    `max_tokens=1` call says the same thing for nothing.

    The model list is checked first because it names what the account can
    actually reach: Groq's catalogue moves, and a model id that was a preview
    model last month is the mistake this catches.
    """
    from openai import OpenAI

    client = OpenAI(base_url=settings.base_url, api_key=settings.api_key, timeout=30)
    try:
        served = [model.id for model in client.models.list().data]
    except Exception as e:
        raise ConfigError(
            f"Groq did not answer at {settings.base_url}: {e}\n"
            "Check GROQ_API_KEY is a live key (https://console.groq.com/keys) "
            "and that this machine can reach api.groq.com."
        ) from e
    if served and settings.model not in served:
        raise ConfigError(
            f"Groq does not serve {settings.model!r} to this account. "
            f"It offers: {', '.join(sorted(served))}"
        )
    try:
        client.chat.completions.create(
            model=settings.model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
            # In extra_body, not as typed arguments: these are Groq's own
            # fields, and the installed OpenAI client refuses a keyword it does
            # not know rather than forwarding it. Sending them here is the
            # point of this call — an effort the model rejects is a 400, and
            # this is where it should land.
            extra_body=reasoning_body(settings.model, settings.reasoning_effort),
        )
    except Exception as e:
        raise ConfigError(
            f"Groq rejected a request for {settings.model!r}: {e}\n"
            "A 400 here usually means the model does not take the reasoning "
            "settings asked of it — see --reasoning-effort."
        ) from e


def apply_reasoning_body(llm, settings: Settings):
    """Send the reasoning fields with every request this chat model makes.

    They go in `extra_body` rather than as typed parameters so they reach Groq
    as top-level request fields whatever the OpenAI client happens to know
    about, and they are set on the model itself rather than bound, because A1
    also calls `with_structured_output` on it and a bound runnable has no such
    method.
    """
    if llm is None:
        return llm
    body = reasoning_body(settings.model, settings.reasoning_effort)
    try:
        llm.extra_body = {**(llm.extra_body or {}), **body}
    except (AttributeError, ValueError):
        # An older langchain-openai has no extra_body field; model_kwargs is
        # forwarded to the client the same way.
        kwargs = dict(getattr(llm, "model_kwargs", None) or {})
        llm.model_kwargs = {
            **kwargs,
            "extra_body": {**(kwargs.get("extra_body") or {}), **body},
        }
    return llm


def reasoning_chat_model(settings: Settings, traces: list[str] | None = None):
    """A ChatOpenAI that keeps the reasoning field the base class throws away.

    langchain's ChatOpenAI tracks the official OpenAI schema and drops anything
    else a provider adds, Groq's separated `reasoning` among it — so the traces
    would never reach us. This subclass copies it back onto the message's
    additional_kwargs, where tb_search.reasoning looks for it, and appends it to
    `traces` when one is given.

    That second copy is what makes an agent run traceable at all: A1 rebuilds
    every step's reply as a fresh AIMessage, so additional_kwargs is gone by the
    time the step reaches the graph state. The list is the only place the
    thinking survives.

    It stays a ChatOpenAI subclass on purpose: biomni's A1 calls
    `with_structured_output` on whatever model it is given. Groq's endpoint is
    OpenAI-compatible, so that is also simply the right client for it.
    """
    from langchain_openai import ChatOpenAI

    class ChatOpenAIKeepingReasoning(ChatOpenAI):
        def _create_chat_result(self, response, generation_info=None):
            result = super()._create_chat_result(response, generation_info)
            # Every reply passes through here, which makes it the one place
            # that sees what each request cost (tb_search.usage).
            record_completion(_as_dict(response), settings.model)
            for generation, choice in zip(result.generations, _choices(response)):
                message = (choice or {}).get("message") or {}
                found = ""
                for key in REASONING_KEYS:
                    text = message.get(key)
                    if isinstance(text, str) and text.strip():
                        generation.message.additional_kwargs[key] = text
                        found = found or text.strip()
                if found and traces is not None:
                    traces.append(found)
            return result

    return ChatOpenAIKeepingReasoning(
        model=settings.model,
        temperature=settings.temperature,
        max_tokens=8192,
        base_url=settings.base_url,
        api_key=settings.api_key,
        # Groq's own 429s are waited out by retry_transient, which can report
        # what it is doing; the client's silent retries would only hide it.
        max_retries=0,
    )


def _as_dict(response) -> dict:
    """The whole response as a plain dict, whatever shape it arrived in."""
    if isinstance(response, dict):
        return response
    try:
        return response.model_dump(
            exclude={"choices": {"__all__": {"message": {"parsed"}}}},
            warnings=False,
        )
    except Exception:  # noqa: BLE001 - no trace is better than a failed call
        return {}


def _choices(response) -> list[dict]:
    """The response's choices as plain dicts, whatever shape it arrived in."""
    body = _as_dict(response)
    return [
        choice for choice in (body.get("choices") or []) if isinstance(choice, dict)
    ]


def _prime_environment(settings: Settings) -> None:
    """Put the key where biomni's own code looks for it, not just ours.

    biomni's Groq branch reads GROQ_API_KEY out of the environment rather than
    taking it as an argument, so a key that came from our own .env handling has
    to be put back there before A1 is built.
    """
    os.environ["GROQ_API_KEY"] = settings.api_key
    # The OpenAI client reads this when nothing is passed explicitly.
    os.environ.setdefault("OPENAI_API_KEY", settings.api_key)


def build_agent(settings: Settings):
    """Construct a Biomni A1 agent wired to Groq.

    This is the pipeline's one model object: screening, retrieval and resistance
    prediction are all A1 runs, so every prompt goes through the same agent and
    the same tool loop.

    The source matters: biomni's auto-detection sees the slash in a model id
    like "openai/gpt-oss-120b" and would route it to Ollama on its own. "Groq"
    is the branch that points langchain at api.groq.com with the key from the
    environment, which is how the agent's tool calling runs against Groq.
    """
    from biomni.agent import A1
    from biomni.config import default_config

    # A1 forwards llm/source to get_llm but not temperature, so that one goes
    # through default_config.
    default_config.llm = settings.model
    default_config.source = settings.source
    default_config.temperature = settings.temperature
    _prime_environment(settings)

    agent = A1(
        path=settings.data_path,
        llm=settings.model,
        source=settings.source,
        use_tool_retriever=settings.use_tool_retriever,
        timeout_seconds=settings.timeout_seconds,
        # An empty list (not None) is biomni's signal to skip its download.
        expected_data_lake_files=None if settings.load_data_lake else [],
    )
    # Where run_agent reads each run's thinking back out. A plain attribute,
    # because A1 is an ordinary class and this is the one piece of per-run state
    # the agent does not already keep for us.
    agent.reasoning_traces = []
    # How many times a rate-limited step is waited out, carried on the agent so
    # every call site retries the way this run was configured to.
    agent.max_retries = settings.max_retries
    # A1 built its own plain ChatOpenAI, which asks for no reasoning and would
    # drop it anyway; swap in the one that asks and keeps it, so agent runs are
    # traceable. Every model call the agent makes goes through this one object:
    # its `generate` node invokes self.llm per step, and its tool retriever is
    # handed self.llm too.
    agent.llm = reasoning_chat_model(settings, agent.reasoning_traces)
    apply_reasoning_body(agent.llm, settings)
    # Then rebuild the graph around it. A1's nodes read self.llm when they run,
    # so the swap already applies — but that is an assumption about biomni's
    # internals, and the failure if it ever stops holding is silent: the run
    # would work, against biomni's own chat model, and every reasoning trace
    # would quietly come back empty. configure() is what A1 itself calls after
    # changing its tools, so calling it here costs a prompt rebuild and makes
    # the swap hold either way.
    reconfigure = getattr(agent, "configure", None)
    if callable(reconfigure):
        try:
            reconfigure()
        except Exception as e:  # noqa: BLE001 - the swap already applies without it
            print(f"! could not rebuild the agent graph after the model swap: {e}")
    return agent


def run_agent(agent, prompt: str) -> tuple[str, str]:
    """One A1 run: (its final answer, the thinking behind it).

    The trace is the whole run's thinking, step by step, rather than the last
    step's alone — with a tool loop in the middle, that is what the answer was
    actually reasoned from. It comes from the list `build_agent` handed the chat
    model, because A1 drops Groq's `reasoning` field on the way into the graph
    state. A reply that left its thinking inline in <think> tags is the
    fallback.
    """
    traces = getattr(agent, "reasoning_traces", None)
    if traces is not None:
        del traces[:]
    _log, content = retry_transient(
        agent.go, prompt, attempts=getattr(agent, "max_retries", None)
    )
    answer, inline = split_thinking(content or "")
    return answer, "\n\n".join(traces or []) or inline


def is_rate_limit_error(exc: Exception) -> bool:
    """Whether Groq refused this request for being over a limit.

    The common failure of a hosted endpoint, and the one worth waiting out at
    length: the per-minute token limit clears within the minute.
    """
    text = str(exc).lower()
    return (
        "429" in text
        or "rate limit" in text
        or "rate_limit" in text
        or "too many requests" in text
    )


def is_transient_error(exc: Exception) -> bool:
    """Whether this looks like a limit or a wobble rather than a bad request.

    Groq's rate limits and its occasional 5xx or dropped connection all clear
    by themselves; a prompt too long for the context window, a bad key or an
    unknown model does not, and retrying it only wastes the time over again.
    """
    if is_token_budget_error(exc):
        # Bigger than a whole minute's allowance: there is nothing to wait for.
        return False
    if is_rate_limit_error(exc):
        return True
    text = str(exc).lower()
    if "503" in text and "capacity" in text:
        # Flex/on-demand capacity, which Groq's own docs say to retry with
        # jittered backoff.
        return True
    return any(
        tag in text
        for tag in (
            "500",
            "502",
            "503",
            "504",
            "timeout",
            "timed out",
            "connection",
            "overloaded",
            "temporarily",
            "service unavailable",
        )
    )


def is_token_budget_error(exc: Exception) -> bool:
    """Whether one request was bigger than the account's per-minute budget.

    Groq's 413 ("Request too large ... on tokens per minute (TPM): Limit 8000,
    Requested 36071"). It reads like a rate limit and is not one: waiting
    changes nothing, because the request does not fit in a whole minute's
    allowance. Kept apart from both the retried errors and the context-window
    ones, since its fix is neither waiting nor a shorter prompt.
    """
    text = str(exc).lower()
    return "413" in text or "request too large" in text


def is_context_error(exc: Exception) -> bool:
    """Whether the run died of a prompt longer than the model's context window."""
    if is_token_budget_error(exc):
        return False
    text = str(exc).lower()
    return (
        "context" in text
        or "too large" in text
        or ("length" in text and "429" not in text)
    )


def failure_hint(exc: Exception) -> str | None:
    """The line to print under a failed agent run, or None when there is none."""
    if is_token_budget_error(exc):
        return TOKEN_BUDGET_HINT
    if is_context_error(exc):
        return CONTEXT_HINT
    return None


def retry_after_seconds(exc: Exception) -> float | None:
    """How long Groq asked us to wait, from the header or the message.

    A 429 carries `retry-after` in seconds, and the body usually repeats it to
    fractions of a second ("Please try again in 7.481s"). Either is better than
    a guess, so whichever is there is used.
    """
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) or {}
    try:
        header = headers.get("retry-after") or headers.get("Retry-After")
    except AttributeError:
        header = None
    if header:
        try:
            return float(str(header).strip())
        except ValueError:
            pass
    found = RETRY_AFTER_IN_MESSAGE.search(str(exc))
    if not found:
        return None
    value = float(found.group(1))
    unit = (found.group(2) or "s").lower()
    return value / 1000 if unit == "ms" else value * 60 if unit == "m" else value


def backoff_seconds(exc: Exception, attempt: int) -> float:
    """What to wait before attempt number `attempt` (1-based), at most a minute
    and a half.

    Groq's own instruction wins when it gave one. Otherwise it doubles per
    attempt from BASE_WAIT_SECONDS, with jitter so several isolates in flight do
    not all come back at the same instant.
    """
    asked = retry_after_seconds(exc)
    if asked is not None:
        # A second of slack, because the window has to have actually rolled
        # over by the time the retry lands.
        return min(MAX_WAIT_SECONDS, asked + 1)
    wait = min(MAX_WAIT_SECONDS, BASE_WAIT_SECONDS * (2 ** (attempt - 1)))
    return wait * random.uniform(0.8, 1.2)


def retry_transient(action, *args, attempts: int | None = None):
    """Run `action`, waiting out Groq's rate limit or a wobble `attempts` times."""
    total = DEFAULT_MAX_RETRIES if attempts is None else max(0, attempts)
    for attempt in range(1, total + 2):
        try:
            return action(*args)
        except Exception as e:
            if attempt > total or not is_transient_error(e):
                raise
            wait = backoff_seconds(e, attempt)
            why = "rate limited" if is_rate_limit_error(e) else "transient failure"
            record_wait(wait, why, attempt)
            print(
                f"    Groq {why} ({type(e).__name__}); waiting {wait:.0f}s"
                f" [attempt {attempt} of {total}]",
                flush=True,
            )
            time.sleep(wait)


def _row(path: Path, verdict: dict, text: str, fallback_reason: str) -> dict:
    """The one result-row shape written to JSON and the summary line.

    `evidence_verified` is the check that separates a screened document from an
    invented one: a quote the excerpt does not contain is the model talking, not
    the paper.
    """
    evidence = str(verdict.get("evidence", "")).strip()
    return {
        "file": path.name,
        "path": str(path),
        "tb_related": as_bool(verdict.get("tb_related")),
        "confidence": as_confidence(verdict.get("confidence")),
        "evidence": evidence,
        "evidence_verified": quote_in_text(evidence, text) if evidence else None,
        "reason": str(verdict.get("reason", "")).strip() or fallback_reason,
    }


def classify_documents_agent(
    agent, docs: list[Path], batch_size: int, max_chars: int
) -> list[dict]:
    """Ask the A1 agent to judge the docs in batches; one row per doc.

    One failed batch is marked unknown rather than taking the run down: a
    screening pass over a whole directory should survive a single agent run that
    ran out of context, came back unparseable, or hit the rate limit for longer
    than the retries covered.
    """
    rows: list[dict] = []
    batches = [docs[i : i + batch_size] for i in range(0, len(docs), batch_size)]

    for batch_number, batch in enumerate(batches, start=1):
        # The opening plus the starts of the methods and closing sections: the
        # opening alone cannot tell a TB methods paper from one that names TB in
        # passing, and it is also what the evidence quote is checked against.
        texts = [section_excerpt(read_text(path), max_chars) for path in batch]
        listing = "\n\n".join(
            f"Doc {index} — {path.name}\n{text}"
            for index, (path, text) in enumerate(zip(batch, texts))
        )
        prompt = BATCH_SCREEN_PROMPT.replace("{DOCS}", listing)
        label = f"{batch_number}/{len(batches)}"

        verdicts: dict[int, dict] = {}
        for strict in (False, True):  # second pass only asks for stricter JSON
            if strict:
                prompt += "\n\nREMINDER: reply with ONLY the JSON list, no other text."
            try:
                answer, _trace = run_agent(agent, prompt)
            except Exception as e:  # noqa: BLE001 - a failed batch must not stop the run
                print(f"  ! agent call {label} failed, marking batch unknown: {e}")
                hint = failure_hint(e)
                if hint:
                    print(hint)
                break
            verdicts = parse_verdict_list(answer, len(batch))
            if len(verdicts) == len(batch):
                break

        print(f"Batch {label}:")
        for index, (path, text) in enumerate(zip(batch, texts)):
            row = _row(
                path,
                verdicts.get(index, {}),
                text,
                "agent verdict could not be parsed" if not verdicts else "",
            )
            rows.append(row)
            print(f"  {verdict_line(row)}")
        print()

    return rows
