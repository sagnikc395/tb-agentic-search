"""Build the biomni-backed LLM or agent, and run the two screening modes."""

import os
import time
from pathlib import Path

from tb_search.config import ConfigError, Settings
from tb_search.documents import extract_text
from tb_search.prompts import BATCH_SCREEN_PROMPT, SCREEN_DOC_PROMPT
from tb_search.report import verdict_line
from tb_search.verdicts import (
    as_bool,
    as_confidence,
    parse_verdict_list,
    parse_verdict_object,
)

# Groq scores requests on a rolling per-minute window, so a TPM rejection
# clears itself if we wait this long and try again.
RATE_LIMIT_WAIT_SECONDS = 65


def check_connection(settings: Settings) -> None:
    """One cheap direct call, so a bad key/model fails fast before biomni boots."""
    from groq import Groq

    try:
        Groq(api_key=settings.api_key).chat.completions.create(
            model=settings.model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
        )
    except Exception as e:
        raise ConfigError(
            f"Groq rejected the request for model {settings.model!r}: {e}"
        ) from e


def build_screener(settings: Settings):
    """A plain Groq chat model via biomni's own LLM plumbing — no A1 prompt.

    This is what fits a free Groq account: ~1k tokens per call instead of the
    A1 agent's ~35k-token system prompt, which trips the 8k-TPM rate limit.
    """
    from biomni.config import default_config
    from biomni.llm import get_llm

    default_config.temperature = settings.temperature
    os.environ["GROQ_API_KEY"] = settings.api_key
    return get_llm(
        model=settings.model, source=settings.source, temperature=settings.temperature
    )


def build_agent(settings: Settings):
    """Construct a Biomni A1 agent wired to Groq.

    `source="Groq"` matters: biomni's auto-detection sees the slash in
    "openai/gpt-oss-120b" and would otherwise route the model to Ollama.
    """
    from biomni.agent import A1
    from biomni.config import default_config

    # A1 forwards llm/source to get_llm but not temperature, so that one goes
    # through default_config. biomni's Groq branch reads GROQ_API_KEY from the
    # environment, so make sure it is there.
    default_config.llm = settings.model
    default_config.source = settings.source
    default_config.temperature = settings.temperature
    os.environ["GROQ_API_KEY"] = settings.api_key

    return A1(
        path=settings.data_path,
        llm=settings.model,
        source=settings.source,
        use_tool_retriever=settings.use_tool_retriever,
        timeout_seconds=settings.timeout_seconds,
        # An empty list (not None) is biomni's signal to skip its download.
        expected_data_lake_files=None if settings.load_data_lake else [],
    )


def _is_rate_limit_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(tag in text for tag in ("413", "rate limit", "tpm", "tokens per minute"))


def _wait_out_rate_limit(action, *args):
    """Groq gives back a 413/TPM error only until the minute window rolls over."""
    for attempt in range(3):
        try:
            return action(*args)
        except Exception as e:
            if not _is_rate_limit_error(e) or attempt == 2:
                raise
            print(
                f"    rate limited; waiting {RATE_LIMIT_WAIT_SECONDS}s and retrying",
                flush=True,
            )
            time.sleep(RATE_LIMIT_WAIT_SECONDS)


def classify_one(screener, name: str, excerpt: str, strict: bool) -> dict | None:
    """Screen a single document excerpt; None when the reply isn't parseable."""
    # .format would eat the JSON braces inside the prompt example; plain replace.
    body = (excerpt[:200] + "...") if strict else excerpt
    prompt = SCREEN_DOC_PROMPT.replace("{name}", name).replace("{DOC}", body)
    if strict:
        prompt += "\n\nREMINDER: reply with ONLY the JSON object, no other text."
    reply = screener.invoke(prompt)
    content = getattr(reply, "content", reply)
    text = (
        content
        if isinstance(content, str)
        else "".join(str(block) for block in content or [])
    )
    return parse_verdict_object(text)


def _row(path: Path, verdict: dict, fallback_reason: str) -> dict:
    """The one result-row shape written to JSON and the summary line."""
    return {
        "file": path.name,
        "path": str(path),
        "tb_related": as_bool(verdict.get("tb_related")),
        "confidence": as_confidence(verdict.get("confidence")),
        "reason": str(verdict.get("reason", "")).strip() or fallback_reason,
    }


def classify_documents_light(screener, docs: list[Path], max_chars: int) -> list[dict]:
    """Screen each document with its own small LLM call; waits out rate limits."""
    rows: list[dict] = []
    for number, path in enumerate(docs, start=1):
        print(f"  {number}/{len(docs)} screening {path.name}...", flush=True)
        excerpt = extract_text(path, max_chars)
        verdict: dict = {}
        for strict in (False, True):  # second pass only asks for stricter JSON
            try:
                verdict = (
                    _wait_out_rate_limit(
                        classify_one, screener, path.name, excerpt, strict
                    )
                    or {}
                )
            except Exception as e:  # noqa: BLE001 - one unparsable doc must not stop the rest
                print(f"    ! screening call failed: {e}")
                break
            if verdict:
                break

        row = _row(
            path, verdict,
            "no parseable verdict returned" if not verdict else "reason missing",
        )
        rows.append(row)
        print(f"    {verdict_line(row)}", flush=True)
    return rows


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
            if _is_rate_limit_error(e):
                print(
                    "    hint: A1's system prompt is huge; try --batch-size 1, or drop --agent"
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
                path, verdict,
                "agent verdict could not be parsed" if not verdicts else "",
            )
            rows.append(row)
            print(f"  {verdict_line(row)}")
        print()

    return rows
