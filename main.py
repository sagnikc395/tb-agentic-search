#!/usr/bin/env python
"""tb-agentic-search — detect TB-related research documents with a Biomni agent.

Single-file pipeline:

  1. Collect the documents under an input path (PDFs and plain text).
  2. Extract a text excerpt from each one.
  3. Ask biomni's Groq-backed LLM whether each document is TB-related.
  4. Save per-document verdicts to a JSON file and print a summary table.

Two screening modes:

  default ("light") — each document gets its own small LLM call, routed through
    biomni.llm.get_llm. Small context (~1k tokens/call), so it works on a free
    Groq account, whose budget is only 8000 tokens per minute.

  --agent — the full Biomni A1 agent loop with biomni's tool stack. A1 ships a
    ~35k-token system prompt, so this needs a Groq tier with enough TPM.

Usage:

    uv run python main.py                 # screens ./papers, else ../papers
    uv run python main.py ../papers       # a directory, or a single PDF/text file
    uv run python main.py --scan-only     # list docs and preview excerpts only
    uv run python main.py --agent         # full A1 agent loop instead (needs TPM)

Requires GROQ_API_KEY in the environment (copy .env.example to .env).
"""

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "openai/gpt-oss-120b"


class ConfigError(RuntimeError):
    """Raised when the environment is not set up well enough to run."""


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    """Everything needed to build the screener/agent. Use `load_settings()`."""

    api_key: str
    model: str = DEFAULT_MODEL
    source: str = "Groq"
    # Low by default: the screener wants tight, mechanical JSON replies.
    temperature: float = 0.2
    data_path: str = "./data"
    timeout_seconds: int = 600
    use_tool_retriever: bool = False
    # Biomni pulls a tens-of-GB data lake on first A1 init. Screening doesn't
    # need it, so it is off by default (`--data-lake` turns it on).
    load_data_lake: bool = False


def load_settings(**overrides) -> Settings:
    """Read settings from .env / environment; kwargs win when not None."""
    load_dotenv()

    values = {
        "api_key": os.getenv("GROQ_API_KEY", "").strip(),
        "model": os.getenv("GROQ_MODEL") or DEFAULT_MODEL,
        "source": os.getenv("LLM_SOURCE") or "Groq",
        "temperature": float(os.getenv("BIOMNI_TEMPERATURE", "0.2")),
        "data_path": os.getenv("BIOMNI_PATH", "./data"),
        "timeout_seconds": int(os.getenv("BIOMNI_TIMEOUT_SECONDS", "600")),
        "use_tool_retriever": _env_bool("BIOMNI_USE_TOOL_RETRIEVER", False),
        "load_data_lake": _env_bool("BIOMNI_LOAD_DATA_LAKE", False),
    }
    values.update({k: v for k, v in overrides.items() if v is not None})

    if not values["api_key"]:
        raise ConfigError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and fill it in, "
            "or export GROQ_API_KEY in your shell."
        )
    return Settings(**values)


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
        raise ConfigError(f"Groq rejected the request for model {settings.model!r}: {e}") from e


# ---------------------------------------------------------------------------
# Biomni-backed LLM and agent builders
# ---------------------------------------------------------------------------


def build_screener(settings: Settings):
    """A plain Groq chat model via biomni's own LLM plumbing — no A1 prompt.

    This is what fits a free Groq account: ~1k tokens per call instead of the
    A1 agent's ~35k-token system prompt, which trips the 8k-TPM rate limit.
    """
    from biomni.config import default_config
    from biomni.llm import get_llm

    default_config.temperature = settings.temperature
    os.environ["GROQ_API_KEY"] = settings.api_key
    return get_llm(model=settings.model, source=settings.source, temperature=settings.temperature)


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


# ---------------------------------------------------------------------------
# Document collection and text extraction
# ---------------------------------------------------------------------------

PDF_SUFFIX = ".pdf"
TEXT_SUFFIXES = (".txt", ".md", ".markdown", ".html", ".htm", ".csv", ".tsv", ".json")

EXCERPT_CHARS = 3000  # per document ~ title + abstract; sized for small API budgets

RATE_LIMIT_WAIT_SECONDS = 65  # Groq scores requests on a rolling per-minute window


def resolve_input(raw: str | None) -> Path | None:
    """Input dir/file given by the user, else ./papers, else ../papers."""
    if raw:
        candidate = Path(raw).expanduser()
        return candidate if candidate.exists() else None
    for candidate in (Path("papers"), Path.cwd().parent / "papers"):
        if candidate.is_dir():
            return candidate
    return None


def collect_docs(input_path: Path) -> list[Path]:
    """All supported documents under `input_path`, sorted by filename."""
    if input_path.is_file():
        return [input_path]

    docs: list[Path] = []
    for path in sorted(input_path.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        if path.suffix.lower() == PDF_SUFFIX or path.suffix.lower() in TEXT_SUFFIXES:
            docs.append(path)
    return docs


def extract_text(path: Path, max_chars: int = EXCERPT_CHARS) -> str:
    """First `max_chars` characters of the document, whitespace-collapsed."""
    import logging

    logging.getLogger("pypdf").setLevel(logging.ERROR)  # pypdf warns exuberantly

    if path.suffix.lower() == PDF_SUFFIX:
        from pypdf import PdfReader

        try:
            reader = PdfReader(str(path))
            text = "\n".join((page.extract_text() or "") for page in reader.pages)
        except Exception as e:  # unreadable / scanned pdf — screen what we can
            return f"(could not extract text: {e})"
    else:
        try:
            text = path.read_text(errors="ignore")
            if path.suffix.lower() in (".html", ".htm"):
                text = re.sub(r"<[^>]+>", " ", text)
        except Exception as e:
            return f"(could not read file: {e})"

    return re.sub(r"\s+", " ", text).strip()[:max_chars]


# ---------------------------------------------------------------------------
# Screening prompts
# ---------------------------------------------------------------------------

SCREEN_DOC_PROMPT = """You are a literature screener for a tuberculosis research project.
Decide whether the document below is fundamentally about tuberculosis (TB) or
Mycobacterium tuberculosis.

TB-related means TB or M. tuberculosis is a central subject or a core component
of the work — e.g. diagnosing TB, its drug resistance, or a genomic/variant
database built around it. A passing mention of TB does not count. Judge by the
excerpt.

Reply with exactly one JSON object and nothing else:

{"tb_related": true, "confidence": "high", "reason": "short justification"}

tb_related must be a boolean, confidence one of high/medium/low, reason under 20
words. Mark borderline calls with confidence "low".

DOCUMENT ("{name}"):

---
{DOC}
---
"""  # {DOC} is replaced with the excerpt; braces elsewhere keep .replace() safe

BATCH_SCREEN_PROMPT = """You are a literature screener for a tuberculosis research project.
For each numbered document below, decide whether it is fundamentally about
tuberculosis (TB) or Mycobacterium tuberculosis.

A document is TB-related when TB or M. tuberculosis is a central subject or a
core component of the work — e.g. diagnosing TB, its drug resistance, or a
genomic/variant database built around it. A passing mention of TB does not
count. Judge each document by its excerpt.

Reply with exactly one JSON list, one object per document id, nothing outside it:

```json
[{"id": 0, "tb_related": true, "confidence": "high", "reason": "short justification"}]
```

Use a boolean for tb_related, one of high/medium/low for confidence, and keep
reasons under 20 words. Mark borderline calls with confidence "low".

DOCUMENTS:

{DOCS}
"""


# ---------------------------------------------------------------------------
# Verdict normalization and JSON salvage
# ---------------------------------------------------------------------------


def _json_candidate_strings(answer: str) -> list[str]:
    candidates: list[str] = []
    fenced = re.search(r"```(?:json)?\s*(.*?)```", answer, re.S)
    if fenced:
        candidates.append(fenced.group(1))
    first, last = answer.find("["), answer.rfind("]")
    if first != -1 and last > first:
        candidates.append(answer[first : last + 1])
    first, last = answer.find("{"), answer.rfind("}")
    if first != -1 and last > first:
        candidates.append(answer[first : last + 1])
    candidates.append(answer)
    return candidates


def _load_json_candidates(answer: str):
    for blob in _json_candidate_strings(answer):
        try:
            yield json.loads(blob)
        except ValueError:
            continue


def _as_bool(value) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        lookup = value.strip().lower()
        if lookup in ("true", "yes", "y", "1", "tb related", "tb-related"):
            return True
        if lookup in ("false", "no", "n", "0", "not related", "not tb related"):
            return False
    return None


def _as_confidence(value) -> str | None:
    if isinstance(value, str) and value.strip().lower() in ("high", "medium", "low"):
        return value.strip().lower()
    return None


# ---------------------------------------------------------------------------
# Light mode — one focused LLM call per document
# ---------------------------------------------------------------------------


def parse_verdict_object(answer: str) -> dict | None:
    """Pull a single {"tb_related": ...} verdict out of the reply, prose and all."""
    for data in _load_json_candidates(answer or ""):
        if isinstance(data, dict) and "tb_related" in data:
            return data
        if isinstance(data, list) and data and isinstance(data[0], dict) and "tb_related" in data[0]:
            return data[0]  # model answered in list form anyway
    return None


def _is_rate_limit_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "413" in text or "rate limit" in text or "tpm" in text or "tokens per minute" in text


def classify_one(light_llm, name: str, excerpt: str, strict: bool) -> dict | None:
    """Screen a single document excerpt; None when the reply isn't parseable."""
    # On the strict retry, ask again with a sterner reminder.
    body = (excerpt[:200] + "...") if strict else excerpt
    prompt = SCREEN_DOC_PROMPT.replace("{name}", name).replace("{DOC}", body)
    if strict:
        prompt += "\n\nREMINDER: reply with ONLY the JSON object, no other text."
    reply = light_llm.invoke(prompt)
    content = getattr(reply, "content", reply)
    text = content if isinstance(content, str) else "".join(str(block) for block in content or [])
    return parse_verdict_object(text)


def classify_documents_light(light_llm, docs: list[Path], max_chars: int) -> list[dict]:
    """Screen each document with its own small LLM call; waits out rate limits."""
    rows: list[dict] = []
    for number, path in enumerate(docs, start=1):
        print(f"  {number}/{len(docs)} screening {path.name}...", flush=True)
        excerpt = extract_text(path, max_chars)
        verdict: dict = {}
        for attempt in range(2):  # second pass only asks for stricter JSON
            try:
                for retry in range(3):  # wait-and-retry on Groq rate limits
                    try:
                        verdict = classify_one(light_llm, path.name, excerpt, strict=attempt > 0) or {}
                        break
                    except Exception as e:
                        if not _is_rate_limit_error(e) or retry == 2:
                            raise
                        print(f"    rate limited; waiting {RATE_LIMIT_WAIT_SECONDS}s "
                              f"and retrying", flush=True)
                        time.sleep(RATE_LIMIT_WAIT_SECONDS)
                if verdict:
                    break
            except Exception as e:
                print(f"    ! screening call failed: {e}")
                break

        row = {
            "file": path.name,
            "path": str(path),
            "tb_related": _as_bool(verdict.get("tb_related")),
            "confidence": _as_confidence(verdict.get("confidence")),
            "reason": str(verdict.get("reason", "")).strip()
            or ("no parseable verdict returned" if not verdict else "reason missing"),
        }
        rows.append(row)
        print(f"    {verdict_line(row)}", flush=True)
    return rows


# ---------------------------------------------------------------------------
# Agent mode — the full Biomni A1 loop, in batches
# ---------------------------------------------------------------------------


def parse_verdict_list(answer: str, count: int) -> dict[int, dict]:
    """Pull {id: verdict-object} out of an agent batch reply, prose and all."""
    for source in _json_candidate_strings(answer):
        data = None
        for blob in (source, source.strip()):
            try:
                data = json.loads(blob)
                break
            except ValueError:
                continue
        if isinstance(data, dict):
            for key in ("documents", "verdicts", "results", "data"):
                if isinstance(data.get(key), list):
                    data = data[key]
                    break
        if not isinstance(data, list):
            continue

        verdicts: dict[int, dict] = {}
        for index, obj in enumerate(data):
            if not isinstance(obj, dict):
                continue
            try:
                ident = int(obj.get("id", obj.get("doc_id", index)))
            except (TypeError, ValueError):
                continue
            if 0 <= ident < count:
                verdicts[ident] = obj
        if verdicts:
            return verdicts
    return {}


def classify_documents_agent(agent, docs: list[Path], batch_size: int, max_chars: int) -> list[dict]:
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
        except Exception as e:
            print(f"  ! agent call {batch_number}/{len(batches)} failed, marking batch unknown: {e}")
            if _is_rate_limit_error(e):
                print("    hint: A1's system prompt is huge; try --batch-size 1, or drop --agent")
            answer = ""

        verdicts = parse_verdict_list(answer or "", len(batch))
        if len(verdicts) < len(batch):  # one retry, with a sterner reminder
            try:
                strict = prompt + "\n\nREMINDER: reply with ONLY the JSON list, no other text."
                _log, answer = agent.go(strict)
                verdicts = parse_verdict_list(answer or "", len(batch))
            except Exception as e:
                print(f"  ! retry failed, leaving unparsed docs as unknown: {e}")

        print(f"Batch {batch_number}/{len(batches)}:")
        for index, path in enumerate(batch):
            verdict = verdicts.get(index, {})
            reason = str(verdict.get("reason", "")).strip()
            if not verdicts:
                reason = "agent verdict could not be parsed"
            rows.append(
                {
                    "file": path.name,
                    "path": str(path),
                    "tb_related": _as_bool(verdict.get("tb_related")),
                    "confidence": _as_confidence(verdict.get("confidence")),
                    "reason": reason,
                }
            )
            print(f"  {verdict_line(rows[-1])}")
        print()

    return rows


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def verdict_line(row: dict) -> str:
    if row["tb_related"] is True:
        verdict = "TB"
    elif row["tb_related"] is False:
        verdict = "--"
    else:
        verdict = "??"
    reason = row["reason"] or "no reason given"
    confidence = f" ('{row['confidence']}')" if row["confidence"] else ""
    return f"[{verdict}] {row['file']}{confidence} — {reason}"


def write_results(rows: list[dict], settings: Settings, input_path: Path, output: Path, mode: str) -> None:
    counts = {"related": 0, "not_related": 0, "unknown": 0}
    for row in rows:
        if row["tb_related"] is True:
            counts["related"] += 1
        elif row["tb_related"] is False:
            counts["not_related"] += 1
        else:
            counts["unknown"] += 1

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": f"{settings.source}/{settings.model}",
        "mode": mode,
        "input": str(input_path),
        "summary": counts,
        "results": rows,
    }
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"Verdicts: {counts['related']} TB-related, {counts['not_related']} not, "
          f"{counts['unknown']} unknown → saved to {output}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tb-agentic-search",
        description="Detect TB-related research documents with a biomni-powered LLM on Groq.",
    )
    parser.add_argument("input", nargs="?", help="Directory or file to screen. Defaults to ./papers, "
                                                 "or ../papers when ./papers does not exist.")
    parser.add_argument("-o", "--output", default="tb_verdicts.json", help="Where to write the JSON verdicts.")
    parser.add_argument("--model", help=f"Groq model id (default: $GROQ_MODEL or {DEFAULT_MODEL}).")
    parser.add_argument("--agent", action="store_true",
                        help="Use the full biomni A1 agent loop instead of small per-doc LLM calls. "
                             "A1's system prompt is large; a free Groq account will be rate limited.")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="[--agent] Documents per agent call (default: 2).")
    parser.add_argument("--max-chars", type=int, default=EXCERPT_CHARS, help="Excerpt length per document.")
    parser.add_argument("--tools", action="store_true",
                        help="[--agent] Let the A1 agent retrieve a relevant tool subset.")
    parser.add_argument("--data-lake", action="store_true",
                        help="[--agent] Let biomni download its full data lake (tens of GB).")
    parser.add_argument("--timeout", type=int,
                        help="[--agent] Per-execution timeout in seconds.")
    parser.add_argument("--temperature", type=float, help="Sampling temperature.")
    parser.add_argument("--scan-only", action="store_true", help="List docs and preview excerpts; no LLM call.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    mode = "agent" if args.agent else "light"

    try:
        settings = load_settings(
            model=args.model,
            temperature=args.temperature,
            timeout_seconds=args.timeout,
            use_tool_retriever=True if args.tools else None,
            load_data_lake=True if args.data_lake else None,
        )
    except ConfigError as e:
        print(e, file=sys.stderr)
        return 1

    input_path = resolve_input(args.input)
    if input_path is None:
        print(f"Input not found: {args.input!r}" if args.input
              else "No papers directory found; pass one, e.g. `python main.py ../papers`", file=sys.stderr)
        return 1
    if input_path.is_dir() and not collect_docs(input_path):
        print(f"No .pdf/.txt/.md/... documents found under {input_path}", file=sys.stderr)
        return 1

    docs = collect_docs(input_path)
    print(f"Screening {len(docs)} document(s) under {input_path} "
          f"[mode: {mode}, {settings.source} {settings.model}]\n")

    if args.scan_only:
        for path in docs:
            preview = extract_text(path, 200) or "(no text)"
            print(f"  {path.name} — {preview}...")
        return 0

    try:
        check_connection(settings)
    except ConfigError as e:
        print(e, file=sys.stderr)
        return 1

    if mode == "agent":
        agent = build_agent(settings)
        rows = classify_documents_agent(agent, docs, max(1, args.batch_size), max(200, args.max_chars))
    else:
        screener = build_screener(settings)
        rows = classify_documents_light(screener, docs, max(200, args.max_chars))

    write_results(rows, settings, input_path, Path(args.output), mode)
    return 0


if __name__ == "__main__":
    sys.exit(main())
