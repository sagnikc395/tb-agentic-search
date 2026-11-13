"""Structured screening: graded relevance, facets, a checked evidence quote, and abstention."""

import re
from pathlib import Path

from tb_search.documents import read_text, section_excerpt
from tb_search.prompts import ASSESS_PROMPT, DEFAULT_TOPIC
from tb_search.report import verdict_line
from tb_search.screening import _wait_out_rate_limit
from tb_search.verdicts import as_bool, parse_assessment

PAPER_TYPES = ("method", "benchmark", "dataset", "review", "clinical", "agent", "other")
FACETS = ("drugs", "genes_variants", "methods", "datasets")


def _reply_text(reply) -> str:
    content = getattr(reply, "content", reply)
    if isinstance(content, str):
        return content
    return "".join(str(block) for block in content or [])


def _squash(text: str) -> str:
    """Lowercased alphanumerics only, so a quote survives re-spacing and re-punctuation."""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _strings(value) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def normalize_assessment(data: dict, text: str) -> dict:
    """Coerce a parsed reply into the one assessment shape, checking the quote against text.

    relevance is None on abstention — "don't know" has to stay distinguishable
    from "not relevant".
    """
    abstain = as_bool(data.get("abstain")) is True
    try:
        relevance = min(3, max(0, int(data.get("relevance"))))
    except (TypeError, ValueError):
        relevance, abstain = None, True
    if abstain:
        relevance = None

    paper_type = str(data.get("paper_type", "")).strip().lower()
    evidence = str(data.get("evidence", "")).strip()
    quote = _squash(evidence)
    assessment = {
        "relevance": relevance,
        "tb_central": None if abstain else as_bool(data.get("tb_central")),
        "abstain": abstain,
        "paper_type": paper_type if paper_type in PAPER_TYPES else "other",
        "organism": str(data.get("organism", "")).strip(),
        "evidence": evidence,
        # A quote the text doesn't contain is the model talking, not the paper.
        "evidence_verified": bool(quote) and quote in _squash(text),
        "reason": str(data.get("reason", "")).strip(),
    }
    for facet in FACETS:
        assessment[facet] = _strings(data.get(facet))
    return assessment


def assess(llm, name: str, text: str, topic: str = DEFAULT_TOPIC) -> dict | None:
    """Assess one text against the topic; None when no reply could be parsed."""
    # .format would eat the JSON braces inside the prompt example; plain replace.
    prompt = (
        ASSESS_PROMPT.replace("{TOPIC}", topic).replace("{name}", name).replace("{DOC}", text)
    )
    for strict in (False, True):  # second pass only asks for stricter JSON
        if strict:
            prompt += "\n\nREMINDER: reply with ONLY the JSON object, no other text."
        reply = _wait_out_rate_limit(llm.invoke, prompt)
        data = parse_assessment(_reply_text(reply))
        if data:
            return normalize_assessment(data, text)
    return None


def assess_documents(llm, docs: list[Path], max_chars: int) -> list[dict]:
    """Screen each local document with its own small LLM call; waits out rate limits."""
    rows: list[dict] = []
    for number, path in enumerate(docs, start=1):
        print(f"  {number}/{len(docs)} screening {path.name}...", flush=True)
        excerpt = section_excerpt(read_text(path), max_chars)
        try:
            assessment = assess(llm, path.name, excerpt)
        except Exception as e:  # noqa: BLE001 - one failed doc must not stop the rest
            print(f"    ! screening call failed: {e}")
            assessment = None

        row = {
            "file": path.name,
            "path": str(path),
            "tb_related": assessment["tb_central"] if assessment else None,
            "confidence": None,
            "reason": (assessment or {}).get("reason") or "no parseable verdict returned",
            **{k: v for k, v in (assessment or {}).items() if k not in ("tb_central", "reason")},
        }
        rows.append(row)
        print(f"    {verdict_line(row)}", flush=True)
    return rows
