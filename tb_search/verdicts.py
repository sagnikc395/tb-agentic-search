"""Salvage verdicts and paper lists out of an LLM reply, whatever shape it came back in."""

import json
import re

# Wrapper keys a model puts its list under instead of returning a bare list,
# ordered by precedence for the parses that share this helper.
LIST_KEYS = ("papers", "documents", "verdicts", "results", "data")
TRUE_WORDS = frozenset({"true", "yes", "y", "1", "tb related", "tb-related"})
FALSE_WORDS = frozenset({"false", "no", "n", "0", "not related", "not tb related"})


def _json_candidates(answer: str) -> list[str]:
    """Likely JSON blobs in the reply: fenced block, bare list, bare object, whole reply."""
    candidates = []
    fenced = re.search(r"```(?:json)?\s*(.*?)```", answer, re.DOTALL)
    if fenced:
        candidates.append(fenced.group(1))
    for opener, closer in (("[", "]"), ("{", "}")):
        first, last = answer.find(opener), answer.rfind(closer)
        if first != -1 and last > first:
            candidates.append(answer[first : last + 1])
    candidates.append(answer)
    return candidates


def _load_json(answer: str):
    for blob in _json_candidates(answer):
        try:
            yield json.loads(blob)
        except ValueError:
            continue


def _as_list(data) -> list | None:
    """The list a reply blob stands for: a known wrapper key, else any list it holds."""
    if isinstance(data, list):
        return data
    if not isinstance(data, dict):
        return None
    for key in LIST_KEYS:
        if isinstance(data.get(key), list):
            return data[key]
    return next((value for value in data.values() if isinstance(value, list)), None)


def as_bool(value) -> bool | None:
    """Coerce whatever the model said into a real bool; None when undecidable."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        wording = value.strip().lower()
        if wording in TRUE_WORDS:
            return True
        if wording in FALSE_WORDS:
            return False
    return None


def as_confidence(value) -> str | None:
    if isinstance(value, str) and value.strip().lower() in ("high", "medium", "low"):
        return value.strip().lower()
    return None


def parse_assessment(answer: str) -> dict | None:
    """Pull a single {"relevance": ...} assessment out of the reply, prose and all."""
    for data in _load_json(answer or ""):
        for item in _as_list(data) or [data]:
            if isinstance(item, dict) and ("relevance" in item or "abstain" in item):
                return item
    return None


def parse_string_list(answer: str) -> list[str]:
    """Pull a JSON list of strings (search queries) out of the reply; [] when none."""
    for data in _load_json(answer or ""):
        found = [
            item.strip()
            for item in _as_list(data) or []
            if isinstance(item, str) and item.strip()
        ]
        if found:
            return found
    return []


def _paper_row(obj: dict) -> dict | None:
    """One paper row; None when the object carries no title."""
    title = str(obj.get("title", "")).strip()
    if not title:  # a paperless row is noise
        return None
    link = obj.get("link") or obj.get("url") or obj.get("doi") or ""
    return {
        "title": title,
        "year": str(obj.get("year", "")).strip(),
        "link": str(link).strip(),
        "source": str(obj.get("source", "")).strip(),
        "relevance": str(obj.get("relevance", "")).strip(),
    }


def parse_paper_list(answer: str) -> list[dict]:
    """Pull paper rows out of an agent reply, prose and all; [] when nothing fits."""
    for data in _load_json(answer or ""):
        rows = []
        for listed in _as_list(data) or []:
            # a row may wrap its papers under a key of its own
            for obj in _as_list(listed) or [listed]:
                row = _paper_row(obj) if isinstance(obj, dict) else None
                if row:
                    rows.append(row)
        if rows:
            return rows
    return []


def parse_verdict_list(answer: str, count: int) -> dict[int, dict]:
    """Pull {id: verdict-object} out of an agent batch reply, prose and all."""
    for data in _load_json(answer or ""):
        verdicts: dict[int, dict] = {}
        for index, obj in enumerate(_as_list(data) or []):
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
