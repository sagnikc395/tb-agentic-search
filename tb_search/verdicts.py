"""Salvage a verdict out of an LLM reply, whatever shape it came back in."""

import json
import re


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


def as_bool(value) -> bool | None:
    """Coerce whatever the model said into a real bool; None when undecidable."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        wording = value.strip().lower()
        if wording in ("true", "yes", "y", "1", "tb related", "tb-related"):
            return True
        if wording in ("false", "no", "n", "0", "not related", "not tb related"):
            return False
    return None


def as_confidence(value) -> str | None:
    if isinstance(value, str) and value.strip().lower() in ("high", "medium", "low"):
        return value.strip().lower()
    return None


def parse_verdict_object(answer: str) -> dict | None:
    """Pull a single {"tb_related": ...} verdict out of the reply, prose and all."""
    for data in _load_json(answer or ""):
        if isinstance(data, dict) and "tb_related" in data:
            return data
        if (
            isinstance(data, list)
            and data
            and isinstance(data[0], dict)
            and "tb_related" in data[0]
        ):
            return data[0]  # the model answered in list form anyway
    return None


def parse_verdict_list(answer: str, count: int) -> dict[int, dict]:
    """Pull {id: verdict-object} out of an agent batch reply, prose and all."""
    for found in _load_json(answer or ""):
        if isinstance(found, dict):
            for key in ("documents", "verdicts", "results", "data"):
                if isinstance(found.get(key), list):
                    found = found[key]
                    break
        if not isinstance(found, list):
            continue

        verdicts: dict[int, dict] = {}
        for index, obj in enumerate(found):
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
