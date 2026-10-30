"""Console summary line and the JSON results file."""

import json
from datetime import datetime
from pathlib import Path

from tb_search.config import Settings


def verdict_line(row: dict) -> str:
    if row["tb_related"] is True:
        verdict = "TB"
    elif row["tb_related"] is False:
        verdict = "--"
    else:
        verdict = "??"
    confidence = f" ('{row['confidence']}')" if row["confidence"] else ""
    return (
        f"[{verdict}] {row['file']}{confidence} — {row['reason'] or 'no reason given'}"
    )


def write_results(
    rows: list[dict], settings: Settings, input_path: Path, output: Path, mode: str
) -> None:
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
    print(
        f"Verdicts: {counts['related']} TB-related, {counts['not_related']} not, "
        f"{counts['unknown']} unknown → saved to {output}"
    )
