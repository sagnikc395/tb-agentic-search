"""Console summary line and the JSON results files."""

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
        "model": settings.label,
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


def paper_line(row: dict) -> str:
    """One console entry for a found paper: title line plus link and relevance."""
    year = f" ({row['year']})" if row.get("year") else ""
    source = f" [{row['source']}]" if row.get("source") else ""
    tail = "".join(
        f"\n      {row[key]}" for key in ("link", "relevance") if row.get(key)
    )
    return f"{row['title']}{year}{source}{tail}"


def discovered_rows(candidates) -> list[dict]:
    """Discovery candidates as paper rows: the --search shape plus routes and assessment."""
    rows = []
    for candidate in candidates:
        paper, assessment = candidate.paper, candidate.assessment or {}
        rows.append(
            {
                "title": paper.title,
                "year": paper.year,
                "link": paper.link,
                "source": paper.source,
                "relevance": assessment.get("reason", ""),
                "relevance_score": candidate.relevance,
                "routes": candidate.routes,
                "assessment": assessment or None,
            }
        )
    return rows


def write_papers(rows: list[dict], model: str, query: str, output: Path) -> None:
    """JSON for a --search run: the found papers plus what was asked."""
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": model,
        "mode": "search",
        "query": query,
        "papers": rows,
    }
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    print(
        f"\nPapers related to the query ({len(rows)}):"
        if rows
        else "\nNo papers found."
    )
    for number, row in enumerate(rows, start=1):
        print(f"  {number:>2}. {paper_line(row)}")
    print(f"\nSaved to {output}")


def write_cohort(cohort, output: Path) -> None:
    """JSON for a --cohort run: per-drug dataset sizes, and how the join went."""
    rows = cohort.summary()
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "isolates": len(cohort.isolates),
        "unmatched_rows": cohort.unmatched_rows,
        "unmatched_vcfs": cohort.unmatched_vcfs,
        "drugs": rows,
    }
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    print(
        f"{len(cohort.isolates)} isolate(s) with a VCF; "
        f"{cohort.unmatched_rows} table row(s) had none.\n"
    )
    print(f"  {'drug':<26}{'R':>7}{'S':>7}{'total':>8}")
    for row in rows:
        print(
            f"  {row['drug']:<26}{row['resistant']:>7}"
            f"{row['susceptible']:>7}{row['total']:>8}"
        )
    print(f"\nSaved to {output}")


def write_predictions(
    rows: list[dict],
    scores: dict,
    model: str,
    drug: str,
    output: Path,
    reasoning_effort: str | None = None,
) -> None:
    """JSON for a --predict run: every prediction with its variants and trace, plus scores."""
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": model,
        "mode": "predict",
        "drug": drug,
        # Recorded because the traces are only comparable across runs at the
        # same effort: it is what the model was asked to spend on thinking.
        "reasoning_effort": reasoning_effort,
        "scores": scores,
        "predictions": rows,
    }
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    confusion, trace = scores["confusion"], scores["trace"]
    print(f"\n{drug}: {scores['answered']} answered of {scores['n']}", end="")
    if scores["unknown"]:
        print(f", {scores['unknown']} unknown", end="")
    if scores["errors"]:
        print(f", {scores['errors']} failed", end="")
    print()
    mismatched = sum(row.get("ref_mismatches") or 0 for row in rows)
    if mismatched:
        print(
            f"  ! {mismatched} record(s) had a REF that disagrees with H37Rv — "
            "check the VCFs were called against NC_000962.3"
        )
    print(
        f"  accuracy {scores['accuracy']}  sensitivity {scores['sensitivity']}  "
        f"specificity {scores['specificity']}  ppv {scores['ppv']}  npv {scores['npv']}"
    )
    print(
        f"  confusion: R->R {confusion['true_resistant_called_resistant']}  "
        f"R->S {confusion['true_resistant_called_susceptible']}  "
        f"S->R {confusion['true_susceptible_called_resistant']}  "
        f"S->S {confusion['true_susceptible_called_susceptible']}"
    )
    print(
        f"  traces: {trace['valid']} valid of {trace['graded']} graded  "
        f"(hallucinated variants {trace['hallucinated_variants']}, "
        f"off-panel genes {trace['off_panel_genes']}, "
        f"ungrounded R calls {trace['ungrounded_resistant_calls']}, "
        f"tier-1 support {trace['tier1_support']})"
    )
    captured = trace.get("thinking_captured")
    if captured:
        print(
            f"  thinking: captured for {captured} of graded calls, "
            f"median {trace.get('median_thinking_words')} words"
        )
    else:
        print(
            "  thinking: none captured — the traces are only what the model wrote "
            "into its answer. Serve the model with a vLLM reasoning parser "
            "(--reasoning-parser) to record its thinking too."
        )
    print(f"\nSaved to {output}")
