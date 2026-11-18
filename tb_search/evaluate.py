"""Measure the harness: retrieval recall against a held-out reference list, screening accuracy."""

import json
import re
from pathlib import Path

from tb_search.discovery import DiscoveryConfig, discover

TB_TITLE = re.compile(r"tubercul|mycobacter|\bM?TB\b", re.IGNORECASE)
CUTOFFS = (10, 25, 50, 100)


def recall_at(
    ranked: list[str], gold: set[str], cutoffs=CUTOFFS
) -> dict[str, float | None]:
    """recall@k per cutoff, plus "pool": recall over everything retrieved."""
    if not gold:
        return {**{f"@{k}": None for k in cutoffs}, "pool": None}
    scores = {
        f"@{k}": round(len(gold & set(ranked[:k])) / len(gold), 3) for k in cutoffs
    }
    scores["pool"] = round(len(gold & set(ranked)) / len(gold), 3)
    return scores


def evaluate_retrieval(
    source,
    seed_query: str,
    llm=None,
    config: DiscoveryConfig | None = None,
    log=print,
) -> dict | None:
    """Hide the seed's reference list, run discovery, and count how much of it comes back.

    The references a paper's authors chose are a free, expert-picked gold set.
    "tb" narrows it to the references that are about TB by title, since a
    methods paper also cites plenty that a TB search should not return.
    None when the seed can't be resolved or has no matched references.
    """
    seed = source.resolve(seed_query)
    if seed is None:
        return None
    references = source.references(seed)
    if not references:
        return None
    gold = {paper.key for paper in references}
    tb_gold = {paper.key for paper in references if TB_TITLE.search(paper.title)}

    config = config or DiscoveryConfig()
    config.use_seed_references = False
    run = discover(source, seed_query, llm, config, log)
    ranked = [candidate.paper.key for candidate in run.ranked()]

    return {
        "seed": {"key": seed.key, "title": seed.title},
        "gold": {"all": len(gold), "tb": len(tb_gold)},
        "retrieved": len(ranked),
        "recall": {"all": recall_at(ranked, gold), "tb": recall_at(ranked, tb_gold)},
        "missed_tb": sorted(
            paper.title
            for paper in references
            if paper.key in tb_gold and paper.key not in set(ranked)
        ),
        "queries": run.queries,
    }


def evaluate_screening(rows: list[dict], gold_path: Path) -> dict:
    """Compare screening rows with hand labels: a JSON object of {filename: tb_related}."""
    gold = json.loads(gold_path.read_text())
    judged = [row for row in rows if row["file"] in gold]
    wrong = [
        row["file"]
        for row in judged
        if row["tb_related"] is not None and row["tb_related"] != gold[row["file"]]
    ]
    abstained = [row["file"] for row in judged if row["tb_related"] is None]
    answered = len(judged) - len(abstained)
    return {
        "labelled": len(judged),
        "answered": answered,
        "accuracy": round((answered - len(wrong)) / answered, 3) if answered else None,
        "wrong": wrong,
        "abstained": abstained,
        "unverified_evidence": [
            row["file"] for row in judged if row.get("evidence_verified") is False
        ],
    }


def _rate(numerator: int, denominator: int) -> float | None:
    """A rounded rate, or None when there is nothing to divide by."""
    return round(numerator / denominator, 3) if denominator else None


def _median(values: list[int]) -> int | None:
    """The middle value, or None for an empty list. Median, because one runaway
    trace should not move the number that describes a run."""
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) // 2


def evaluate_resistance(rows: list[dict]) -> dict:
    """Score resistance predictions against the DST phenotype, and grade the traces.

    Accuracy is computed over answered rows only, so abstention shows up as a
    lower `answered` rather than as silently wrong calls. The trace block is the
    other half of the picture: a run can be accurate and still be reasoning from
    variants that were never in the VCF.
    """
    calls = {"resistant": "R", "not_resistant": "S"}
    matrix = {"RR": 0, "RS": 0, "SR": 0, "SS": 0}  # truth then call
    unknown = {"R": 0, "S": 0}
    errors = [row for row in rows if row.get("error")]

    for row in rows:
        truth = row.get("phenotype")
        if truth not in ("R", "S"):
            continue
        called = calls.get(row.get("call", ""))
        if called is None:
            unknown[truth] += 1
        else:
            matrix[truth + called] += 1

    answered = sum(matrix.values())
    correct = matrix["RR"] + matrix["SS"]
    resistant = matrix["RR"] + matrix["RS"]
    susceptible = matrix["SS"] + matrix["SR"]

    graded = [row["trace"] for row in rows if isinstance(row.get("trace"), dict)]
    valid = sum(1 for trace in graded if trace.get("valid"))
    cited = [trace for trace in graded if trace.get("cited_variants")]

    return {
        "n": len(rows),
        "answered": answered,
        "unknown": sum(unknown.values()),
        "unknown_by_phenotype": unknown,
        "errors": len(errors),
        "confusion": {
            "true_resistant_called_resistant": matrix["RR"],
            "true_resistant_called_susceptible": matrix["RS"],
            "true_susceptible_called_resistant": matrix["SR"],
            "true_susceptible_called_susceptible": matrix["SS"],
        },
        "accuracy": _rate(correct, answered),
        # Sensitivity is the number that matters clinically: a missed resistant
        # isolate is a patient put on a drug that will not work.
        "sensitivity": _rate(matrix["RR"], resistant),
        "specificity": _rate(matrix["SS"], susceptible),
        "ppv": _rate(matrix["RR"], matrix["RR"] + matrix["SR"]),
        "npv": _rate(matrix["SS"], matrix["SS"] + matrix["RS"]),
        "trace": {
            "graded": len(graded),
            "valid": _rate(valid, len(graded)),
            "hallucinated_variants": _rate(
                sum(1 for trace in graded if not trace.get("variants_resolve")),
                len(graded),
            ),
            "off_panel_genes": _rate(
                sum(1 for trace in graded if not trace.get("genes_on_panel")),
                len(graded),
            ),
            "ungrounded_resistant_calls": _rate(
                sum(1 for trace in graded if not trace.get("call_grounded")),
                len(graded),
            ),
            "tier1_support": _rate(
                sum(1 for trace in cited if trace.get("tier1_support")), len(cited)
            ),
            # How many calls came with the model's own thinking attached, and
            # how long it ran: with no separated trace there is nothing to read
            # through for biological validity.
            "thinking_captured": _rate(
                sum(1 for trace in graded if trace.get("thinking_captured")),
                len(graded),
            ),
            "median_thinking_words": _median(
                [
                    trace.get("thinking_words") or 0
                    for trace in graded
                    if trace.get("thinking_captured")
                ]
            ),
        },
        "misses": [
            {
                "isolate": row["isolate"],
                "phenotype": row["phenotype"],
                "call": row["call"],
                "reasoning": row.get("reasoning", ""),
            }
            for row in rows
            if row.get("phenotype") in ("R", "S")
            and calls.get(row.get("call", "")) not in (None, row["phenotype"])
        ],
    }
