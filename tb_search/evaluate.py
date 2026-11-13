"""Measure the harness: retrieval recall against a held-out reference list, screening accuracy."""

import json
import re
from pathlib import Path

from tb_search.discovery import DiscoveryConfig, discover
from tb_search.store import Store

TB_TITLE = re.compile(r"tubercul|mycobacter|\bM?TB\b", re.IGNORECASE)
CUTOFFS = (10, 25, 50, 100)


def recall_at(ranked: list[str], gold: set[str], cutoffs=CUTOFFS) -> dict[str, float | None]:
    """recall@k per cutoff, plus "pool": recall over everything retrieved."""
    if not gold:
        return {**{f"@{k}": None for k in cutoffs}, "pool": None}
    scores = {f"@{k}": round(len(gold & set(ranked[:k])) / len(gold), 3) for k in cutoffs}
    scores["pool"] = round(len(gold & set(ranked)) / len(gold), 3)
    return scores


def evaluate_retrieval(
    source, store: Store, seed_query: str, llm=None, model: str = "",
    config: DiscoveryConfig | None = None, log=print,
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
    run = discover(source, store, seed_query, llm, model, config, log)
    ranked = [candidate.paper.key for candidate in run.ranked()]

    return {
        "seed": {"key": seed.key, "title": seed.title},
        "gold": {"all": len(gold), "tb": len(tb_gold)},
        "retrieved": len(ranked),
        "recall": {"all": recall_at(ranked, gold), "tb": recall_at(ranked, tb_gold)},
        "missed_tb": sorted(
            paper.title for paper in references
            if paper.key in tb_gold and paper.key not in set(ranked)
        ),
        "queries": run.queries,
    }


def evaluate_screening(rows: list[dict], gold_path: Path) -> dict:
    """Compare screening rows with hand labels: a JSON object of {filename: tb_related}."""
    gold = json.loads(gold_path.read_text())
    judged = [row for row in rows if row["file"] in gold]
    wrong = [
        row["file"] for row in judged
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
