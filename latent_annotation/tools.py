from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

from .schema import CATEGORIES, LatentRole, ToolCall


# A deliberately small fixture: it lets the architecture be tested before GPU/data work.
PROTEINS = {
    "DNAJA1": {
        "sequence": "MSTAGKL AKD AAKKAGVEAG DDDDDHPDLA VDQVEKAAE AAGVEAKKLA LAA".replace(" ", ""),
        "structure": "2LO1",
        "chain": "A",
        "motif_spans": {"HPD": [30, 31, 32]},
    }
}


def parse_mutation(mutation: str) -> tuple[int, str, str]:
    mutation = mutation.strip().upper()
    if len(mutation) < 3 or not mutation[0].isalpha() or not mutation[-1].isalpha() or not mutation[1:-1].isdigit():
        raise ValueError("mutation must look like L7D")
    position = int(mutation[1:-1])
    if position < 1:
        raise ValueError("mutation position must be positive")
    return position, mutation[0], mutation[-1]


def _call(name: str, arguments: dict[str, Any], result: dict[str, Any]) -> ToolCall:
    return ToolCall(name=name, arguments=arguments, result=result)


def ground(protein: str, mutation: str) -> tuple[dict[str, Any], ToolCall]:
    protein = protein.upper()
    if protein not in PROTEINS:
        raise ValueError(f"no offline fixture for {protein}; use DNAJA1 for the prototype")
    position, wild_type, replacement = parse_mutation(mutation)
    record = PROTEINS[protein]
    sequence = record["sequence"]
    observed = sequence[position - 1] if position <= len(sequence) else "?"
    if observed != wild_type:
        raise ValueError(
            f"fixture says position {position} is {observed}, not {wild_type}; "
            "the prototype refuses to guess"
        )
    # Synthetic contact set stands in for the future PDB heavy-atom calculation.
    contacts = sorted({position - 1, position, position + 1, 22, 48, 51})
    result = {
        "protein": protein,
        "mutation": f"{wild_type}{position}{replacement}",
        "sequence_length": len(sequence),
        "structure": record["structure"],
        "chain": record["chain"],
        "wild_type": wild_type,
        "contacts_within_5A": contacts,
        "motif_spans": record["motif_spans"],
        "status": "synthetic fixture; replace with PDB/UniProt tools in Stage 1",
    }
    return result, _call("ground_structure", {"protein": protein, "mutation": mutation}, result)


def score(mutation: str) -> tuple[dict[str, Any], ToolCall]:
    position, wild_type, replacement = parse_mutation(mutation)
    # A stable mock score makes loop/report development reproducible.
    score_value = round(-0.35 - (abs(ord(replacement) - ord(wild_type)) / 100), 3)
    result = {"mutation": f"{wild_type}{position}{replacement}", "masked_marginal_llr": score_value, "source": "mock"}
    return result, _call("mock_masked_marginal_llr", {"mutation": mutation}, result)


def _category(position: int, residue: int, contacts: set[int], motif: set[int], window: int) -> str:
    if residue == position:
        return "mutation_site"
    if residue in contacts:
        return "3d_contact"
    if abs(residue - position) <= window:
        return "sequence_local"
    if residue in motif:
        return "hpd_motif"
    return "other_residues"


def categorize_latent(
    latent_id: str,
    attribution: dict[int, float],
    contacts: Iterable[int],
    mut_site: int,
    motif_spans: dict[str, Iterable[int]],
    seq_local_window: int = 5,
) -> LatentRole:
    """Assign attribution mass using the plan's deterministic H.3 priority order."""
    if not attribution or any(value < 0 for value in attribution.values()):
        raise ValueError("attribution must contain non-negative values")
    total = sum(attribution.values())
    if total <= 0:
        return LatentRole(latent_id, "unresolved_diffuse", {c: 0.0 for c in CATEGORIES}, [], 1.0, "categorize_latent")
    contact_set = set(contacts)
    motif_set = {residue for span in motif_spans.values() for residue in span}
    masses = {category: 0.0 for category in CATEGORIES}
    by_category: dict[str, list[int]] = {category: [] for category in CATEGORIES}
    for residue, value in attribution.items():
        category = _category(mut_site, residue, contact_set, motif_set, seq_local_window)
        masses[category] += value / total
        by_category[category].append(residue)
    entropy = -sum(mass * math.log(mass) for mass in masses.values() if mass > 0)
    diffuseness = entropy / math.log(len(CATEGORIES)) if len(CATEGORIES) > 1 else 0.0
    category = max(CATEGORIES[:-1], key=lambda item: masses[item])
    if diffuseness >= 0.82:
        category = "unresolved_diffuse"
    supporting = sorted(by_category.get(category, []))
    return LatentRole(latent_id, category, {k: round(v, 4) for k, v in masses.items()}, supporting, round(diffuseness, 4), "categorize_latent")


def mock_latents(mut_site: int) -> list[dict[str, Any]]:
    """Return three interpretable attributions representing a tiny circuit."""
    return [
        {"latent_id": "L-contact-01", "attribution": {mut_site: 0.2, 22: 0.6, 48: 0.2}},
        {"latent_id": "L-local-02", "attribution": {mut_site - 2: 0.5, mut_site: 0.3, mut_site + 2: 0.2}},
        {"latent_id": "L-motif-03", "attribution": {30: 0.7, 31: 0.2, 48: 0.1}},
    ]
