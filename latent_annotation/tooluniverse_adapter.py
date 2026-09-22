from __future__ import annotations

from typing import Any

from .schema import ToolCall
from .tools import parse_mutation

ACCESSION = "P31689"  # reviewed human DNAJA1 entry


class ToolUniverseUnavailable(RuntimeError):
    """Raised when the live CLI is requested without ToolUniverse installed."""


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)


def _find_string(value: Any, names: tuple[str, ...]) -> str | None:
    if isinstance(value, str) and value:
        return value
    if isinstance(value, dict):
        for name in names:
            candidate = value.get(name)
            if isinstance(candidate, str) and candidate:
                return candidate
        for child in value.values():
            found = _find_string(child, names)
            if found:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_string(child, names)
            if found:
                return found
    return None


def fetch_dnaj1(protein: str, mutation: str) -> tuple[dict[str, Any], list[ToolCall]]:
    """Fetch UniProt and InterPro evidence through ToolUniverse's Python API."""
    if protein.upper() != "DNAJA1":
        raise ValueError("the live prototype currently supports DNAJA1 only")
    position, wild_type, replacement = parse_mutation(mutation)
    try:
        from tooluniverse import ToolUniverse
    except ImportError as exc:
        raise ToolUniverseUnavailable(
            "ToolUniverse is not installed in this Python environment. "
            "Run the repository with its existing uv environment, then retry."
        ) from exc

    tu = ToolUniverse()
    tu.load_tools(
        include_tools=[
            "UniProt_search",
            "UniProt_get_sequence_by_accession",
            "UniProt_get_features_by_accession",
            "InterPro_get_protein_domains",
            "InterPro_get_residue_annotations",
        ],
        quiet=True,
    )

    def run(name: str, arguments: dict[str, Any]) -> tuple[Any, ToolCall]:
        result = tu.run({"name": name, "arguments": arguments})
        return result, ToolCall(name, arguments, _jsonable(result))

    search, search_call = run(
        "UniProt_search",
        {"query": "gene:DNAJA1 AND organism_id:9606 AND reviewed:true", "limit": 5},
    )
    sequence_result, sequence_call = run(
        "UniProt_get_sequence_by_accession", {"accession": ACCESSION}
    )
    features, features_call = run(
        "UniProt_get_features_by_accession", {"accession": ACCESSION}
    )
    domains, domains_call = run(
        "InterPro_get_protein_domains", {"protein_id": ACCESSION}
    )
    residues, residues_call = run(
        "InterPro_get_residue_annotations", {"protein_id": ACCESSION}
    )

    sequence = _find_string(sequence_result, ("sequence", "value"))
    if not sequence:
        raise ValueError("ToolUniverse returned no sequence for DNAJA1/P31689")
    sequence = "".join(sequence.split()).upper()
    if position > len(sequence):
        raise ValueError(f"UniProt {ACCESSION} has no residue at position {position}")
    observed = sequence[position - 1]
    numbering_warning = None
    if observed != wild_type:
        numbering_warning = (
            f"Mutation uses {wild_type}{position}{replacement}, but UniProt {ACCESSION} "
            f"has {observed}{position}; this is likely a construct/UniProt numbering mismatch."
        )

    motif_spans: dict[str, list[int]] = {}
    for index in range(len(sequence) - 2):
        if sequence[index : index + 3] == "HPD":
            motif_spans["HPD"] = [index + 1, index + 2, index + 3]
            break
    result = {
        "protein": protein.upper(),
        "accession": ACCESSION,
        "mutation": f"{wild_type}{position}{replacement}",
        "sequence_length": len(sequence),
        "wild_type": wild_type,
        "uniprot_observed_residue": observed,
        "numbering_warning": numbering_warning,
        "structure": "2LO1 (plan reference; contacts not computed by this slice)",
        "chain": "A",
        "contacts_within_5A": [position],
        "motif_spans": motif_spans,
        "uniprot_search": _jsonable(search),
        "features": _jsonable(features),
        "interpro_domains": _jsonable(domains),
        "interpro_residue_annotations": _jsonable(residues),
        "status": "live UniProt/InterPro evidence via ToolUniverse; structural contacts pending",
    }
    return result, [search_call, sequence_call, features_call, domains_call, residues_call]
