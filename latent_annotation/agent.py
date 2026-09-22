from __future__ import annotations

import json
from collections import Counter
from typing import Any

from .schema import ToolCall
from .tools import categorize_latent, ground, mock_latents, parse_mutation, score
from .tooluniverse_adapter import fetch_dnaj1


def explain(protein: str, mutation: str, use_tooluniverse: bool = False) -> dict[str, Any]:
    """Run the prototype loop, optionally grounding it with live ToolUniverse calls."""
    if use_tooluniverse:
        grounding, grounding_calls = fetch_dnaj1(protein, mutation)
    else:
        grounding, ground_call = ground(protein, mutation)
        grounding_calls = [ground_call]
    scoring, score_call = score(mutation)
    position, _, _ = parse_mutation(mutation)
    roles = [
        categorize_latent(
            latent["latent_id"],
            latent["attribution"],
            grounding["contacts_within_5A"],
            position,
            grounding["motif_spans"],
        )
        for latent in mock_latents(position)
    ]
    annotation_call = ToolCall(
        name="categorize_latents",
        arguments={"latent_count": len(roles), "mutation": mutation},
        result={"roles": [role.as_dict() for role in roles]},
    )
    counts = Counter(role.category for role in roles)
    claims = [
        {
            "claim": f"The prototype implicates {counts.get('3d_contact', 0)} contact-enriched latent(s).",
            "supported_by": ["categorize_latents"],
        },
        {
            "claim": f"The mutation has a mock masked-marginal LLR of {scoring['masked_marginal_llr']}.",
            "supported_by": ["mock_masked_marginal_llr"],
        },
    ]
    return {
        "protein": protein.upper(),
        "mutation": grounding["mutation"],
        "mode": "tooluniverse" if use_tooluniverse else "dry-run",
        "grounding": grounding,
        "score": scoring,
        "latent_roles": [role.as_dict() for role in roles],
        "claims": claims,
        "verification": {"claims_checked": len(claims), "unsupported_claims_dropped": 0},
        "trace": [call.as_dict() for call in (*grounding_calls, score_call, annotation_call)],
        "limitations": [
            "ESM, SAE activations, and agent hypothesis generation are mocked.",
            "This validates orchestration and provenance only; it is not a biological result.",
        ],
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        f"# Mechanistic latent report: {report['mutation']}",
        "",
        "> **Prototype:** ToolUniverse mode uses live UniProt/InterPro evidence; ESM, SAE, and structural contacts remain mocked or incomplete.",
        "",
        "## Grounding",
        f"- Protein: `{report['protein']}`",
        f"- Structure fixture: `{report['grounding']['structure']}` chain `{report['grounding']['chain']}`",
        f"- Contacts reported: `{report['grounding']['contacts_within_5A']}`",
        f"- HPD span: `{report['grounding']['motif_spans'].get('HPD', [])}`",
        *([f"- **Numbering warning:** {report['grounding']['numbering_warning']}"] if report['grounding'].get('numbering_warning') else []),
        "",
        "## Score",
        f"- Mock masked-marginal LLR: `{report['score']['masked_marginal_llr']}`",
        "",
        "## Latent annotations",
    ]
    for role in report["latent_roles"]:
        lines.append(f"- `{role['latent_id']}` → **{role['category']}**, residues `{role['supporting_residues']}` (citation: `{role['citation']}`)")
    lines.extend([
        "",
        "## Verified claims",
        *[f"- {claim['claim']} [{', '.join(claim['supported_by'])}]" for claim in report["claims"]],
        "",
        "## Limitations",
        *[f"- {limitation}" for limitation in report["limitations"]],
    ])
    return "\n".join(lines)


def json_report(report: dict[str, Any]) -> str:
    return json.dumps(report, indent=2, sort_keys=True)
