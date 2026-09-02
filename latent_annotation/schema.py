"""Define and validate the latent-annotation output.

A free-text label cannot be scored, compared across SAE runs, or audited. The
schema requires a source for each claim, a falsification condition, and at least
one rejected hypothesis.

HuggingFace providers do not handle JSON schemas consistently. Some ignore
``response_format``; some models wrap JSON in a code fence. ``extract_json``
recovers the object, and ``validate_annotation`` checks it before returning it to
the caller.
"""

from __future__ import annotations

import json
import re
from typing import Any

CONCEPT_TYPES = [
    "structural_motif",
    "sequence_motif",
    "domain_family",
    "active_or_binding_site",
    "post_translational_modification",
    "subcellular_localization",
    "taxonomic_or_phylogenetic",
    "biophysical_property",
    "disease_or_variant_associated",
    "positional_or_terminal",
    "polysemantic",
    "uninterpretable",
]

CONFIDENCE_LEVELS = ["high", "medium", "low"]

MUTATION_SENSITIVITY = ["high", "moderate", "low", "unknown"]

ANNOTATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "latent_id": {
            "type": "string",
            "description": "The latent this annotation describes.",
        },
        "label": {
            "type": "string",
            "description": "Plain-language name of about eight words or fewer, such as 'zinc-finger C2H2 coordinating residues'.",
        },
        "description": {
            "type": "string",
            "description": "Two to four sentences describing when the latent fires and how the cited records support the label.",
        },
        "concept_type": {"type": "string", "enum": CONCEPT_TYPES},
        "confidence": {"type": "string", "enum": CONFIDENCE_LEVELS},
        "evidence": {
            "type": "array",
            "description": "Claims with the tool and record that support them. At least one entry must cite a database tool, not only an activation window.",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "source_tool": {
                        "type": "string",
                        "description": "Name of the tool that produced this evidence.",
                    },
                    "source_id": {
                        "type": "string",
                        "description": "Accession, entry id, or latent example id the evidence came from.",
                    },
                },
                "required": ["claim", "source_tool", "source_id"],
                "additionalProperties": False,
            },
        },
        "rejected_hypotheses": {
            "type": "array",
            "description": "Other explanations considered for the activations and the evidence against each one.",
            "items": {
                "type": "object",
                "properties": {
                    "hypothesis": {"type": "string"},
                    "why_rejected": {"type": "string"},
                },
                "required": ["hypothesis", "why_rejected"],
                "additionalProperties": False,
            },
        },
        "disconfirming_evidence": {
            "type": "string",
            "description": "An observation that would falsify the label, such as a sequence that should fire but does not or a firing site the label cannot explain.",
        },
        "mutation_effect": {
            "type": "object",
            "description": "Expected relationship between this latent and variant effects.",
            "properties": {
                "is_relevant": {"type": "boolean"},
                "predicted_sensitivity": {
                    "type": "string",
                    "enum": MUTATION_SENSITIVITY,
                    "description": "Expected fitness effect of substitutions at residues marked by this latent.",
                },
                "rationale": {"type": "string"},
            },
            "required": ["is_relevant", "predicted_sensitivity", "rationale"],
            "additionalProperties": False,
        },
    },
    "required": [
        "latent_id",
        "label",
        "description",
        "concept_type",
        "confidence",
        "evidence",
        "rejected_hypotheses",
        "disconfirming_evidence",
        "mutation_effect",
    ],
    "additionalProperties": False,
}


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> tuple[dict[str, Any] | None, str | None]:
    """Recover a JSON object from a model reply as ``(object, error)``.

    Accept clean JSON, fenced JSON, or a JSON object surrounded by prose.
    """
    text = (text or "").strip()
    if not text:
        return None, "model returned an empty message"

    candidates = [text]
    fenced = _FENCE.search(text)
    if fenced:
        candidates.insert(0, fenced.group(1).strip())
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed, None
    return None, "no JSON object found in the reply"


def validate_annotation(annotation: dict[str, Any]) -> list[str]:
    """Check an annotation against ANNOTATION_SCHEMA and return readable errors.

    If `jsonschema` is unavailable, check only the required keys.
    """
    try:
        import jsonschema
    except ImportError:
        return [
            f"missing required field: {key}"
            for key in ANNOTATION_SCHEMA["required"]
            if key not in annotation
        ]

    validator = jsonschema.Draft202012Validator(ANNOTATION_SCHEMA)
    return [
        f"{'/'.join(str(part) for part in error.path) or '<root>'}: {error.message}"
        for error in sorted(
            validator.iter_errors(annotation), key=lambda e: list(e.path)
        )
    ]


SYSTEM_PROMPT = """\
Annotate one latent from a sparse autoencoder trained on protein-language-model \
activations. A latent fires on a subset of residues. Describe what those residues \
have in common using the evidence retrieved in this session.

For each latent:
1. Call `get_latent_evidence`. Read the strongest activation windows and record \
their accessions.
2. Inspect the bracketed peak residues. Check residue identity, local sequence, \
position in the chain, and repeated motifs. It is acceptable to find no pattern.
3. Look up several of the returned accessions with the available UniProt, InterPro, \
Pfam, or Reactome tools. Compare each peak index with annotated domains, binding \
sites, active sites, and post-translational modifications. A protein-level label \
does not by itself explain a residue-level activation.
4. Test at least one competing explanation. For example, activations in several \
kinases could mark the kinase fold, an ATP-binding region, or a corpus artifact. \
Find an activation that the proposed label explains poorly.

Use `low` confidence when the records are sparse or inconsistent. Use concept_type \
`uninterpretable` when no single explanation fits. Use `polysemantic` when two \
distinct patterns fit and describe both.

Every claim must cite the tool and record id that support it. Do not state database \
facts that were not retrieved in this session.

Return one JSON object that matches the annotation schema. Include every required \
field. Do not add prose or a code fence.\
"""
