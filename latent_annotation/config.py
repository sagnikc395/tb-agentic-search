"""Project configuration.

Code settings cover tool lists, evidence windows, loop limits, and result length.
Environment settings select the provider, model, and API token. Functions use
these values as defaults and may override them per call.

This module imports no other package module, which keeps it free of dependency
cycles.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_PATH = REPO_ROOT / "examples" / "fixtures" / "latents.json"
OUTPUT_PATH = REPO_ROOT / "examples" / "annotations.jsonl"
ENV_FILE = ".env"

ENV_LLM_PROVIDER = "LLM_PROVIDER"
ENV_LLM_MODEL = "LLM_MODEL"
ENV_HF_TOKEN = "HF_TOKEN"
ENV_HF_API_KEY = "HUGGINGFACE_API_KEY"

# ---------------------------------------------------------------------------
# Model layer (latent_annotation/llm.py)
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"
DEFAULT_HF_PROVIDER = "auto"  # HF routing: let the Inference API pick the provider
TEMPERATURE = 0.2
MAX_TOKENS = 4096
TIMEOUT_SECONDS = 180.0
# Only request `response_format` once tool calling is done; some HF providers
# emit the schema instead of the tool call when both are active.
USE_RESPONSE_FORMAT = True

# MockChatClient calls this database tool after fetching evidence. It must appear
# in PROTEIN_CONTEXT_TOOLS below.
MOCK_LOOKUP_TOOL = "UniProt_get_function_by_accession"

# ---------------------------------------------------------------------------
# Agent loop (latent_annotation/agent.py)
# ---------------------------------------------------------------------------


@dataclass
class AgentConfig:
    """Settings for one latent's tool-use loop."""

    max_turns: int = 12
    require_evidence_tool: bool = True  # nudge once if the model answers ungrounded
    max_repair_attempts: int = 2  # retries when the final message is not valid JSON


# ---------------------------------------------------------------------------
# Evidence extraction (latent_annotation/evidence.py)
# ---------------------------------------------------------------------------

TOP_K_EXAMPLES = 8  # how many top-activating windows the agent gets to see
WINDOW_SIZE = 24  # residues on each side of the peak in a window
HIGHLIGHT_SPAN = 2  # residues bracketed on each side of the peak
ACTIVE_THRESHOLD = 0.0  # activation above this counts as "firing"

# ---------------------------------------------------------------------------
# Tool surface (latent_annotation/tools.py)
# ---------------------------------------------------------------------------
# Curated starting sets. To search the full catalog, call
# `ToolUniverse().load_tools()` and then `find_tools_by_pattern("UniProt")`.
# Keep these lists short and ordered. Each entry uses context on every request,
# and reordering invalidates the prompt cache.

PROTEIN_CONTEXT_TOOLS: tuple[str, ...] = (
    "UniProt_get_recommended_name_by_accession",
    "UniProt_get_function_by_accession",
    "UniProt_get_features_by_accession",
    "UniProt_get_subcellular_location_by_accession",
    "UniProt_get_ptm_processing_by_accession",
    "InterPro_get_entries_for_protein",
    "InterPro_get_protein_domain_architecture",
    "Pfam_get_protein_annotations",
    "Reactome_map_uniprot_to_pathways",
)

MUTATION_EFFECT_TOOLS: tuple[str, ...] = (
    "UniProt_get_disease_variants_by_accession",
    "MaveDB_search_score_sets",
    "MaveDB_get_variant_scores",
    "MaveDB_get_effect_matrix",
    "ClinVar_search_variants",
    "gnomad_get_constraint",
)

# Stop serialized tool results at this length so one UniProt entry cannot fill
# the context window.
TOOL_RESULT_LIMIT = 8_000
