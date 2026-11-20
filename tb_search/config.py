"""Settings for the Groq endpoint: which model, how it thinks, how hard to retry.

Everything runs against Groq's hosted inference endpoint — there is no model to
serve and no GPU in the loop. Groq is one of biomni's own backends, and it
drives the A1 agent: an OpenAI-compatible API at
https://api.groq.com/openai/v1 with tool calling and separated reasoning, which
is all A1 needs.

What that moves is the thing that can go wrong. A server you serve yourself can
be down; a hosted endpoint is up but rate-limited, so `max_retries` and the
backoff in tb_search.screening matter here in the way `--max-model-len` used to.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

from tb_search.reasoning import normalize_effort

# GPT-OSS 120B is the default because every piece this pipeline needs is
# supported for it at once on Groq: tool calling (which is what A1 does), a
# 131k context window (A1's system prompt alone is ~35k tokens), reasoning
# returned in its own `reasoning` field, and a low/medium/high effort scale.
DEFAULT_MODEL = "openai/gpt-oss-120b"
# biomni's name for its Groq branch.
GROQ_SOURCE = "Groq"
DEFAULT_SOURCE = GROQ_SOURCE
# The only backend offered. Biomni also knows OpenAI, Anthropic, Gemini,
# Bedrock, Azure, Ollama and Custom; this pipeline runs on Groq.
SOURCES = (GROQ_SOURCE,)
# Groq's OpenAI-compatible endpoint.
DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
# How many times a rate-limited or briefly failing request is retried before a
# batch or an isolate is given up on.
DEFAULT_MAX_RETRIES = 5


class ConfigError(RuntimeError):
    """Raised when the environment is not set up well enough to run."""


@dataclass(frozen=True)
class Settings:
    """Everything needed to build the screener or agent. Use load_settings()."""

    model: str = DEFAULT_MODEL
    source: str = DEFAULT_SOURCE
    # Groq's OpenAI-compatible endpoint. Overridable only so a proxy or a
    # region-specific host can be pointed at; the default is the real one.
    base_url: str = DEFAULT_BASE_URL
    # From GROQ_API_KEY. Required: unlike a model we served ourselves, this
    # endpoint will not answer without one.
    api_key: str = ""
    # Low by default: the screener wants tight, mechanical JSON replies.
    temperature: float = 0.2
    data_path: str = "./data"
    timeout_seconds: int = 600
    use_tool_retriever: bool = False
    # Biomni pulls a tens-of-GB data lake on first A1 init; screening doesn't
    # need it (`--data-lake` turns it on).
    load_data_lake: bool = False
    # Passed straight through as the request's `reasoning_effort`, so this is
    # how much reasoning the captured traces are made of. None leaves Groq's
    # own default for the model alone. Which values a model takes is the
    # model's business — see tb_search.reasoning.EFFORTS.
    reasoning_effort: str | None = None
    # Groq rate-limits per minute and per day, so a sweep long enough to be
    # worth running will meet a 429. This is how many times one request waits
    # it out before its batch or isolate is given up on.
    max_retries: int = DEFAULT_MAX_RETRIES

    @property
    def label(self) -> str:
        """How a run names its backend in its JSON output."""
        return f"{self.source}/{self.model}"

    @property
    def description(self) -> str:
        """The one console line that says what is answering the requests."""
        parts = [self.source, self.model]
        if self.reasoning_effort:
            parts.append(f"effort {self.reasoning_effort}")
        return " ".join(parts)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    return default if raw is None else raw.strip().lower() in ("1", "true", "yes", "on")


def _env_first(*names: str) -> str:
    """The first of these environment variables that is set and non-empty."""
    for name in names:
        value = (os.getenv(name) or "").strip()
        if value:
            return value
    return ""


def _resolve_source(source: str, from_env: bool) -> str:
    """Fold a source to the way biomni spells it, and refuse any other one.

    A stale `LLM_SOURCE` in someone's .env — this pipeline used to serve its own
    model with vLLM — is a leftover, not an instruction, so it is reported and
    ignored rather than taking the run down. Asked for on the command line, it
    is an error.
    """
    wanted = str(source or "").strip()
    for known in SOURCES:
        if wanted.lower() == known.lower():
            return known
    complaint = (
        f"{wanted!r} is not the backend this pipeline runs on. Use "
        f"{' or '.join(SOURCES)}: inference is Groq's hosted endpoint, not a "
        "model served locally."
    )
    if not from_env:
        raise ConfigError(complaint)
    print(f"! LLM_SOURCE={wanted} ignored: {complaint}")
    return DEFAULT_SOURCE


def load_settings(**overrides) -> Settings:
    """Read settings from .env / environment; kwargs win when not None."""
    load_dotenv()

    values = {
        "model": _env_first("GROQ_MODEL", "LLM_MODEL") or DEFAULT_MODEL,
        "source": _env_first("LLM_SOURCE") or DEFAULT_SOURCE,
        "base_url": _env_first("GROQ_BASE_URL", "LLM_BASE_URL") or DEFAULT_BASE_URL,
        "api_key": _env_first("GROQ_API_KEY"),
        "temperature": float(os.getenv("BIOMNI_TEMPERATURE", "0.2")),
        "data_path": os.getenv("BIOMNI_PATH", "./data"),
        "timeout_seconds": int(os.getenv("BIOMNI_TIMEOUT_SECONDS", "600")),
        "use_tool_retriever": _env_bool("BIOMNI_USE_TOOL_RETRIEVER", False),
        "load_data_lake": _env_bool("BIOMNI_LOAD_DATA_LAKE", False),
        "reasoning_effort": _env_first("LLM_REASONING_EFFORT", "GROQ_REASONING_EFFORT")
        or None,
        "max_retries": int(os.getenv("GROQ_MAX_RETRIES", str(DEFAULT_MAX_RETRIES))),
    }
    values.update({k: v for k, v in overrides.items() if v is not None})

    values["source"] = _resolve_source(
        values["source"], from_env=overrides.get("source") is None
    )

    try:
        values["reasoning_effort"] = normalize_effort(values["reasoning_effort"])
    except ValueError as e:
        raise ConfigError(str(e)) from e

    if not values["api_key"]:
        raise ConfigError(
            "No GROQ_API_KEY. Every mode but --cohort calls Groq, so put the key "
            "in .env (see .env.example) or export it:\n"
            "  export GROQ_API_KEY=gsk_...\n"
            "Keys are at https://console.groq.com/keys"
        )
    values["max_retries"] = max(0, values["max_retries"])
    return Settings(**values)
