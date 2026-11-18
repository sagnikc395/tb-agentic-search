"""Settings for a locally served model: which engine, which model, how it thinks.

Everything runs against a model we serve ourselves — no hosted provider. Two of
biomni's backends do that, and both drive the A1 agent:

- Custom (the default): any OpenAI-compatible server, which is what vLLM is.
  This is the one to use on Unity, and the only one that can hand back separated
  reasoning traces.
- Ollama: biomni's other local branch, for a laptop-sized model. It is reached
  on its own default port rather than by URL.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

from tb_search.reasoning import normalize_effort

# Qwen3 is the default because every piece this pipeline needs is supported for
# it at once: vLLM serves it with a `qwen3` reasoning parser and with tool
# calling, and biomni reaches it as an ordinary OpenAI-compatible model, which
# is what A1 needs to call its tools.
DEFAULT_MODEL = "Qwen/Qwen3-8B"
# biomni's name for "an OpenAI-compatible endpoint I am handed the URL of",
# which is what a vLLM server is.
CUSTOM_SOURCE = "Custom"
OLLAMA_SOURCE = "Ollama"
DEFAULT_SOURCE = CUSTOM_SOURCE
# The biomni backends that serve a model locally. The hosted ones it also knows
# (OpenAI, Anthropic, Gemini, Bedrock, Azure) are deliberately not offered.
LOCAL_SOURCES = (CUSTOM_SOURCE, OLLAMA_SOURCE)
# Where `vllm serve` listens unless told otherwise.
DEFAULT_BASE_URL = "http://localhost:8000/v1"


class ConfigError(RuntimeError):
    """Raised when the environment is not set up well enough to run."""


@dataclass(frozen=True)
class Settings:
    """Everything needed to build the screener or agent. Use load_settings()."""

    model: str = DEFAULT_MODEL
    source: str = DEFAULT_SOURCE
    # The OpenAI-compatible endpoint serving the model — on Unity, the vLLM
    # job's own URL, e.g. "http://gpu042:8000/v1". Unused by the Ollama branch,
    # which biomni reaches on its own default port.
    base_url: str | None = DEFAULT_BASE_URL
    # A locally served model wants no credential, but the OpenAI client insists
    # on sending one; biomni's own placeholder is this same string.
    api_key: str = "EMPTY"
    # Low by default: the screener wants tight, mechanical JSON replies.
    temperature: float = 0.2
    data_path: str = "./data"
    timeout_seconds: int = 600
    use_tool_retriever: bool = False
    # Biomni pulls a tens-of-GB data lake on first A1 init; screening doesn't
    # need it (`--data-lake` turns it on).
    load_data_lake: bool = False
    # Passed straight through as the request's `reasoning_effort`. vLLM maps it
    # onto the chat template's thinking switch, so this is how much reasoning
    # the traces are made of. None leaves the server's own default alone.
    reasoning_effort: str | None = None

    @property
    def is_custom(self) -> bool:
        """Whether the model is reached over an OpenAI-compatible URL."""
        return self.source == CUSTOM_SOURCE

    @property
    def label(self) -> str:
        """How a run names its backend in its JSON output."""
        return f"{self.source}/{self.model}"

    @property
    def description(self) -> str:
        """The one console line that says what is answering the requests."""
        parts = [self.source, self.model]
        if self.is_custom and self.base_url:
            parts.append(f"@ {self.base_url}")
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
    """Fold a source to the way biomni spells it, and refuse a non-local one.

    A stale `LLM_SOURCE` in someone's .env — this pipeline used to run on Groq —
    is a leftover, not an instruction, so it is reported and ignored rather than
    taking the run down. Asked for on the command line, it is an error.
    """
    wanted = str(source or "").strip()
    for known in LOCAL_SOURCES:
        if wanted.lower() == known.lower():
            return known
    complaint = (
        f"{wanted!r} is not a local biomni backend. Use "
        f"{' or '.join(LOCAL_SOURCES)}: this pipeline runs against a model you "
        "serve yourself, not a hosted API."
    )
    if not from_env:
        raise ConfigError(complaint)
    print(f"! LLM_SOURCE={wanted} ignored: {complaint}")
    return DEFAULT_SOURCE


def load_settings(**overrides) -> Settings:
    """Read settings from .env / environment; kwargs win when not None."""
    load_dotenv()

    values = {
        "model": _env_first("VLLM_MODEL", "LLM_MODEL") or DEFAULT_MODEL,
        "source": _env_first("LLM_SOURCE") or DEFAULT_SOURCE,
        "base_url": _env_first(
            "VLLM_BASE_URL", "LLM_BASE_URL", "BIOMNI_CUSTOM_BASE_URL"
        )
        or DEFAULT_BASE_URL,
        "api_key": _env_first("VLLM_API_KEY", "BIOMNI_CUSTOM_API_KEY") or "EMPTY",
        "temperature": float(os.getenv("BIOMNI_TEMPERATURE", "0.2")),
        "data_path": os.getenv("BIOMNI_PATH", "./data"),
        "timeout_seconds": int(os.getenv("BIOMNI_TIMEOUT_SECONDS", "600")),
        "use_tool_retriever": _env_bool("BIOMNI_USE_TOOL_RETRIEVER", False),
        "load_data_lake": _env_bool("BIOMNI_LOAD_DATA_LAKE", False),
        "reasoning_effort": _env_first("LLM_REASONING_EFFORT", "VLLM_REASONING_EFFORT")
        or None,
    }
    values.update({k: v for k, v in overrides.items() if v is not None})

    # A URL was given, so the OpenAI-compatible branch is meant even if
    # LLM_SOURCE in the environment still says Ollama.
    if overrides.get("base_url"):
        values["source"] = CUSTOM_SOURCE
    values["source"] = _resolve_source(
        values["source"], from_env=overrides.get("source") is None
    )

    try:
        values["reasoning_effort"] = normalize_effort(values["reasoning_effort"])
    except ValueError as e:
        raise ConfigError(str(e)) from e

    if values["source"] == CUSTOM_SOURCE and not values["base_url"]:
        raise ConfigError(
            "No endpoint to call. Start the model server and pass its URL with "
            "--base-url, or set VLLM_BASE_URL in .env "
            f"(default: {DEFAULT_BASE_URL})."
        )
    return Settings(**values)
