import os
from dataclasses import dataclass

from dotenv import load_dotenv

DEFAULT_MODEL = "openai/gpt-oss-120b"
DEFAULT_SOURCE = "Groq"


class ConfigError(RuntimeError):
    """Raised when the environment is not set up well enough to run."""


@dataclass(frozen=True)
class Settings:
    """Everything needed to build the screener or agent. Use load_settings()."""

    api_key: str
    model: str = DEFAULT_MODEL
    source: str = DEFAULT_SOURCE
    # Low by default: the screener wants tight, mechanical JSON replies.
    temperature: float = 0.2
    data_path: str = "./data"
    timeout_seconds: int = 600
    use_tool_retriever: bool = False
    # Biomni pulls a tens-of-GB data lake on first A1 init; screening doesn't
    # need it (`--data-lake` turns it on).
    load_data_lake: bool = False


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    return default if raw is None else raw.strip().lower() in ("1", "true", "yes", "on")


def load_settings(**overrides) -> Settings:
    """Read settings from .env / environment; kwargs win when not None."""
    load_dotenv()

    values = {
        "api_key": os.getenv("GROQ_API_KEY", "").strip(),
        "model": os.getenv("GROQ_MODEL") or DEFAULT_MODEL,
        "source": os.getenv("LLM_SOURCE") or DEFAULT_SOURCE,
        "temperature": float(os.getenv("BIOMNI_TEMPERATURE", "0.2")),
        "data_path": os.getenv("BIOMNI_PATH", "./data"),
        "timeout_seconds": int(os.getenv("BIOMNI_TIMEOUT_SECONDS", "600")),
        "use_tool_retriever": _env_bool("BIOMNI_USE_TOOL_RETRIEVER", False),
        "load_data_lake": _env_bool("BIOMNI_LOAD_DATA_LAKE", False),
    }
    values.update({k: v for k, v in overrides.items() if v is not None})

    if not values["api_key"]:
        raise ConfigError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and fill it in, "
            "or export GROQ_API_KEY in your shell."
        )
    return Settings(**values)
