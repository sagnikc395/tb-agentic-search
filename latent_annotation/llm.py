"""Chat clients for HuggingFace models and offline tests.

The agent depends on `ChatClient`, not a vendor SDK. `MockChatClient` runs without
a model call, while `HuggingFaceChatClient` sends the same message and tool shapes
to a Qwen-3 endpoint. Both use OpenAI-style chat messages and function definitions.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

from . import config


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ChatResponse:
    """A normalized reply containing tool calls or final text."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class ChatClient(Protocol):
    def complete(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
        *,
        response_schema: dict[str, Any] | None = None,
    ) -> ChatResponse: ...


# ---------------------------------------------------------------------------
# HuggingFace / Qwen-3
# ---------------------------------------------------------------------------


class HuggingFaceChatClient:
    """Call Qwen-3 or another tool-calling model through HuggingFace.

    With `provider="auto"`, HuggingFace selects a provider that serves the model.
    Support for tool calls and `response_format` differs by provider. The schema
    module therefore parses and validates the returned text on the client side.
    """

    def __init__(
        self,
        model: str = config.DEFAULT_MODEL,
        *,
        token: str | None = None,
        provider: str = config.DEFAULT_HF_PROVIDER,
        temperature: float = config.TEMPERATURE,
        max_tokens: int = config.MAX_TOKENS,
        use_response_format: bool = config.USE_RESPONSE_FORMAT,
        timeout: float = config.TIMEOUT_SECONDS,
    ):
        from huggingface_hub import InferenceClient

        token = (
            token or os.getenv(config.ENV_HF_TOKEN) or os.getenv(config.ENV_HF_API_KEY)
        )
        if not token:
            raise RuntimeError(
                "No HuggingFace token. Set HF_TOKEN in .env, or use the mock provider."
            )
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_response_format = use_response_format
        self._client = InferenceClient(
            provider=provider, api_key=token, timeout=timeout
        )

    def complete(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
        *,
        response_schema: dict[str, Any] | None = None,
    ) -> ChatResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if tools:
            kwargs["tools"] = list(tools)
            kwargs["tool_choice"] = "auto"
        # Only constrain the format once the model has stopped calling tools.
        # When tools and a schema are both active, some providers
        # emit the schema instead of the tool call.
        if response_schema and self.use_response_format and not tools:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "annotation", "schema": response_schema},
            }

        completion = self._client.chat_completion(**kwargs)
        choice = completion.choices[0]
        message = choice.message

        tool_calls = []
        for index, call in enumerate(getattr(message, "tool_calls", None) or []):
            tool_calls.append(
                ToolCall(
                    id=getattr(call, "id", None) or f"call_{index}",
                    name=call.function.name,
                    arguments=_coerce_arguments(call.function.arguments),
                )
            )

        usage = {}
        if completion.usage:
            usage = {
                "prompt_tokens": completion.usage.prompt_tokens or 0,
                "completion_tokens": completion.usage.completion_tokens or 0,
            }

        return ChatResponse(
            text=(message.content or "").strip(), tool_calls=tool_calls, usage=usage
        )


def _coerce_arguments(arguments: Any) -> dict[str, Any]:
    """Normalize tool arguments returned as either a JSON string or a dictionary."""
    if isinstance(arguments, dict):
        return arguments
    if not arguments:
        return {}
    try:
        parsed = json.loads(arguments)
    except (TypeError, json.JSONDecodeError):
        return {"_raw": str(arguments)}
    return parsed if isinstance(parsed, dict) else {"_raw": parsed}


# ---------------------------------------------------------------------------
# Mock
# ---------------------------------------------------------------------------


class MockChatClient:
    """Run the full agent loop without a model call.

    The mock fetches latent evidence, looks up the first accession, and returns an
    annotation built from those tool results. It tests tool dispatch and schema
    handling without testing biological interpretation.
    """

    def __init__(self, *, lookup_tool: str = config.MOCK_LOOKUP_TOOL):
        self.lookup_tool = lookup_tool

    def complete(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
        *,
        response_schema: dict[str, Any] | None = None,
    ) -> ChatResponse:
        available = {_tool_name(tool) for tool in tools}
        seen = _tool_results(messages)
        latent_id = _requested_latent(messages)

        if "get_latent_evidence" not in seen and "get_latent_evidence" in available:
            return ChatResponse(
                tool_calls=[
                    ToolCall("mock-1", "get_latent_evidence", {"latent_id": latent_id})
                ],
            )

        evidence = _first_json(seen.get("get_latent_evidence", ""))
        accessions = (evidence or {}).get("accessions", [])
        if (
            accessions
            and self.lookup_tool in available
            and self.lookup_tool not in seen
        ):
            return ChatResponse(
                tool_calls=[
                    ToolCall("mock-2", self.lookup_tool, {"accession": accessions[0]})
                ],
            )

        return ChatResponse(
            text=json.dumps(
                _mock_annotation(latent_id, evidence, self.lookup_tool, seen)
            ),
            usage={"prompt_tokens": 0, "completion_tokens": 0},
        )


def _tool_name(tool: dict[str, Any]) -> str:
    return tool.get("function", {}).get("name", tool.get("name", ""))


def _tool_results(messages: Sequence[dict[str, Any]]) -> dict[str, str]:
    """Map each tool name to its latest result in the transcript."""
    names: dict[str, str] = {}
    by_id: dict[str, str] = {}
    for message in messages:
        if message.get("role") == "assistant":
            for call in message.get("tool_calls") or []:
                by_id[call["id"]] = call["function"]["name"]
        elif message.get("role") == "tool":
            name = message.get("name") or by_id.get(message.get("tool_call_id", ""), "")
            if name:
                names[name] = str(message.get("content", ""))
    return names


def _requested_latent(messages: Sequence[dict[str, Any]]) -> str:
    for message in messages:
        if message.get("role") == "user":
            match = re.search(r"latent\s+(\S+)", str(message.get("content", "")))
            if match:
                return match.group(1).rstrip(".")
    return "unknown"


def _first_json(text: str) -> dict[str, Any] | None:
    try:
        return json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return None


def _mock_annotation(
    latent_id: str,
    evidence: dict[str, Any] | None,
    lookup_tool: str,
    seen: dict[str, str],
) -> dict[str, Any]:
    evidence = evidence or {}
    accessions = evidence.get("accessions", [])
    rate = evidence.get("firing_rate", 0.0)
    examples = evidence.get("examples", [])
    windows = ", ".join(ex.get("highlighted", "") for ex in examples[:3])

    # Vary the fixture outputs by firing rate instead of returning one fixed type.
    if rate > 0.06:
        concept_type, confidence = "uninterpretable", "low"
    elif rate > 0.04:
        concept_type, confidence = "sequence_motif", "medium"
    else:
        concept_type, confidence = "polysemantic", "low"

    evidence_items = [
        {
            "claim": f"Top activations occur in windows: {windows or 'none recorded'}",
            "source_tool": "get_latent_evidence",
            "source_id": latent_id,
        }
    ]
    if lookup_tool in seen and accessions:
        evidence_items.append(
            {
                "claim": f"Retrieved the annotated record for {accessions[0]}.",
                "source_tool": lookup_tool,
                "source_id": accessions[0],
            }
        )

    return {
        "latent_id": latent_id,
        "label": f"MOCK annotation for {latent_id}",
        "description": (
            "MockChatClient produced this placeholder. It reports the "
            f"observed firing rate ({rate:.4f}) across {evidence.get('n_proteins_scanned', 0)} "
            "proteins and the accessions behind the top windows, but performs no "
            "biological interpretation. Set LLM_PROVIDER=huggingface to run a model."
        ),
        "concept_type": concept_type,
        "confidence": confidence,
        "evidence": evidence_items,
        "rejected_hypotheses": [
            {
                "hypothesis": "The latent encodes a specific domain family.",
                "why_rejected": "The mock client does not evaluate hypotheses.",
            }
        ],
        "disconfirming_evidence": "Any real annotation would supersede this placeholder.",
        "mutation_effect": {
            "is_relevant": False,
            "predicted_sensitivity": "unknown",
            "rationale": "Not assessed by the mock client.",
        },
    }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def load_client(
    provider: str | None = None,
    *,
    model: str | None = None,
    env_file: str = config.ENV_FILE,
) -> ChatClient:
    """Build a chat client from `.env` / environment.

    LLM_PROVIDER = mock | huggingface   (default: huggingface when HF_TOKEN is set)
    LLM_MODEL    = HF model id          (default: Qwen-3)
    HF_TOKEN     = HuggingFace API key
    """
    _load_dotenv(env_file)
    provider = (provider or os.getenv(config.ENV_LLM_PROVIDER) or "").strip().lower()
    if not provider:
        provider = "huggingface" if os.getenv(config.ENV_HF_TOKEN) else "mock"

    if provider == "mock":
        return MockChatClient()
    if provider in {"huggingface", "hf", "qwen"}:
        return HuggingFaceChatClient(
            model=model or os.getenv(config.ENV_LLM_MODEL) or config.DEFAULT_MODEL
        )
    raise ValueError(
        f"unknown LLM provider {provider!r} (expected 'mock' or 'huggingface')"
    )


def _load_dotenv(env_file: str) -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(env_file, override=False)
