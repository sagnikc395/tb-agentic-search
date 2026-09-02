"""Build and dispatch the annotation agent's tools.

ToolUniverse tools query scientific databases. Local tools read SAE evidence from
this process. `ToolBox` presents both as OpenAI-style functions and dispatches each
call by name.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from . import config
from .evidence import LatentEvidenceStore

# The ToolUniverse allowlists are in
# `PROTEIN_CONTEXT_TOOLS` / `MUTATION_EFFECT_TOOLS` in config.py, and
# notes/02-tooluniverse.md for the selection criteria.


@dataclass
class LocalTool:
    """A tool backed by a Python callable in this process."""

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], Any]


class ToolBox:
    """Assembles tool definitions and dispatches calls by name."""

    def __init__(
        self,
        *,
        tooluniverse: Any | None = None,
        tooluniverse_tools: Sequence[str] = (),
        local_tools: Sequence[LocalTool] = (),
    ):
        self._tu = tooluniverse
        self._tu_names = tuple(tooluniverse_tools)
        self._local = {tool.name: tool for tool in local_tools}
        if self._tu_names and self._tu is None:
            raise ValueError("tooluniverse_tools given without a tooluniverse instance")
        self._definitions: list[dict[str, Any]] | None = None

    # -- definitions -------------------------------------------------------

    def definitions(self) -> list[dict[str, Any]]:
        """Return a cached list of OpenAI-style function definitions."""
        if self._definitions is None:
            defs = [
                _as_function(tool.name, tool.description, tool.input_schema)
                for tool in self._local.values()
            ]
            defs.extend(self._tooluniverse_definitions())
            self._definitions = defs
        return self._definitions

    def names(self) -> list[str]:
        return [definition["function"]["name"] for definition in self.definitions()]

    def _tooluniverse_definitions(self) -> list[dict[str, Any]]:
        definitions = []
        for name in self._tu_names:
            spec = self._tu.tool_specification(name, format="openai")
            if not spec:
                raise ValueError(f"ToolUniverse has no tool named {name!r}")
            definitions.append(
                _as_function(
                    spec["name"],
                    spec.get("description", ""),
                    spec.get("parameters", {"type": "object", "properties": {}}),
                )
            )
        return definitions

    # -- dispatch ----------------------------------------------------------

    def run(self, name: str, arguments: dict[str, Any]) -> tuple[str, bool]:
        """Execute a tool call. Returns ``(result_text, is_error)``.

        Errors become tool results so the model can try another accession.
        """
        try:
            if name in self._local:
                result = self._local[name].handler(arguments)
            elif name in self._tu_names:
                result = self._tu.run(
                    {"name": name, "arguments": arguments}, verbose=False
                )
            else:
                return f"Error: unknown tool {name!r}.", True
        except Exception as exc:  # noqa: BLE001 - surfaced to the model, not swallowed
            return f"Error calling {name}: {type(exc).__name__}: {exc}", True
        return _stringify(result), False


def _as_function(
    name: str, description: str, parameters: dict[str, Any]
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


def _stringify(result: Any, *, limit: int = config.TOOL_RESULT_LIMIT) -> str:
    """Serialize a tool result and stop it at the configured character limit."""
    if isinstance(result, str):
        text = result
    else:
        try:
            text = json.dumps(result, indent=2, default=str)
        except (TypeError, ValueError):
            text = str(result)
    if len(text) > limit:
        text = text[:limit] + f"\n... [truncated at {limit} characters]"
    return text


def evidence_tools(store: LatentEvidenceStore) -> list[LocalTool]:
    """Local tools that expose the SAE evidence store to the agent."""

    def get_evidence(args: dict[str, Any]) -> Any:
        return store.get(args["latent_id"]).to_dict()

    def list_latents(_: dict[str, Any]) -> Any:
        return {"latent_ids": store.latent_ids, "count": len(store)}

    return [
        LocalTool(
            name="get_latent_evidence",
            description=(
                "Return one SAE latent's strongest sequence windows, with brackets "
                "around each peak, plus their UniProt accessions and activation "
                "statistics. Call this before annotating a latent. It is the source "
                "of activation data, and its accessions are inputs to the database "
                "tools."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "latent_id": {
                        "type": "string",
                        "description": "Latent identifier, e.g. 'L-1042'.",
                    }
                },
                "required": ["latent_id"],
            },
            handler=get_evidence,
        ),
        LocalTool(
            name="list_latents",
            description="List every latent id that has evidence available in this run.",
            input_schema={"type": "object", "properties": {}},
            handler=list_latents,
        ),
    ]


def load_tooluniverse(tool_names: Sequence[str]):
    """Load ToolUniverse and verify the tools selected for this run.

    The lazy import lets the evidence and schema modules load without
    ToolUniverse.
    """
    from tooluniverse import ToolUniverse

    tu = ToolUniverse()
    tu.load_tools()
    missing = [name for name in tool_names if name not in tu.all_tool_dict]
    if missing:
        raise ValueError(f"tools not found in ToolUniverse: {missing}")
    return tu
