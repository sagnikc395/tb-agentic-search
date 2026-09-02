# Codebase map and configuration

This document describes the current repository: file responsibilities, module
dependencies, configuration, previous cleanup work, and code that may look unused
but is intentional.

## 1. Module map

```
latent_annotation/
  config.py     project configuration; see section 3
  evidence.py   SAE activations -> top-activating windows + firing stats (no torch/numpy)
  tools.py      ToolUniverse + local tools behind one dispatcher (ToolBox)
  llm.py        ChatClient protocol; HuggingFaceChatClient (Qwen-3) + MockChatClient
  agent.py      the per-latent tool-use loop (LatentAnnotationAgent)
  schema.py     annotation JSON schema, system prompt, JSON extraction/validation
  __init__.py   empty by design; see section 6
examples/
  make_fixture.py       generates examples/fixtures/latents.json (synthetic activations)
  annotate_latents.py   end-to-end demo: evidence -> agent -> annotations.jsonl
main.py                 ToolUniverse discovery helper (search the full catalog)
notes/                  these documents
```

The six modules and two scripts contain roughly 1,000 lines. Three interfaces
separate the core loop from model calls, database requests, and SAE code:

| Seam | Module | What it lets you do without the expensive thing |
|---|---|---|
| `ChatClient` protocol | `llm.py` | Run the whole loop offline with `MockChatClient` |
| `LatentEvidenceStore` | `evidence.py` | Build evidence from any activation dump, no SAE code in this repo |
| `ToolBox` | `tools.py` | Run with `--no-tooluniverse` when the databases are down or slow |

## 2. Data flow of one annotation

```
fixture (proteins + activations)
  -> evidence.LatentEvidenceStore.from_json        build windows + firing stats
  -> tools.evidence_tools(store)                   get_latent_evidence / list_latents
  -> tools.ToolBox                                 local tools + curated ToolUniverse tools
  -> agent.LatentAnnotationAgent.annotate(latent)  loop over llm.ChatClient
  -> schema.validate_annotation                    enforce ANNOTATION_SCHEMA client-side
  -> annotations.jsonl                             annotation + tool trace + usage
```

The loop inside `annotate` has three branches:

1. **The model requests tools:** run them, append the results, and continue.
2. **The model answers before calling `get_latent_evidence`:** send
   `NUDGE_MISSING_EVIDENCE` once.
3. **The reply is not JSON or fails the schema:** send the specific errors
   back and ask for a correction, up to `max_repair_attempts`.

`config` imports no package modules. Other modules depend on it in this direction:

```
config.py            (imports nothing from the package)
   ^-- evidence.py, tools.py, llm.py, agent.py, examples/, main.py
evidence.py  <- tools.py      (LatentEvidenceStore type hint + evidence_tools)
llm.py       <- agent.py      (ChatClient, ChatResponse)
schema.py    <- agent.py      (ANNOTATION_SCHEMA, SYSTEM_PROMPT, extract/validate)
tools.py     <- agent.py      (ToolBox)
```

There are no cycles and no module imports the package root.

## 3. The central config (`config.py`)

`latent_annotation/config.py` groups settings by the module that reads them:

| Group | Constants | Consumed by |
|---|---|---|
| Paths | `REPO_ROOT`, `FIXTURE_PATH`, `OUTPUT_PATH`, `ENV_FILE` | both example scripts, `llm.load_client` |
| Env var names | `ENV_LLM_PROVIDER`, `ENV_LLM_MODEL`, `ENV_HF_TOKEN`, `ENV_HF_API_KEY` | `llm.load_client` |
| Model layer | `DEFAULT_MODEL`, `DEFAULT_HF_PROVIDER`, `TEMPERATURE`, `MAX_TOKENS`, `TIMEOUT_SECONDS`, `USE_RESPONSE_FORMAT`, `MOCK_LOOKUP_TOOL` | `HuggingFaceChatClient`, `MockChatClient` |
| Agent loop | `AgentConfig` (`max_turns`, `require_evidence_tool`, `max_repair_attempts`) | `LatentAnnotationAgent` |
| Evidence | `TOP_K_EXAMPLES`, `WINDOW_SIZE`, `HIGHLIGHT_SPAN`, `ACTIVE_THRESHOLD` | `build_latent_evidence` |
| Tool surface | `PROTEIN_CONTEXT_TOOLS`, `MUTATION_EFFECT_TOOLS`, `TOOL_RESULT_LIMIT` | `ToolBox` assembly, `_stringify`, example, `main.py` |

Settings fall into two groups:

- **Code settings** (tool lists, windowing, and loop limits) live in `config.py`.
  Functions use these values as defaults, so an experiment can still pass
  `build_latent_evidence(..., top_k=4)` without changing the project default.
- **Environment settings** (provider, model id, token): set the variable in
  `.env` (see `.env.example`). The config constants are the *names* of those
  variables plus the fallback defaults; the runtime values are read by
  `llm.load_client`.

Some constants stay beside the code that gives them meaning:

- **Prompts.** `SYSTEM_PROMPT` sits next to `ANNOTATION_SCHEMA` in `schema.py`
  because its output instructions restate that schema. `NUDGE_MISSING_EVIDENCE`
  sits next to the branch that sends it in `agent.py`.
- **The schema enums** (`CONCEPT_TYPES`, `CONFIDENCE_LEVELS`,
  `MUTATION_SENSITIVITY`). They are the output contract, not knobs.
- **CLI defaults** (`--limit 25` in `main.py`). Argparse already documents them.

Keep `config.py` free of package imports and application logic.

## 4. Previous cleanup

Two cleanup passes removed unused state and duplicate settings. Each symbol was
searched across the source and documentation before removal.

### Pass 1: consolidate settings

| Removed | From | Why |
|---|---|---|
| `ChatResponse.finish_reason` | `llm.py` | Both clients wrote it, but the loop reads `wants_tools` and `text` instead. |
| `LatentAnnotationAgent.annotate_many` | `agent.py` | It had no caller, and the example script already contained the same sequential loop. |
| `LatentEvidenceStore.__contains__` | `evidence.py` | Magic method never invoked; lookups go through `get`. |
| Package-root re-exports (`from latent_annotation import ...`) | `main.py`, `examples/`, README | `__init__.py` is empty, so these imports raised `ImportError`. |
| Module-local `DEFAULT_MODEL` and friends | `llm.py`, `tools.py`, `evidence.py`, `agent.py` | Moved to `config.py`, so each setting lives in exactly one file. |

Pass 1 collected settings that had been scattered across four files: the model
defaults and env-var strings in `llm.py`; `AgentConfig`'s defaults in `agent.py`;
`top_k` / `window` / `highlight_span` / `active_threshold` in `evidence.py`; the
two tool tuples and the truncation limit in `tools.py`; and the fixture/output
paths in `examples/annotate_latents.py`.

### Pass 2: remove unused state and duplication

| Removed | From | Why |
|---|---|---|
| `AnnotationResult.ok` | `agent.py` | It had no readers. The demo derives pass or fail from `annotation is None` and `schema_errors`. |
| `AnnotationResult.transcript` | `agent.py` | It was written at every exit and never read. `tool_calls` already records the trace saved to `annotations.jsonl`. |
| `MockChatClient.calls` | `llm.py` | Counter incremented every turn, never read. |
| `LatentExample.window_sequence` | `evidence.py` | `highlighted` contains the same sequence with brackets around the peak, so serializing both fields sent each window twice. |
| `AgentConfig` imported via `agent.py` | `examples/annotate_latents.py` | Re-export indirection: `AgentConfig` is defined in `config.py`, and importing it through `agent.py` contradicted the repo's own "name the defining module" convention. |
| Explicit `config=AgentConfig()` argument | `examples/annotate_latents.py` | Restated the constructor default. |
| Duplicate fixture path (`OUT = ...`) | `examples/make_fixture.py` | It repeated `config.FIXTURE_PATH`, which could let the writer and reader drift apart. |
| Hardcoded `"UniProt_get_function_by_accession"` | `llm.py` | The mock now reads `config.MOCK_LOOKUP_TOOL`, beside `PROTEIN_CONTEXT_TOOLS`, which must contain the selected tool. |

After pass 2, `ruff check --select F,ARG,ERA,SIM,B` reported only the intentional
`ARG002` described below. Both `--show-evidence` with 11 tools and the mock run
completed; all three fixture annotations passed schema validation.

## 5. Code kept on purpose

- **`MUTATION_EFFECT_TOOLS`** is read only by `main.py`, for display. It is the
  reserved tool surface for the planned mutation-effect agent (notes 03 and 04).
- **`MockChatClient.complete`'s unused `response_schema` parameter** (the one
  remaining ruff `ARG002`). Required to satisfy the `ChatClient` protocol; the
  mock does not use a schema that it is hardcoded to match.
- **`LatentExample.peak_index` / `window_start` / `description`** are never read
  by our code, but they are serialized into the evidence bundle the model reads.
  Position in the chain helps distinguish a `positional_or_terminal` latent from a
  `sequence_motif` latent.
- **`use_response_format`, `timeout`, `temperature`, `max_tokens` constructor
  params** currently use their config defaults. Benchmark scripts can override
  them, and `use_response_format` disables the option for HF providers that
  mishandle it.
- **`ToolBox.names()`** has a single caller (`--show-evidence`), which is the
  only way to see the tool surface without spending a model call on it.
- **`list_latents`** is not called in fixture runs because the demo passes latent
  ids explicitly. It is needed when an agent receives a store without a latent id.

## 6. Conventions

- **`__init__.py` stays empty.** No re-exports, no side effects. Import from the
  defining submodule: `from latent_annotation.evidence import LatentEvidenceStore`.
  This makes the defining module visible at each import site and avoids a second
  package-level API. Pass 2 removed the indirect `AgentConfig` import for this
  reason.
- **Config is a leaf.** See section 3.
- **Defaults come from config, overrides are per call.** A signature names its
  config default (`top_k: int = config.TOP_K_EXAMPLES`); callers who need
  something else pass it explicitly.
- **No heavy dependencies in the core.** `evidence.py` is plain lists and JSON,
  so the evidence layer runs and tests without torch or numpy. `tooluniverse`
  and `huggingface_hub` are imported lazily, so the rest of the package imports
  when neither dependency is installed.
- **Tool errors are results, not exceptions.** `ToolBox.run` returns
  `(text, is_error)` so a failed accession lookup makes the agent try another
  accession instead of killing the run.
- **The agent loop is explicit, not a framework.** Tool traces are research
  artifacts. The code handles unsupported answers and malformed JSON at the point
  where they occur.
- **Keep coupled values together.** This rule puts `MOCK_LOOKUP_TOOL` and
  `FIXTURE_PATH` in config.

## 7. How to extend

- **Add a ToolUniverse tool to the agent surface:** add its name to
  `PROTEIN_CONTEXT_TOOLS` (or `MUTATION_EFFECT_TOOLS`) in `config.py`. Before you
  do, identify the annotation field it supports. If it supports none, leave it out.
  Use `uv run python main.py <pattern>` to discover names and `--spec <name>` for
  the full spec. (The catalog is ~2,600
  tools and changes with each ToolUniverse release, so use `main.py`'s printed
  count over any number written in these notes.)
- **Swap the model:** set `LLM_PROVIDER` or `LLM_MODEL` in `.env`. To support
  another service, add a `ChatClient` implementation and connect it in
  `llm.load_client`; the agent does not use a vendor SDK directly.
- **Change evidence presentation:** window size, example count, highlight span,
  firing threshold are all in `config.py`, with no call sites to chase.
- **Add a second agent** (e.g. the mutation-effect agent): a tool tuple, a
  schema, and a system prompt, assembled as in `examples/annotate_latents.py`. The
  loop in `agent.py` can be reused when the new agent follows the same sequence:
  evidence, database calls, then structured output.
- **Plug in your own SAE:** the only coupling point is
  `LatentEvidenceStore.from_activation_dump`, which takes
  `{latent_id: {protein_id: [per-residue activation, ...]}}` plus a list of
  `ProteinRecord`. Nothing downstream changes.

## 8. Known risks

- **`MockChatClient` recovers state by rereading the message list,** including
  a regex over the user turn to find the latent id (`_requested_latent`). It is
  coupled to the exact wording of the task string in `agent.annotate`. Change
  that string and the mock silently annotates `"unknown"`.
- **`TOOL_RESULT_LIMIT` truncates mid-JSON.** An 8,000-character cut of a UniProt
  record is not valid JSON, and the model sees the fragment. A tool that returns
  one large object may therefore lose the relevant field. Prefer narrow accessor
  tools.
- **Nothing is tested.** There is no test suite; the mock run is the smoke test.
  `evidence.py` has no third-party dependencies and is the easiest place to add
  unit tests first (see notes 04).
