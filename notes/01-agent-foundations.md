# Building the annotation agent

## An agent is a loop, not a model

The agent is this loop:

```
messages = [system, user_task]
while True:
    reply = model(messages, tools)
    if reply has no tool calls: return reply.text
    results = [run(call) for call in reply.tool_calls]
    messages += [reply, *results]
```

`latent_annotation/agent.py` adds bookkeeping and error handling to these seven
lines. The loop stays in the repository because its failures are part of the
experiment. We need to record when a model labels a latent before reading its
activations. A framework could hide that branch behind retries or callbacks.

The model does not run tools. It emits a structured request ("call
`UniProt_get_function_by_accession` with `accession=P00533`"); your code runs it and
hands the text back. Every capability an agent has is a capability you wrote and
exposed.

## Three design decisions

**1. Which tools the model sees.** Every tool definition consumes context on every
request. Models also make more selection errors when many tools have similar names
and descriptions. The default surface therefore exposes nine ToolUniverse tools,
not the full catalog. A description should tell the model when to call the tool.
For example, `get_latent_evidence` says that it must be called first and that it is
the source of activation data. "Returns latent evidence" would not give the model
that routing instruction.

**2. Which evidence the model sees.** Each run contains one latent. The model pulls
its evidence through `get_latent_evidence`; the system prompt does not contain the
whole store. Prompt size per annotation therefore does not grow with the number of
latents, and evidence from one latent cannot leak into the next run.

**3. What the model must return.** A free-text label is hard to score, compare
across SAE runs, or audit. The JSON schema in `schema.py` requires a source for
each claim, at least one rejected hypothesis, and a falsification condition.

## Grounding is the point

An LLM can answer "what does a latent firing on GxGxxG mean?" from memory. That
answer gives us no record to inspect. With a UniProt tool, the model must cite an
accession and use the retrieved entry. We can then rerun the lookup and check the
claim.

Two guardrails in this repo enforce it:

- `require_evidence_tool`: if the model answers without calling
  `get_latent_evidence`, the loop pushes back once before accepting anything. An
  annotation made without activation data is unsupported, even if the label sounds
  plausible.
- Every `evidence` entry carries `source_tool` and `source_id`, so a reviewer can
  re-run the lookup.

## Working with an open-weights model

Qwen-3 responses vary across HuggingFace providers. The loop handles the cases we
have observed:

| Behaviour | Handling in this repo |
|---|---|
| Wraps JSON in a ```json fence, or adds prose | `extract_json` tries fenced, raw, and brace-slice candidates |
| Ignores `response_format` on some HF providers | Schema validated client-side by `validate_annotation` |
| Emits tool arguments as a JSON *string* | `_coerce_arguments` normalizes both shapes |
| Answers without calling tools | one nudge; an unsupported final answer remains visible in the result |
| Returns malformed JSON | up to `max_repair_attempts` correction turns with the specific errors |

Schema validation and repair are part of the agent loop. They cannot be deferred
to a later cleanup step because downstream code expects a valid annotation object.

## Failure modes to watch for in this project

- **A label for every latent.** A model can find *a* pattern in almost any set of sequences.
  Broad, high-firing-rate latents should come back `polysemantic` or
  `uninterpretable`. The fixture in `examples/make_fixture.py` includes an aromatic-
  residue latent to test this behavior.
- **Confirmation loops.** The model forms a hypothesis from example 1 and looks up
  only proteins that confirm it. Mitigation: the schema requires a disconfirming
  case, and the prompt asks for the site the label does *not* explain.
- **Context exhaustion.** A UniProt entry can contain tens of thousands of
  characters. `_stringify` stops at 8,000 so one lookup cannot fill the context.
- **Silent tool errors.** Tool failures return as `is_error` results the model can
  see and route around, not exceptions that abort the run.

## Where to go deeper

- Anthropic, *Building Effective Agents* (2024), on when to use an agent instead of
  a fixed workflow.
- The ReAct paper (Yao et al., 2022), which describes the interleaved reasoning and
  action pattern used here.
- Toolformer (Schick et al., 2023), on tool use as a learned capability.
- ToolUniverse / *Democratizing Biomedical AI Agents*, the scientific-tool layer
  used here (see [02-tooluniverse.md](02-tooluniverse.md)).
