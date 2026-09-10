# Repository deep dive

This note is a code-first map of the repository as inspected on 2026-09-03. It
describes what runs today, what is deliberately only a prototype, and where the
main scientific and engineering risks are.

## Executive summary

`latent-annotation` is a small research prototype for assigning grounded,
structured interpretations to sparse-autoencoder (SAE) features in protein
language-model activations. It does not train a protein model or an SAE. Its input
boundary is a set of protein sequences plus per-residue activation vectors. Its
output is one JSONL record per latent containing an LLM-generated annotation,
database evidence, tool-call trace, turn count, and token usage.

The central workflow is:

```text
protein sequences + activation dump
    -> top activation windows and firing statistics
    -> local evidence tool + curated ToolUniverse database tools
    -> explicit model/tool loop per latent
    -> JSON extraction and schema validation
    -> annotations.jsonl
```

The repository is intentionally separated at three seams:

| Seam | Purpose |
|---|---|
| `LatentEvidenceStore` | Lets any SAE export feed the pipeline without importing torch or numpy. |
| `ToolBox` | Presents local evidence and remote ToolUniverse functions through one OpenAI-shaped interface. |
| `ChatClient` | Allows the same loop to run with a mock client or a HuggingFace-served Qwen model. |

The result is a good experiment scaffold, not yet a validated annotation system:
the mock tests plumbing rather than biology, the live model path has no automated
evaluation, and there is no test suite.

## Repository contents

### Package modules

- `latent_annotation/config.py` is the configuration leaf. It owns repository
  paths, environment variable names, model defaults, loop limits, evidence
  windowing, tool allowlists, and the 8,000-character tool-result cap. The two
  allowlists are important: `PROTEIN_CONTEXT_TOOLS` is active in the demo, while
  `MUTATION_EFFECT_TOOLS` is reserved for future work and only displayed by
  `main.py`.
- `latent_annotation/evidence.py` converts raw activation vectors into
  `LatentEvidence`. For every scanned protein it counts values above the threshold,
  selects that protein's maximum activation if positive, sorts proteins by peak,
  returns up to eight windows of 24 residues on either side, and brackets two
  residues on either side of each peak. It also computes firing rate, mean active
  activation, maximum activation, scanned-protein count, and unique accessions.
- `latent_annotation/tools.py` defines `LocalTool` and `ToolBox`. Local tools are
  Python handlers; remote tools are resolved from ToolUniverse specifications and
  executed through `tu.run`. The dispatcher catches tool exceptions and returns
  them as model-visible error results. Definitions are cached after first build.
- `latent_annotation/llm.py` defines the normalized `ToolCall`, `ChatResponse`, and
  `ChatClient` protocol. `HuggingFaceChatClient` uses the HuggingFace Inference
  API, normalizes tool-call arguments, collects token usage, and requests the JSON
  schema only after tool definitions are removed. `MockChatClient` follows a fixed
  path: get latent evidence, look up the first accession if possible, then emit a
  placeholder annotation.
- `latent_annotation/agent.py` owns the per-latent loop. It creates a fresh
  transcript, calls the client, executes every requested tool, nudges a model once
  if it answers before calling `get_latent_evidence`, repairs malformed or
  schema-invalid JSON up to two times, and returns an `AnnotationResult` with
  annotation, trace, usage, and errors.
- `latent_annotation/schema.py` owns the output contract and prompt. The schema
  restricts concept type, confidence, and mutation-sensitivity enums; requires
  evidence, rejected hypotheses, a falsification condition, and mutation-effect
  rationale; and disallows additional properties. `extract_json` tolerates clean
  JSON, fenced JSON, and JSON surrounded by prose.
- `latent_annotation/__init__.py` is intentionally empty. Imports are made from
  defining modules rather than package-level re-exports.

### Scripts and data

- `examples/make_fixture.py` creates synthetic, illustrative activation data for
  eight real accessions. The sequences are fragments, not fetched full UniProt
  records. The three latent patterns are a glycine-rich motif (`L-0142`), acidic
  pairs (`L-0731`), and any aromatic residue (`L-1503`). The patterns are designed
  to exercise narrow, broad, and uninterpretable cases.
- `examples/annotate_latents.py` is the end-to-end CLI. It loads the fixture,
  builds a toolbox, optionally prints evidence, selects a client, annotates each
  requested latent sequentially, and writes JSONL. `--no-tooluniverse` is the
  reliable fully local mode.
- `main.py` is a ToolUniverse discovery helper, not part of annotation execution.
  It loads the whole catalog, prints the selected allowlists, searches tool names,
  or prints an OpenAI spec for one named tool.
- `examples/fixtures/latents.json` is the checked-in generated fixture: 8 protein
  records and activation vectors for 3 latents.
- `examples/annotations.jsonl` is a checked-in/generated three-line example output
  and is ignored for future changes by `.gitignore`.
- `uv.lock` pins the Python dependency resolution. `pyproject.toml` requires
  Python >=3.13 and depends on ToolUniverse, HuggingFace Hub, dotenv, and
  jsonschema.
- `notes/01` through `05` contain the design rationale, ToolUniverse selection,
  SAE/mutation-effect research plan, roadmap, and a prior code/config map.

## Detailed execution path

1. The CLI loads JSON and constructs `ProteinRecord` objects.
2. `LatentEvidenceStore.from_activation_dump` calls
   `build_latent_evidence` once per latent. The evidence is retained in memory,
   keyed by latent id.
3. `evidence_tools` exposes `get_latent_evidence` and `list_latents`. The CLI
   normally adds nine remote database functions from ToolUniverse.
4. `LatentAnnotationAgent.annotate("L-0142")` starts with a system prompt and a
   user instruction requiring evidence first.
5. The model either asks for tools or emits a candidate final answer. Tool calls
   are serialized into the transcript, executed by `ToolBox`, and returned as
   `role=tool` messages.
6. The local evidence result supplies peak windows and accessions. The prompt asks
   the model to compare residue positions with database features, domains,
   localizations, PTMs, and pathways, and to test an alternative explanation.
7. Once the model stops calling tools, the result is parsed and validated. Failed
   parsing or validation gets a targeted repair prompt. Exhausted retries return
   an error rather than pretending the record is clean.
8. The example writes the annotation and audit metadata as one JSON object per
   line. It returns a nonzero exit code if any latent fails or has schema errors.

## Evidence semantics and limitations

The evidence algorithm chooses at most one example per protein because it retains
only each protein's single maximum. Thus `TOP_K_EXAMPLES=8` means up to eight
proteins, not eight independent firing sites. It has no non-maximum suppression or
diversity criterion beyond protein identity. Examples are therefore an extreme,
protein-level sample and can be dominated by corpus composition.

The activation vector is assumed to be aligned with the sequence, but the code
does not validate equal lengths, residue alphabet, finite numeric values, or
duplicate protein ids. A malformed longer activation vector can produce a peak
index outside the sequence, while a shorter one silently changes the denominator
of the firing rate. These should be explicit input checks before real SAE sweeps.

The threshold is `> 0.0`, so zero is inactive and negative values are inactive.
`firing_rate` is active activation entries divided by activation-vector entries,
not necessarily by sequence residues. The peak index is zero-based; the output
does not state that explicitly, so downstream feature-position comparisons need a
documented conversion to database conventions.

## Tool and model behavior

The nine default database tools are deliberately narrow: recommended name,
function, features, subcellular location, PTM/processing, InterPro entries and
architecture, Pfam annotations, and Reactome pathways. This is appropriate for
protein context but not enough for mutation-effect evaluation. The future mutation
agent has disease variants, MaveDB, ClinVar, and gnomAD tools, but no code currently
assembles or runs that agent.

Tool results are serialized and truncated at 8,000 characters. This protects the
context window but can cut a JSON object in the middle, which makes the returned
text hard for a model to use. Narrower endpoint-specific accessors or structured
field extraction would be safer than simply increasing the limit.

The HuggingFace client uses `provider="auto"` by default, so the serving provider
can vary between runs. Tool calling and `response_format` support can vary with
that provider. The code intentionally sends `response_format` only on turns with
no tools, then validates locally anyway. For comparable experiments, provider,
model, temperature, and data version should be recorded and pinned.

The mock is a plumbing smoke test, not a biology baseline. It never interprets
sequences, and it usually performs no database call in `--no-tooluniverse` mode.
Its annotation text explicitly says it is a placeholder; a clean mock result must
not be reported as a successful scientific annotation.

## Findings from the runnable checks

Using the existing `.venv` directly:

- `examples/annotate_latents.py --show-evidence --no-tooluniverse` loaded all
  three latents and printed two local tools plus the calculated windows.
- The fully local mock run produced 3/3 clean schema-valid JSONL records. The
  observed firing rates were approximately 0.0449 (`L-0142`), 0.0336 (`L-0731`),
  and 0.0710 (`L-1503`); the mock mapped these to sequence motif, polysemantic,
  and uninterpretable respectively.
- `compileall` completed successfully.
- There is no test directory or test dependency in the repository. Ruff is not
  installed as `.venv/bin/ruff`, so lint status could not be independently rerun.
- `uv run` could not initialize its default cache in this restricted environment;
  this is an environment permission issue, not a source-code failure.
- The worktree already contained a user modification to `latent_annotation/config.py`.
  This investigation did not change that file.

## Highest-priority risks and fixes

1. **No evaluation loop.** Add held-out activation prediction, random-label
   controls, repeated annotations, and a human-audited set before using labels as
   scientific measurements.
2. **No automated tests.** Start with unit tests for `_highlight`, empty/no-active
   activations, ties, top-k behavior, malformed lengths, JSON extraction, schema
   repair, and tool-error routing. Add one deterministic end-to-end mock test.
3. **Evidence is not enforced strongly enough.** The agent requires the local
   evidence call, but the schema does not actually require a database evidence
   entry despite its description. A local-only run can therefore pass validation
   with activation evidence alone. Add a schema-level or agent-level policy if
   remote grounding is mandatory.
4. **Environment alias inconsistency.** `HUGGINGFACE_API_KEY` is accepted when
   constructing `HuggingFaceChatClient`, but `load_client` only auto-selects
   HuggingFace when `HF_TOKEN` is present. A user who sets only the documented
   alias can unexpectedly get the mock client. Detect both variables in the
   provider fallback.
5. **Synthetic fixture can be over-read.** Its accessions are real but its
   sequences are illustrative fragments. Their database records cannot validate
   exact positions in the fragments without an explicit fragment offset and full
   sequence provenance.
6. **Operational scaling is incomplete.** The CLI overwrites output, runs
   sequentially, has no resume/checkpoint skip, and does not persist ToolUniverse
   cache. These are acceptable for a demo but unsafe for thousands of latents.
7. **Reproducibility metadata is thin.** JSONL lacks fixture hash, model/provider,
   prompt/schema version, tool catalog version, timestamps, and wall time. Add
   these before comparing experiments.

## Recommended implementation order

1. Add validation and focused tests around the evidence and schema seams.
2. Fix provider fallback and make the offline/live mode distinction explicit in
   the README and CLI output.
3. Add resumable JSONL execution, cache persistence, request throttling, and
   bounded concurrency.
4. Replace fragments with a versioned real SAE export plus sequence offsets and
   corpus metadata.
5. Add annotation-quality evaluation and controls.
6. Build the separate mutation-effect agent only after annotation quality and
   reproducibility are measured.

The existing `notes/04-roadmap.md` has the broader research roadmap; this note is
the implementation-oriented companion and should be updated when the code changes.
