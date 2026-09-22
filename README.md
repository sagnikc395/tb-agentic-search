# latent-annotation

An agent that takes `(protein, mutation)` and returns a mechanistic, structurally
grounded, falsifiable explanation of the SAE latents involved — with no human in
PyMOL at any point.

The architectural bet from [`PLAN.md`](PLAN.md) is a split of labour:

| layer | who does it |
|---|---|
| structural facts (contacts, burial, motif spans, latent category) | deterministic code |
| orchestration (which layer, which latents, when to escalate/stop) | agent |
| hypothesis + narration | agent, as a *proposal to be tested* |
| verification (every claim → a recorded tool return) | deterministic code |

Protein evidence comes from UniProt, InterPro, and related resources through
[ToolUniverse](https://github.com/mims-harvard/ToolUniverse). Claims that cannot be
traced to a tool call in the recorded trace are dropped rather than softened.

## Status

**Stage 0 of the plan, plus the first slice of Stage 1.** The full loop
(`GROUND → SCORE → PLAN → CIRCUIT → ANNOTATE → HYPOTHESIZE → TEST → VERIFY → REPORT`)
is walked end to end, but only `ground` (in ToolUniverse mode) and `categorize_latent`
are real. ESM scores, SAE activations, circuit construction, 3D contacts, and agent
hypothesis generation are mocked. Nothing here is a biological result yet; it exists
so the orchestration, the deterministic annotation, the citations, and the trace are
showable before any GPU or API time is spent.

## Run it

Not installable yet — run from the repository root.

```bash
python examples/annotate_latents.py --dry-run --mut L7D
python examples/annotate_latents.py --dry-run --mut L7D --json
```

Live evidence through ToolUniverse (DNAJA1 only for now):

```bash
python examples/annotate_latents.py --tooluniverse --protein DNAJA1 --mut L7D
```

That path calls `UniProt_search`, `UniProt_get_sequence_by_accession`,
`UniProt_get_features_by_accession`, `InterPro_get_protein_domains`, and
`InterPro_get_residue_annotations` for reviewed human DNAJA1 (`P31689`). It fails
loudly on a missing ToolUniverse install or an API error, and records — rather than
hides — numbering mismatches: the live UniProt sequence reports Y7 where the plan's
construct mutation is L7D. That discrepancy surfaces in the report as a numbering
warning and is exactly what Stage 1's `numbering.align_constructs` must settle.

## Layout

```
latent_annotation/
  schema.py                 LatentRole, ToolCall, the six H.3 categories
  tools.py                  mutation parsing, offline fixture grounding, mock LLR,
                            mock latents, and categorize_latent (deterministic)
  tooluniverse_adapter.py   live UniProt/InterPro grounding via ToolUniverse
  agent.py                  the loop: explain() plus Markdown/JSON reporting
examples/
  annotate_latents.py       CLI over the loop
PLAN.md                     staged plan this repo is being built against
```

`categorize_latent` is the core deliverable in miniature: it assigns per-token
attribution mass to `mutation_site`, `3d_contact`, `sequence_local`, `hpd_motif`,
`other_residues`, or `unresolved_diffuse`, applying the paper's H.3 priority order
with an entropy-based diffuseness threshold ahead of the argmax. It already runs on
synthetic attribution grids, as the plan's write order requires, so it is ready the
day a real circuit exists.

## What comes next

Per [`PLAN.md`](PLAN.md):

- **Stage 1** — real `fetch_pdb`, `align_constructs` (asserting construct 7 = Leu and
  30–32 = HPD), heavy-atom `contact_map`, `rel_sasa`, programmatic motif spans.
- **Stage 2** — ESM-2 masked-marginal LLR gated on Spearman ≈ 0.803 vs ProteinGym,
  then attribution → greedy restore → τ=0.8 → backward prune.
- **Stage 3** — random-latent and random-residue nulls, giving a p-value for the
  39.1% contact-enrichment claim.
- **Stage 4** — `categorize_latent` validated against hand labels, then run over all
  circuit nodes rather than the top 15.
- **Stage 5–8** — MCP registry alongside ToolUniverse, claim-drop rate as a result in
  its own right, the ToolUniverse panel swap, falsification tools, second protein.

Two rules hold throughout: every tool must be usable without the agent, and every
artifact must carry provenance.
