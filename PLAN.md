---
date: 09/22/2026
---

# plan of code

Concrete implementation plan derived from [[plan of action]] and [[central problem]].

**Working assumption: no code exists.** Sai's pipeline is not available, so everything below is built from scratch. That is a cost but also a license — I am not constrained to reproduce their architecture, and the thing worth building is not a script that produces one circuit for one mutation. It is an **agent that takes `(protein, mutation)` and returns a mechanistic, structurally-grounded, falsifiable explanation**, with no human in PyMOL at any point.

That directly answers the plan-of-action item: *"tool universe have to right now manually annotate the protein, but how can we make it work without the manual annotation."*

---

## the architectural bet

The tempting design is "give an LLM the tools and let it figure out the biology." That is the wrong bet and it will produce Table 2 all over again — plausible prose, wrong role. Latent 1695 gets called "SAM domains" because a language model was asked to do a job that geometry should do.

So the split is:

| layer | who does it | why |
|---|---|---|
| **structural facts** | deterministic code | contact sets, burial, motif spans, category assignment are functions of a PDB file. An LLM adds noise, not information. |
| **orchestration** | agent | which layer, which mutation, which latents, when to escalate, when to stop, what to re-run when a result looks wrong. This is genuinely a search problem and it is what currently eats my time. |
| **hypothesis + narration** | agent | "L7D buries a charge next to Y22/Y48" is a claim worth generating in language — but only as a *proposal that then gets tested*. |
| **verification** | deterministic code | every sentence the agent emits must carry a citation to a computed number, or it does not ship. |

The agent is therefore a **planner over verified tools**, not an oracle. Every claim in the final report is traceable to a tool call with a recorded return value. If it isn't, the report generator drops it.

---

## repo layout

```
dnaja1-agent/
  pyproject.toml              # uv, python 3.11
  src/dja/
    tools/                    # THE DETERMINISTIC LAYER. every fn is @tool-decorated,
      __init__.py             #   typed, pure, cached, and independently unit-tested.
      structure.py            #   fetch_pdb, contact_map, rel_sasa, secondary_structure
      numbering.py            #   align_constructs, map_index  (uniprot <-> pdb <-> ProteinGym)
      annotations.py          #   interpro_domains, uniprot_features, motif_spans
      esm.py                  #   masked_marginal_llr, logit_distribution
      sae.py                  #   encode_latents, latent_token_grid
      ablate.py               #   ablate_pair, restore_pair, js_recovery
      attrib.py               #   integrated_gradients
      circuit.py              #   build_circuit(tau), prune_minimal
      annotate.py             #   categorize_latent  <- the automation deliverable
      nulls.py                #   random_latent_null, random_residue_null
      steer.py                #   clamp_latent, measure_llr_shift
    registry.py               # tool schemas -> MCP server + ToolUniverse registration
    agent/
      loop.py                 # the orchestration loop
      policies.py             # stopping rules, escalation, budget
      verify.py               # claim -> supporting tool call, or drop
      report.py               # structured result -> markdown + JSON
    provenance.py             # git sha + config hash + timing on every artifact
  data/{raw,interim,results}/
  tests/                      # golden tests per tool, synthetic-input tests for agent
  scripts/explain.py          # CLI: explain --protein DNAJA1 --mut L7D
  slurm/
```

Two hard rules:
1. **Every tool is usable without the agent.** `python -m dja.tools.circuit --mut L7D --layer 16` must work. The agent is a convenience layer over a working pipeline, never a dependency of it. If the agent idea collapses, Stages 1–4 still stand as a paper.
2. **Every artifact carries provenance.** No notebook is ever the source of a number in a writeup.

---

## the agent loop

```
explain(protein, mutation):
  ├─ GROUND      deterministic, no LLM. numbering table, structure, contacts,
  │              burial, domain + motif spans. Fails loudly on any assertion.
  ├─ SCORE       ESM-2 masked-marginal LLR; DMS value if available; flag
  │              agreement/disagreement — this is fed to the agent as context.
  ├─ PLAN        agent picks layers + node budget. Not fixed at 16. Policy:
  │              start at a cheap layer, escalate on low recovery or high
  │              cross-layer Jaccard instability.
  ├─ CIRCUIT     attribution -> greedy restore -> tau -> backward prune.
  ├─ ANNOTATE    deterministic categorize_latent over ALL nodes (not top-15).
  ├─ HYPOTHESIZE agent reads the annotation profile + structural facts and
  │              proposes a mechanism in language, with named residues.
  ├─ TEST        agent selects falsification tests from a fixed menu:
  │                - steer the implicated latents, predict LLR sign, check
  │                - null comparison (random latent / residue sets)
  │                - homolog re-ablation
  │                - leave-category-out recovery
  ├─ VERIFY      every claim must map to a recorded tool return. unsupported
  │              claims are dropped, not softened.
  └─ REPORT      markdown + JSON, with the failed tests included.
```

The `TEST` step is the part that makes this more than an automation wrapper. The paper's weakness is that plausibility substitutes for falsification; here, an explanation that survives no test is reported as *unsupported* rather than as a finding.

---

## Stage 0 — skeleton + tool contracts (days 1–3)

Build the shape before the science, so every later stage drops into a slot.

- [ ] `uv` env; `torch`, HF `facebook/esm2_t33_650M_UR50D`, `biotite`, `httpx`, an agent SDK (Claude API tool-use loop is enough — no framework needed).
- [ ] `@tool` decorator: JSON schema from type hints, disk cache keyed on args, provenance stamping, structured errors (`ToolError` with a `retryable` flag the agent can act on).
- [ ] Write **every tool signature as a stub that raises NotImplementedError**, plus its test. The agent loop can then be developed against mocked tools on day 3, in parallel with filling the stubs.
- [ ] Unity: GPU allocation, `slurm/` templates. Paper's budget is ~41 min/mutation/layer, worst case 3 h → 67 subs x 8 layers ≈ a GPU-week. Everything checkpoints per `(mutation, layer)` and is resumable; the agent must be able to *submit and poll* rather than block.

Deliverable: `scripts/explain.py --mut L7D --dry-run` walks the full loop against mocked tools and prints a report skeleton. Showable immediately, and it makes the architecture legible to Sai before any GPU time is spent.

## Stage 1 — the grounding tools (days 3–7, no GPU)

This is the plan-of-action item *"look at interprot and protein bank data for DNAJA1."* Pure code, high value, zero risk.

- [ ] `structure.fetch_pdb` — 2LO1 from RCSB, chain + model 1, sequence out.
- [ ] `numbering.align_constructs` — align 2LO1, UniProt P31689, ProteinGym construct. Emit `data/interim/numbering.tsv` (`construct_idx, uniprot_idx, pdb_resnum, aa`). **Assert** construct 7 is Leu and 30–32 is HPD; fail loudly otherwise. Settles the numbering ambiguity the paper leaves implicit.
- [ ] Identity check: is BC008182 actually DNAJA1 or a paralog? Align against P31689, record % identity in the artifact. One line, kills a reviewer question.
- [ ] `structure.contact_map` — heavy-atom min distance ≤ 5 Å. Confirm Y22/Y48 ∈ contacts(L7). `rel_sasa` to confirm L7/L51 buried, L27 not.
- [ ] `annotations.interpro_domains` — J-domain boundaries vs the 67-residue fragment; `motif_spans` returns HPD programmatically rather than hard-coded.

Deliverable: `notes/substrate.md` — numbering table, 2LO1 identity call, contact sets for all four sites, burial numbers, InterPro boundaries. **Week-1 showable.**

## Stage 2 — the scoring + circuit tools (week 2)

- [ ] `esm.masked_marginal_llr` (Eq. 1). Sanity gate: Spearman vs ProteinGym DNAJA1 DMS ≈ 0.803. If I can't hit that, nothing downstream is trustworthy — **this is the first real number to show Sai.**
- [ ] `sae.latent_token_grid` — InterProt SAE, the 266,112 candidate pairs.
- [ ] `ablate` + `attrib` + `circuit.build_circuit` — zero-ablate a pair, indirect effect on LLR, IG ranking, greedy restore, `RS_JS` normalized by empty-circuit divergence, stop at τ=0.8.
- [ ] Target: L7D, layer 16 → ~535 nodes (0.20%). Write `data/results/circuits/L7D_L16.json`.
- [ ] `circuit.prune_minimal` — backward prune while `RS_JS ≥ τ`. Greedy IG order gives *sufficiency*, not minimality; report minimal size vs 535. Cheap, and it is a number the paper doesn't have.

Deliverable: recovery curve vs nodes restored, my run against the paper's Fig. 6 numbers, plus a minimality number.

## Stage 3 — nulls (week 3)

The cheapest real contribution, and the agent needs it as a `TEST` primitive anyway.

- [ ] `nulls.random_latent_null` — sample N=535 pairs, measure `RS_JS`, 1000 draws. Also an **IG-magnitude-matched** variant (sample within the same attribution quantile band) so the comparison isn't trivially won.
- [ ] `nulls.random_residue_null` — size-matched residue sets, fraction within 5 Å of the mutation site. Gives a p-value for the paper's 39.1% contact enrichment.
- [ ] Report observed vs null mean/CI and empirical p, for L7D and L51R.

Deliverable: one figure, two p-values, one paragraph. Lead with this in a progress update.

## Stage 4 — automated annotation (weeks 4–5) — **the core deliverable**

The six H.3 categories are a deterministic function of geometry. Write them as one:

```python
@tool
def categorize_latent(latent_attr: np.ndarray,   # per-token attribution, len L
                      contacts: set[int],        # 5 A neighbours of mut_site
                      mut_site: int,
                      motif_spans: dict[str, range],
                      seq_local_window: int = 5) -> LatentRole:
    """-> mass fraction per category, argmax under the paper's H.3 priority
    order, the residue list carrying the mass, and a diffuseness score
    (entropy of normalized attribution over tokens)."""
```

- [ ] Categories: `mutation_site`, `3d_contact`, `sequence_local`, `hpd_motif`, `other_residues`, `unresolved_diffuse` (entropy threshold applied *before* argmax).
- [ ] H.3 priority order preserved exactly, so numbers stay comparable to the paper.
- [ ] **Validation against the human:** run on the top 10–15 latents for L7D, L51R, H30P at layer 16; confusion matrix vs the paper's hand labels. >~80% agreement means the manual bottleneck is gone.
- [ ] **Then exceed the human:** annotate *all* circuit nodes, not just the top 15 — which hand annotation could never do. Redo the Fig. 3 composition over the full 535 and check whether 39.1% survives.

Deliverable: `annotate.py` + confusion matrix + full-circuit composition. Strongest single artifact of the semester, and it generalizes to any protein with a structure.

## Stage 5 — wire the agent to real tools (week 5–6)

Now the mocked loop from Stage 0 gets its actual backends.

- [ ] `registry.py`: expose all tools over MCP; register alongside ToolUniverse's protein tools so the agent sees one namespace.
- [ ] `agent/policies.py`: layer-selection policy (escalate when recovery is low or cross-layer Jaccard is unstable — Table 3 swings 38.3%→100%, so "layer 16 is representative" is an assumption the agent should *test* rather than inherit), node budget, wall-clock budget, retry-on-`ToolError`.
- [ ] `agent/verify.py`: parse the agent's claims, require each to name a tool call in the trace. Drop unsupported ones. Log the drop rate — **that number is itself a result** about whether LLMs can be trusted here.
- [ ] Evaluation of the agent, not just the tools: on L7D/L51R/L27x/H30P, does the agent's report match what I'd write by hand from the same artifacts? Score on (a) correct residues named, (b) correct category profile, (c) no unsupported claims.

Deliverable: `scripts/explain.py --mut L7D` end-to-end, report + trace. This is the demo.

## Stage 6 — ToolUniverse, properly (week 6)

- [ ] Wrap `ESM_describe_sae_feature`; reproduce Table 2's failure (latent 1695 → "SAM domains").
- [ ] Swap the ten-protein default panel (TP53, EGFR, KRAS, … — no J-domain protein among them) for a J-domain-rich panel: DNAJA1/A2, DNAJB1, E. coli DnaJ. Re-run.
- [ ] Either outcome is publishable: improvement ⇒ the tool was misconfigured, not broken; no improvement ⇒ **corpus-level labels cannot recover per-mutation causal role**, which is the cleaner and more interesting claim, and it is the justification for the whole structural-annotation approach.
- [ ] Head-to-head table: ToolUniverse's language label vs `categorize_latent`'s structural label, same latents. That table is the figure.

Docs: https://zitniklab.hms.harvard.edu/ToolUniverse/ · skill: https://explainx.ai/skills/mims-harvard/tooluniverse/tooluniverse-protein-structure-retrieval

## Stage 7 — falsification tools (week 7+, gated on Sai)

These are the `TEST` menu entries. Each is a tool the agent can select.

- [ ] `steer.clamp_latent` — clamp top `3d_contact` latents, check the LLR moves in the predicted direction with the predicted sign per substitution class (hydrophobic vs charged). Log effect sizes, not just signs.
- [ ] **Agreement split** — partition the 67 substitutions by |ESM-2 rank − DMS rank|, build circuits for both halves, test whether contact-latent fraction differs. If contact-enriched circuits appear where ESM-2 is *wrong*, the plausibility bridge breaks. This is the falsification test the paper lacks.
- [ ] `leave_category_out` — ablate all latents of one category, measure recovery loss. Gives per-category necessity, complementing per-latent sufficiency.

## Stage 8 — stretch: second protein

Any ProteinGym DMS with a solved structure and a non-helical-bundle fold. If Stages 1–4 were written protein-generically (they are, by construction — nothing hard-codes DNAJA1 outside `data/`), the cost is compute, not code. **This is the real payoff of the agent framing:** a new protein should be one CLI invocation, not a week in PyMOL. Tests whether "contacts dominate the circuit at buried sites" is about J-domains or about ESM-2.

---

## write order

1. `tools/@tool` decorator + every stub + its test — day 1, unblocks everything.
2. `agent/loop.py` against mocked tools — day 2–3. Architecture is showable before any science.
3. `structure.py` / `numbering.py` — real, no GPU, week-1 deliverable.
4. `esm.py` — the ρ=0.803 gate.
5. `annotate.py` and `nulls.py` — **write and unit-test against synthetic attribution grids before any real circuit exists.** They only *depend* on circuits for inputs, not for development. Then the day Stage 2 produces a real circuit, the analysis runs the same day.
6. `circuit.py` — the GPU-heavy one, last, once everything that consumes it is already tested.

## checkpoints to show

| when | artifact |
|---|---|
| end wk 1 | mocked agent loop + `notes/substrate.md` (numbering, contacts, burial, InterPro) |
| end wk 2 | reproduced ρ ≈ 0.803; L7D layer-16 circuit + recovery curve + minimality |
| end wk 3 | null distributions, p-value for the 39.1% contact claim |
| end wk 5 | `categorize_latent` + confusion matrix vs hand labels + full-circuit composition |
| end wk 6 | `explain --mut L7D` end-to-end; ToolUniverse panel-swap table; claim-drop rate |
| end wk 8 | falsification results (steering, agreement split), or second protein |

## the risk I should name out loud

The agent layer is the part most likely to be judged as engineering rather than research. The defence is Stage 5's evaluation and the claim-drop rate: *how often does an LLM assert an unsupported mechanism when it has verified structural tools available?* That is a real question about agent reliability in a domain with ground truth, and it is measurable here in a way it usually isn't. If Sai wants the thesis to be validation rather than automation, that number is the bridge between the two.
