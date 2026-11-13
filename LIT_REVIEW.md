# Literature review and v2 plan

Notes on the six papers in `papers/`, what they imply for this project, and how
the v2 harness on this branch acts on them. "Discovery" here means two things in
order: finding the right papers, then pulling TB-resistance findings out of them.

## The papers

| File | Paper | What it is |
| --- | --- | --- |
| `2026.01.30.702134v1.full.pdf` | BIG-TB (Tasmin, Mohanty et al., bioRxiv 2026) | Benchmark of sequence models on 17,000+ *M. tuberculosis* isolates and 11 drugs, scoring both resistance prediction and attribution to known resistance variants. |
| `nihms-2147410.pdf` | Benchmarking generative AI on the WHO TB mutation catalogue (Moreno-Molina et al., BMC Digit Health 2026) | Four chat models answering questions over the 2023 WHO catalogue, scored by hand for accuracy, completeness, citation and hallucination. |
| `2026.01.16.696667v3.full.pdf` | Medea (Sui, Li et al., bioRxiv 2026) | Therapeutic-reasoning agent with four modules: planning, tool-grounded analysis, literature reasoning, and a multi-round discussion that reconciles evidence or abstains. |
| `2405.17631v3.pdf` | BioDiscoveryAgent (Roohani, Lee, Huang et al., ICLR 2025) | Closed-loop agent that picks genes to perturb over five rounds, using previous results, literature search, and a critic agent. |
| `2026.05.12.724604v1.full.pdf` | BiomniBench (Qu, Lu et al., bioRxiv 2026) | 100 data-analysis tasks graded on the agent's whole trace against expert rubrics, across four agent harnesses. |
| `2606.12736v1.pdf` | SciAgentArena (Liu et al., arXiv 2026) | About 200 tasks in five biomedical fields with stepwise verification, run on 18 agents, including tasks whose premise is invalid. |

Two are TB papers; four are about how to build and evaluate scientific agents.

## What each one says that matters here

**BIG-TB.** DNA foundation models do not beat simple baselines (mean test AUC
0.888 for the best CNN against 0.846 for the best DNABERT variant), and the
models that predict best are not the ones that best recover canonical resistance
variants. The paper reads this as evidence of non-causal associations. For this
project it defines the payoff: a high-attribution variant that is not in the
catalogue is either a shortcut or a lead, and literature evidence is one way to
tell them apart.

**WHO catalogue benchmark.** Given the full catalogue as context, every model
hallucinated to some degree, and none was judged fit for clinical use. The
study's notion of fidelity (answer only from the supplied source) and its
definition of hallucination (anything stated as fact that is not in the
source, whether or not it is true) are both directly usable. The catalogue
should be a table the system looks things up in, not text a model reads.

**Medea.** The literature module retrieves from Semantic Scholar and OpenAlex,
screens for contextual relevance, and only then synthesises. The ablation is the
useful part: the literature-only configuration abstains in 79.1% of disease
contexts, while the LLM-only configuration abstains in 1.8% and produces the
most wrong answers (49% to 80% depending on disease). Abstention has to be a
first-class output, and retrieval alone is not enough.

**BioDiscoveryAgent.** Each round's prompt carries the previous rounds' results,
which is what makes the loop closed. Two details transfer. With a weaker model
(Claude 3 Haiku) the literature tool "often resulted in fixation on a few simple
keywords and irrelevant papers". And the agent cites specific papers and line
numbers, and says so when a paper has nothing actionable.

**BiomniBench.** The harness shifts scores by more than the gap between
successive model generations, and frontier and open-weight models land within a
few points of each other. By dimension, source reliability scores 88–98% while
method selection scores 44–67%. So the pipeline is where the effort should go,
and a final-answer score would hide where it fails.

**SciAgentArena.** Agents do well on well-specified workflows and poorly on
open-ended ones. Its sharpest finding for us: doing an analysis well does not
mean the agent notices when the premise is invalid, so validity detection
should be evaluated separately from task completion.

## Where v1 stood

- The one saved retrieval run (`tb_papers.json`) returned no papers. Retrieval
  was a single Biomni A1 run, whose system prompt is about 35k tokens, against
  Groq's per-minute token limit.
- Nothing checked that a returned paper existed. "Never invent titles" lived
  only in the prompt.
- Screening read the first 3,000 characters and returned a yes/no with no
  evidence, so it could not rank papers or say what they were about.
- There was no labelled data and no test suite, so no change could be shown to
  help.

## Design decisions, and where they come from

| Decision in v2 | Source |
| --- | --- |
| Deterministic pipeline; the LLM only writes queries and scores real records | BiomniBench (harness over model); v1's empty run |
| Every returned paper is an API record | WHO benchmark (fidelity); BiomniBench (source reliability) |
| Citation-graph expansion alongside keyword search | BioDiscoveryAgent (keyword fixation); Medea (graph-backed sources) |
| Each round expands from what the last round accepted, with queries required to take new angles | BioDiscoveryAgent (closed loop) |
| Graded relevance, facets, and explicit abstention | Medea (calibrated abstention) |
| Evidence quote checked against the text shown | WHO benchmark (fidelity); BioDiscoveryAgent (line citations) |
| Evaluation built first, with the trace stored per run | BiomniBench (process-level); SciAgentArena (stepwise verification) |
| An unresolvable request returns nothing rather than a guess | SciAgentArena (validity checks) |

## What is built on this branch

1. **Evaluation** (`tb_search/evaluate.py`, `eval/`, `tests/`).
   `--eval-retrieval SEED` hides the seed paper's reference list, runs
   discovery, and reports recall@k of that list, overall and for the TB
   references. `--gold` scores screening against hand labels.
2. **Retrieval pipeline** (`tb_search/sources.py`, `tb_search/discovery.py`).
   Europe PMC resolves the seed, then candidates come from its references,
   its citing papers and keyword searches. They are deduplicated, ranked by
   text overlap with the seed, and the top of the list is assessed by the LLM.
   `--search` now runs this; `--search --agent` keeps the A1 path.
3. **Structured screening** (`tb_search/assess.py`). Relevance 0–3, paper type,
   drugs, genes and variants, methods, datasets, a verbatim quote that is
   checked against the excerpt, and abstention. Local documents are read as
   opening plus methods plus conclusion rather than the first 3,000 characters.
4. **Discovery loop and store** (`tb_search/discovery.py`, `tb_search/store.py`).
   Accepted papers seed the next round; the loop stops when a round's
   acceptance rate drops below 15%. SQLite keeps papers, assessments (cached
   across runs) and a step-by-step trace.

Europe PMC replaced the OpenAlex/Semantic Scholar pairing Medea uses because it
is free without a key, indexes bioRxiv preprints with their reference lists,
and OpenAlex now meters keyless requests against a small shared daily budget.

## First measurements

Run on 2026-11-13 with BIG-TB as the seed.

| Check | Result |
| --- | --- |
| Retrieval, no LLM, 2 rounds, references held out | 52 gold references (28 about TB). Recall of TB references: 0.0 @10, 0.036 @25, 0.107 @100, 0.107 over the whole pool. All references: 0.135 over the pool. |
| Retrieval, Groq `openai/gpt-oss-120b`, 2 rounds, 8 assessments per round | 10 of 16 assessed papers accepted. Of the 8 returned, 6 are TB resistance prediction or benchmark papers by title; the other 2 are about antibiotic resistance and model generalisability more broadly. Recall not yet measured in this mode. |
| Screening, 6 labelled papers in `papers/` | 6 of 6 correct, none abstained. The evidence quote failed verification on 3 of the 4 non-TB papers. |

The retrieval number is the baseline to beat. The pool recall is the limit:
only about one in ten of the held-out TB references is reached at all, so
ranking is not the problem yet. Without an LLM the next round expands from the
papers most similar to the seed by wording, which are mostly 2026 papers with
few indexed references. The six-paper screening set is too small and too easy to
mean much.

## Not built yet

- **Findings extraction.** Variant–drug claims from accepted papers, checked
  against the WHO catalogue loaded as a table, with claims absent from the
  catalogue surfaced as candidates. This needs the catalogue file.
- **BIG-TB hook.** Taking high-attribution, non-canonical variants from BIG-TB
  models as search topics.
- **Critic step.** A second model pass over accepted papers, as in
  BioDiscoveryAgent. The topic-anchored relevance score is the only drift
  guard at present.
- **Trace rubric.** The trace is stored, but nothing grades it yet for query
  diversity or relevance reasoning.
- **Invalid-premise queries.** One unit test covers the empty case; there is
  no evaluation set of requests that should return nothing.
- **A real screening set.** About 50 hand-labelled papers, including
  borderline ones that mention TB in passing.

## Next steps, in order

1. Raise pool recall: run the recall evaluation with the LLM on, expand from
   more accepted papers per round, and add older, highly cited papers to the
   frontier since they carry the reference lists that reach the gold set.
2. Add a second seed with a longer citation history, so the recall number is
   not tied to one very recent preprint.
3. Grow `eval/screening_gold.json` to about 50 labels.
4. Start findings extraction once the WHO catalogue table is in the repo.
