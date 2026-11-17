# tb-knowledge-discovery

![flow-1](./assets/v2-1.png)
![flow-2](./assets/v2-2.png)
![flow-3](./assets/v2-3.png)
![relevance_check](./assets/paper_relevance_check.png)

Three capabilities, with every LLM call going through
[Biomni](https://github.com/snap-stanford/Biomni)'s Groq plumbing:

- **Retrieval** (`--search`) — give it a request such as "find me relevant papers
  like this paper <paper-url>". It resolves the seed paper on Europe PMC,
  Semantic Scholar or Crossref, gathers candidates from the seed's references,
  the papers citing it and keyword searches, and has the LLM score them. Papers
  it accepts seed the next round, and in this default path every returned paper
  comes straight from a source API, so none can be invented. `--agent` swaps the
  small per-paper calls for the Biomni A1 agent, whose own tool calls do the
  retrieval.
- **Screening** — hand it a folder of documents (PDF or plain text) and it
  returns an assessment per document: `tb_related`, a 0–3 `relevance`, paper
  type, drugs, genes/variants, methods, datasets, and an `evidence` quote that
  is checked against the text (`evidence_verified`).
- **Resistance prediction from VCFs** (`--cohort`, `--predict`) — ingest the
  BIG-TB isolate VCFs, and for a given drug predict `resistant` /
  `not_resistant` / `unknown` with a reasoning trace. Calls are scored against
  the phenotypic DST labels, and each trace is graded against the variants the
  model was actually shown.

Screening has two modes:

- **light** (default) — one small LLM call per document (~1.5k tokens), so a
  free Groq account's 8000-tokens-per-minute budget holds.
- **`--agent`** — the full Biomni A1 agent tool loop. A1's system prompt is
  ~35k tokens, so this needs a Groq tier with a bigger TPM budget.

## Setup

```bash
uv sync
cp .env.example .env   # then fill in GROQ_API_KEY
```

Every mode calls Groq, so `GROQ_API_KEY` is required; there is no offline mode.
`SEMANTIC_SCHOLAR_API_KEY` and `CROSSREF_MAILTO` are optional: both sources
answer without them, only slower (`CROSSREF_MAILTO` is just an email address and
puts requests in Crossref's faster "polite" pool).

## Resistance prediction from VCFs

The dataset comes from two files in `data/cleaned_vcf_paths/`:
`master_resistance_table.csv` (phenotypic DST, R/S per isolate per drug) and
`big_tb_file_paths.txt` (the BIG-TB VCF paths on the cluster). They join on the
table's `New_ID` column, which matches the VCF filename stem for all 17,942
BIG-TB isolates, giving a labelled dataset per drug — 17.6k isolates for
rifampicin, 17.4k for isoniazid, down to a few dozen for the rarely tested
drugs.

Variants are named without needing an annotated VCF. Each position is looked up
in `data/h37rv_genes.tsv` (NCBI's feature table for H37Rv, NC_000962.3) to get
the gene, the codon number, or a promoter offset, and the codon is then
translated against the reference in `data/h37rv.fna.gz` to get the actual
protein consequence. So `761155 C>T` is presented as `rpoB S450L`, `761156 G>A`
as `rpoB S450S synonymous`, `1673425 C>T` as `fabG1 -15 (promoter)` and a 1 bp
deletion in pncA as a frameshift. When the VCF carries its own `ANN`/`CSQ`
field, that annotation is used instead.

Naming the protein change matters more than it looks: shown only "rpoB codon
453", the model calls any hotspot codon resistant, because it cannot tell a
silent change from a missense one. Shown "rpoB S450S synonymous", it correctly
rules the variant out.

Because the reference is loaded anyway, every record's `REF` is checked against
H37Rv. A run that reports mismatches was called against a different reference or
has shifted coordinates, and its gene and codon assignments cannot be trusted.

Variants are then filtered to the genes the WHO catalogue grades for that drug,
split into tier 1 (established mechanism) and tier 2 (secondary).

**Each trace is checked, not just read.** Every prediction is stored next to the
variants it was shown, so three things are verified mechanically:

| check | what it catches |
| --- | --- |
| `variants_resolve` | the model cited a variant that was never in the VCF |
| `genes_on_panel` | the model reasoned from a gene not graded for that drug |
| `call_grounded` | a `resistant` call that rests on no variant at all |

`tier1_support` additionally records whether the call rests on a tier 1 gene
rather than only a secondary one. A trace is `valid` when nothing in it was
invented or self-contradictory — accuracy and trace validity are reported
separately, because a run can be accurate and still be reasoning from variants
that were not there.

```bash
# the labelled dataset size per drug; no LLM calls
uv run cli.py --cohort

# predict, on a balanced sample, with the VCFs mounted locally
uv run cli.py --predict isoniazid --limit 40 --balance --vcf-root /path/to/vcfs
uv run cli.py --predict rifampicin --limit 100 --balance --seed 1
```

The BIG-TB VCFs live on the cluster, so `--vcf-root` points at wherever they
actually are; without it the recorded cluster path is used as-is. A run prints
one line per isolate (the call, whether it matched the DST, and any trace
problem) and ends with accuracy, sensitivity, specificity, PPV/NPV, the
confusion matrix and the trace rates. `tb_resistance_<drug>.json` holds every
prediction with its variant list, reasoning and trace grade.

## Usage

**Screening local documents**

```bash
uv run cli.py                        # screens ./papers, or ../papers when ./papers is absent
uv run cli.py ../papers               # any directory or single PDF/text file
uv run cli.py -o out.json             # default output: tb_verdicts.json
uv run cli.py --agent                 # full A1 agent loop instead (needs TPM)
uv run cli.py --gold eval/screening_gold.json   # plus accuracy against hand labels
```

**Retrieval**

```bash
uv run cli.py --search "find me relevant papers like this paper https://www.biorxiv.org/content/..."
uv run cli.py --search "..." -n 20 -o found.json        # at most 20 papers, most relevant first
uv run cli.py --search "..." --sources europepmc        # one literature source only
uv run cli.py --search "..." --sources semanticscholar,crossref
uv run cli.py --search "..." --rounds 3 --budget 10     # more rounds, fewer LLM calls per round
uv run cli.py --search "..." --agent --verbose          # the Biomni A1 agent instead
```

Candidates are ranked by structure, not by text: a paper reached by several
routes (both reference lists and a keyword search, say) is scored before one a
single search turned up. The LLM then gives it the 0–3 relevance.

**Evaluation**

```bash
uv run cli.py --eval-retrieval "https://www.biorxiv.org/content/..."
uv run cli.py --eval-retrieval "https://..." --sources europepmc   # score one source
```

A search run echoes each lookup as it goes (the round number in brackets),
then each assessed paper with its 0–3 relevance, then the papers it settled on:

```
Searching [2 round(s), Groq/openai/gpt-oss-120b]
  query: find me relevant papers like this paper https://...

  seed: BIG-TB: A benchmark for ... (2026) [PPR:PPR1226747]
  [1] cited-by:PPR:PPR1226747 -> 53
  [1] search:rifampicin katG mutation -> 25
  [1] [3] Benchmarking the translational potential of AI-based drug-resistance ...
  [1] 148 new candidates, 8 judged, 4 accepted

Papers related to the query (8):
   1. TB-Bench: A Systematic Benchmark of ... (2026) [europepmc]
      https://doi.org/10.64898/2026.04.08.717138
      Provides a TB drug resistance benchmark similar to BIG-TB.
```

## How a run flows

Both modes enter through `cli.py`, which parses flags, loads `Settings` from
`.env` and picks a path. Everything below the CLI is plain Python. In the
default paths the model only plans queries and judges text; under `--agent` it
also drives the retrieval tools and its reply is parsed into rows.

**Screening (the default mode)**

```
cli.py
  load_settings          config.py      .env -> Settings
  resolve_input          documents.py   the directory or single file to read
  collect_docs           documents.py   every PDF/text under it
  check_connection       screening.py   one 1-token Groq call, so a bad key fails here
  build_screener         screening.py   biomni.llm.get_llm -> chat model
  assess_documents       assess.py      one model call per document
    section_excerpt      documents.py   opening + methods + conclusion
    assess               assess.py      ASSESS_PROMPT -> one JSON object
    parse_assessment     verdicts.py    salvage that JSON from prose or code fences
    normalize_assessment assess.py      quote checked against the text, facets, abstention
  write_results          report.py      console lines + tb_verdicts.json
```

**Retrieval (`--search`, and `--eval-retrieval` which wraps it)**

```
cli.py run_discovery
  build_source           sources.py     Europe PMC / Semantic Scholar / Crossref, or Combined
  discover               discovery.py   repeats --rounds times:
    source.resolve       sources.py     seed paper from a URL/DOI; with no link, the query is the topic
    generate_queries     discovery.py   the LLM writes 3 queries, each from a different angle
    lookup               sources.py     references + citations of the frontier, then every query,
                                        sent to every selected source and merged by DOI
    source.hydrate       sources.py     batch-fetch the abstracts
    rank                 discovery.py   by Candidate.support: routes that reached it, then citations
    assess               assess.py      the top --budget papers, 0-3 relevance, 2+ accepted
    next round           discovery.py   the top 5 accepted papers seed the next round
  discovered_rows        report.py      rows -> tb_papers.json
```

`--agent` replaces the model step, not the pipeline: screening hands document
batches to A1, and `--search --agent` runs A1 once and reads the paper list out
of its reply (`retrieval.py`). `--eval-retrieval` wraps `discover()`: it hides
the seed's reference list, runs the same loop, and reports recall@10/25/50/100
against that list (`evaluate.py`).

## How it uses Biomni

Biomni supplies the model and the optional agent; the pipeline around them is
this repo's, including every prompt (`prompts.py`). The touchpoints are few:

| Biomni piece | Where | Used for |
| --- | --- | --- |
| `biomni.llm.get_llm` | `screening.build_screener` | the chat model behind every non-agent call: one prompt per document, and query writing |
| `biomni.config.default_config` | `build_screener`, `build_agent` | temperature, model and provider; A1 forwards `llm`/`source` but not `temperature` |
| `biomni.agent.A1` | `screening.build_agent` | the `--agent` path: tool loop, tool retriever, data-lake switch |
| `agent.go(prompt)` | `screening.classify_documents_agent` | screening in batches of `--batch-size` documents |
| `agent.go_stream(prompt)` | `retrieval._run_streamed` | search step by step, so its tool calls can be echoed |

Four details are easy to get wrong, which is why the code spells them out:

- `source="Groq"` is always passed explicitly. Biomni's auto-detection sees the
  slash in `openai/gpt-oss-120b` and would otherwise send the model to Ollama.
- `GROQ_API_KEY` is copied into the environment before `A1` is built, because
  Biomni's Groq branch reads it from there, not from our `Settings`.
- `expected_data_lake_files=[]` (not `None`) is what tells A1 to skip its
  tens-of-GB data-lake download; `--data-lake` turns it back on.
- `timeout_seconds` comes from `BIOMNI_TIMEOUT_SECONDS` (default 600) and is
  passed only to A1.

A1's system prompt is ~35k tokens and Groq's free tier allows 8k per minute, so
the default paths avoid A1 entirely. Where a call can still be rejected,
`_wait_out_rate_limit` retries a 413/TPM error after 65 seconds, and a failure
is reported per document or per lookup instead of ending the run.

## Flags

| flag | meaning |
| --- | --- |
| `input` (optional) | Directory or file to screen; defaults to `./papers`, else `../papers` |
| `--search QUERY` | Retrieval request; typically names a seed paper by URL/DOI. Output: `tb_papers.json` |
| `--eval-retrieval SEED` | Recall of SEED's held-out reference list. Output: `eval_retrieval.json` |
| `-n`, `--top` | `[--search]` Upper bound on papers returned, most relevant first (default 10) |
| `--sources` | `[--search]`, `[--eval-retrieval]` Literature sources to query and merge, comma-separated, from `europepmc`, `semanticscholar`, `crossref` (default all three) |
| `--rounds` | `[--search]` Discovery rounds (default 2) |
| `--budget` | `[--search]` Papers the LLM assesses per round (default 20) |
| `--gold JSON` | `[screening]` Hand labels `{filename: true/false}`; prints accuracy |
| `--cohort` | Join the resistance table to the listed VCFs; per-drug dataset sizes. Output: `tb_cohort.json` |
| `--predict DRUG` | Predict resistance to DRUG from each isolate's VCF, with a checked trace. Output: `tb_resistance_<drug>.json` |
| `--vcf-root DIR` | `[--predict]` Where the VCFs actually live, replacing the recorded cluster path |
| `--data-dir DIR` | `[--cohort]`, `[--predict]` Where the cohort files live (default `data/cleaned_vcf_paths`) |
| `--limit N` | `[--predict]` Isolates to predict over (default 20) |
| `--balance` | `[--predict]` Equal numbers of resistant and susceptible isolates |
| `--seed N` | `[--predict]` Sampling seed, so a run repeats (default 0) |
| `--min-af` | `[--predict]` Drop variants below this alternate-allele frequency (default 0, keep all) |
| `--keep-filtered` | `[--predict]` Also use variants the caller did not mark PASS |
| `-o, --output` | JSON results path (default `tb_verdicts.json`, or `tb_papers.json` with `--search`) |
| `--model` | Groq model id (default: `$GROQ_MODEL` or `openai/gpt-oss-120b`) |
| `--agent` | Use the full A1 agent tool loop, for screening or `--search` |
| `--batch-size` | `[--agent]` Documents per agent call (default 2) |
| `--max-chars` | Excerpt length per document (default 3000) |
| `--tools` | `[--agent]` Let the agent retrieve a relevant tool subset |
| `--data-lake` | `[--agent]` Let biomni pull its full data lake (tens of GB) |
| `--temperature` | Sampling temperature (default 0.2) |
| `--verbose` | `[--search --agent]` Also print biomni's full step-by-step agent output |

## Layout

```
cli.py             CLI entry point
tb_search/
  config.py        Settings loaded from .env / environment
  documents.py     collect files, extract text and section-aware excerpts
  prompts.py       the assessment, query-writing and agent prompts
  verdicts.py      salvage JSON out of whatever the model returned
  sources.py       Europe PMC, Semantic Scholar, Crossref: seed lookup, references, citations, search
  discovery.py     the discovery loop: expand, assess, repeat
  assess.py        structured assessment of a paper or local document
  evaluate.py      retrieval recall, screening accuracy, resistance scoring
  screening.py     LLM/agent construction and agent-mode screening
  retrieval.py     one A1 agent run: query (+ seed paper URL) -> paper rows
  report.py        summary line + JSON output (verdicts, searches, predictions)
  genome.py        H37Rv position -> gene, codon, promoter offset, protein change
  catalogue.py     WHO drug -> resistance gene panels, tier 1 and tier 2
  vcf.py           read a VCF into named, panel-filtered variants
  cohort.py        join the resistance table to the VCF paths; per-drug datasets
  resistance.py    predict resistance for one isolate/drug, and grade the trace
data/
  h37rv_genes.tsv        H37Rv gene coordinates (from NCBI NC_000962.3)
  h37rv.fna.gz           the H37Rv reference sequence, for protein consequences
  cleaned_vcf_paths/     the resistance table and the BIG-TB VCF path list
eval/              hand labels for screening
tests/             offline tests for the VCF pipeline
```
