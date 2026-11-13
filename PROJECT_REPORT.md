# Project report

A rough outline of the project: what it does, what each part of the code is
for, and how the search works. The papers behind the design are covered in
[LIT_REVIEW.md](./LIT_REVIEW.md); usage and flags are in [README.md](./README.md).

## What the project does

It does two things for TB literature: it **finds** papers related to a seed
paper or topic, and it **screens** documents you already have. The LLM (Groq,
through Biomni's plumbing) only writes search queries and scores real records.

## Layout

```
cli.py                 entry point, picks the mode
tb_search/
  sources.py           Europe PMC client
  discovery.py         the search loop
  assess.py            LLM scoring of one paper or document
  store.py             SQLite persistence
  evaluate.py          recall and accuracy measurements
  documents.py         read PDFs/text, build excerpts
  prompts.py           all prompt text
  verdicts.py          pull JSON out of messy model replies
  report.py            console lines and JSON output
  screening.py         build the LLM / A1 agent, agent-mode screening
  retrieval.py         old path: one Biomni A1 agent run
  config.py            settings from .env
eval/                  hand labels for screening
tests/                 18 tests on parsing, discovery, evaluation
```

## What each part does

| File | Job |
| --- | --- |
| `cli.py` | Parses flags and routes to search (`--search`), evaluation (`--eval-retrieval`) or screening (default). |
| `sources.py` | Four Europe PMC lookups: resolve a URL/DOI to a paper, get its references, get papers citing it, and keyword search. Also fetches abstracts in batches. |
| `discovery.py` | Runs the search rounds: gathers candidates, ranks them, has them scored, and picks what to expand next. |
| `assess.py` | Sends one paper to the LLM and normalises the reply: relevance 0–3, paper type, drugs, genes/variants, methods, datasets, a quote checked against the text, or an abstention. |
| `store.py` | Saves papers, candidates per run, a step-by-step trace, and a cache of assessments so a paper is not re-scored for the same topic. |
| `evaluate.py` | Hides a seed's reference list and measures how much of it the search recovers; compares screening output to hand labels. |
| `documents.py` | Collects files, extracts text, and builds an excerpt from the opening, methods and conclusion. |
| `prompts.py` | The assessment prompt, the query-writing prompt, and the prompts for the A1 agent paths. |
| `verdicts.py` | Finds the JSON in a reply even when the model wraps it in prose or code fences. |
| `report.py` | Prints the per-paper console lines and writes `tb_papers.json` and `tb_verdicts.json`. |
| `screening.py`, `retrieval.py` | The Biomni A1 agent paths, kept behind `--agent`. `screening.py` also builds the plain LLM used everywhere else. |
| `config.py` | Reads the Groq key, model and other settings from `.env`. |

## How the search works

1. **Resolve the seed.** A DOI, bioRxiv, PubMed or arXiv link is pulled out of
   the request and looked up on Europe PMC. The seed's title and abstract
   become the topic. With no link, the request text itself is the topic.
2. **Write queries.** One query is the seed title's keywords joined with OR.
   With an LLM, it adds three more, each asked to take a different angle
   (question, method, dataset, drug or gene).
3. **Gather candidates** from three routes:
   - papers the seed cites
   - papers that cite the seed
   - the top 25 results for each query
4. **Rank cheaply.** Each candidate is scored by how much of the topic's
   vocabulary its title and abstract cover, weighted so rare words count more.
   A paper reached by several routes gets a boost.
5. **Assess the top of the list.** The best `--budget` candidates (default 20)
   go to the LLM one at a time for a 0–3 relevance score. A score of 2 or more
   is accepted.
6. **Next round.** The top five accepted papers become the new starting
   points: their references and citing papers are fetched, and the LLM writes
   new queries from what was accepted. This repeats for `--rounds` (default 2),
   or stops early if under 15% of a round is accepted or nothing new turns up.
7. **Return.** Accepted papers are sorted by relevance then overlap score, a
   preprint and its published version are collapsed to one, and the top
   `--max-papers` are written to `tb_papers.json`.

With `--no-llm`, the LLM-written queries and the assessment step are skipped:
ranking is by word overlap only, and the top five by overlap are expanded.

Every paper in the output comes from a Europe PMC record, so the model cannot
invent one.

## Screening local files

For each file in `papers/`, `documents.py` builds the excerpt and `assess.py`
scores it against a fixed TB topic. The result goes to `tb_verdicts.json`, with
`tb_related` taken from whether the model judged TB central to the paper.

## How it is evaluated

- **Retrieval:** `--eval-retrieval SEED` hides the seed's reference list, runs
  the search without it, and reports what share of those references came back
  in the top 10, 25, 50 and 100, and anywhere in the candidate pool. It reports
  this for all references and for the ones about TB by title.
- **Screening:** `--gold eval/screening_gold.json` compares `tb_related` with
  hand labels and lists wrong answers, abstentions and unverified quotes.

First results, run on 2026-11-13 with BIG-TB as the seed:

| Check | Result |
| --- | --- |
| Retrieval recall, no LLM, 2 rounds | 10.7% of the 28 held-out TB references reached; 13.5% of all 52 |
| Retrieval with the Groq model, 8 assessments per round | 10 of 16 assessed papers accepted; 6 of the 8 returned are TB resistance prediction or benchmark papers by title |
| Screening, 6 labelled papers | 6 of 6 correct |

## Not built yet

Findings extraction against the WHO catalogue, the hook to BIG-TB attributions,
a separate critic pass, a rubric for grading run traces, and a set of
invalid-premise queries. `LIT_REVIEW.md` has the detail and the next steps.
