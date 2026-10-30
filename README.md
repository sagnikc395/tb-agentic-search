# tb-agentic-search

![flow](./assets/v1.png)

Screen a folder of research documents (PDF or plain text) for tuberculosis /
*Mycobacterium tuberculosis* relevance, with an LLM wired through
[Biomni](https://github.com/snap-stanford/Biomni)'s Groq plumbing.

Each document gets a verdict in the output JSON: `tb_related`, `confidence`, `reason`.

Two modes:

- **light** (default) — one small LLM call per document (~1k tokens), so a free
  Groq account's 8000-tokens-per-minute budget holds.
- **`--agent`** — the full Biomni A1 agent tool loop. A1's system prompt is
  ~35k tokens, so this needs a Groq tier with a bigger TPM budget.

## Setup

```bash
uv sync
cp .env.example .env   # then fill in GROQ_API_KEY
```

## Usage

```bash
uv run cli.py                               # screens ./papers, or ../papers when ./papers is absent
uv run cli.py ../papers                    # any directory or single PDF/text file
uv run cli.py --scan-only                  # list docs and preview excerpts, no API call
uv run cli.py -o out.json                  # default output: tb_verdicts.json
uv run cli.py --agent                      # full A1 agent loop instead (needs TPM)
```

## Flags

| flag | meaning |
| --- | --- |
| `input` (optional) | Directory or file to screen; defaults to `./papers`, else `../papers` |
| `-o, --output` | JSON results path (default `tb_verdicts.json`) |
| `--model` | Groq model id (default: `$GROQ_MODEL` or `openai/gpt-oss-120b`) |
| `--agent` | Use the full A1 agent tool loop instead of small per-doc LLM calls |
| `--batch-size` | `[--agent]` Documents per agent call (default 2) |
| `--max-chars` | Excerpt length per document (default 3000) |
| `--tools` | `[--agent]` Let the agent retrieve a relevant tool subset |
| `--data-lake` | `[--agent]` Let biomni pull its full data lake (tens of GB) |
| `--timeout` | `[--agent]` Per-execution timeout in seconds |
| `--temperature` | Sampling temperature (default 0.2) |
| `--scan-only` | List docs and preview excerpts; no LLM call |

## Layout

```
tb_search/
  cli.py           CLI entry point; also installed as the `tb-search` script
  config.py        Settings loaded from .env / environment
  documents.py     collect files, extract text excerpts
  prompts.py       the two screening prompts
  verdicts.py      salvage a verdict out of whatever JSON the model returned
  screening.py     light and agent screening loops
  report.py        summary line + JSON output
tests/             pytest suite for the parsing and extraction logic
```

Run the tests with `uv run pytest`, lint with `uv run ruff check .`.
