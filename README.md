# tb-agentic-search

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
tb-search                              # screens ./papers, or ../papers when ./papers is absent
tb-search ../papers                    # any directory or single PDF/text file
tb-search --scan-only                  # list docs and preview excerpts, no API call
tb-search -o out.json                  # default output: tb_verdicts.json
tb-search --agent                      # full A1 agent loop instead (needs TPM)
```

`python main.py` is the same thing — `main.py` just calls `tb_search.cli.main`.

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
main.py            shim → tb_search.cli (keeps the original command working)
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

## Dependencies

The list is exactly what the code (and biomni's used paths) import: `biomni`,
`groq`, `langchain-openai` (biomni's Groq backend), `pypdf`, `python-dotenv`,
`tooluniverse` (biomni's agent tools).

`gradio` and `fonttools` used to be declared and are gone — nothing in this repo
or in biomni's screening paths imports them. The only loss: biomni's built-in
`launch_gradio_demo()` demo UI now raises its own ImportError if you ever call it.

## How code is written here

House style, borrowed from the lazy-senior-dev school
([ponytail](https://github.com/DietrichGebert/ponytail)): the smallest change
that fully works, and nothing a busy human can't read in one pass.

- Read what a change touches before writing it: the callers, tests and fixtures
  it would break. Fix the root cause once in the shared code, not at every
  call site.
- Check what already exists before adding anything: a helper in this repo,
  then the standard library, then an installed dependency. A new dependency
  only when it clearly saves more lines than it costs.
- No abstraction, wrapper, option, or "for later" code nobody asked for.
  Deletion beats addition.
- Comments say only what the code cannot show, one line each. A comment that
  restates the code is wrong; delete the comment. A deliberate shortcut gets
  `shortcut: <limit>, <when to upgrade>` so it can be found later.
- Docstrings on modules and non-obvious functions; one line when that's enough.
- Non-trivial logic (parsers, normalization, filtering) ships with a small test
  in `tests/`. Trivial changes need none.
- Prose — this README, docstrings, commit messages — is direct and
  imperative: "Run X", not "It is recommended to run X". No emoji, no section
  banners, no filler.
- Never cut: validation at trust boundaries, error handling that prevents
  data loss, tests.
