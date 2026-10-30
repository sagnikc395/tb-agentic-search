## tb-agentic-search

Use a [Biomni](https://github.com/sugar-biomni/biomni) A1 agent (running on Groq) to detect
which research documents are related to tuberculosis / *Mycobacterium tuberculosis*.

## Setup

```bash
uv sync
cp .env.example .env   # then fill in GROQ_API_KEY
```

## Usage

```bash
uv run python main.py --scan-only      # see which docs were found + excerpt previews
uv run python main.py                  # screens ./papers, or ../papers when ./papers is absent
uv run python main.py ../papers        # any directory or single PDF/text file
uv run python main.py -o out.json      # default output: tb_verdicts.json
```

The pipeline: collect PDFs/text under the input path, extract a short excerpt from each,
send them to biomni's Groq-backed LLM one document at a time, and save a per-document
verdict (`tb_related`, `confidence`, `reason`) to JSON plus a console summary.

Two modes: the default ("light") screener makes one small LLM call per document —
small enough for a free Groq account's 8000 tokens/minute budget. `--agent` runs the
full Biomni A1 agent tool loop instead, but A1's system prompt is ~35k tokens, so it
needs a Groq tier with a bigger TPM budget.

## Flags

| flag | meaning |
| --- | --- |
| `input` (optional) | Directory or file to screen; defaults to `./papers`, else `../papers` |
| `-o, --output` | JSON results path (default `tb_verdicts.json`) |
| `--tools` | `[--agent]` Enable biomni's tool retriever so the agent may double-check with its bio tools |
| `--data-lake` | `[--agent]` Let biomni pull its full data lake (tens of GB; not needed for screening) |
| `--agent` | Use the full A1 agent tool loop instead of the light per-doc calls |
| `--batch-size` | `[--agent]` Docs per agent call (default 2) |
| `--max-chars` | Excerpt length per document (default 6000) |
| `--scan-only` | List docs and preview excerpts without calling the agent |
