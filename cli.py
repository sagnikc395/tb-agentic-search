"""Screen research documents for tuberculosis relevance with a Groq-backed LLM.

Two screening modes: light (default) makes one small LLM call per document,
sized for a free Groq account; --agent runs the full Biomni A1 agent tool loop,
which needs a Groq tier with a bigger tokens-per-minute budget.

Separately, --search runs the A1 agent on a retrieval request — often one naming
a seed paper by URL or DOI — and saves the related papers it finds.
"""

import argparse
import sys
from pathlib import Path

from tb_search.config import DEFAULT_MODEL, ConfigError, load_settings
from tb_search.documents import EXCERPT_CHARS, collect_docs, extract_text, resolve_input
from tb_search.report import write_papers, write_results
from tb_search.retrieval import retrieve_papers
from tb_search.screening import (
    build_agent,
    build_screener,
    check_connection,
    classify_documents_agent,
    classify_documents_light,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tb-search",
        description="Detect TB-related research documents with a biomni-powered LLM on Groq.",
    )
    parser.add_argument(
        "input",
        nargs="?",
        help="Directory or file to screen. Defaults to ./papers, "
        "or ../papers when ./papers does not exist.",
    )
    parser.add_argument(
        "--search",
        metavar="QUERY",
        help="Retrieval request for the A1 agent — typically a sentence naming a "
        "seed paper by URL or DOI, e.g. 'find me relevant papers like this paper "
        "<url>'. Output defaults to tb_papers.json.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Where to write the JSON output (default: tb_verdicts.json, "
        "or tb_papers.json with --search).",
    )
    parser.add_argument(
        "--max-papers",
        type=int,
        default=10,
        help="[--search] Cap on papers the agent should return (default: 10).",
    )
    parser.add_argument(
        "--model", help=f"Groq model id (default: $GROQ_MODEL or {DEFAULT_MODEL})."
    )
    parser.add_argument(
        "--agent",
        action="store_true",
        help="Use the full biomni A1 agent loop instead of small per-doc LLM calls. "
        "A1's system prompt is large; a free Groq account will be rate limited.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="[--agent] Documents per agent call (default: 2).",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=EXCERPT_CHARS,
        help="Excerpt length per document.",
    )
    parser.add_argument(
        "--tools",
        action="store_true",
        help="[--agent] Let the A1 agent retrieve a relevant tool subset.",
    )
    parser.add_argument(
        "--data-lake",
        action="store_true",
        help="[--agent] Let biomni download its full data lake (tens of GB).",
    )
    parser.add_argument(
        "--timeout", type=int, help="[--agent] Per-execution timeout in seconds."
    )
    parser.add_argument("--temperature", type=float, help="Sampling temperature.")
    parser.add_argument(
        "--scan-only",
        action="store_true",
        help="List docs and preview excerpts; no LLM call.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    mode = "search" if args.search else ("agent" if args.agent else "light")
    output = Path(args.output or ("tb_papers.json" if mode == "search" else "tb_verdicts.json"))

    try:
        settings = load_settings(
            model=args.model,
            temperature=args.temperature,
            timeout_seconds=args.timeout,
            use_tool_retriever=args.tools or None,
            load_data_lake=args.data_lake or None,
        )
    except ConfigError as e:
        print(e, file=sys.stderr)
        return 1

    if mode == "search":
        print(f"Searching [agent, {settings.source} {settings.model}]\n")
        try:
            check_connection(settings)
            rows = retrieve_papers(build_agent(settings), args.search)
        except ConfigError as e:
            print(e, file=sys.stderr)
            return 1
        write_papers(rows, settings, args.search, output)
        return 0

    input_path = resolve_input(args.input)
    if input_path is None:
        print(
            f"Input not found: {args.input!r}"
            if args.input
            else "No papers directory found; pass one, e.g. `tb-search ../papers`",
            file=sys.stderr,
        )
        return 1

    docs = collect_docs(input_path)
    if not docs:
        print(
            f"No .pdf/.txt/.md/... documents found under {input_path}", file=sys.stderr
        )
        return 1

    print(
        f"Screening {len(docs)} document(s) under {input_path} "
        f"[mode: {mode}, {settings.source} {settings.model}]\n"
    )

    if args.scan_only:
        for path in docs:
            preview = extract_text(path, 200) or "(no text)"
            print(f"  {path.name} — {preview}...")
        return 0

    try:
        check_connection(settings)
    except ConfigError as e:
        print(e, file=sys.stderr)
        return 1

    if mode == "agent":
        agent = build_agent(settings)
        rows = classify_documents_agent(
            agent, docs, max(1, args.batch_size), max(200, args.max_chars)
        )
    else:
        screener = build_screener(settings)
        rows = classify_documents_light(screener, docs, max(200, args.max_chars))

    write_results(rows, settings, input_path, output, mode)
    return 0


if __name__ == "__main__":
    main()
