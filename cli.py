"""Screen research documents for tuberculosis relevance with a Groq-backed LLM.

Two screening modes: light (default) makes one small LLM call per document,
sized for a free Groq account; --agent runs the full Biomni A1 agent tool loop,
which needs a Groq tier with a bigger tokens-per-minute budget.

Separately, --search runs the discovery loop on a retrieval request — often one
naming a seed paper by URL or DOI — and saves the related papers it finds;
--eval-retrieval scores that loop against the seed's held-out reference list.
"""

import argparse
import json
import sys
from pathlib import Path

from tb_search.assess import assess_documents
from tb_search.config import DEFAULT_MODEL, ConfigError, load_settings
from tb_search.discovery import DiscoveryConfig, discover
from tb_search.documents import EXCERPT_CHARS, collect_docs, resolve_input
from tb_search.evaluate import evaluate_retrieval, evaluate_screening
from tb_search.report import discovered_rows, write_papers, write_results
from tb_search.retrieval import retrieve_papers
from tb_search.screening import (
    build_agent,
    build_screener,
    check_connection,
    classify_documents_agent,
)
from tb_search.sources import SOURCE_NAMES, build_source
from tb_search.store import Store


def fail(message: str) -> int:
    """Report an error on stderr and return the CLI's exit code."""
    print(message, file=sys.stderr)
    return 1


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
        help="Retrieval request — typically a sentence naming a seed paper by URL "
        "or DOI, e.g. 'find me relevant papers like this paper <url>'. Runs the "
        "discovery loop, or the A1 agent with --agent. Output defaults to "
        "tb_papers.json.",
    )
    parser.add_argument(
        "--eval-retrieval",
        metavar="SEED",
        help="Score the discovery loop: hide SEED's reference list, search, and "
        "report how much of it is recovered. Output defaults to eval_retrieval.json.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=2,
        help="[--search] Discovery rounds (default: 2).",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=20,
        help="[--search] Papers the LLM assesses per round (default: 20).",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="[--search] Rank by text overlap only; needs no API key.",
    )
    parser.add_argument(
        "--db",
        default="tb_search.db",
        help="[--search] SQLite file that accumulates papers, assessments and "
        "run traces (default: tb_search.db).",
    )
    parser.add_argument(
        "--gold",
        metavar="JSON",
        help="[screening] Hand labels as {filename: true/false}; prints accuracy "
        "against them, e.g. eval/screening_gold.json.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Where to write the JSON output (default: tb_verdicts.json, "
        "or tb_papers.json with --search).",
    )
    parser.add_argument(
        "-n",
        "--top",
        dest="max_papers",
        type=int,
        default=10,
        help="[--search] Upper bound on papers returned, most relevant first "
        "(default: 10). Fewer come back when fewer are judged relevant.",
    )
    parser.add_argument(
        "--sources",
        default=",".join(SOURCE_NAMES),
        help="[--search] Comma-separated literature sources to query and merge "
        f"(default: {','.join(SOURCE_NAMES)}).",
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
    parser.add_argument("--temperature", type=float, help="Sampling temperature.")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="[--search --agent] Also print biomni's full step-by-step agent output.",
    )
    return parser


def run_discovery(args, output: Path | None) -> int:
    """--search and --eval-retrieval: the discovery loop, with or without an LLM."""
    request = args.search or args.eval_retrieval
    llm, model = None, "none (lexical ranking)"
    if not args.no_llm:
        try:
            settings = load_settings(model=args.model, temperature=args.temperature)
            check_connection(settings)
        except ConfigError as e:
            return fail(f"{e}\n(--no-llm runs the search without a model)")
        llm, model = build_screener(settings), f"{settings.source}/{settings.model}"

    config = DiscoveryConfig(
        rounds=max(1, args.rounds),
        budget=max(1, args.budget),
        max_papers=max(1, args.max_papers),
    )
    try:
        source = build_source(
            [name.strip() for name in args.sources.split(",") if name.strip()]
        )
    except ValueError as e:
        return fail(e)
    print(f"Searching [{config.rounds} round(s), {model}]\n  query: {request}\n")
    store = Store(args.db)
    try:
        if args.eval_retrieval:
            report = evaluate_retrieval(source, store, request, llm, model, config)
            if report is None:
                print(
                    "Seed paper not found, or it has no matched references.",
                    file=sys.stderr,
                )
                return 1
            output = output or Path("eval_retrieval.json")
            output.write_text(
                json.dumps({"model": model, **report}, indent=2, ensure_ascii=False)
            )
            print(
                f"\nGold: {report['gold']['all']} references, {report['gold']['tb']} about TB"
            )
            for name, scores in report["recall"].items():
                line = "  ".join(
                    f"{cutoff} {value}" for cutoff, value in scores.items()
                )
                print(f"  recall ({name}): {line}")
            print(f"Saved to {output}")
            return 0

        run = discover(source, store, request, llm, model, config)
        rows = discovered_rows(run.papers(config.max_papers, config.accept))
        write_papers(rows, model, request, output or Path("tb_papers.json"))
        return 0
    finally:
        store.close()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.eval_retrieval or (args.search and not args.agent):
        return run_discovery(args, Path(args.output) if args.output else None)

    mode = "search" if args.search else ("agent" if args.agent else "light")
    output = Path(
        args.output or ("tb_papers.json" if mode == "search" else "tb_verdicts.json")
    )

    try:
        settings = load_settings(
            model=args.model,
            temperature=args.temperature,
            use_tool_retriever=args.tools or None,
            load_data_lake=args.data_lake or None,
        )
    except ConfigError as e:
        return fail(e)

    if mode == "search":
        max_papers = max(1, args.max_papers)
        print(
            f"Searching for up to {max_papers} paper(s) "
            f"[A1 agent, {settings.source} {settings.model}]\n"
            f"  query: {args.search}\n"
        )
        try:
            check_connection(settings)
            rows = retrieve_papers(
                build_agent(settings), args.search, max_papers, args.verbose
            )
        except ConfigError as e:
            return fail(e)
        write_papers(rows, f"{settings.source}/{settings.model}", args.search, output)
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

    try:
        check_connection(settings)
    except ConfigError as e:
        return fail(e)

    if mode == "agent":
        agent = build_agent(settings)
        rows = classify_documents_agent(
            agent, docs, max(1, args.batch_size), max(200, args.max_chars)
        )
    else:
        screener = build_screener(settings)
        rows = assess_documents(screener, docs, max(200, args.max_chars))

    write_results(rows, settings, input_path, output, mode)
    if args.gold:
        scores = evaluate_screening(rows, Path(args.gold))
        print(
            f"Against {args.gold}: {scores['accuracy']} accuracy on {scores['answered']} "
            f"answered of {scores['labelled']} labelled; wrong: {scores['wrong'] or 'none'}; "
            f"abstained: {scores['abstained'] or 'none'}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
