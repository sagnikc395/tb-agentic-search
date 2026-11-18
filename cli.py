"""Screen research documents and predict TB drug resistance with an LLM-backed agent.

Every mode runs against a model you serve yourself: a vLLM job holding it, with
this process (and biomni inside it) talking to that engine over HTTP. Start the
server with a reasoning parser and --reasoning-effort is honoured and the
model's thinking is recorded next to each prediction:

    vllm serve Qwen/Qwen3-32B --reasoning-parser qwen3 \
        --enable-auto-tool-choice --tool-call-parser hermes --port 8000
    tb-search --predict rifampicin --base-url http://$NODE:8000/v1 \
        --model Qwen/Qwen3-32B --reasoning-effort high

Two screening modes: light (default) makes one small LLM call per document;
--agent runs the full Biomni A1 agent tool loop, whose ~35k-token system prompt
needs the server started with a context window to match.

Separately, --search runs the discovery loop on a retrieval request — often one
naming a seed paper by URL or DOI — and saves the related papers it finds;
--eval-retrieval scores that loop against the seed's held-out reference list.

The VCF side is its own pair of modes. --cohort joins the BIG-TB resistance
table to the listed VCFs and reports the labelled dataset size per drug;
--predict DRUG reads each isolate's VCF, asks the model for a resistant /
not resistant / unknown call with a reasoning trace, and scores both the calls
against the DST phenotype and the traces against the variants actually present.
"""

import argparse
import json
import sys
from pathlib import Path

from tb_search.assess import assess_documents
from tb_search.catalogue import drug_key, known_drugs, panel_for
from tb_search.cohort import DATA_DIR, CohortError, load_cohort
from tb_search.config import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DEFAULT_SOURCE,
    LOCAL_SOURCES,
    ConfigError,
    load_settings,
)
from tb_search.discovery import DiscoveryConfig, discover
from tb_search.documents import EXCERPT_CHARS, collect_docs, resolve_input
from tb_search.evaluate import (
    evaluate_resistance,
    evaluate_retrieval,
    evaluate_screening,
)
from tb_search.reasoning import EFFORTS
from tb_search.report import (
    discovered_rows,
    write_cohort,
    write_papers,
    write_predictions,
    write_results,
)
from tb_search.resistance import predict_samples
from tb_search.retrieval import retrieve_papers
from tb_search.screening import (
    build_agent,
    build_screener,
    check_connection,
    classify_documents_agent,
)
from tb_search.sources import SOURCE_NAMES, build_source


def provider_options(args) -> dict:
    """The provider flags shared by every mode, as load_settings overrides."""
    return {
        "model": args.model,
        "temperature": args.temperature,
        "base_url": args.base_url,
        "reasoning_effort": args.reasoning_effort,
        "source": args.source,
    }


def fail(message: str) -> int:
    """Report an error on stderr and return the CLI's exit code."""
    print(message, file=sys.stderr)
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tb-search",
        description="Screen TB literature and predict drug resistance from VCFs "
        "with a biomni-powered LLM, served locally with vLLM or Ollama.",
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
        "--cohort",
        action="store_true",
        help="Join the resistance table to the listed BIG-TB VCFs and report the "
        "labelled dataset size per drug. No LLM calls. Output defaults to "
        "tb_cohort.json.",
    )
    parser.add_argument(
        "--predict",
        metavar="DRUG",
        help="Predict resistance to DRUG from each isolate's VCF, with a reasoning "
        f"trace. Graded drugs: {', '.join(known_drugs())}. Output defaults to "
        "tb_resistance_<drug>.json.",
    )
    parser.add_argument(
        "--vcf-root",
        metavar="DIR",
        help="[--predict] Directory the VCFs actually live in, replacing the "
        "cluster path recorded in big_tb_file_paths.txt.",
    )
    parser.add_argument(
        "--data-dir",
        metavar="DIR",
        default=str(DATA_DIR),
        help=f"[--cohort --predict] Where the cohort files live (default: {DATA_DIR}).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="[--predict] Isolates to predict over (default: 20).",
    )
    parser.add_argument(
        "--balance",
        action="store_true",
        help="[--predict] Take equal numbers of resistant and susceptible isolates, "
        "so accuracy is not dominated by the susceptible majority.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="[--predict] Sampling seed, so a run can be repeated (default: 0).",
    )
    parser.add_argument(
        "--min-af",
        type=float,
        default=0.0,
        help="[--predict] Drop variants below this alternate-allele frequency; "
        "0 keeps them all (default: 0).",
    )
    parser.add_argument(
        "--keep-filtered",
        action="store_true",
        help="[--predict] Also use variants the caller did not mark PASS.",
    )
    parser.add_argument(
        "--model",
        help="Model id to request: on vLLM the id the server was started with, "
        "on Ollama the pulled tag "
        f"(default: $VLLM_MODEL or {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--base-url",
        metavar="URL",
        help="Where the model is served — the vLLM job's URL, e.g. "
        f"http://gpu042:8000/v1 (default: $VLLM_BASE_URL or {DEFAULT_BASE_URL}). "
        "This is biomni's Custom source, which the A1 agent takes too, so "
        "--agent does its tool calling against that same server.",
    )
    parser.add_argument(
        "--reasoning-effort",
        metavar="LEVEL",
        choices=EFFORTS,
        help="Reasoning effort sent with every request: "
        f"{', '.join(EFFORTS)}. vLLM maps it onto the chat template's thinking "
        "switch, so it sets how much reasoning the captured traces are made of "
        "(default: $LLM_REASONING_EFFORT, else the server's own default).",
    )
    parser.add_argument(
        "--source",
        choices=LOCAL_SOURCES,
        help="Which biomni backend serves the model: Custom (any "
        "OpenAI-compatible server, i.e. vLLM) or Ollama "
        f"(default: $LLM_SOURCE or {DEFAULT_SOURCE}). Both drive the A1 agent.",
    )
    parser.add_argument(
        "--agent",
        action="store_true",
        help="Use the full biomni A1 agent loop instead of small per-doc LLM calls. "
        "A1's system prompt is ~35k tokens, so serve the model with a context "
        "window well past that and with tool calling enabled.",
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
    """--search and --eval-retrieval: the discovery loop."""
    request = args.search or args.eval_retrieval
    try:
        settings = load_settings(**provider_options(args))
        check_connection(settings)
    except ConfigError as e:
        return fail(e)
    llm, model = build_screener(settings), settings.label

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
    if args.eval_retrieval:
        report = evaluate_retrieval(source, request, llm, config)
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
            line = "  ".join(f"{cutoff} {value}" for cutoff, value in scores.items())
            print(f"  recall ({name}): {line}")
        print(f"Saved to {output}")
        return 0

    run = discover(source, request, llm, config)
    rows = discovered_rows(run.papers(config.max_papers, config.accept))
    write_papers(rows, model, request, output or Path("tb_papers.json"))
    return 0


def run_cohort(args, output: Path | None) -> int:
    """--cohort: join the resistance table to the VCF list, no LLM involved."""
    try:
        cohort = load_cohort(args.data_dir)
    except CohortError as e:
        return fail(e)
    write_cohort(cohort, output or Path("tb_cohort.json"))
    return 0


def run_predict(args, output: Path | None) -> int:
    """--predict DRUG: VCF in, resistance call plus a checked reasoning trace out."""
    drug = drug_key(args.predict)
    panel = panel_for(drug)
    try:
        cohort = load_cohort(args.data_dir)
    except CohortError as e:
        return fail(e)

    if drug not in cohort.drugs:
        return fail(
            f"No phenotype column for {args.predict!r} in the resistance table. "
            f"Graded drugs: {', '.join(known_drugs())}"
        )
    samples = cohort.dataset(
        drug,
        limit=max(1, args.limit),
        balance=args.balance,
        seed=args.seed,
        vcf_root=args.vcf_root,
    )
    if not samples:
        return fail(f"No isolate has both a VCF and a {drug} phenotype.")

    try:
        settings = load_settings(**provider_options(args))
        check_connection(settings)
    except ConfigError as e:
        return fail(e)

    counts = {call: sum(1 for s in samples if s.phenotype == call) for call in "RS"}
    print(
        f"Predicting {drug} resistance for {len(samples)} isolate(s) "
        f"({counts['R']} R, {counts['S']} S) "
        f"[{settings.description}]"
    )
    print(f"  panel: tier 1 {', '.join(panel.tier1) or 'none'}", end="")
    print(f"; tier 2 {', '.join(panel.tier2) or 'none'}" if panel.tier2 else "")
    if panel.is_fallback:
        print(f"  ! no graded panel for {drug}; using the union of all panels")
    missing = next((s for s in samples if not Path(s.vcf_path).exists()), None)
    if missing is not None:
        print(
            f"  ! {missing.vcf_path} is not readable here. The BIG-TB VCFs live on "
            "the cluster; pass --vcf-root to point at a local copy."
        )
    print()

    rows = predict_samples(
        build_screener(settings),
        samples,
        drug,
        pass_only=not args.keep_filtered,
        min_allele_frequency=args.min_af,
    )
    scores = evaluate_resistance(rows)
    write_predictions(
        rows,
        scores,
        settings.label,
        drug,
        output or Path(f"tb_resistance_{drug}.json"),
        reasoning_effort=settings.reasoning_effort,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = Path(args.output) if args.output else None
    if args.cohort:
        return run_cohort(args, output)
    if args.predict:
        return run_predict(args, output)
    if args.eval_retrieval or (args.search and not args.agent):
        return run_discovery(args, Path(args.output) if args.output else None)

    mode = "search" if args.search else ("agent" if args.agent else "light")
    output = Path(
        args.output or ("tb_papers.json" if mode == "search" else "tb_verdicts.json")
    )

    try:
        settings = load_settings(
            **provider_options(args),
            use_tool_retriever=args.tools or None,
            load_data_lake=args.data_lake or None,
        )
    except ConfigError as e:
        return fail(e)

    if mode == "search":
        max_papers = max(1, args.max_papers)
        print(
            f"Searching for up to {max_papers} paper(s) "
            f"[A1 agent, {settings.description}]\n"
            f"  query: {args.search}\n"
        )
        try:
            check_connection(settings)
            rows = retrieve_papers(
                build_agent(settings), args.search, max_papers, args.verbose
            )
        except ConfigError as e:
            return fail(e)
        write_papers(rows, settings.label, args.search, output)
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
        f"[mode: {mode}, {settings.description}]\n"
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
