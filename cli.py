"""Screen research documents and predict TB drug resistance with the Biomni A1 agent.

Every LLM mode is one A1 run. The agent runs in this process and talks to Groq's
hosted endpoint, so there is no server to start and no GPU involved — only a key
in the environment:

    export GROQ_API_KEY=gsk_...
    tb-search --predict rifampicin --model openai/gpt-oss-120b \
        --reasoning-effort high

The model has to do two things for any of this to work: call tools, because A1
calls its tools as code, and carry A1's ~35k-token system prompt on every step,
which the default openai/gpt-oss-120b has 131k of context for. Its thinking
comes back in its own field and is recorded next to each prediction.

The default mode screens a directory of documents, handing the agent
--batch-size of them per run. --search runs a retrieval request instead — often
one naming a seed paper by URL or DOI — and saves the related papers the agent
found through its own search tools.

The VCF side is its own pair of modes. --cohort joins the BIG-TB resistance
table to the listed VCFs and reports the labelled dataset size per drug, with no
model involved; --predict DRUG reads each isolate's VCF, asks the agent for a
resistant / not resistant / unknown call with a reasoning trace, and scores both
the calls against the DST phenotype and the traces against the variants actually
present.
"""

import argparse
import sys
from pathlib import Path

from tb_search.catalogue import drug_key, known_drugs, panel_for
from tb_search.cohort import DATA_DIR, CohortError, load_cohort
from tb_search.config import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_MODEL,
    DEFAULT_SOURCE,
    SOURCES,
    ConfigError,
    load_settings,
)
from tb_search.documents import EXCERPT_CHARS, collect_docs, resolve_input
from tb_search.evaluate import evaluate_resistance, evaluate_screening
from tb_search.reasoning import EFFORTS
from tb_search.report import (
    write_cohort,
    write_papers,
    write_predictions,
    write_results,
)
from tb_search.resistance import predict_samples
from tb_search.retrieval import retrieve_papers
from tb_search.screening import (
    build_agent,
    check_connection,
    classify_documents_agent,
)


def provider_options(args) -> dict:
    """The provider flags shared by every mode, as load_settings overrides."""
    return {
        "model": args.model,
        "temperature": args.temperature,
        "base_url": args.base_url,
        "reasoning_effort": args.reasoning_effort,
        "source": args.source,
        "max_retries": args.max_retries,
        "use_tool_retriever": args.tools or None,
        "load_data_lake": args.data_lake or None,
    }


def fail(message: str) -> int:
    """Report an error on stderr and return the CLI's exit code."""
    print(message, file=sys.stderr)
    return 1


def connected_agent(args):
    """Settings checked, Groq pinged, agent built — the one model path.

    Raises ConfigError, so a missing key or a model this account cannot reach
    fails here rather than part-way through a sweep.
    """
    settings = load_settings(**provider_options(args))
    check_connection(settings)
    return settings, build_agent(settings)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tb-search",
        description="Screen TB literature and predict drug resistance from VCFs "
        "with the Biomni A1 agent, running on Groq's inference endpoint.",
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
        "or DOI, e.g. 'find me relevant papers like this paper <url>'. The agent "
        "searches with its own tools. Output defaults to tb_papers.json.",
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
        "(default: 10). Fewer come back when the agent confirmed fewer.",
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
        help="Groq model id to request, e.g. openai/gpt-oss-20b or "
        f"qwen/qwen3.8-27b (default: $GROQ_MODEL or {DEFAULT_MODEL}). It has to "
        "support tool calling, and its context window has to clear A1's "
        "~35k-token system prompt.",
    )
    parser.add_argument(
        "--base-url",
        metavar="URL",
        help="Groq's OpenAI-compatible endpoint (default: $GROQ_BASE_URL or "
        "https://api.groq.com/openai/v1). Only worth passing to reach Groq "
        "through a proxy.",
    )
    parser.add_argument(
        "--reasoning-effort",
        metavar="LEVEL",
        choices=EFFORTS,
        help="Reasoning effort sent with every request: "
        f"{', '.join(EFFORTS)}. It sets how much reasoning the captured traces "
        "are made of, and which values are accepted depends on the model — "
        "low/medium/high for the GPT-OSS models, none/default for Qwen "
        "(default: $LLM_REASONING_EFFORT, else Groq's default for the model).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        metavar="N",
        help="How many times a rate-limited or briefly failing request waits and "
        f"tries again (default: $GROQ_MAX_RETRIES or {DEFAULT_MAX_RETRIES}). "
        "Groq's 429 says how long to wait and that is obeyed; otherwise the "
        "wait doubles per attempt.",
    )
    parser.add_argument(
        "--source",
        choices=SOURCES,
        help="Which biomni backend answers the agent's calls "
        f"(default: $LLM_SOURCE or {DEFAULT_SOURCE}). Only Groq is offered: "
        "this pipeline does not serve its own model.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="[screening] Documents per agent run (default: 2).",
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
        help="Let A1 retrieve a relevant tool subset instead of carrying them all.",
    )
    parser.add_argument(
        "--data-lake",
        action="store_true",
        help="Let biomni download its full data lake (tens of GB).",
    )
    parser.add_argument("--temperature", type=float, help="Sampling temperature.")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="[--search] Also print biomni's full step-by-step agent output.",
    )
    return parser


def run_cohort(args, output: Path | None) -> int:
    """--cohort: join the resistance table to the VCF list, no agent involved."""
    try:
        cohort = load_cohort(args.data_dir)
    except CohortError as e:
        return fail(e)
    write_cohort(cohort, output or Path("tb_cohort.json"))
    return 0


def run_search(args, output: Path | None) -> int:
    """--search: one A1 run on a retrieval request, its paper list saved."""
    max_papers = max(1, args.max_papers)
    try:
        settings, agent = connected_agent(args)
    except ConfigError as e:
        return fail(e)
    print(
        f"Searching for up to {max_papers} paper(s) "
        f"[A1 agent, {settings.description}]\n"
        f"  query: {args.search}\n"
    )
    rows = retrieve_papers(agent, args.search, max_papers, args.verbose)
    write_papers(rows, settings.label, args.search, output or Path("tb_papers.json"))
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
        settings, agent = connected_agent(args)
    except ConfigError as e:
        return fail(e)

    counts = {call: sum(1 for s in samples if s.phenotype == call) for call in "RS"}
    print(
        f"Predicting {drug} resistance for {len(samples)} isolate(s) "
        f"({counts['R']} R, {counts['S']} S) "
        f"[A1 agent, {settings.description}]"
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
        agent,
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


def run_screening(args, output: Path | None) -> int:
    """The default mode: screen a directory of documents in agent batches."""
    input_path = resolve_input(args.input)
    if input_path is None:
        return fail(
            f"Input not found: {args.input!r}"
            if args.input
            else "No papers directory found; pass one, e.g. `tb-search ../papers`"
        )

    docs = collect_docs(input_path)
    if not docs:
        return fail(f"No .pdf/.txt/.md/... documents found under {input_path}")

    try:
        settings, agent = connected_agent(args)
    except ConfigError as e:
        return fail(e)

    print(
        f"Screening {len(docs)} document(s) under {input_path} "
        f"[A1 agent, {settings.description}]\n"
    )
    rows = classify_documents_agent(
        agent, docs, max(1, args.batch_size), max(200, args.max_chars)
    )
    write_results(rows, settings, input_path, output or Path("tb_verdicts.json"))

    if args.gold:
        scores = evaluate_screening(rows, Path(args.gold))
        print(
            f"Against {args.gold}: {scores['accuracy']} accuracy on {scores['answered']} "
            f"answered of {scores['labelled']} labelled; wrong: {scores['wrong'] or 'none'}; "
            f"abstained: {scores['abstained'] or 'none'}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = Path(args.output) if args.output else None
    if args.cohort:
        return run_cohort(args, output)
    if args.predict:
        return run_predict(args, output)
    if args.search:
        return run_search(args, output)
    return run_screening(args, output)


if __name__ == "__main__":
    sys.exit(main())
