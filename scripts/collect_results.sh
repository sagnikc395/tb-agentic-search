#!/bin/bash
# Turn a run folder into something another person can read: SUMMARY.md with the
# headline numbers, metrics.csv with the per-drug scores, and a tarball of the
# whole folder to attach to an email.
#
#   scripts/collect_results.sh                      # the newest run under results/
#   scripts/collect_results.sh results/<run-id>     # a particular one
#
# run_experiment.sh calls this at the end, so a finished run is already
# packaged; running it again after editing nothing is harmless, and running it
# on an interrupted run summarises whatever stages did finish.
#
# The raw JSON stays exactly as the pipeline wrote it — every prediction with
# its variants and its reasoning trace — because the summary is for reading and
# the JSON is what anyone would re-check it against.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS="${REPO}/results"

RUN_DIR="${1:-}"
if [ -z "${RUN_DIR}" ]; then
    RUN_DIR="$(ls -1d "${RESULTS}"/*/ 2>/dev/null | sort | tail -1 || true)"
    [ -n "${RUN_DIR}" ] || { echo "No run folders under ${RESULTS}." >&2; exit 1; }
fi
RUN_DIR="${RUN_DIR%/}"
[ -d "${RUN_DIR}" ] || { echo "Not a directory: ${RUN_DIR}" >&2; exit 1; }

python3 - "${RUN_DIR}" <<'PY'
import csv, json, pathlib, re, sys

run = pathlib.Path(sys.argv[1])


def load(path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def pct(value):
    """Rates come back as fractions, or as None when nothing was counted."""
    return "—" if value is None else f"{value * 100:.1f}%"


meta = load(run / "run_metadata.json") or {}
lines = [f"# TB agentic search — run `{meta.get('run_id', run.name)}`", ""]

lines += [
    "| | |",
    "| --- | --- |",
    f"| started | {meta.get('started_at', '—')} |",
    f"| model | `{meta.get('model', '—')}` |",
    f"| reasoning effort | {meta.get('reasoning_effort', '—')} |",
    f"| isolates per drug | {meta.get('isolates_per_drug', '—')}"
    f"{' (balanced R/S)' if meta.get('balanced') else ''} |",
    f"| sampling seed | {meta.get('seed', '—')} |",
    f"| host | {meta.get('host', '—')}"
    f"{', Slurm job ' + meta['slurm_job_id'] if meta.get('slurm_job_id') else ''} |",
    f"| commit | `{(meta.get('git_commit') or '—')[:12]}`"
    f"{' (uncommitted changes)' if meta.get('git_dirty') else ''} |",
    "",
]

gpu = run / "logs" / "gpu.txt"
if gpu.exists():
    card = [line for line in gpu.read_text().splitlines() if line.strip()][1:]
    if card:
        lines += [f"GPU: {'; '.join(card)}", ""]

stages = run / "logs" / "stages.txt"
if stages.exists():
    failed = [
        line.split("\t")[0]
        for line in stages.read_text().splitlines()
        if line.endswith("FAILED")
    ]
    if failed:
        lines += [f"**Stages that failed: {', '.join(failed)}** (see `logs/`).", ""]

# Resistance prediction: the table the whole run exists to produce.
metric_rows = []
for path in sorted(run.glob("tb_resistance_*.json")):
    payload = load(path)
    if not payload:
        continue
    scores = payload.get("scores", {})
    trace = scores.get("trace", {})
    confusion = scores.get("confusion", {})
    metric_rows.append(
        {
            "drug": payload.get("drug", path.stem),
            "isolates": scores.get("n"),
            "answered": scores.get("answered"),
            "accuracy": scores.get("accuracy"),
            "sensitivity": scores.get("sensitivity"),
            "specificity": scores.get("specificity"),
            "ppv": scores.get("ppv"),
            "npv": scores.get("npv"),
            "trace_valid": trace.get("valid"),
            "hallucinated_variants": trace.get("hallucinated_variants"),
            "thinking_captured": trace.get("thinking_captured"),
            "errors": scores.get("errors"),
            **{f"confusion_{k}": v for k, v in confusion.items()},
            "file": path.name,
        }
    )

if metric_rows:
    lines += [
        "## Resistance prediction",
        "",
        "Accuracy is over answered isolates only, so abstentions show up as a "
        "lower *answered* rather than as wrong calls. Sensitivity is the "
        "clinical number: a missed resistant isolate is a patient put on a drug "
        "that will not work. *Trace valid* and *hallucinated* grade the model's "
        "own reasoning against the VCF it was given — a run can be accurate and "
        "still be reasoning from variants that were never there.",
        "",
        "| drug | isolates | answered | accuracy | sensitivity | specificity | "
        "trace valid | hallucinated | thinking |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in metric_rows:
        lines.append(
            f"| {row['drug']} | {row['isolates']} | {row['answered']} | "
            f"{pct(row['accuracy'])} | {pct(row['sensitivity'])} | "
            f"{pct(row['specificity'])} | {pct(row['trace_valid'])} | "
            f"{pct(row['hallucinated_variants'])} | {pct(row['thinking_captured'])} |"
        )
    lines.append("")

    with open(run / "metrics.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)

# Screening: the scores against the hand labels are printed, not stored, so
# they are read back out of the stage log.
verdicts = load(run / "tb_verdicts.json")
if verdicts:
    counts = verdicts.get("summary", {})
    lines += [
        "## Literature screening",
        "",
        f"{counts.get('related', 0)} TB-related, {counts.get('not_related', 0)} not, "
        f"{counts.get('unknown', 0)} unknown, over `{verdicts.get('input', '?')}`.",
        "",
    ]
    log = run / "logs" / "screening.log"
    if log.exists():
        against = [
            line.strip()
            for line in log.read_text().splitlines()
            if re.match(r"^Against .*accuracy", line.strip())
        ]
        if against:
            lines += [f"{against[-1]}", ""]

cohort = load(run / "tb_cohort.json")
if cohort:
    drugs = {row["drug"]: row for row in cohort.get("drugs", [])}
    lines += [
        "## Cohort",
        "",
        f"{cohort.get('isolates', 0)} isolate(s) with a VCF; "
        f"{cohort.get('unmatched_rows', 0)} table row(s) had none.",
        "",
        "| drug | resistant | susceptible | total |",
        "| --- | ---: | ---: | ---: |",
    ]
    wanted = meta.get("drugs") or list(drugs)
    for drug in wanted:
        row = drugs.get(drug)
        if row:
            lines.append(
                f"| {drug} | {row['resistant']} | {row['susceptible']} | {row['total']} |"
            )
    lines.append("")

lines += [
    "## What is in this folder",
    "",
    "| file | what it holds |",
    "| --- | --- |",
    "| `SUMMARY.md` | this file |",
    "| `metrics.csv` | the per-drug scores above, plus the confusion counts |",
    "| `tb_resistance_<drug>.json` | every prediction: variants given, call made, "
    "reasoning trace, trace grade |",
    "| `tb_verdicts.json` | the screening verdict for each document |",
    "| `tb_cohort.json` | the isolate/phenotype join the predictions sampled from |",
    "| `run_metadata.json` | model, commit, seed, host — what the numbers are of |",
    "| `logs/` | the full stdout of every stage |",
    "",
]

(run / "SUMMARY.md").write_text("\n".join(lines))
print(f"Wrote {run / 'SUMMARY.md'}")
if metric_rows:
    print(f"Wrote {run / 'metrics.csv'} ({len(metric_rows)} drug(s))")
PY

# One file to attach. Built from the parent, so it unpacks into its own folder
# rather than over whatever directory someone happens to be in.
TARBALL="${RESULTS}/$(basename "${RUN_DIR}").tar.gz"
tar -czf "${TARBALL}" -C "$(dirname "${RUN_DIR}")" "$(basename "${RUN_DIR}")"
echo "Wrote ${TARBALL}"
echo
echo "Send ${RUN_DIR} as it is, or just the tarball — SUMMARY.md reads first."
