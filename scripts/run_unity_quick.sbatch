#!/bin/bash
#SBATCH --job-name=tb-quick
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/tb-quick-%j.out
# The same experiment, cut down to show that the pipeline runs end to end and
# what the model's responses look like — one drug, a handful of isolates, an
# hour of walltime. These are not the numbers to report; re-run
# scripts/run_unity_cpu.sbatch for those.
#
#   sbatch scripts/run_unity_quick.sbatch
#   tail -f logs/tb-quick-<jobid>.out
#
# On Groq the three things that decide whether this finishes inside the
# walltime are the drug count, the isolate count and the reasoning effort —
# every step of every A1 run is a request, carrying A1's ~35k-token system
# prompt, counted against the account's per-minute token limit. A smaller model
# is the fourth lever, and the one that costs the least in quality here:
# openai/gpt-oss-20b has the same 131k context and the same low/medium/high
# effort scale as the 120B.
#
# Every knob is overridable at submit time:
#
#   QUICK_DRUGS="rifampicin isoniazid" sbatch scripts/run_unity_quick.sbatch
#   QUICK_LIMIT=4 sbatch scripts/run_unity_quick.sbatch
#   MODEL=openai/gpt-oss-120b QUICK_EFFORT=high sbatch scripts/run_unity_quick.sbatch
#
# Note sbatch only reads the directives of the file it is given, so flags on
# the command line (--partition, --time) override the ones above.

set -euo pipefail

# The 20B rather than the 120B: same context window, same effort scale, fewer
# tokens per second of the account's limit spent on a run whose point is that
# it finishes.
export MODEL="${MODEL:-openai/gpt-oss-20b}"

# One drug, few isolates, less thinking per call.
export DRUGS="${QUICK_DRUGS:-rifampicin}"
export LIMIT="${QUICK_LIMIT:-8}"
export EFFORT="${QUICK_EFFORT:-medium}"
export SEED="${SEED:-0}"
# Named so it is obvious in results/ which runs are the cut-down ones.
export RUN_ID="${RUN_ID:-quick-${SLURM_JOB_ID:-local}}"

echo "Quick run: ${MODEL}, drugs=${DRUGS}, limit=${LIMIT}, effort=${EFFORT}"
echo "Results will land in results/${RUN_ID}/"

# sbatch copies the submitted file to the node's spool directory, so $0 is not
# in the repo and cannot locate its sibling. SLURM_SUBMIT_DIR is where sbatch
# was run from, which is the repo root or scripts/ inside it.
FULL=""
for dir in "${SLURM_SUBMIT_DIR:-$PWD}" "${SLURM_SUBMIT_DIR:-$PWD}/scripts" \
           "${SLURM_SUBMIT_DIR:-$PWD}/.."; do
    if [ -f "${dir}/scripts/run_unity_cpu.sbatch" ]; then
        FULL="${dir}/scripts/run_unity_cpu.sbatch"
        break
    elif [ -f "${dir}/run_unity_cpu.sbatch" ]; then
        FULL="${dir}/run_unity_cpu.sbatch"
        break
    fi
done
if [ -z "${FULL}" ]; then
    echo "Cannot find run_unity_cpu.sbatch next to ${SLURM_SUBMIT_DIR:-$PWD}." >&2
    echo "Submit this from the repo: sbatch scripts/run_unity_quick.sbatch" >&2
    exit 1
fi

# Checking the key, finding the repo and running the pipeline are all identical
# to the full job, so that script does them — with the environment above in
# place of its defaults.
exec bash "${FULL}" experiment
