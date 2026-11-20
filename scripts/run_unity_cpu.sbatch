#!/bin/bash
#SBATCH --job-name=tb-groq
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/tb-groq-%j.out
# One Slurm job for the full experiment. Inference is Groq's, so this job holds
# no model and asks for no GPU: it is a long-running client, and what it needs
# is wall time and a working network route to api.groq.com.
#
#   sbatch scripts/run_unity_cpu.sbatch experiment     # the whole experiment
#   sbatch scripts/run_unity_cpu.sbatch --search "fluoroquinolone resistance"
#   sbatch scripts/run_unity_cpu.sbatch --predict rifampicin --limit 40
#
# The word `experiment` runs run_experiment.sh, which is the usual way in;
# anything else is passed straight to cli.py. Watch it with
#   tail -f logs/tb-groq-<jobid>.out
#
# This replaces the A100 job the pipeline used to need. The resources are now
# the small ones: 4 cores and 16G is generous for a process whose work is HTTP
# requests and VCF parsing, and `--partition=cpu` is Unity's default partition,
# so this queues in minutes rather than sitting in Reason=Priority behind every
# other GPU job. Eight hours is the walltime a four-drug run at --limit 20
# fits inside with room for the rate limit; a longer sweep wants more:
#
#   sbatch --time=24:00:00 scripts/run_unity_cpu.sbatch experiment
#
# What used to be the hard part — a card big enough, the right compute
# capability, a quantization that runs on it, a context window past A1's 35k
# system prompt — is now the provider's problem. What is left is the account's
# rate limit, which is why run_experiment.sh records every second spent waiting
# one out (logs/groq_usage.jsonl) and SUMMARY.md reports it.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

# The key has to reach the job. sbatch exports the submitting environment by
# default, so an exported GROQ_API_KEY is already here; .env is the fallback
# and run_experiment.sh reads it itself.
if [ -z "${GROQ_API_KEY:-}" ] && [ ! -f .env ]; then
    echo "No GROQ_API_KEY in the job environment and no .env in $(pwd)." >&2
    echo "Export it before submitting, or put it in .env:" >&2
    echo "  export GROQ_API_KEY=gsk_...; sbatch scripts/run_unity_cpu.sbatch experiment" >&2
    exit 1
fi

MODEL="${MODEL:-${GROQ_MODEL:-openai/gpt-oss-120b}}"
echo "Groq model: ${MODEL}  (job ${SLURM_JOB_ID:-local} on $(hostname))"

if [ "$#" -eq 0 ] || [ "$1" = "experiment" ]; then
    [ "$#" -gt 0 ] && shift
    # LOCAL=1 so the script runs the stages here rather than submitting again;
    # SLURM_JOB_ID alone already says that, and this makes it explicit.
    LOCAL=1 MODEL="${MODEL}" ./run_experiment.sh "$@"
else
    uv run cli.py --model "${MODEL}" "$@"
fi
