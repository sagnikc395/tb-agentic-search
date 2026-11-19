#!/bin/bash
#SBATCH --job-name=tb-quick
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --constraint="vram23&[sm_80|sm_86|sm_89|sm_90]"
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=logs/tb-quick-%j.out
# The same experiment, sized to actually start. run_unity_a100.sbatch asks for
# a non-preemptible A100 for eight hours with 96G and 16 cores, and on a busy
# day that sits in Reason=Priority for most of an afternoon. Everything here is
# relaxed in the four directions that decide when Slurm can backfill you:
#
#   partition   gpu-preempt, not gpu-highvram — preemptible, so far shorter
#               queue. --requeue above puts the job straight back in line if it
#               is killed, and a rerun starts the run from the top.
#   gpu         gpu:1 + vram23 + Ampere-or-newer, not gpu:a100:1 — any recent
#               card with >=23 GB, which is a far larger pool than the A100
#               nodes rather than a handful of them. A 4-bit 14B is ~9 GB of
#               weights, so it fits with room for the KV cache.
#
#               The architecture half of that constraint is not optional. The
#               vram23 pool includes the Tesla M40 24GB, which is Maxwell
#               (sm_52): vLLM needs sm_70 to start at all, and the awq_marlin
#               kernel that reads these 4-bit weights needs sm_80. Land on an
#               M40 and the job spends ten minutes building the venv and then
#               dies at `vllm serve` on an unsupported device. The sm_* list
#               above is Ampere, Ada and Hopper; add sm_70/sm_75 only with a
#               non-AWQ model, since those cards fall back to the slow AWQ
#               path. `sinfo -o '%P %G %f'` settles the feature names if any
#               of them are ever renamed.
#   time        2h, not 8h. Short jobs slot into gaps before the next big
#               reservation; this is usually the single biggest win.
#   cpu/mem     4 cores and 32G. vLLM at TP=1 is GPU-bound and the weights
#               never touch host RAM, so the old 16/96G only blocked packing.
#
# What you give up: a smaller model and one drug instead of four. These are
# partial results to show that the pipeline runs end to end and what the
# responses look like — not the numbers to report. Re-run
# scripts/run_unity_a100.sbatch for those.
#
#   sbatch scripts/run_unity_quick.sbatch
#   tail -f logs/tb-quick-<jobid>.out
#
# Every knob below is overridable at submit time, so the same script covers
# "as fast as possible" and "the real model, still relaxed":
#
#   QUICK_DRUGS="rifampicin isoniazid" sbatch scripts/run_unity_quick.sbatch
#   QUICK_LIMIT=4 sbatch scripts/run_unity_quick.sbatch       # 4 isolates
#   MODEL=Qwen/Qwen3-32B-AWQ MAX_MODEL_LEN=65536 \
#     sbatch --constraint=vram40 scripts/run_unity_quick.sbatch
#
# Note sbatch only reads the directives of the file it is given, so flags on
# the command line (--constraint, --partition, --time) override the ones above.

set -euo pipefail

# A 4-bit 14B instead of the 32B: ~9 GB of weights, which is what lets this run
# on a 23 GB card. awq_marlin still applies on sm_80 and newer.
export MODEL="${MODEL:-Qwen/Qwen3-14B-AWQ}"
# A1's system prompt is ~35k tokens on its own, so this is close to the floor —
# anything tighter and agent mode fails on context length rather than on the
# task. Paired with a higher utilisation because the card is smaller.
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
export GPU_UTIL="${GPU_UTIL:-0.95}"

# One drug, few isolates, less thinking per call: the three things that decide
# whether the run finishes inside the walltime.
export DRUGS="${QUICK_DRUGS:-rifampicin}"
export LIMIT="${QUICK_LIMIT:-8}"
export EFFORT="${QUICK_EFFORT:-medium}"
export SEED="${SEED:-0}"
# Named so it is obvious in results/ which runs are the cut-down ones, and
# stable across a requeue so a preempted job overwrites itself instead of
# leaving half-finished folders behind.
export RUN_ID="${RUN_ID:-quick-${SLURM_JOB_ID:-local}}"

echo "Quick run: ${MODEL}, drugs=${DRUGS}, limit=${LIMIT}, effort=${EFFORT}"
echo "Results will land in results/${RUN_ID}/"

# sbatch copies the submitted file to the node's spool directory, so $0 is not
# in the repo and cannot locate its sibling. SLURM_SUBMIT_DIR is where sbatch
# was run from, which is the repo root or scripts/ inside it.
SERVE=""
for dir in "${SLURM_SUBMIT_DIR:-$PWD}" "${SLURM_SUBMIT_DIR:-$PWD}/scripts" \
           "${SLURM_SUBMIT_DIR:-$PWD}/.."; do
    if [ -f "${dir}/scripts/run_unity_a100.sbatch" ]; then
        SERVE="${dir}/scripts/run_unity_a100.sbatch"
        break
    elif [ -f "${dir}/run_unity_a100.sbatch" ]; then
        SERVE="${dir}/run_unity_a100.sbatch"
        break
    fi
done
if [ -z "${SERVE}" ]; then
    echo "Cannot find run_unity_a100.sbatch next to ${SLURM_SUBMIT_DIR:-$PWD}." >&2
    echo "Submit this from the repo: sbatch scripts/run_unity_quick.sbatch" >&2
    exit 1
fi

# Serving, waiting for the server, running the pipeline and cleaning up the
# card are all identical to the full job, so that script does them — with the
# environment above in place of its defaults.
exec bash "${SERVE}" experiment
