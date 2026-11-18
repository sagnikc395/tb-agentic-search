#!/bin/bash
# Set up and serve vLLM in an interactive GPU session with 32 GB of VRAM.
#
# The batch job next door (serve_vllm_unity.sbatch) wants two large GPUs and
# runs unattended; this one is for a single 32 GB card you sit in front of,
# which is what a shorter allocation usually gets you. It builds the venv the
# first time, then serves, so a fresh account needs nothing else.
#
#   scripts/setup_vllm_32g.sh              # grabs the session, then serves
#   srun --pty ... bash                    # or: get a session yourself,
#   scripts/setup_vllm_32g.sh              #     and this serves inside it
#
# Run from the login node it re-execs itself inside `srun --pty`, so the server
# stays attached to your terminal and dies with it. Already inside a job
# (SLURM_JOB_ID set), it skips straight to serving.
#
# 32 GB does not hold Qwen3-32B in bf16 — the weights alone are ~64 GB — so the
# default here is the AWQ 4-bit build of the same model: ~18 GB of weights,
# leaving room for the KV cache at the context length A1 needs. It is the same
# family, so the `qwen3` reasoning parser and the tool-call parser are
# unchanged, and tb_search only has to be told the served model id. For an
# unquantized model on this card, set MODEL=Qwen/Qwen3-8B instead.

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-32B-AWQ}"
# Must match the model family; see the table under
# https://docs.vllm.ai/en/stable/features/reasoning_outputs/
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
PORT="${PORT:-8000}"
# One card, so no tensor parallelism.
TP_SIZE="${TP_SIZE:-1}"
# Slurm's way of asking for a card with this much VRAM; Unity spells it
# vram32 / vram40 / vram80. Change it to whatever your cluster uses.
GPU_CONSTRAINT="${GPU_CONSTRAINT:-vram32}"
PARTITION="${PARTITION:-gpu}"
CPUS="${CPUS:-8}"
# Host memory, not VRAM: enough to stage the weights while they load.
HOST_MEM="${HOST_MEM:-64G}"
TIME_LIMIT="${TIME_LIMIT:-04:00:00}"
VLLM_VENV="${VLLM_VENV:-$HOME/vllm-venv}"
# Leave a slice of the 32 GB to the CUDA context and fragmentation; raising
# this past ~0.92 is how you get an OOM minutes into a run rather than at load.
GPU_UTIL="${GPU_UTIL:-0.90}"
# Has to clear A1's own system prompt (~35k tokens) with room for the task, or
# agent mode fails on context length.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"

# Step 1: be inside a GPU session. Re-exec into one if we are not.
if [ -z "${SLURM_JOB_ID:-}" ]; then
    if ! command -v srun >/dev/null 2>&1; then
        echo "No srun here and no Slurm job around us: nothing to allocate a GPU." >&2
        echo "Run this on the cluster, or start the server by hand inside a session." >&2
        exit 1
    fi
    echo "Requesting a ${GPU_CONSTRAINT} GPU for ${TIME_LIMIT} on partition ${PARTITION}..."
    exec srun \
        --partition="${PARTITION}" \
        --gres=gpu:1 \
        --constraint="${GPU_CONSTRAINT}" \
        --cpus-per-task="${CPUS}" \
        --mem="${HOST_MEM}" \
        --time="${TIME_LIMIT}" \
        --job-name=vllm-tb-32g \
        --pty "$0" "$@"
fi

# Step 2: the venv holding vLLM, built once.
module load cuda/12.6 2>/dev/null || true
if [ ! -x "${VLLM_VENV}/bin/vllm" ]; then
    echo "Installing vLLM into ${VLLM_VENV} (first run only)..."
    python3 -m venv "${VLLM_VENV}"
    # shellcheck disable=SC1091
    source "${VLLM_VENV}/bin/activate"
    pip install --upgrade pip
    # autoawq is what reads the 4-bit default model; harmless for a bf16 one.
    pip install vllm autoawq
else
    # shellcheck disable=SC1091
    source "${VLLM_VENV}/bin/activate"
fi

# Step 3: serve, and say where.
mkdir -p logs
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "Serving ${MODEL} on $(hostname) port ${PORT}"
echo "Point the client at:  --base-url http://$(hostname):${PORT}/v1 --model ${MODEL}"
echo "From off-cluster:     ssh -N -L ${PORT}:$(hostname):${PORT} unity.rc.umass.edu"

exec vllm serve "${MODEL}" \
    --port "${PORT}" \
    --host 0.0.0.0 \
    --tensor-parallel-size "${TP_SIZE}" \
    --gpu-memory-utilization "${GPU_UTIL}" \
    --reasoning-parser "${REASONING_PARSER}" \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --max-model-len "${MAX_MODEL_LEN}"
