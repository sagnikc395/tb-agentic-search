#!/bin/bash
#SBATCH --job-name=tb-a100
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/tb-a100-%j.out
# One Slurm job on a Unity A100: serve a quantized Qwen3 with vLLM, then run
# the biomni A1 pipeline against it over HTTP. The agent only ever knows the
# inference engine by URL, so nothing here calls a hosted API.
#
#   sbatch scripts/run_unity_a100.sbatch experiment  # the whole experiment
#   sbatch scripts/run_unity_a100.sbatch --search "fluoroquinolone resistance"
#   sbatch scripts/run_unity_a100.sbatch --predict rifampicin --limit 40
#   sbatch scripts/run_unity_a100.sbatch            # no args: just serve, and hold
#
# The word `experiment` runs run_experiment.sh, which is the usual way in;
# anything else is passed straight to cli.py. Watch it with
#   tail -f logs/tb-a100-<jobid>.out
#
# Why this model on this card. Unity's a100 nodes are compute capability 8.0,
# which has no native FP8, so a 4-bit AWQ build is the quantization that runs:
# Qwen3-32B-AWQ is ~18 GB of weights, leaving the rest of a 40 GB card (and
# most of an 80 GB one) to the KV cache. vLLM reads the AWQ config out of the
# checkpoint and picks the awq_marlin kernel itself, so no --quantization flag.
#
# Two serve flags carry the weight. --reasoning-parser splits the model's
# thinking out of the answer into the response's `reasoning` field, which is
# what tb_search stores beside every prediction; without it the thinking stays
# inside the answer text and only part of it survives.
# --enable-auto-tool-choice is what lets biomni's A1 agent call its tools
# against this server. --max-model-len has to clear A1's own system prompt
# (~35k tokens) with room for the task, or agent mode fails on context length.

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-32B-AWQ}"
# Must match the model family; see the table under
# https://docs.vllm.ai/en/stable/features/reasoning_outputs/
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
PORT="${PORT:-8000}"
# Tensor-parallel size has to match --gres=gpu:N above; a 4-bit 32B fits on one
# card, so there is nothing to split.
TP_SIZE="${TP_SIZE:-1}"
# Leave a slice of the card to the CUDA context and fragmentation. On a 40 GB
# A100 this is the number to drop first if the KV cache will not allocate.
GPU_UTIL="${GPU_UTIL:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
VLLM_VENV="${VLLM_VENV:-$HOME/vllm-venv}"
# How long to wait for the weights to load before giving up on the server.
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-1200}"

mkdir -p logs
module load cuda/12.6 2>/dev/null || true

# The venv holding vLLM, built once. autoawq is what reads the 4-bit weights.
if [ ! -x "${VLLM_VENV}/bin/vllm" ]; then
    echo "Installing vLLM into ${VLLM_VENV} (first run only)..."
    python3 -m venv "${VLLM_VENV}"
    # shellcheck disable=SC1091
    source "${VLLM_VENV}/bin/activate"
    pip install --upgrade pip
    pip install vllm autoawq
else
    # shellcheck disable=SC1091
    source "${VLLM_VENV}/bin/activate"
fi

BASE_URL="http://127.0.0.1:${PORT}/v1"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "Serving ${MODEL} on $(hostname) port ${PORT}"
echo "Point a client at:  --base-url http://$(hostname):${PORT}/v1 --model ${MODEL}"
echo "From off-cluster:   ssh -N -L ${PORT}:$(hostname):${PORT} unity.rc.umass.edu"

vllm serve "${MODEL}" \
    --port "${PORT}" \
    --host 0.0.0.0 \
    --tensor-parallel-size "${TP_SIZE}" \
    --gpu-memory-utilization "${GPU_UTIL}" \
    --reasoning-parser "${REASONING_PARSER}" \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --max-model-len "${MAX_MODEL_LEN}" &
VLLM_PID=$!
# Kill the server on any exit, so the job does not sit on the card after the
# agent is done or has failed.
trap 'kill "${VLLM_PID}" 2>/dev/null || true' EXIT

# Wait for the weights to load. A 4-bit 32B takes a couple of minutes.
echo "Waiting for ${BASE_URL} ..."
for ((waited = 0; waited < STARTUP_TIMEOUT; waited += 5)); do
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "vLLM exited before it was ready; see the log above." >&2
        exit 1
    fi
    if curl -sf "${BASE_URL}/models" >/dev/null; then
        echo "Server up after ${waited}s."
        break
    fi
    sleep 5
done
curl -sf "${BASE_URL}/models" >/dev/null || {
    echo "Server did not come up within ${STARTUP_TIMEOUT}s." >&2
    exit 1
}

# No pipeline arguments: hold the server for the rest of the walltime so a
# client elsewhere (or an ssh tunnel) can use it.
if [ "$#" -eq 0 ]; then
    echo "No cli.py arguments given — holding the server until the job ends."
    wait "${VLLM_PID}"
    exit $?
fi

# biomni runs here, in this process, and reaches the model as an ordinary
# OpenAI-compatible server; --source Custom is the backend that hands back the
# separated reasoning traces.
# Not exec: the trap above has to run afterwards, or the server keeps the card
# until the walltime expires.
cd "${SLURM_SUBMIT_DIR:-$PWD}"

if [ "$1" = "experiment" ]; then
    shift
    BASE_URL="${BASE_URL}" MODEL="${MODEL}" ./run_experiment.sh "$@"
else
    uv run cli.py \
        --base-url "${BASE_URL}" \
        --model "${MODEL}" \
        --source Custom \
        "$@"
fi
