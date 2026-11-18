#!/bin/bash
#SBATCH --job-name=vllm-tb
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/vllm-%j.out
# Serve the model on Unity, so the agent runs as a separate process that only
# knows the inference engine by URL. This is the only way the pipeline gets a
# model: nothing calls a hosted API.
#
#   sbatch scripts/serve_vllm_unity.sbatch
#   tail -f logs/vllm-<jobid>.out      # prints the --base-url to use
#
# Two flags carry the weight. --reasoning-parser splits the model's thinking out
# of the answer into the response's `reasoning` field, which is what tb_search
# stores beside every prediction; without it the thinking stays inside the
# answer text and only part of it survives. --enable-auto-tool-choice is what
# lets biomni's A1 agent call its tools against this server.
#
# --max-model-len has to clear A1's own system prompt (~35k tokens) with room
# for the task, or agent mode fails on context length.

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-32B}"
# Must match the model family; see the table under
# https://docs.vllm.ai/en/stable/features/reasoning_outputs/
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
PORT="${PORT:-8000}"
# Tensor-parallel size has to match --gres=gpu:N above.
TP_SIZE="${TP_SIZE:-2}"

mkdir -p logs
module load cuda/12.6 2>/dev/null || true
source "${VLLM_VENV:-$HOME/vllm-venv}/bin/activate"

echo "Serving ${MODEL} on $(hostname) port ${PORT}"
echo "Point the client at:  --base-url http://$(hostname):${PORT}/v1 --model ${MODEL}"
echo "From off-cluster:     ssh -N -L ${PORT}:$(hostname):${PORT} unity.rc.umass.edu"

exec vllm serve "${MODEL}" \
    --port "${PORT}" \
    --host 0.0.0.0 \
    --tensor-parallel-size "${TP_SIZE}" \
    --reasoning-parser "${REASONING_PARSER}" \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --max-model-len 65536
