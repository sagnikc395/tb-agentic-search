#!/bin/bash
# Sample a run while it is happening, so every run leaves a record of what the
# card and the server were doing and not just what the pipeline printed.
#
#   scripts/monitor_run.sh start results/<run-id>   # background sampler
#   scripts/monitor_run.sh stop  results/<run-id>   # stop it
#   scripts/monitor_run.sh status results/<run-id>  # last sample, human-readable
#
# run_experiment.sh starts and stops it around the stages, so there is nothing
# to run by hand for a normal run. Started twice on the same folder the second
# call is a no-op, which is what makes it safe to call from a requeued job.
#
# Two files land in <run-id>/logs/:
#
#   gpu_timeline.csv     utilisation, memory, power and temperature per tick.
#                        The question it answers is the one a stage log cannot:
#                        a run that is slow because the card is pinned at 100%
#                        is a different problem from one that is slow because
#                        the card is idle waiting on the agent's tool calls.
#   server_timeline.csv  vLLM's own counters off its Prometheus endpoint:
#                        requests running and queued, KV-cache occupancy, and
#                        generation tokens per second derived from the token
#                        counter between ticks. A queue that grows while
#                        throughput is flat means the KV cache is the ceiling,
#                        which is a GPU_UTIL or MAX_MODEL_LEN decision.
#
# Both are plain CSV with a header, appended to, so a preempted-and-requeued
# job adds to the same timeline rather than starting a new one.

set -uo pipefail

INTERVAL="${MONITOR_INTERVAL:-15}"

ACTION="${1:-}"
RUN_DIR="${2:-}"
if [ -z "${ACTION}" ] || [ -z "${RUN_DIR}" ]; then
    echo "usage: monitor_run.sh start|stop|status <run-dir>" >&2
    exit 2
fi
RUN_DIR="${RUN_DIR%/}"
LOG_DIR="${RUN_DIR}/logs"
PID_FILE="${LOG_DIR}/monitor.pid"
GPU_CSV="${LOG_DIR}/gpu_timeline.csv"
SRV_CSV="${LOG_DIR}/server_timeline.csv"

# BASE_URL is the OpenAI-compatible path; the metrics endpoint sits beside it
# at the server root, not under /v1.
METRICS_URL="${MONITOR_METRICS_URL:-${BASE_URL:-http://localhost:8000/v1}}"
METRICS_URL="${METRICS_URL%/}"
METRICS_URL="${METRICS_URL%/v1}/metrics"

sample_gpu() {
    # One row per GPU per tick. nounits keeps the CSV numeric so anything can
    # read it without stripping " MiB" off every field.
    local stamp="$1"
    nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
        --format=csv,noheader,nounits 2>/dev/null \
        | sed "s/^/${stamp}, /" \
        | tr -d ' ' | sed 's/,/, /g' >> "${GPU_CSV}"
}

sample_server() {
    # vLLM publishes counters, not rates, so tokens/s is the delta between this
    # tick and the last one. State lives in these two globals rather than a
    # file: a restarted monitor just records one empty rate and carries on.
    local stamp="$1" body
    body="$(curl -sf --max-time 5 "${METRICS_URL}" 2>/dev/null)" || {
        echo "${stamp}, , , , , , unreachable" >> "${SRV_CSV}"
        return
    }

    local running waiting cache prompt gen
    running="$(metric "${body}" vllm:num_requests_running)"
    waiting="$(metric "${body}" vllm:num_requests_waiting)"
    cache="$(metric "${body}" vllm:gpu_cache_usage_perc)"
    prompt="$(metric "${body}" vllm:prompt_tokens_total)"
    gen="$(metric "${body}" vllm:generation_tokens_total)"

    local now rate=""
    now="$(date +%s)"
    if [ -n "${gen}" ] && [ -n "${LAST_GEN}" ] && [ -n "${LAST_TIME}" ]; then
        rate="$(awk -v a="${gen}" -v b="${LAST_GEN}" -v t="${now}" -v u="${LAST_TIME}" \
            'BEGIN { d = t - u; if (d > 0 && a >= b) printf "%.1f", (a - b) / d }')"
    fi
    LAST_GEN="${gen}"
    LAST_TIME="${now}"

    echo "${stamp}, ${running}, ${waiting}, ${cache}, ${gen}, ${rate}, ok" >> "${SRV_CSV}"
    # Printed, not just filed: this is the line that tells someone tailing the
    # job log that the model is generating rather than wedged.
    echo "[monitor ${stamp}] running=${running:-?} queued=${waiting:-?} kv=${cache:-?} tok/s=${rate:-?}"
}

metric() {
    # The last field of the first non-comment line whose name matches. vLLM
    # labels every series with model_name=..., so an exact-match grep would
    # never hit; anchoring on the name and a following { or space does.
    awk -v name="$2" '
        /^#/ { next }
        index($0, name) == 1 {
            c = substr($0, length(name) + 1, 1)
            if (c == "{" || c == " ") { print $NF; exit }
        }' <<< "$1"
}

case "${ACTION}" in
start)
    mkdir -p "${LOG_DIR}"
    if [ -f "${PID_FILE}" ] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
        echo "Monitor already running for ${RUN_DIR} (pid $(cat "${PID_FILE}"))."
        exit 0
    fi
    [ -s "${GPU_CSV}" ] || echo "timestamp, gpu, util_gpu_pct, util_mem_pct, mem_used_mib, mem_total_mib, power_w, temp_c" > "${GPU_CSV}"
    [ -s "${SRV_CSV}" ] || echo "timestamp, requests_running, requests_waiting, kv_cache_usage, generation_tokens_total, generation_tok_per_s, status" > "${SRV_CSV}"

    # The sampler itself, backgrounded and disowned so it outlives the shell
    # that started it but still dies on `stop` or on the job's exit trap.
    (
        LAST_GEN=""
        LAST_TIME=""
        while :; do
            stamp="$(date '+%Y-%m-%dT%H:%M:%S')"
            command -v nvidia-smi >/dev/null 2>&1 && sample_gpu "${stamp}"
            sample_server "${stamp}"
            sleep "${INTERVAL}"
        done
    ) &
    echo $! > "${PID_FILE}"
    echo "Monitoring ${RUN_DIR} every ${INTERVAL}s (pid $(cat "${PID_FILE}")); metrics from ${METRICS_URL}"
    ;;
stop)
    if [ -f "${PID_FILE}" ]; then
        kill "$(cat "${PID_FILE}")" 2>/dev/null || true
        rm -f "${PID_FILE}"
        echo "Monitor stopped for ${RUN_DIR}."
    fi
    ;;
status)
    # Stage progress first: a stage that is still running has no output file
    # yet, so the listing of the run folder does not show it, but its start
    # time is already here.
    if [ -s "${LOG_DIR}/stages.jsonl" ]; then
        printf '\n--- stages ---\n'
        python3 - "${LOG_DIR}/stages.jsonl" <<'PYEOF'
import json, sys

for line in open(sys.argv[1]):
    try:
        record = json.loads(line)
    except ValueError:
        continue
    seconds = int(record.get("seconds") or 0)
    print(
        "{:<7} {:>3}m{:02d}s  {}".format(
            record.get("status", "?"),
            seconds // 60,
            seconds % 60,
            record.get("stage", "?"),
        )
    )
PYEOF
    fi
    for csv in "${GPU_CSV}" "${SRV_CSV}"; do
        if [ -s "${csv}" ]; then
            printf '\n--- %s ---\n' "$(basename "${csv}")"
            head -1 "${csv}"
            tail -n +2 "${csv}" | tail -3
        fi
    done
    ;;
*)
    echo "usage: monitor_run.sh start|stop|status <run-dir>" >&2
    exit 2
    ;;
esac
