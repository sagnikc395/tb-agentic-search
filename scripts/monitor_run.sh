#!/bin/bash
# Sample a run while it is happening, so every run leaves a record of what Groq
# was doing for it and not just what the pipeline printed.
#
#   scripts/monitor_run.sh start results/<run-id>   # background sampler
#   scripts/monitor_run.sh stop  results/<run-id>   # stop it
#   scripts/monitor_run.sh status results/<run-id>  # last sample, human-readable
#
# run_experiment.sh starts and stops it around the stages, so there is nothing
# to run by hand for a normal run. Started twice on the same folder the second
# call is a no-op, which is what makes it safe to call from a requeued job.
#
# On a model we served ourselves this sampled the card and vLLM's own metrics
# endpoint. Groq publishes neither to us: the only record of what a request cost
# is the response we already received, so tb_search.usage files each one into
# <run-id>/logs/groq_usage.jsonl and this script rolls that up per tick. One
# file lands in <run-id>/logs/:
#
#   groq_timeline.csv    per tick: requests completed, tokens in and out, how
#                        much of the output was thinking, mean queue time, and
#                        the seconds the run spent waiting out a rate limit.
#                        The question it answers is the one a stage log cannot:
#                        a run that is slow because the model is thinking hard
#                        (completion tokens up, queue time flat) is a different
#                        problem from one that is slow because the account is
#                        throttled (queue time and wait seconds up), and the
#                        second is fixed by a smaller batch or a higher tier
#                        rather than by a lower effort.
#
# It is plain CSV with a header, appended to, so a requeued job adds to the same
# timeline rather than starting a new one.

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
GROQ_CSV="${LOG_DIR}/groq_timeline.csv"
# Where tb_search.usage writes; run_experiment.sh exports the same path.
USAGE_LOG="${TB_SEARCH_USAGE_LOG:-${LOG_DIR}/groq_usage.jsonl}"

sample_groq() {
    # One row per tick, covering the requests that finished since the last one.
    # The reader is python because the source is JSONL, and it has to tolerate
    # a last line that is still being written.
    local stamp="$1"
    python3 - "${USAGE_LOG}" "${GROQ_CSV}" "${stamp}" "${LOG_DIR}/monitor.offset" <<'PYEOF'
import json, pathlib, sys

usage, csv_path, offset_path = (pathlib.Path(p) for p in (sys.argv[1], sys.argv[2], sys.argv[4]))
stamp = sys.argv[3]

# Only the lines that are new since the last tick, so every row is a rate over
# that tick rather than a running total. The offset is a byte count, which is
# also how a half-written final line is left for the next tick: it is not
# counted as consumed.
start = 0
if offset_path.exists():
    try:
        start = int(offset_path.read_text().strip() or 0)
    except ValueError:
        start = 0

if not usage.exists():
    sys.exit(0)

with open(usage, "rb") as handle:
    handle.seek(start)
    chunk = handle.read()

complete, _, _ = chunk.rpartition(b"\n")
if complete:
    offset_path.write_text(str(start + len(complete) + 1))

events = []
for line in complete.decode("utf-8", "replace").splitlines():
    try:
        events.append(json.loads(line))
    except ValueError:
        continue

requests = [e for e in events if e.get("event") == "completion"]
waits = [e for e in events if e.get("event") == "wait"]


def total(key, rows):
    return sum(row.get(key) or 0 for row in rows)


def mean(key, rows):
    values = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
    return sum(values) / len(values) if values else ""


row = [
    stamp,
    str(len(requests)),
    str(total("prompt_tokens", requests)),
    str(total("completion_tokens", requests)),
    str(total("reasoning_tokens", requests)),
    f"{mean('queue_time', requests):.2f}" if requests else "",
    f"{mean('total_time', requests):.2f}" if requests else "",
    f"{total('seconds', waits):.0f}",
]
with open(csv_path, "a") as handle:
    handle.write(", ".join(row) + "\n")

# Printed, not just filed: this is the line that tells someone tailing the job
# log that the run is making requests rather than wedged.
print(
    f"[monitor {stamp}] requests={len(requests)} "
    f"in={total('prompt_tokens', requests)} out={total('completion_tokens', requests)} "
    f"thinking={total('reasoning_tokens', requests)} "
    f"queue={mean('queue_time', requests) or 0:.2f}s "
    f"waited={total('seconds', waits):.0f}s"
)
PYEOF
}

case "${ACTION}" in
start)
    mkdir -p "${LOG_DIR}"
    if [ -f "${PID_FILE}" ] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
        echo "Monitor already running for ${RUN_DIR} (pid $(cat "${PID_FILE}"))."
        exit 0
    fi
    [ -s "${GROQ_CSV}" ] || echo "timestamp, requests, prompt_tokens, completion_tokens, reasoning_tokens, mean_queue_s, mean_total_s, rate_limit_wait_s" > "${GROQ_CSV}"

    # The sampler itself, backgrounded and disowned so it outlives the shell
    # that started it but still dies on `stop` or on the job's exit trap.
    (
        while :; do
            stamp="$(date '+%Y-%m-%dT%H:%M:%S')"
            sample_groq "${stamp}"
            sleep "${INTERVAL}"
        done
    ) &
    echo $! > "${PID_FILE}"
    echo "Monitoring ${RUN_DIR} every ${INTERVAL}s (pid $(cat "${PID_FILE}")); usage from ${USAGE_LOG}"
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
    if [ -s "${GROQ_CSV}" ]; then
        printf '\n--- %s ---\n' "$(basename "${GROQ_CSV}")"
        head -1 "${GROQ_CSV}"
        tail -n +2 "${GROQ_CSV}" | tail -3
    fi
    ;;
*)
    echo "usage: monitor_run.sh start|stop|status <run-dir>" >&2
    exit 2
    ;;
esac
