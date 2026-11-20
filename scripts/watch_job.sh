#!/bin/bash
# Watch a running job from a Unity login node: queue state, what the run has
# cost at Groq so far, and the tail of the log, refreshed in place.
#
#   scripts/watch_job.sh              # the newest job of yours still running
#   scripts/watch_job.sh 1234567      # a specific job id
#   INTERVAL=5 scripts/watch_job.sh   # refresh faster (default 15s)
#
# It keeps running after the job ends, printing the sacct line (exit code,
# whether it was preempted, how much of the walltime it used) so a job that
# dies while you are away still leaves its reason on screen.

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INTERVAL="${INTERVAL:-15}"
JOB="${1:-}"

# No id given: the most recent job in the queue, whatever its state, so this
# also attaches to one still sitting in PENDING.
if [ -z "${JOB}" ]; then
    JOB="$(squeue -h -u "${USER}" -o '%i' --sort=-V | head -1)"
fi
if [ -z "${JOB}" ]; then
    echo "No jobs in the queue for ${USER}. Pass a job id to look at a finished one." >&2
    exit 1
fi

LOG=""
for candidate in "${REPO}/logs/tb-quick-${JOB}.out" "${REPO}/logs/tb-groq-${JOB}.out"; do
    [ -f "${candidate}" ] && LOG="${candidate}" && break
done

while :; do
    clear
    printf '=== job %s  %s ===\n\n' "${JOB}" "$(date '+%H:%M:%S')"

    # STATE/REASON is the line that matters while pending: Priority means
    # waiting your turn, Resources means nothing free matches the request.
    squeue -j "${JOB}" -o '%.10i %.9P %.10T %.10M %.10l %.6D %R' 2>/dev/null

    STATE="$(squeue -h -j "${JOB}" -o '%T' 2>/dev/null)"
    NODE="$(squeue -h -j "${JOB}" -o '%N' 2>/dev/null)"

    if [ -z "${STATE}" ]; then
        printf '\nNot in the queue any more:\n'
        sacct -j "${JOB}" --format=JobID%14,State%20,ExitCode,Elapsed,MaxRSS -X 2>/dev/null
    fi

    # The log only exists once the job starts, so find it late if we have to.
    if [ -z "${LOG}" ]; then
        for candidate in "${REPO}/logs/tb-quick-${JOB}.out" "${REPO}/logs/tb-groq-${JOB}.out"; do
            [ -f "${candidate}" ] && LOG="${candidate}" && break
        done
    fi

    # Results appear stage by stage, so this is the honest progress bar.
    RUN_DIR="$(ls -dt "${REPO}"/results/*"${JOB}"* 2>/dev/null | head -1)"
    if [ -n "${RUN_DIR}" ]; then
        printf '\n--- %s ---\n' "${RUN_DIR#${REPO}/}"

        # Stages done and how long each took, plus the sampler's last few
        # ticks: requests, tokens and rate-limit waits as of seconds ago. There
        # is no card to ask about any more — the run's cost is the measurement.
        "${REPO}/scripts/monitor_run.sh" status "${RUN_DIR}" 2>/dev/null \
            | sed 's/^/  /'

        ls -lt "${RUN_DIR}" | tail -n +2 | head -5
    fi

    if [ -n "${LOG}" ]; then
        printf '\n--- %s ---\n' "${LOG#${REPO}/}"
        tail -20 "${LOG}"
    else
        printf '\n(no log yet — the job has not started)\n'
    fi

    sleep "${INTERVAL}"
done
