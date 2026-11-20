#!/bin/bash
# One command, start to finish: submit the job, wait for it, show what it is
# doing, and print where the results landed.
#
#   ./run.sh                    # the quick job: one drug, 8 isolates, ~1h cap
#   ./run.sh full               # the full four-drug run
#   ./run.sh local              # don't submit anything; run it right here
#
# Then wait. It prints the queue position while the job is pending, follows
# the log once it starts, and ends with the run's summary and the tarball to
# send on. Ctrl-C detaches and leaves the job running — it does not cancel it;
# the message on the way out says how to come back.
#
# Everything run_experiment.sh takes still works, because it is passed through
# the environment:
#
#   DRUGS="rifampicin isoniazid" LIMIT=20 ./run.sh full
#   QUICK_LIMIT=4 ./run.sh                      # smaller quick run
#   MODEL=openai/gpt-oss-20b ./run.sh full
#   SBATCH_ARGS="--partition=cpu-preempt" ./run.sh
#
# Inference is Groq's, so the jobs these submit hold no model and ask for no
# GPU — they are long-running clients, and the queue for CPUs is short. Off a
# cluster there is nothing to submit to and the pipeline runs here.

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO}"

MODE="${1:-quick}"
POLL="${POLL:-20}"
ATTACH_JOB=""

case "${MODE}" in
quick) SCRIPT="scripts/run_unity_quick.sbatch"; PREFIX="tb-quick" ;;
full)  SCRIPT="scripts/run_unity_cpu.sbatch";   PREFIX="tb-groq"  ;;
# Nothing to submit: run the stages in this shell. The same thing happens
# anyway where there is no sbatch.
local) SCRIPT=""; PREFIX=""; export LOCAL=1 ;;
# Re-attach to a job that is already in the queue, which is what Ctrl-C tells
# you to run. Same waiting, following and summary as a fresh submit, minus the
# submit; with no id, the newest job of yours.
--attach)
    ATTACH_JOB="${2:-$(squeue -h -u "${USER}" -o '%i' --sort=-V 2>/dev/null | head -1)}"
    [ -n "${ATTACH_JOB}" ] || { echo "No job to attach to." >&2; exit 1; }
    SCRIPT=""; PREFIX=""
    ;;
*)     echo "usage: ./run.sh [quick|full|local|--attach [jobid]]" >&2; exit 2 ;;
esac

mkdir -p logs results

# --- preflight -------------------------------------------------------------
# The three things that make a job die ten minutes in for a reason that was
# visible before it was ever submitted. Checked here, where the message lands
# on your terminal instead of at the bottom of a log you have to go find.
fail=0
for path in cli.py run_experiment.sh ${SCRIPT:+"${SCRIPT}"} \
            data/cleaned_vcf_paths/master_resistance_table.csv; do
    [ -e "${path}" ] || { echo "Missing ${path}" >&2; fail=1; }
done
command -v uv >/dev/null 2>&1 || {
    echo "uv is not on PATH; the pipeline stages run under it." >&2
    echo "On Unity: module load uv, or install it into your home." >&2
    fail=1
}
# The key is the one thing with no default, and a job that does not have it
# dies on its first Groq call — twenty minutes of queue later. run_experiment.sh
# reads .env itself, so either is enough.
if [ -z "${GROQ_API_KEY:-}" ] && ! grep -qs '^[[:space:]]*GROQ_API_KEY[[:space:]]*=' .env; then
    echo "No GROQ_API_KEY, in the environment or in .env." >&2
    echo "  export GROQ_API_KEY=gsk_...   (keys: https://console.groq.com/keys)" >&2
    fail=1
fi
[ "${fail}" -eq 0 ] || exit 1

# --- nothing to submit to: run it here ------------------------------------
if [ "${MODE}" = "local" ] || ! command -v sbatch >/dev/null 2>&1; then
    [ "${MODE}" = "local" ] || echo "No sbatch here, so running the pipeline in this shell."
    exec ./run_experiment.sh
fi

# --- attach, or submit -----------------------------------------------------
if [ -n "${ATTACH_JOB}" ]; then
    JOB="${ATTACH_JOB}"
    LOG=""
    for candidate in "logs/tb-quick-${JOB}.out" "logs/tb-groq-${JOB}.out"; do
        [ -f "${candidate}" ] && LOG="${candidate}" && break
    done
    # A job still pending has written no log yet; guess by the job's name so
    # tail -F has the right path to wait on.
    if [ -z "${LOG}" ]; then
        NAME="$(squeue -h -j "${JOB}" -o '%j' 2>/dev/null)"
        LOG="logs/${NAME:-tb-quick}-${JOB}.out"
    fi
    echo "Attaching to job ${JOB}; log ${LOG}"
    echo
else

echo "Submitting ${SCRIPT}..."
# sbatch writes advisory notes ("INFO: Note that 32 GB per node will...") to
# stderr, so stderr is kept out of the id and parked in a file for the error
# path. Folding it in with 2>&1 makes JOB the whole notice plus the number,
# and every squeue, log path and results glob after this point misses.
SBATCH_ERR="$(mktemp)"
# shellcheck disable=SC2086
JOB="$(sbatch --parsable ${SBATCH_ARGS:-} "${SCRIPT}" 2>"${SBATCH_ERR}")" || {
    echo "Submission failed: $(cat "${SBATCH_ERR}")" >&2
    rm -f "${SBATCH_ERR}"
    echo >&2
    echo "If that says the partition does not exist, name one that does — these" >&2
    echo "jobs need CPUs only. What exists right now:" >&2
    sinfo -h -o '%20P %10l %8D %6t' 2>/dev/null | head -20 >&2
    echo >&2
    echo '  SBATCH_ARGS="--partition=<partition>" ./run.sh '"${MODE}" >&2
    exit 1
}
# Show the notes, since they are worth reading, then reduce --parsable's
# "jobid" or "jobid;cluster" to the bare number the rest of the script uses.
[ -s "${SBATCH_ERR}" ] && sed 's/^/  /' "${SBATCH_ERR}" >&2
rm -f "${SBATCH_ERR}"
JOB="${JOB%%;*}"
JOB="$(printf '%s' "${JOB}" | tr -cd '0-9')"
[ -n "${JOB}" ] || { echo "sbatch returned no job id." >&2; exit 1; }
LOG="logs/${PREFIX}-${JOB}.out"
echo "Job ${JOB} submitted; its log will be ${LOG}"
echo

fi

# Detaching must not take the job with it, so the trap says so rather than
# cleaning anything up. scancel is the deliberate act, never the accident.
trap 'echo; echo "Detached. Job ${JOB} is still running."; \
      echo "  Come back with:  ./run.sh --attach ${JOB}   (or scripts/watch_job.sh ${JOB})"; \
      echo "  Stop it with:    scancel ${JOB}"; exit 0' INT

# --- wait for it to start --------------------------------------------------
# Pending is the normal state, not an error: Priority means waiting your turn,
# Resources means nothing free matches. Both are printed as they change rather
# than every tick, so a two-hour wait does not fill the screen.
LAST=""
while :; do
    STATE="$(squeue -h -j "${JOB}" -o '%T' 2>/dev/null)"
    [ -z "${STATE}" ] && break                      # gone from the queue
    [ "${STATE}" = "RUNNING" ] && break
    REASON="$(squeue -h -j "${JOB}" -o '%r' 2>/dev/null)"
    if [ "${STATE} ${REASON}" != "${LAST}" ]; then
        printf '%s  %s (%s)\n' "$(date '+%H:%M:%S')" "${STATE}" "${REASON}"
        LAST="${STATE} ${REASON}"
    fi
    sleep "${POLL}"
done

NODE="$(squeue -h -j "${JOB}" -o '%N' 2>/dev/null)"
[ -n "${NODE}" ] && echo "$(date '+%H:%M:%S')  RUNNING on ${NODE}"
echo

# --- follow it -------------------------------------------------------------
# -F rather than -f: the file does not exist yet at this point, and on a
# requeue Slurm replaces it. The follower is backgrounded so this shell can
# watch the queue at the same time and stop cleanly when the job ends.
tail -n +1 -F "${LOG}" 2>/dev/null &
TAIL=$!
trap 'kill "${TAIL}" 2>/dev/null; wait "${TAIL}" 2>/dev/null; \
      echo; echo "Detached. Job ${JOB} is still running."; \
      echo "  Come back with:  ./run.sh --attach ${JOB}"; \
      echo "  Stop it with:    scancel ${JOB}"; exit 0' INT

while squeue -h -j "${JOB}" -o '%T' 2>/dev/null | grep -q .; do
    sleep "${POLL}"
done
sleep 3                      # let the last of the log flush through tail
# Reaped with wait, and both redirected: killing a background job otherwise
# leaves bash printing "Terminated" across the summary that follows.
kill "${TAIL}" 2>/dev/null
wait "${TAIL}" 2>/dev/null
trap - INT

# --- what happened ---------------------------------------------------------
echo
echo "=== job ${JOB} finished ==="
sacct -j "${JOB}" --format=JobID%14,State%20,ExitCode,Elapsed,MaxRSS -X 2>/dev/null

RUN_DIR="$(ls -dt results/*"${JOB}"* 2>/dev/null | grep -v '\.tar\.gz$' | head -1)"
if [ -z "${RUN_DIR}" ]; then
    echo
    echo "No results folder for this job. The job died before the pipeline" >&2
    echo "started — the end of ${LOG} says why:" >&2
    # Not `2>/dev/null >&2`: that dups stdout onto the already-redirected
    # stderr, which is /dev/null, and the log tail disappears.
    [ -f "${LOG}" ] && tail -20 "${LOG}" >&2
    exit 1
fi

echo
echo "=== ${RUN_DIR}/SUMMARY.md ==="
if [ -f "${RUN_DIR}/SUMMARY.md" ]; then
    cat "${RUN_DIR}/SUMMARY.md"
else
    # Interrupted before collect_results.sh ran — salvage what did finish, so
    # a preempted run still ends with a readable folder.
    echo "(no summary; the run did not reach the end — building one now)"
    scripts/collect_results.sh "${RUN_DIR}" || true
    [ -f "${RUN_DIR}/SUMMARY.md" ] && cat "${RUN_DIR}/SUMMARY.md"
fi

echo
echo "Folder:   ${RUN_DIR}"
[ -f "${RUN_DIR}.tar.gz" ] && echo "Tarball:  ${RUN_DIR}.tar.gz"
echo "Ledger:   results/runs.csv (one row per run)"
grep -q "FAILED" "${RUN_DIR}/logs/stages.txt" 2>/dev/null && {
    echo
    echo "Some stages failed:"
    grep "FAILED" "${RUN_DIR}/logs/stages.txt" | sed 's/^/  /'
    exit 1
}
exit 0
