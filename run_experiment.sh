#!/bin/bash
# The whole experiment, one command. Everything it produces lands in
# results/<run-id>/, which is the folder to send on.
#
#   ./run_experiment.sh                 # runs here, against Groq
#   LOCAL=1 ./run_experiment.sh         # ... even on a cluster login node
#
# Inference is Groq's hosted endpoint, so there is no server to start and no
# GPU to wait for: all this needs is GROQ_API_KEY in the environment or in
# .env. On a Unity login node it still submits itself as a batch job, because a
# four-drug sweep takes hours of wall time and a login node is the wrong place
# to spend them — but the job asks for CPUs only.
#
# Knobs, all optional:
#   DRUGS="rifampicin isoniazid"        which drugs to predict (default: four below)
#   LIMIT=20                            isolates per drug
#   BALANCE=1                           equal R and S isolates (default on)
#   SEED=0                              sampling seed, so a run repeats
#   EFFORT=high                         reasoning effort asked of the model
#   MODEL=openai/gpt-oss-20b            which Groq model answers
#   MAX_RETRIES=5                       how long to wait out a 429
#   TOOLS=1                             let A1 retrieve a tool subset instead of
#                                       carrying the whole catalogue: ~36k tokens
#                                       per request down to ~11k, which is what
#                                       decides whether a low tier fits at all
#   VCF_ROOT=/path/to/vcfs              if the VCFs are not where the cohort file says
#   RUN_ID=my-run                       name the results folder
#
# The three stages: the cohort join (no model involved, so it is the cheap
# check that the data is readable), resistance prediction per drug, and the
# literature screening against the hand labels in eval/. Each stage writes its
# JSON and its full log into the run folder; a stage that fails is recorded and
# the rest still run, because a four-drug run is too expensive in tokens and
# wall time to throw away over one bad drug.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO}"

DRUGS="${DRUGS:-rifampicin isoniazid ethambutol pyrazinamide}"
LIMIT="${LIMIT:-20}"
BALANCE="${BALANCE:-1}"
SEED="${SEED:-0}"
EFFORT="${EFFORT:-high}"
BASE_URL="${BASE_URL:-${GROQ_BASE_URL:-https://api.groq.com/openai/v1}}"
MODEL="${MODEL:-${GROQ_MODEL:-openai/gpt-oss-120b}}"
MAX_RETRIES="${MAX_RETRIES:-${GROQ_MAX_RETRIES:-5}}"
PAPERS="${PAPERS:-papers}"
GOLD="${GOLD:-eval/screening_gold.json}"

# The key is the one thing with no default, and it is worth reading out of .env
# here as well: the stages do it themselves, but this script pings Groq before
# any of them start.
if [ -z "${GROQ_API_KEY:-}" ] && [ -f "${REPO}/.env" ]; then
    GROQ_API_KEY="$(sed -n 's/^[[:space:]]*GROQ_API_KEY[[:space:]]*=[[:space:]]*//p' \
        "${REPO}/.env" | tail -1 | tr -d '"'\''' | tr -d '[:space:]')"
    export GROQ_API_KEY
fi
if [ -z "${GROQ_API_KEY:-}" ]; then
    echo "No GROQ_API_KEY. Every stage but the cohort join calls Groq." >&2
    echo "  export GROQ_API_KEY=gsk_...   (or put it in .env; see .env.example)" >&2
    echo "Keys: https://console.groq.com/keys" >&2
    exit 1
fi

# One call before anything is submitted or written: a dead key fails here, in
# two seconds, instead of four stages deep.
if ! curl -sf --max-time 20 -H "Authorization: Bearer ${GROQ_API_KEY}" \
        "${BASE_URL}/models" >/dev/null 2>&1; then
    echo "Groq did not answer at ${BASE_URL}/models with this key." >&2
    echo "Check the key is live (https://console.groq.com/keys) and that this" >&2
    echo "machine can reach api.groq.com." >&2
    exit 1
fi

# On a login node, submit instead of running: the sweep is long, and nothing
# about it needs a GPU any more. LOCAL=1 overrides, and inside a job this is
# already the submitted run.
if [ -z "${SLURM_JOB_ID:-}" ] && [ -z "${LOCAL:-}" ] \
   && command -v sbatch >/dev/null 2>&1; then
    echo "Submitting scripts/run_unity_cpu.sbatch to run this in a batch job."
    echo "  (LOCAL=1 ./run_experiment.sh runs it here instead.)"
    mkdir -p logs
    # shellcheck disable=SC2086
    if ! sbatch ${SBATCH_ARGS:-} "${REPO}/scripts/run_unity_cpu.sbatch" experiment; then
        echo >&2
        echo "Submission failed. CPU partitions available right now:" >&2
        sinfo -h -o '%20P %10l %8D %6t' 2>/dev/null | head -10 >&2
        echo >&2
        echo "Re-submit against one of those, or run here:" >&2
        echo '  SBATCH_ARGS="--partition=<partition>" ./run_experiment.sh' >&2
        echo '  LOCAL=1 ./run_experiment.sh' >&2
        exit 1
    fi
    echo "Watch it with:  tail -f logs/tb-groq-<jobid>.out"
    exit 0
fi

RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)-${SLURM_JOB_ID:-local}}"
RUN_DIR="${REPO}/results/${RUN_ID}"
mkdir -p "${RUN_DIR}/logs"
echo "Writing everything to ${RUN_DIR}"

# Where tb_search.usage files what every request cost. This is the Groq-side
# replacement for the GPU timeline: tokens per request, how much of the output
# was thinking, how long each request sat in Groq's queue, and every second the
# run spent waiting out a 429.
export TB_SEARCH_USAGE_LOG="${RUN_DIR}/logs/groq_usage.jsonl"
: > "${TB_SEARCH_USAGE_LOG}"

# What was run, on what, against which model — the context a result is
# meaningless without when someone reads it weeks later.
python3 - "${RUN_DIR}/run_metadata.json" <<PY
import json, subprocess, sys, platform, datetime

def git(*args):
    try:
        return subprocess.check_output(("git", *args), text=True).strip()
    except Exception:
        return None

json.dump({
    "run_id": "${RUN_ID}",
    "started_at": datetime.datetime.now().isoformat(timespec="seconds"),
    "provider": "Groq",
    "model": "${MODEL}",
    "base_url": "${BASE_URL}",
    "reasoning_effort": "${EFFORT}",
    "max_retries": int("${MAX_RETRIES}"),
    "drugs": "${DRUGS}".split(),
    "isolates_per_drug": int("${LIMIT}"),
    "balanced": "${BALANCE}" == "1",
    "seed": int("${SEED}"),
    "host": platform.node(),
    "slurm_job_id": "${SLURM_JOB_ID:-}" or None,
    "git_commit": git("rev-parse", "HEAD"),
    "git_dirty": bool(git("status", "--porcelain")),
}, open(sys.argv[1], "w"), indent=2)
PY

# Every run is sampled while it runs, not just summarised after it. The trap
# covers the failure paths too — a stage that dies, a preemption, a Ctrl-C on a
# login node — so no run leaves a sampler behind holding the folder open.
BASE_URL="${BASE_URL}" "${REPO}/scripts/monitor_run.sh" start "${RUN_DIR}" || true
trap '"${REPO}/scripts/monitor_run.sh" stop "${RUN_DIR}" >/dev/null 2>&1 || true' EXIT

# Stage results are collected here rather than failing the run, so one bad drug
# does not cost the others.
STATUS_FILE="${RUN_DIR}/logs/stages.txt"
: > "${STATUS_FILE}"
# stages.txt stays the two-column file everything already reads; the JSONL
# beside it is the one with the numbers you need when a run was slow rather
# than wrong — when each stage started, how long it took, what it exited with.
STAGE_LOG="${RUN_DIR}/logs/stages.jsonl"
: > "${STAGE_LOG}"

stage() {
    # stage <name> <logfile> <command...>
    local name="$1" log="$2"
    shift 2
    local started elapsed code status
    started="$(date +%s)"
    echo
    echo "=== ${name}  ($(date '+%H:%M:%S')) ==="
    # errexit off for exactly this pipeline: a failing stage is recorded and
    # the run continues, which is the whole point of collecting status here.
    # PIPESTATUS, not $?, because the last element is always tee succeeding —
    # and `|| true` cannot be used instead, since it overwrites PIPESTATUS.
    set +e
    "$@" 2>&1 | tee "${RUN_DIR}/logs/${log}"
    code="${PIPESTATUS[0]}"
    set -e
    elapsed=$(( $(date +%s) - started ))
    if [ "${code}" -eq 0 ]; then
        status="ok"
    else
        status="FAILED"
        echo "!! ${name} failed (exit ${code}) after ${elapsed}s; see logs/${log}. Continuing."
    fi
    echo "${name}	${status}" >> "${STATUS_FILE}"
    printf '{"stage": "%s", "status": "%s", "exit_code": %s, "started_at": "%s", "seconds": %s, "log": "%s"}\n' \
        "${name}" "${status}" "${code}" \
        "$(date -d "@${started}" '+%Y-%m-%dT%H:%M:%S' 2>/dev/null \
            || date -r "${started}" '+%Y-%m-%dT%H:%M:%S' 2>/dev/null)" \
        "${elapsed}" "${log}" >> "${STAGE_LOG}"
    echo "--- ${name}: ${status} in ${elapsed}s ---"
    "${REPO}/scripts/monitor_run.sh" status "${RUN_DIR}" 2>/dev/null || true
}

COMMON=(--model "${MODEL}" --base-url "${BASE_URL}" --max-retries "${MAX_RETRIES}")
# Off by default, because it changes what the agent can reach and so what the
# run measures — but it is the only real lever on the per-request token cost.
[ "${TOOLS:-0}" = "1" ] && COMMON+=(--tools)

# 1. The cohort join: no LLM calls, so it fails fast if the data is not here.
stage "cohort" "cohort.log" \
    uv run cli.py --cohort -o "${RUN_DIR}/tb_cohort.json"

# 2. Resistance prediction, one run per drug.
PREDICT_FLAGS=(--limit "${LIMIT}" --seed "${SEED}" --reasoning-effort "${EFFORT}")
[ "${BALANCE}" = "1" ] && PREDICT_FLAGS+=(--balance)
[ -n "${VCF_ROOT:-}" ] && PREDICT_FLAGS+=(--vcf-root "${VCF_ROOT}")

for drug in ${DRUGS}; do
    stage "predict ${drug}" "predict-${drug}.log" \
        uv run cli.py --predict "${drug}" "${PREDICT_FLAGS[@]}" "${COMMON[@]}" \
        -o "${RUN_DIR}/tb_resistance_${drug}.json"
done

# 3. Literature screening, scored against the hand labels when they are there.
if [ -d "${PAPERS}" ]; then
    SCREEN_FLAGS=()
    [ -f "${GOLD}" ] && SCREEN_FLAGS+=(--gold "${GOLD}")
    stage "screening" "screening.log" \
        uv run cli.py "${PAPERS}" "${SCREEN_FLAGS[@]}" "${COMMON[@]}" \
        -o "${RUN_DIR}/tb_verdicts.json"
else
    echo "No ${PAPERS}/ directory; skipping the screening stage."
    echo "screening	skipped" >> "${STATUS_FILE}"
fi

# 4. Turn the run folder into something readable. The sampler stops first, so
# collect_results.sh summarises a finished timeline rather than a moving one.
"${REPO}/scripts/monitor_run.sh" stop "${RUN_DIR}" || true
"${REPO}/scripts/collect_results.sh" "${RUN_DIR}"

# Summarise first, then fail: a partial run is still worth packaging, but the
# job should not come back green when a stage did not finish.
if grep -q "FAILED" "${STATUS_FILE}"; then
    echo
    echo "Stages that failed:"
    grep "FAILED" "${STATUS_FILE}" | sed 's/^/  /'
    exit 1
fi
