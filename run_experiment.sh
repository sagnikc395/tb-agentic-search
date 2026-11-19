#!/bin/bash
# The whole experiment, one command. Everything it produces lands in
# results/<run-id>/, which is the folder to send on.
#
#   ./run_experiment.sh                 # from a Unity login node: submits the
#                                       # A100 job, which runs this again inside
#   ./run_experiment.sh                 # with a server already up: runs here
#
# Knobs, all optional:
#   DRUGS="rifampicin isoniazid"        which drugs to predict (default: four below)
#   LIMIT=20                            isolates per drug
#   BALANCE=1                           equal R and S isolates (default on)
#   SEED=0                              sampling seed, so a run repeats
#   EFFORT=high                         reasoning effort asked of the model
#   VCF_ROOT=/path/to/vcfs              if the VCFs are not where the cohort file says
#   BASE_URL=... MODEL=...              point at a server yourself
#   RUN_ID=my-run                       name the results folder
#
# The three stages: the cohort join (no model involved, so it is the cheap
# check that the data is readable), resistance prediction per drug, and the
# literature screening against the hand labels in eval/. Each stage writes its
# JSON and its full log into the run folder; a stage that fails is recorded and
# the rest still run, because a four-drug run is too expensive to throw away
# over one bad drug.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO}"

DRUGS="${DRUGS:-rifampicin isoniazid ethambutol pyrazinamide}"
LIMIT="${LIMIT:-20}"
BALANCE="${BALANCE:-1}"
SEED="${SEED:-0}"
EFFORT="${EFFORT:-high}"
BASE_URL="${BASE_URL:-${VLLM_BASE_URL:-http://localhost:8000/v1}}"
MODEL="${MODEL:-${VLLM_MODEL:-Qwen/Qwen3-32B-AWQ}}"
PAPERS="${PAPERS:-papers}"
GOLD="${GOLD:-eval/screening_gold.json}"

# No server answering: on a login node that means the job has not been
# submitted yet, so submit it. The job runs this same script with the server up.
if ! curl -sf --max-time 10 "${BASE_URL}/models" >/dev/null 2>&1; then
    if [ -n "${SLURM_JOB_ID:-}" ]; then
        echo "No server at ${BASE_URL} inside job ${SLURM_JOB_ID}." >&2
        exit 1
    fi
    if ! command -v sbatch >/dev/null 2>&1; then
        echo "Nothing is serving ${MODEL} at ${BASE_URL}, and there is no sbatch" >&2
        echo "here to start one. Run this on Unity, or start a server yourself." >&2
        exit 1
    fi
    echo "No server at ${BASE_URL}; submitting the A100 job to run this there."
    mkdir -p logs
    # shellcheck disable=SC2086
    if ! sbatch ${SBATCH_ARGS:-} "${REPO}/scripts/run_unity_a100.sbatch" experiment; then
        # "Requested node configuration is not available" means nothing on the
        # cluster matches the resources asked for, so print what does and let
        # the reader pick. SBATCH_ARGS overrides the job's own directives.
        echo >&2
        echo "Submission failed. Partitions that hold A100s right now:" >&2
        sinfo -h -o '%20P %10l %8D %6t %30G' 2>/dev/null \
            | awk '/gpu:a100/' >&2 || true
        echo >&2
        echo "Re-submit against one of those:" >&2
        echo '  SBATCH_ARGS="--partition=<partition>" ./run_experiment.sh' >&2
        exit 1
    fi
    echo "Watch it with:  tail -f logs/tb-a100-<jobid>.out"
    exit 0
fi

RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)-${SLURM_JOB_ID:-local}}"
RUN_DIR="${REPO}/results/${RUN_ID}"
mkdir -p "${RUN_DIR}/logs"
echo "Writing everything to ${RUN_DIR}"

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
    "model": "${MODEL}",
    "base_url": "${BASE_URL}",
    "reasoning_effort": "${EFFORT}",
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

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,driver_version \
        --format=csv > "${RUN_DIR}/logs/gpu.txt" 2>&1 || true
fi

# Stage results are collected here rather than failing the run, so one bad drug
# does not cost the others.
STATUS_FILE="${RUN_DIR}/logs/stages.txt"
: > "${STATUS_FILE}"

stage() {
    # stage <name> <logfile> <command...>
    local name="$1" log="$2"
    shift 2
    echo
    echo "=== ${name} ==="
    if "$@" 2>&1 | tee "${RUN_DIR}/logs/${log}"; then
        echo "${name}	ok" >> "${STATUS_FILE}"
    else
        echo "${name}	FAILED" >> "${STATUS_FILE}"
        echo "!! ${name} failed; see logs/${log}. Continuing."
    fi
}

COMMON=(--base-url "${BASE_URL}" --model "${MODEL}" --source Custom)

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

# 4. Turn the run folder into something readable.
"${REPO}/scripts/collect_results.sh" "${RUN_DIR}"

# Summarise first, then fail: a partial run is still worth packaging, but the
# job should not come back green when a stage did not finish.
if grep -q "FAILED" "${STATUS_FILE}"; then
    echo
    echo "Stages that failed:"
    grep "FAILED" "${STATUS_FILE}" | sed 's/^/  /'
    exit 1
fi
