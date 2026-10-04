# Shared sweep launcher. Call after CONFIG + run_one() are defined:
#
#   source "${_SBATCH}/common/launch_sweep.sh"
#
# Expects: SIZES, JOB_NAME, PROFILE/JOBS_PER_GPU (optional), and function run_one.
# Optional: run_analyze() — if defined, login-node submit queues train then analyze with
#   --dependency=afterok:<train>; LOCAL=1 runs analyze after all train ids finish.
# Does: plan_array → maybe_submit → env → (analyze | run_ids_this_task → run_parallel run_one).

_LAUNCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck disable=SC1091
source "${_LAUNCH_DIR}/jobs_per_gpu.sh"
# shellcheck disable=SC1091
source "${_LAUNCH_DIR}/sweep.sh"

if ! declare -F run_one >/dev/null; then
  echo "error: define run_one() before sourcing launch_sweep.sh" >&2
  exit 1
fi

if declare -F run_analyze >/dev/null; then
  : "${SUBMIT_ANALYZE:=1}"
else
  : "${SUBMIT_ANALYZE:=0}"
fi

N_RUNS="$(product_size SIZES)"
echo "N_RUNS=${N_RUNS} DIM_ORDER=(${DIM_ORDER[*]:-}) SIZES=(${SIZES[*]}) SUBMIT_ANALYZE=${SUBMIT_ANALYZE}"

plan_array "${N_RUNS}"
# shellcheck disable=SC1091
source "${_LAUNCH_DIR}/maybe_submit.sh"
# shellcheck disable=SC1091
source "${_LAUNCH_DIR}/env.sh"

: "${TRAINER:=${HYDRA_TRAINER:-gpu}}"
: "${PHASE:=train}"

if [[ "${PHASE}" == "analyze" ]]; then
  if ! declare -F run_analyze >/dev/null; then
    echo "error: PHASE=analyze but run_analyze() is not defined" >&2
    exit 1
  fi
  echo "=== PHASE=analyze ==="
  run_analyze
  unset _LAUNCH_DIR
  exit 0
fi

IDS_THIS_JOB=()
run_ids_this_task "${N_RUNS}" IDS_THIS_JOB
_log_day="$(date +%Y-%m-%d)"
_run_log_dir="logs/slurm/${_log_day}"
run_parallel "${_run_log_dir}" "${JOB_NAME:-sweep}" run_one "${IDS_THIS_JOB[@]}"

# LOCAL path: no SLURM dependency — run analyze after this process finishes all train ids.
if [[ -n "${LOCAL:-}" && "${LOCAL}" != "0" ]] && declare -F run_analyze >/dev/null; then
  echo "=== LOCAL analyze (after train) ==="
  run_analyze
fi

unset _LAUNCH_DIR _log_day _run_log_dir
