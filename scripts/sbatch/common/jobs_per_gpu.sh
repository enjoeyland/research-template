# JOBS_PER_GPU — how many runs share one GPU allocation (independent of profiles).
# MAX_GPUS     — max concurrent SLURM array tasks (= concurrent GPU allocations).
#                Appends %MAX_GPUS to --array (e.g. 0-14%4). Unset = no throttle.
#
#   JOBS_PER_GPU=1 ./scripts/isic2024_train.sh
#   JOBS_PER_GPU=3 PROFILE=gpu96 ./scripts/isic2024_train.sh
#   MAX_GPUS=4 ./scripts/dsprites_train.sh
#
# Source after SEEDS (or any item count) is known, before maybe_submit:
#   : "${JOBS_PER_GPU:=1}"
#   source .../jobs_per_gpu.sh
#   plan_array ${#SEEDS[@]}
#   source .../maybe_submit.sh

: "${JOBS_PER_GPU:=1}"

if ! [[ "${JOBS_PER_GPU}" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: JOBS_PER_GPU must be a positive integer (got: ${JOBS_PER_GPU})" >&2
  exit 1
fi

if [[ -n "${MAX_GPUS:-}" ]] && ! [[ "${MAX_GPUS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: MAX_GPUS must be a positive integer (got: ${MAX_GPUS})" >&2
  exit 1
fi

# If ARRAY is unset and more than one GPU allocation is needed, set ARRAY=0-(n_tasks-1)
# (optionally %MAX_GPUS for concurrent GPU cap).
plan_array() {
  local n_items="${1:?}"
  if [[ -n "${ARRAY:-}" ]]; then
    return 0
  fi
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    return 0
  fi
  if [[ -n "${LOCAL:-}" && "${LOCAL}" != "0" ]]; then
    return 0
  fi
  local n_tasks=$(( (n_items + JOBS_PER_GPU - 1) / JOBS_PER_GPU ))
  if (( n_tasks > 1 )); then
    ARRAY="0-$((n_tasks - 1))"
    if [[ -n "${MAX_GPUS:-}" ]]; then
      ARRAY="${ARRAY}%${MAX_GPUS}"
    fi
  fi
}

# Fill dst with items for this allocation (array slice, or all items for LOCAL / single task).
# Usage: items_this_task SRC_ARRAY_NAME DST_ARRAY_NAME
items_this_task() {
  local -n _src="$1"
  local -n _dst="$2"
  local n=${#_src[@]}
  _dst=()

  if [[ -n "${LOCAL:-}" && "${LOCAL}" != "0" ]]; then
    _dst=("${_src[@]}")
    return 0
  fi

  local tid=${SLURM_ARRAY_TASK_ID:-0}
  local start=$((tid * JOBS_PER_GPU))
  local i
  for ((i = start; i < start + JOBS_PER_GPU && i < n; i++)); do
    _dst+=("${_src[i]}")
  done
}

# Run fn once per item, at most JOBS_PER_GPU concurrent (waves).
# Logs: <log_dir>/<log_prefix>_<jobid>[_<array_task>]_run<item>.log
#   jobid = SLURM_ARRAY_JOB_ID or SLURM_JOB_ID or "local"
# Usage: run_parallel log_dir log_prefix fn item [item ...]
run_parallel() {
  local log_dir="${1:?}"
  local log_prefix="${2:?}"
  local fn="${3:?}"
  shift 3
  local items=("$@")
  if ((${#items[@]} == 0)); then
    echo "run_parallel: no items for this task (array id=${SLURM_ARRAY_TASK_ID:-none})" >&2
    return 0
  fi

  mkdir -p "${log_dir}"
  local job_id="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
  local name_mid="${job_id}"
  if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    name_mid="${job_id}_${SLURM_ARRAY_TASK_ID}"
  fi

  echo "JOBS_PER_GPU=${JOBS_PER_GPU} items=${items[*]} job=${name_mid}"

  local i=0
  local -a statuses=()
  while ((i < ${#items[@]})); do
    local -a pids=()
    local started=0
    while ((started < JOBS_PER_GPU && i < ${#items[@]})); do
      local item="${items[i]}"
      local log_file="${log_dir}/${log_prefix}_${name_mid}_run${item}.log"
      "${fn}" "${item}" > "${log_file}" 2>&1 &
      pids+=("$!")
      ((++i, ++started))
    done
    local pid
    for pid in "${pids[@]}"; do
      wait "${pid}"
      statuses+=("$?")
    done
  done

  echo "Exit codes: ${statuses[*]}"
  local s
  for s in "${statuses[@]}"; do
    if [[ "${s}" != "0" ]]; then
      return 1
    fi
  done
  return 0
}
