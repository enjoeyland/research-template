# Hyperparameter sweep helpers (ImgEdit-style mixed-radix run_id decoding).
#
# Define axes in the run script, set DIM_ORDER (first = innermost / changes fastest),
# then map run_id -> indices:
#
#   EXPERIMENTS=(a b)
#   SEEDS=(0 1 2)
#   DIM_ORDER=(seed experiment)          # documentation / your decode switch
#   SIZES=(${#SEEDS[@]} ${#EXPERIMENTS[@]})  # same order as DIM_ORDER
#   N_RUNS=$(product_size SIZES)
#   decode_run_id "$run_id" DIM_ORDER SIZES IDX   # IDX[seed], IDX[experiment], ...


# Product of positive integers in array named by $1.
product_size() {
  local -n _sizes="$1"
  local n=1 s
  for s in "${_sizes[@]}"; do
    n=$((n * s))
  done
  echo "${n}"
}

# Decode run_id into associative IDX keyed by DIM_ORDER names.
# Usage: declare -A IDX=(); decode_run_id RUN_ID DIM_ORDER_NAME SIZES_NAME IDX_NAME
decode_run_id() {
  local run_id="${1:?}"
  local -n _order="$2"
  local -n _sizes="$3"
  local -n _idx="$4"
  local tmp="${run_id}"
  local i size dim
  _idx=()
  if ((${#_order[@]} != ${#_sizes[@]})); then
    echo "error: DIM_ORDER length (${#_order[@]}) != SIZES length (${#_sizes[@]})" >&2
    return 1
  fi
  for i in "${!_order[@]}"; do
    dim="${_order[i]}"
    size="${_sizes[i]}"
    _idx["${dim}"]="$((tmp % size))"
    tmp=$((tmp / size))
  done
}

# Run ids [0, n_runs) for this GPU allocation (all if LOCAL; else array slice).
# Usage: run_ids_this_task N_RUNS DST_ARRAY_NAME
run_ids_this_task() {
  local n_runs="${1:?}"
  local -n _dst="$2"
  _dst=()

  if [[ -n "${LOCAL:-}" && "${LOCAL}" != "0" ]]; then
    local i
    for ((i = 0; i < n_runs; i++)); do
      _dst+=("${i}")
    done
    return 0
  fi

  local tid=${SLURM_ARRAY_TASK_ID:-0}
  local start=$((tid * JOBS_PER_GPU))
  local i
  for ((i = start; i < start + JOBS_PER_GPU && i < n_runs; i++)); do
    _dst+=("${i}")
  done
}

# Per-group array-task-id ranges for one DIM_ORDER dimension (2026-09-08, added so a run
# script's login-node submit can give each group its OWN analyze job with a
# --dependency=afterok:<train_job>_<i>:<train_job>_<i+1>:... on just ITS array task ids,
# instead of one analyze job waiting on the WHOLE train array -- see maybe_submit.sh's
# ANALYZE_GROUPS handling).
#
# `dim` MUST be the OUTERMOST (last) entry in DIM_ORDER: decode_run_id's mixed-radix scheme
# only gives an outermost dimension's values contiguous run_id blocks -- any inner dimension's
# values interleave across the whole run_id space and can't be expressed as a single
# start:end range per value.
#
# Usage: group_ranges_for_dim DIM_ORDER_NAME SIZES_NAME dim JOBS_PER_GPU DST_ARRAY_NAME
# DST_ARRAY_NAME is filled with one "start:end" (both inclusive, array-task-id space) per value
# of `dim`, in that dimension's index order (0, 1, 2, ...).
group_ranges_for_dim() {
  local -n _order="$1"
  local -n _sizes="$2"
  local dim="${3:?}"
  local jobs_per_gpu="${4:?}"
  local -n _dst="$5"
  _dst=()

  local last=$((${#_order[@]} - 1))
  if [[ "${_order[last]}" != "${dim}" ]]; then
    echo "error: group_ranges_for_dim: '${dim}' must be the outermost (last) entry in" \
      "DIM_ORDER (got DIM_ORDER=(${_order[*]}))" >&2
    return 1
  fi

  local inner_size=1 i
  for ((i = 0; i < last; i++)); do
    inner_size=$((inner_size * _sizes[i]))
  done
  if ((inner_size % jobs_per_gpu != 0)); then
    echo "error: group_ranges_for_dim: JOBS_PER_GPU=${jobs_per_gpu} does not evenly divide" \
      "${dim}'s group size (${inner_size} run_ids/value) -- a task would straddle two" \
      "${dim} values" >&2
    return 1
  fi

  local group_count="${_sizes[last]}" g run_start run_end
  for ((g = 0; g < group_count; g++)); do
    run_start=$((g * inner_size))
    run_end=$(((g + 1) * inner_size))
    _dst+=("$((run_start / jobs_per_gpu)):$((run_end / jobs_per_gpu - 1))")
  done
}
