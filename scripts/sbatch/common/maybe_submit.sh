# Self-submit helper. Source from a run script (or via launch_sweep.sh).
#
# Behavior:
#   - Already inside a SLURM job (SLURM_JOB_ID set) → return, continue the run script
#   - LOCAL=1 → return, run on the current machine (no sbatch)
#   - SUBMIT_ANALYZE=1, ANALYZE_GROUPS unset → sbatch train (optional --array), then ONE sbatch
#     analyze with --dependency=afterok:<train_jobid> (ANALYZE_PROFILE, default cpu; no array),
#     then exit
#   - SUBMIT_ANALYZE=1, ANALYZE_GROUPS set (array of "label:start:end", array-task-id space,
#     both ends inclusive -- see sweep.sh's group_ranges_for_dim) → sbatch train, then ONE
#     analyze job PER entry, each depending only on ITS OWN train array task ids
#     (--dependency=afterok:<train_jobid>_<start>:<train_jobid>_<start+1>:...:<train_jobid>_<end>)
#     instead of the whole train array -- so e.g. one experiment's analyze can start as soon as
#     THAT experiment's seeds finish, without waiting on every other experiment in the sweep.
#     Each analyze job gets ANALYZE_GROUP_LABEL=<label> exported (via --export=ALL, same as
#     every other var already exported in the submitting shell) -- the run script's own
#     run_analyze() is responsible for reading it and restricting itself to that group. Without this, no analysis could
#     start until the slowest of several unrelated experiments in one sweep had finished.
#   - Otherwise → load profiles/<PROFILE>.sh and sbatch the top-level run script, then exit
#
# Overrides (env): PROFILE, ARRAY, JOB_NAME, TIME, OUTPUT, JOBS_PER_GPU, SLURM_EXCLUDE, SLURM_EXCLUDE_EXTRA, NODELIST, QOS,
#                  SUBMIT_ANALYZE, ANALYZE_PROFILE, PHASE, ANALYZE_GROUPS (array), TEST_ONLY
#   NODELIST=node45   run the train job on these node(s) only (e.g. a job that needs a lot of host RAM); not applied to analyze
#   QOS=base_qos      submit the train job with this QOS instead of the profile's (e.g. big_qos for gpu48); not applied to analyze
#   TEST_ONLY=1       print the estimated start time (sbatch --test-only) and exit without submitting
# Before a train job is submitted, common/preflight.sh checks NODELIST, warns about risky concurrency and (for profiles that
# ask for it, like gpu48) prints the estimated start time.

if [[ -n "${LOCAL:-}" && "${LOCAL}" != "0" ]]; then
  return 0
fi

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  return 0
fi

: "${PROFILE:=gpu96}"
: "${JOBS_PER_GPU:=1}"
: "${SUBMIT_ANALYZE:=0}"
: "${ANALYZE_PROFILE:=cpu}"

_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_profiles_dir="${_dir}/../profiles"
_repo_root="$(cd "${_dir}/../../.." && pwd)"
# shellcheck disable=SC1091
source "${_dir}/preflight.sh"
# Outermost script in the source chain (works when nested: run -> launch_sweep -> here).
_caller="${BASH_SOURCE[-1]:-}"

if [[ -z "${_caller}" || ! -f "${_caller}" ]]; then
  echo "error: maybe_submit.sh must be sourced from a run script" >&2
  exit 1
fi

_caller_abs="$(cd "$(dirname "${_caller}")" && pwd)/$(basename "${_caller}")"

_load_profile() {
  local profile="${1:?}"
  local profile_file="${_profiles_dir}/${profile}.sh"
  if [[ ! -f "${profile_file}" ]]; then
    echo "error: unknown profile '${profile}' (expected ${profile_file})" >&2
    echo "available:" >&2
    ls -1 "${_profiles_dir}"/*.sh 2>/dev/null \
      | xargs -n1 basename | sed 's/\.sh$//' | sed 's/^/  /' >&2
    exit 1
  fi
  preflight_reset_knobs   # knobs of the previously loaded profile must not leak into this one
  # shellcheck disable=SC1090
  source "${profile_file}"
}

# --exclude value: the profile's SLURM_EXCLUDE plus SLURM_EXCLUDE_EXTRA, which adds nodes for THIS submission only.
# Profiles are shared by every session, so one failure should not add a node there; the sweep script uses this variable
# instead (a SLURM_EXCLUDE given through the environment would be overwritten by the profile). Off by default.
_effective_exclude() {
  local excl="${SLURM_EXCLUDE:-}"
  if [[ -n "${SLURM_EXCLUDE_EXTRA:-}" ]]; then
    excl="${excl:+${excl},}${SLURM_EXCLUDE_EXTRA}"
  fi
  echo "${excl}"
}

# --nodelist value: NODELIST pins the TRAIN job to specific node(s). Never for the analyze job (a different, usually CPU,
# partition where the GPU node does not exist).
_effective_nodelist() {
  if [[ "${PHASE:-train}" == "analyze" ]]; then
    return 0
  fi
  echo "${NODELIST:-}"
}

# --qos value: QOS overrides the profile's SLURM_QOS for the TRAIN job only (the analyze job keeps its own profile's QOS).
_effective_qos() {
  if [[ "${PHASE:-train}" == "analyze" || -z "${QOS:-}" ]]; then
    echo "${SLURM_QOS}"
  else
    echo "${QOS}"
  fi
}

# Fill _sbatch_args (and _resolved_time / _resolved_nodelist) from the currently sourced profile + JOB_NAME/ARRAY/TIME/OUTPUT.
_build_sbatch_args() {
  _resolved_time="${TIME:-${SLURM_TIME}}"
  local output excl
  if [[ -n "${OUTPUT:-}" ]]; then
    output="${OUTPUT}"
  elif [[ -n "${ARRAY:-}" ]]; then
    output="${_log_dir}/${JOB_NAME}_%A_%a.log"
  else
    output="${_log_dir}/${JOB_NAME}_%j.log"
  fi
  excl="$(_effective_exclude)"
  _resolved_nodelist="$(_effective_nodelist)"

  _sbatch_args=(
    -J "${JOB_NAME}"
    -p "${SLURM_PARTITION}"
    -q "$(_effective_qos)"
    --time="${_resolved_time}"
    --output="${output}"
  )
  [[ -n "${SLURM_CPUS:-}" ]] && _sbatch_args+=(--cpus-per-task="${SLURM_CPUS}")
  [[ -n "${SLURM_GRES:-}" ]] && _sbatch_args+=(--gres="${SLURM_GRES}")
  [[ -n "${excl}" ]] && _sbatch_args+=(--exclude="${excl}")
  [[ -n "${_resolved_nodelist}" ]] && _sbatch_args+=(--nodelist="${_resolved_nodelist}")
  [[ -n "${ARRAY:-}" ]] && _sbatch_args+=(--array="${ARRAY}")

  echo "profile=${PROFILE_NAME} partition=${SLURM_PARTITION} qos=$(_effective_qos) gres=${SLURM_GRES:-(none)} exclude=${excl:-(none)} nodelist=${_resolved_nodelist:-(any)} time=${_resolved_time} jobs_per_gpu=${JOBS_PER_GPU} phase=${PHASE:-train}"
  echo "output=${output}"
}

# Checks before a TRAIN job is submitted (common/preflight.sh): NODELIST, concurrency warning, estimated start.
_preflight_train() {
  preflight "${ARRAY:-}" "${_resolved_time}" "${_resolved_nodelist}" "${SLURM_PARTITION}" \
    "${_sbatch_args[@]}" "${_sbatch_export[@]}" "${_caller_abs}" "$@"
}

_load_profile "${PROFILE}"

cd "${_repo_root}"

: "${JOB_NAME:=$(basename "${_caller_abs}" | sed -E 's/\.(sbatch|sh)$//')}"

_log_day="$(date +%Y-%m-%d)"
_log_dir="logs/slurm/${_log_day}"
mkdir -p "${_log_dir}"

export HYDRA_TRAINER="${HYDRA_TRAINER:-gpu}"
export SLURM_PROFILE="${PROFILE_NAME}"
export JOBS_PER_GPU

_sbatch_export=(--export=ALL,HYDRA_TRAINER,SLURM_PROFILE,JOBS_PER_GPU,PHASE)

if [[ "${SUBMIT_ANALYZE}" == "1" ]]; then
  # Train array (or single job), then a dependent analyze job. Login-node only; exits after both
  # are queued so the user still runs one script.
  export PHASE=train
  _build_sbatch_args
  _preflight_train "$@"
  echo "job=${_caller_abs} $* (train)"
  _train_job="$(sbatch --parsable "${_sbatch_args[@]}" "${_sbatch_export[@]}" \
    "${_caller_abs}" "$@")"
  echo "submitted train job=${_train_job}"

  _train_job_name="${JOB_NAME}"
  _train_array="${ARRAY:-}"

  if declare -p ANALYZE_GROUPS &>/dev/null && ((${#ANALYZE_GROUPS[@]} > 0)); then
    # Per-group analyze jobs -- see this file's header comment. Each ANALYZE_GROUPS entry is
    # "label:start:end" (array-task-id space, both ends inclusive, from group_ranges_for_dim).
    for _group in "${ANALYZE_GROUPS[@]}"; do
      _label="${_group%%:*}"
      _range="${_group#*:}"
      _start="${_range%%:*}"
      _end="${_range#*:}"
      _dep="afterok"
      for ((_i = _start; _i <= _end; _i++)); do
        _dep+=":${_train_job}_${_i}"
      done
      JOB_NAME="${_train_job_name}-analyze-${_label}"
      ARRAY=""
      OUTPUT=""
      TIME=""
      _load_profile "${ANALYZE_PROFILE}"
      export HYDRA_TRAINER="${HYDRA_TRAINER:-cpu}"
      export SLURM_PROFILE="${PROFILE_NAME}"
      export PHASE=analyze
      export ANALYZE_GROUP_LABEL="${_label}"
      _build_sbatch_args
      echo "job=${_caller_abs} $* (analyze group=${_label}, dependency=${_dep})"
      _analyze_job="$(sbatch --parsable \
        --dependency="${_dep}" \
        "${_sbatch_args[@]}" "${_sbatch_export[0]},ANALYZE_GROUP_LABEL" \
        "${_caller_abs}" "$@")"
      echo "submitted analyze job=${_analyze_job} group=${_label} dependency=${_dep}"
    done
    JOB_NAME="${_train_job_name}"
    ARRAY="${_train_array}"
    exit 0
  fi

  JOB_NAME="${_train_job_name}-analyze"
  ARRAY=""
  OUTPUT=""
  TIME=""
  _load_profile "${ANALYZE_PROFILE}"
  export HYDRA_TRAINER="${HYDRA_TRAINER:-cpu}"
  export SLURM_PROFILE="${PROFILE_NAME}"
  export PHASE=analyze
  _build_sbatch_args
  echo "job=${_caller_abs} $* (analyze, dependency=afterok:${_train_job})"
  _analyze_job="$(sbatch --parsable \
    --dependency="afterok:${_train_job}" \
    "${_sbatch_args[@]}" "${_sbatch_export[@]}" \
    "${_caller_abs}" "$@")"
  echo "submitted analyze job=${_analyze_job} dependency=afterok:${_train_job}"
  # Restore in case this file is ever sourced without exiting (shouldn't happen).
  JOB_NAME="${_train_job_name}"
  ARRAY="${_train_array}"
  exit 0
fi

export PHASE="${PHASE:-train}"
_build_sbatch_args
if [[ "${PHASE}" != "analyze" ]]; then
  _preflight_train "$@"
fi
echo "job=${_caller_abs} $*"

exec sbatch "${_sbatch_args[@]}" "${_sbatch_export[@]}" \
  "${_caller_abs}" "$@"
