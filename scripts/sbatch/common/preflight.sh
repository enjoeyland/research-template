# Checks run right before a TRAIN job is submitted (sourced by maybe_submit.sh). Defining the functions has no side effects.
#
# What a profile can ask for (optional knobs in profiles/<name>.sh, reset before every profile is loaded):
#   SLURM_CHECK_START=1        scarce GPUs: print `sbatch --test-only`'s estimated start time before submitting, so the
#                              wait is known (and a bad request fails here, not after queuing)
#   SLURM_WARN_CONCURRENCY=N   warn when N or more allocations may run at once ...
#   SLURM_WARN_AFTER=HH:MM:SS  ... with a requested time limit longer than this (the profile's own reason is in its comments)
# Environment (any profile):
#   TEST_ONLY=1                print the start estimate and exit WITHOUT submitting
#   NODELIST=node45            run on specific node(s) only (maybe_submit.sh adds --nodelist; checked here)

preflight_reset_knobs() {
  unset SLURM_CHECK_START SLURM_WARN_CONCURRENCY SLURM_WARN_AFTER
}

# [D-]HH:MM:SS | MM:SS | MM  ->  seconds (the formats sbatch --time accepts)
time_to_seconds() {
  local t="${1:?}" days=0 h=0 m=0 s=0
  if [[ "${t}" == *-* ]]; then
    days="${t%%-*}"
    t="${t#*-}"
  fi
  local IFS=':'
  # shellcheck disable=SC2206
  local -a p=(${t})
  case "${#p[@]}" in
    3) h="${p[0]}"; m="${p[1]}"; s="${p[2]}" ;;
    2) m="${p[0]}"; s="${p[1]}" ;;
    1) m="${p[0]}" ;;
    *) echo "error: cannot parse time '$1'" >&2; return 1 ;;
  esac
  echo $(( ((10#${days} * 24 + 10#${h}) * 60 + 10#${m}) * 60 + 10#${s} ))
}

# `--array` spec -> how many array tasks can run at once: "0-14%4" -> 4, "0-3" -> 4, "0,2,5%2" -> 2, "" -> 1
array_concurrency() {
  local spec="${1:-}"
  if [[ -z "${spec}" ]]; then
    echo 1
    return 0
  fi
  local cap="" ids="${spec}" n=0 part step a b
  if [[ "${spec}" == *%* ]]; then
    cap="${spec##*%}"
    ids="${spec%%%*}"
  fi
  local IFS=','
  for part in ${ids}; do
    step=1
    if [[ "${part}" == *:* ]]; then
      step="${part##*:}"
      part="${part%%:*}"
    fi
    if [[ "${part}" == *-* ]]; then
      a="${part%%-*}"
      b="${part##*-}"
      n=$(( n + (b - a) / step + 1 ))
    else
      n=$(( n + 1 ))
    fi
  done
  if [[ -n "${cap}" ]] && (( cap < n )); then
    n="${cap}"
  fi
  echo "${n}"
}

# preflight_warn_concurrency <array spec> <time limit>
preflight_warn_concurrency() {
  local n="${SLURM_WARN_CONCURRENCY:-}" after="${SLURM_WARN_AFTER:-}"
  if [[ -z "${n}" || -z "${after}" ]]; then
    return 0
  fi
  local conc
  conc="$(array_concurrency "${1:-}")"
  if (( conc >= n )) && (( $(time_to_seconds "${2:?}") > $(time_to_seconds "${after}") )); then
    {
      echo "WARNING: profile ${PROFILE_NAME}: up to ${conc} allocations may run at once (>= ${n}) with --time=${2} (> ${after})."
      echo "         Jobs may be stopped midway once they run longer than ${after}. Lower the concurrency (MAX_GPUS=$((n - 1)))"
      echo "         and make long runs resumable (callbacks=default_resumable)."
    } >&2
  fi
}

# preflight_check_nodelist <NODELIST> <partitions>   warn when a requested node is not in the profile's partitions
preflight_check_nodelist() {
  local nodelist="${1:-}" partitions="${2:-}" allowed node
  if [[ -z "${nodelist}" ]] || ! command -v scontrol >/dev/null 2>&1 || ! command -v sinfo >/dev/null 2>&1; then
    return 0
  fi
  allowed="$(sinfo -h -N -p "${partitions}" -o '%N' 2>/dev/null | sort -u)"
  for node in $(scontrol show hostnames "${nodelist}" 2>/dev/null); do
    if ! grep -qx "${node}" <<< "${allowed}"; then
      echo "WARNING: NODELIST node '${node}' is not in the partitions of profile ${PROFILE_NAME} (${partitions}): the job would never start." >&2
    fi
  done
}

# preflight_start_estimate <sbatch args...>   `sbatch --test-only`: nothing is queued
preflight_start_estimate() {
  local out
  if ! out="$(sbatch --test-only "$@" 2>&1)"; then
    echo "error: 'sbatch --test-only' rejected this request (the real submit would fail too):" >&2
    echo "${out}" >&2
    return 1
  fi
  echo "start estimate: ${out#sbatch: }"
}

# preflight <array spec> <time limit> <nodelist> <partitions> <sbatch args...>
preflight() {
  local array="${1:-}" time_limit="${2:?}" nodelist="${3:-}" partitions="${4:-}"
  shift 4
  preflight_check_nodelist "${nodelist}" "${partitions}"
  preflight_warn_concurrency "${array}" "${time_limit}"
  if [[ "${SLURM_CHECK_START:-}" == "1" || "${TEST_ONLY:-}" == "1" ]]; then
    preflight_start_estimate "$@" || exit 1
  fi
  if [[ "${TEST_ONLY:-}" == "1" ]]; then
    echo "TEST_ONLY=1: nothing was submitted."
    exit 0
  fi
}
