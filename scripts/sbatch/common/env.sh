# Shared runtime for job scripts under scripts/.
# Source from scripts/:           source "$(dirname "$0")/sbatch/common/env.sh"
# Source from scripts/sbatch/:    source "$(dirname "$0")/common/env.sh"
#
# Sets: REPO_ROOT, VENV, PYTHON, CKPT_ROOT; cds to REPO_ROOT; activates venv if present.
# Does NOT source the project .env wholesale (it may hold API keys) — only PROJECT_NAME, VENV and CHECKPOINT_DIR.

_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${_ENV_DIR}/../../.." && pwd)"
cd "$REPO_ROOT"

# Read ONE variable from .env with its ${...} references expanded (e.g. CHECKPOINT_DIR=/lustre/${USER}/.../${PROJECT_NAME}).
# Only PROJECT_NAME, CHECKPOINT_DIR and VENV are evaluated (never the whole file, it may hold API keys).
_dotenv() {
  ( eval "$(grep -E '^(PROJECT_NAME|CHECKPOINT_DIR|VENV)=' .env 2>/dev/null || true)"; printf '%s' "${!1:-}" )
}

# venv: $VENV env var > VENV= in .env > /scratch2/$USER/venvs/<PROJECT_NAME in .env | repo folder name>
if [[ -z "${VENV:-}" ]]; then
  VENV="$(_dotenv VENV)"
  PROJECT_NAME="${PROJECT_NAME:-$(_dotenv PROJECT_NAME)}"
  VENV="${VENV:-/scratch2/${USER}/venvs/${PROJECT_NAME:-$(basename "${REPO_ROOT}")}}"
fi
PYTHON="${VENV}/bin/python"

if [[ -f "${VENV}/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${VENV}/bin/activate"
elif [[ -d "${VENV}/bin" ]]; then
  export PATH="${VENV}/bin:${PATH}"
else
  echo "WARNING: VENV=${VENV} missing; using python on PATH" >&2
  PYTHON="$(command -v python3 || command -v python)"
fi

# CHECKPOINT_DIR: the environment wins over .env (same as the python side, where rootutils does not override set variables)
CKPT_ROOT="${CHECKPOINT_DIR:-$(_dotenv CHECKPOINT_DIR)}"
CKPT_ROOT="${CKPT_ROOT:-logs}"

mkdir -p "logs/slurm/$(date +%Y-%m-%d)"

unset _ENV_DIR

# Checkpoint path helpers -- one folder per experiment, seed<N>_epoch_*.ckpt filenames
# (see configs/callbacks/default.yaml).

# EXTRA_ARGS = extra hydra overrides the launcher passes to train.py (e.g. EXTRA_ARGS="debug=smoke").
is_debug_run() { [[ " ${EXTRA_ARGS:-} " == *" debug="* ]]; }
is_smoke_run() { [[ " ${EXTRA_ARGS:-} " == *" debug=smoke "* ]]; }

checkpoint_dir() {
  local experiment="${1:?}"
  # debug=smoke isolates the whole run tree under logs/smoke (configs/debug/smoke.yaml) and never writes to
  # CHECKPOINT_DIR, so the helpers (resume, "was a checkpoint written") must look there too.
  if is_smoke_run; then
    echo "logs/smoke/runs/${experiment}/checkpoints"
  else
    echo "${CKPT_ROOT}/runs/${experiment}/checkpoints"
  fi
}

seed_ckpt_path() {
  local experiment="${1:?}" seed="${2:?}"
  ls -1 "$(checkpoint_dir "${experiment}")"/seed"${seed}"_epoch_*.ckpt 2>/dev/null | sort | tail -1
}

seed_ckpt_exists() {
  [[ -n "$(seed_ckpt_path "$1" "$2")" ]]
}

# Waits for a seed's checkpoint to appear -- may still be training in a *different* array
# task / GPU allocation, so this is a plain poll, not an error if it's not there yet.
wait_for_seed_ckpt() {
  local experiment="${1:?}" seed="${2:?}" poll_sec="${3:-30}"
  if seed_ckpt_exists "${experiment}" "${seed}"; then
    return 0
  fi
  echo "Waiting for checkpoint: $(checkpoint_dir "${experiment}")/seed${seed}_epoch_*.ckpt"
  while ! seed_ckpt_exists "${experiment}" "${seed}"; do
    sleep "${poll_sec}"
  done
  echo "Found checkpoint: experiment=${experiment} seed=${seed}"
}

wait_for_all_seed_ckpts() {
  local experiment="${1:?}"
  shift
  local seed
  for seed in "$@"; do
    wait_for_seed_ckpt "${experiment}" "${seed}"
  done
}
