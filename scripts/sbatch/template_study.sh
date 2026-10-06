#!/bin/bash
# Template: launcher of ONE study (a one-off check / diagnostic), run on SLURM through the shared infra: profiles, shared
# SLURM_EXCLUDE, logs/slurm, resubmission, TEST_ONLY. The launcher lives NEXT TO THE STUDY'S CODE:
#
#   cp scripts/sbatch/template_study.sh src/studies/<YYMMDD_topic>/run.sh      # then edit STUDY, ITEMS and run_one
#   ./src/studies/<YYMMDD_topic>/run.sh                                         # submit (login node)
#   PROFILE=cpu ./src/studies/<YYMMDD_topic>/run.sh                             # analysis without a GPU
#   JOBS_PER_GPU=2 MAX_GPUS=3 PROFILE=gpu24 ./src/studies/<YYMMDD_topic>/run.sh
#   TEST_ONLY=1 PROFILE=gpu48 ./src/studies/<YYMMDD_topic>/run.sh               # estimated start only, nothing submitted
#
# Outputs (figures, tables, logs of the study) go to logs/studies/<YYMMDD_topic>/, not next to the code (logs/ is gitignored);
# only what a document needs is copied to docs/figures/. A study that becomes a routine moves to src/analysis/<YYMMDD_topic>/
# together with its run.sh. scripts/ keeps only the training sweep launchers (template.sh).
set -euo pipefail

# Locate the repo root at ANY depth (this file may live in scripts/, src/studies/<topic>/ or src/analysis/<topic>/). sbatch runs a COPY
# of this file from the slurm spool dir, so inside a job the root comes from REPO_ROOT (exported by the submitting shell and carried
# by --export=ALL) or, for a plain `sbatch`, from the submit directory.
_find_root() {
  local d="${1:?}"
  while [[ "${d}" != "/" && ! -f "${d}/.project-root" ]]; do d="$(dirname "${d}")"; done
  if [[ -f "${d}/.project-root" ]]; then printf '%s' "${d}"; fi
  return 0
}
REPO_ROOT="${REPO_ROOT:-$(_find_root "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)")}"
REPO_ROOT="${REPO_ROOT:-$(_find_root "${SLURM_SUBMIT_DIR:-${PWD}}")}"
[[ -n "${REPO_ROOT}" ]] || { echo "error: .project-root not found above this script or the submit directory" >&2; exit 1; }
export REPO_ROOT
_SBATCH="${REPO_ROOT}/scripts/sbatch"

: "${PROFILE:=gpu24}"
: "${JOBS_PER_GPU:=1}"
: "${MAX_GPUS:=}"

# Name of THIS folder. Written out because inside a job this file is a copy in the slurm spool dir.
STUDY="YYMMDD_topic"
: "${JOB_NAME:=${STUDY}}"

# What to run: one array task per JOBS_PER_GPU items (variants, models, seeds ... whatever the study loops over).
ITEMS=(a b)
DIM_ORDER=(item)
SIZES=("${#ITEMS[@]}")

run_one () {
  local run_id="$1"
  local -A IDX=()
  decode_run_id "${run_id}" DIM_ORDER SIZES IDX
  local item="${ITEMS[${IDX[item]}]}"

  local out="${REPO_ROOT}/logs/studies/${STUDY}/${item}"
  mkdir -p "${out}"
  echo "=== study ${STUDY}: item=${item} -> ${out} ==="
  python "src/studies/${STUDY}/main.py" --item "${item}" --out "${out}"
}

# shellcheck disable=SC1091
source "${_SBATCH}/common/launch_sweep.sh"
