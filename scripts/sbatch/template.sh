#!/bin/bash
# Template: train + analyze over a hyperparameter grid.
# Copy to scripts/, edit axes + run_one / run_analyze, keep the launch_sweep line.
#
# Layout (one folder per experiment):
#   logs/runs/<experiment>/checkpoints/seed<N>_epoch_XXX.ckpt
#   logs/runs/<experiment>/train/    -- hydra config, log file, csv metrics
#   logs/runs/<experiment>/analyze/  -- csv/png, mean +/- std over all seeds, aggregated
#                                       once per experiment (not per-seed files)
#
# Submit: one command queues a train array, then an analyze job with
#   --dependency=afterok:<train_job> (ANALYZE_PROFILE=cpu by default). No ckpt polling.
#   define run_analyze() to opt in; scripts without it are unchanged.
#
#   cp scripts/sbatch/template.sh scripts/my_sweep.sh
#   ./scripts/my_sweep.sh
#   JOBS_PER_GPU=3 PROFILE=gpu96 ./scripts/my_sweep.sh
#   MAX_GPUS=4 ./scripts/my_sweep.sh
#   LOCAL=1 ./scripts/my_sweep.sh
#   ANALYZE_PROFILE=cpu ./scripts/my_sweep.sh
#   LOGGER=csv ./scripts/my_sweep.sh      # no wandb (smoke tests)
#   EXTRA_ARGS="debug=smoke" LOGGER=csv ./scripts/my_sweep.sh   # smoke test: isolated under logs/smoke, never resumes
set -euo pipefail

# sbatch runs a copy of this file from the slurm spool dir, so locate common/ via SLURM_SUBMIT_DIR
# (submit from the repo root) when inside a job, and via this file's own path otherwise.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  _SBATCH="${SLURM_SUBMIT_DIR}/scripts/sbatch"
else
  _HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  if [[ -d "${_HERE}/common" ]]; then
    _SBATCH="${_HERE}"        # this file still lives in scripts/sbatch/
  else
    _SBATCH="${_HERE}/sbatch" # copied to scripts/<name>.sh
  fi
fi

: "${PROFILE:=gpu24}"
: "${JOBS_PER_GPU:=1}"
: "${MAX_GPUS:=}"
: "${JOB_NAME:=my-sweep}"

WANDB_GROUP="${JOB_NAME}"
# config paths under configs/experiment/train/ (without .yaml): <model>/<YYMMDD_topic>/<YYMMDD-name>
EXPERIMENTS=(toy/261004_example/261004-toy-example)
# What's actually being swept is the CV fold -- seed is just set
# equal to it (most data configs default data.fold: ${seed}, so seed="${fold}" alone is enough,
# but data.fold is passed explicitly too so the sweep axis reads as fold, not a seed-variance
# replicate). Checkpoint filenames stay "seed<N>_epoch_*.ckpt" (configs/callbacks/default.yaml),
# so seed_ckpt_exists below still works unchanged.
FOLDS=(0 1)

DIM_ORDER=(fold experiment)
SIZES=("${#FOLDS[@]}" "${#EXPERIMENTS[@]}")

run_one () {
  local run_id="$1"
  local -A IDX=()
  decode_run_id "${run_id}" DIM_ORDER SIZES IDX

  local fold="${FOLDS[${IDX[fold]}]}"
  local experiment="${EXPERIMENTS[${IDX[experiment]}]}"
  local exp_name="${experiment##*/}"   # = experiment_name = checkpoint folder (the config file stem)
  local wandb_name="${exp_name}-fold${fold}"

  # wandb naming only for the default logger; LOGGER=csv (smoke / debugging) skips it
  local -a WANDB_ARGS=()
  if [[ "${LOGGER:-wandb}" == "wandb" ]]; then
    WANDB_ARGS=(+logger.wandb.name="${wandb_name}" logger.wandb.group="${WANDB_GROUP}")
  fi

  # mid-run resume + wandb run id (scripts/sbatch/common/resume.sh): fills RESUME_ARGS
  prepare_resume "${exp_name}" "${fold}" "${wandb_name}"

  # extra hydra overrides, e.g. EXTRA_ARGS="debug=smoke" (split on spaces)
  local -a EXTRA=()
  read -r -a EXTRA <<< "${EXTRA_ARGS:-}"

  echo "=== train: experiment=${experiment} fold=${fold} ==="
  python src/train.py experiment/train="${experiment}" seed="${fold}" data.fold="${fold}" \
    experiment_name="${exp_name}" \
    trainer="${TRAINER}" \
    logger="${LOGGER:-wandb}" ${LOGGER:+extras.enforce_tags=False} \
    "${WANDB_ARGS[@]}" "${RESUME_ARGS[@]}" "${EXTRA[@]}"

  # debug runs other than smoke turn the callbacks off, so no checkpoint is written (smoke writes under logs/smoke)
  if is_debug_run && ! is_smoke_run; then
    return 0
  fi
  if ! seed_ckpt_exists "${exp_name}" "${fold}"; then
    echo "error: no checkpoint for fold=${fold} under $(checkpoint_dir "${exp_name}")" >&2
    return 1
  fi
}

# Runs once after the whole train array succeeds (SLURM --dependency=afterok), or after all
# LOCAL train ids finish. Mean +/- std over folds per experiment — not per-fold files.
run_analyze () {
  local experiment folds_csv
  folds_csv="$(IFS=,; echo "${FOLDS[*]}")"
  for experiment in "${EXPERIMENTS[@]}"; do
    echo "=== analyze: experiment=${experiment} folds=[${folds_csv}] ==="
    python src/analyze.py experiment/train="${experiment}" seeds="[${folds_csv}]" \
      experiment_name="${experiment##*/}"
  done
}

# shellcheck disable=SC1091
source "${_SBATCH}/common/resume.sh"
# shellcheck disable=SC1091
source "${_SBATCH}/common/launch_sweep.sh"
