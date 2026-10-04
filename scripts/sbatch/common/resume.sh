#!/bin/bash
# Mid-run resume + wandb run id (shared by sweep scripts).
#
# Use inside run_one:
#   prepare_resume "${exp_name}" "${seed}" "${wandb_name}"
#   python src/train.py ... "${RESUME_ARGS[@]}"
# Result: global array RESUME_ARGS = ( +logger.wandb.id='<id>' [ckpt_path=<resume ckpt>] ). checkpoint_dir comes from env.sh.
# With LOGGER=csv (or any non-wandb logger) the wandb id is left out.
#
# Resume: if the experiment config uses `callbacks=default_resumable`, `seed<N>_resume.ckpt` is overwritten every
# 10 epochs (configs/callbacks/model_checkpoint_resume.yaml). If it exists it is passed as ckpt_path and training
# continues from that step (optimizer / scheduler included) -- just resubmit the same command.
#
# Corruption guard: a failed overwrite (quota exceeded etc.) leaves a TRUNCATED file, and resuming from it dies with
# `failed finding central directory`. Instead of a hand-set size threshold (which goes wrong when the model changes),
# actually open the zip central directory. A corrupt file is kept as .corrupt, not deleted, and training starts over.
#
# wandb run id: stored in wandb_id_seed<N>.txt in the checkpoint dir so a resubmission appends to the SAME wandb run
# (when deleting a run in wandb, delete the checkpoint dir too -- else wandb refuses: "previously created and deleted").
#   - file exists -> that id. Else, if only a resume file exists (pre-existing run) -> md5(name)[:8]. Else a new unique id.
#
# ***Pass the id QUOTED***: an 8-hex md5 like `78261e27` (digits+e+digits) is parsed by Hydra as a float (7.8261e+31)
# and wandb Settings (pydantic) dies at start with `run_id Input should be a valid string` (~0.2% of ids). Hydra reads a
# single-quoted value as str.

_resume_ckpt_ok() {
  "${PYTHON:-python}" -c 'import sys, zipfile; zipfile.ZipFile(sys.argv[1]).close()' "$1" 2>/dev/null
}

prepare_resume() {
  local experiment="${1:?}" seed="${2:?}" wandb_name="${3:?}"
  local ckdir resume_ckpt wandb_id id_file
  ckdir="$(checkpoint_dir "${experiment}")"
  resume_ckpt="${ckdir}/seed${seed}_resume.ckpt"
  id_file="${ckdir}/wandb_id_seed${seed}.txt"   # per seed -- seeds of one experiment share the dir without mixing ids

  if [[ -f "${resume_ckpt}" ]] && ! _resume_ckpt_ok "${resume_ckpt}"; then
    echo "=== RESUME ignored: cannot open ${resume_ckpt} as zip (truncated/corrupt? $(stat -c %s "${resume_ckpt}") bytes) -> starting over. File kept as .corrupt ===" >&2
    mv -f "${resume_ckpt}" "${resume_ckpt}.corrupt"
  fi

  if [[ -f "${id_file}" ]]; then
    wandb_id="$(<"${id_file}")"
  elif [[ -f "${resume_ckpt}" ]]; then
    wandb_id="$(printf '%s' "${wandb_name}" | md5sum | cut -c1-8)"
  else
    wandb_id="$(printf '%s' "${wandb_name}-$(date +%s%N)" | md5sum | cut -c1-8)"
  fi
  mkdir -p "${ckdir}"
  [[ -f "${id_file}" ]] || printf '%s\n' "${wandb_id}" > "${id_file}"

  RESUME_ARGS=()
  if [[ "${LOGGER:-wandb}" == "wandb" ]]; then
    RESUME_ARGS+=(+logger.wandb.id="'${wandb_id}'")
  fi
  if [[ -f "${resume_ckpt}" ]]; then
    echo "=== RESUME: ${resume_ckpt} (wandb id=${wandb_id}) ==="
    RESUME_ARGS+=(ckpt_path="${resume_ckpt}")
  fi
}
