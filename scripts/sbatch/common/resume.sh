#!/bin/bash
# 중간 재개 + wandb run id 처리 (2026-09-22: milk10k_train_{scm_cascade,causalcgm,c2bm}.sh 에 복붙돼 있던 블록을 여기로 통합).
#
# 사용: run_one 안에서
#   prepare_resume "${experiment}" "${fold}" "${wandb_name}"
#   python src/train.py ... "${RESUME_ARGS[@]}"
# 결과: 전역 배열 RESUME_ARGS = (+logger.wandb.id='<id>' [ckpt_path=<resume ckpt>]).  checkpoint_dir 은 env.sh 의 것을 쓴다.
#
# 재개: config 가 callbacks=default_resumable 이면 10 epoch 마다 `fold{fold}_resume.ckpt` 가 덮어쓰기로 남는다
# (configs/callbacks/model_checkpoint_resume.yaml). 있으면 ckpt_path 로 넘겨 그 step 에서(optimizer/scheduler 포함) 이어간다 --
# 같은 명령으로 재제출만 하면 된다.
#
# 손상 방어 (2026-09-22 방식 변경): quota 초과 등으로 덮어쓰기 저장이 도중에 실패하면 **잘린 파일**이 남고(2026-09-20 실측: 609MB -> 8MB),
# 그 상태로 재개하면 `failed finding central directory` 로 죽는다. 예전에는 "정상 크기의 절반 미만"을 스크립트마다 손으로 정한 상수
# (scm 300MB / baseline 4MB)로 판정했는데, 모델이 바뀌면 상수가 틀려진다(ResNet fine-tune 은 정상이 ~135MB 라 4MB 기준으로는 8MB 로
# 잘려도 통과한다). 크기 대신 **zip 중앙 디렉토리를 실제로 열어 보는 것**으로 바꿔 모델과 무관하게 잘림을 잡는다.
# 손상 파일은 지우지 않고 .corrupt 로 보존한다.
#
# wandb run id: 체크포인트 디렉토리의 wandb_id_fold{fold}.txt 에 저장해 재제출 때 같은 run 에 이어 쓴다 (wandb 에서 run 을 지울 때는 체크포인트
# 디렉토리도 같이 지울 것 -- 안 그러면 "previously created and deleted" 로 거부된다).
#   - 파일이 있으면 그 id. 없고 재개 파일만 있으면(이 방식 도입 전 런) 옛 규칙 md5(이름)[:8]. 둘 다 없으면 새 유일 id 를 만들어 기록.
#
# ***id 는 반드시 quote 해서 넘긴다***: md5 8자리 hex 가 `78261e27` 처럼 숫자+e+숫자 꼴이면 Hydra 가 float(7.8261e+31)로 파싱해서
# wandb Settings(pydantic)가 `run_id Input should be a valid string` 으로 시작하자마자 죽는다 (2026-09-21 ReconCBM v1-alt 잡 2305457 이 이렇게 죽었다;
# 16^8 중 약 0.2% 확률이라 몇 번 잘 돌다가 어느 날 터진다). Hydra 는 작은따옴표로 감싼 값을 str 로 읽는다.

_resume_ckpt_ok() {
  "${PYTHON:-python}" -c 'import sys, zipfile; zipfile.ZipFile(sys.argv[1]).close()' "$1" 2>/dev/null
}

prepare_resume() {
  local experiment="${1:?}" fold="${2:?}" wandb_name="${3:?}"
  local ckdir resume_ckpt wandb_id id_file
  ckdir="$(checkpoint_dir "${experiment}")"
  resume_ckpt="${ckdir}/fold${fold}_resume.ckpt"
  id_file="${ckdir}/wandb_id_fold${fold}.txt"   # fold 별 -- 한 experiment 디렉토리를 여러 fold 가 공유해도 id 가 안 섞인다

  if [[ -f "${resume_ckpt}" ]] && ! _resume_ckpt_ok "${resume_ckpt}"; then
    echo "=== RESUME 무시: ${resume_ckpt} 를 zip 으로 열 수 없다(잘림/손상 의심, $(stat -c %s "${resume_ckpt}") bytes) -> 처음부터 시작. 파일은 .corrupt 로 보존 ===" >&2
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

  RESUME_ARGS=(+logger.wandb.id="'${wandb_id}'")
  if [[ -f "${resume_ckpt}" ]]; then
    echo "=== RESUME: ${resume_ckpt} (wandb id=${wandb_id}) ==="
    RESUME_ARGS+=(ckpt_path="${resume_ckpt}")
  fi
}
