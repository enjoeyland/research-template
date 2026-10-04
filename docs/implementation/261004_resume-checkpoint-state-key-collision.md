---
title: default_resumable이 max_epochs=10에서 시작하지 못하던 문제 (ModelCheckpoint state_key 충돌)
created: 2026-10-04
updated: 2026-10-05
status: 수정됨
commits: [bf93fd2]
files: [src/utils/callbacks.py, configs/callbacks/model_checkpoint_resume.yaml, tests/test_configs.py]
related: [../experiments/261004_mccg-port-simulation.md]
---

# default_resumable이 max_epochs=10에서 시작하지 못하던 문제 (ModelCheckpoint state_key 충돌)

> **상태 (2026-10-05)**: 수정됨. resume용 체크포인트가 고정 `state_key`를 쓰고, `max_epochs`를 10/11/50으로 바꿔 가며 Trainer를 만드는 회귀 테스트가 있다.
> MCCG의 같은 설정에도 있는 잠재 버그로 보이지만 MCCG는 고치지 않았다(§6).

## 1. 증상
SLURM 스윕(`scripts/sbatch/template.sh`)의 학습 job이 시작하자마자 실패:
`RuntimeError: Found more than one stateful callback of type ModelCheckpoint ... The callback.state_key must be unique among all callbacks`.
toy 예제 실험(`261004-toy-example`, `callbacks=default_resumable`, `trainer.max_epochs: 10`)에서만 났다. 앞서 resume을 검증한 직접 실행은
`trainer.max_epochs=12`/`15`로 덮어써서 문제를 피해 갔다.

## 2. 원인
| 가설 | 확인 방법 | 결과 |
|---|---|---|
| 새 실험 폴더 구조(`logs/runs/...`) 변경이 원인 | 로그 트레이스백 확인 | ❌ 오류는 `Trainer.__init__`의 콜백 검증에서 나며 경로와 무관 |
| 같은 이름의 콜백이 중복 등록됨 | 콜백 목록 출력 | ❌ 이름은 서로 다름(`model_checkpoint`, `model_checkpoint_last`, `model_checkpoint_resume`) |
| Lightning이 `state_key`를 (monitor, mode, every_n_*)로 만들고, `last`(`every_n_epochs=${trainer.max_epochs}`)와 resume(`every_n_epochs=10`)이 `max_epochs=10`에서 같아짐 | 두 콜백 설정을 비교 | ✅ 원인 |

## 3. 근거
- `default_resumable`은 `ModelCheckpoint` 3개: best(`monitor: val/acc`), last(`monitor: null`, `every_n_epochs: ${trainer.max_epochs}`), resume(`monitor: null`, `every_n_epochs: 10`).
- monitor가 둘 다 `null`이고 `every_n_epochs`만 다른데, `max_epochs=10`이면 그 값마저 같아 `state_key`가 동일해진다.

## 4. 수정
resume용 체크포인트를 `ResumeModelCheckpoint(ModelCheckpoint)`로 바꾸고 `state_key`를 고정 문자열(`ModelCheckpoint{resume}`)로 오버라이드했다
(`configs/callbacks/model_checkpoint_resume.yaml`의 `_target_`). 간격을 바꿔 피하는 방법은 다른 `max_epochs`에서 같은 충돌이 다시 생길 수 있어 택하지 않았다.

## 5. 검증
- `tests/test_configs.py::test_resumable_callbacks_build_for_any_max_epochs`: `max_epochs` 10/11/50에서 콜백을 만들어 `Trainer`를 생성(검증은 생성 시점에 일어난다).
- 스윕(`PROFILE=cpu LOGGER=csv ./scripts/sbatch/template.sh`)이 `seed0_resume.ckpt`, `seed1_resume.ckpt`를 만들며 끝까지 통과.

## 6. 영향 범위 / 남은 위험
- MCCG의 `configs/callbacks/model_checkpoint_resume.yaml`과 `model_checkpoint_last.yaml`도 같은 구성이라 `max_epochs`가 resume 간격과 같으면 같은 오류가 난다(MCCG 실험은 70~420 epoch이라 드러나지 않았다). MCCG에는 반영하지 않았다.
- 새 `ModelCheckpoint`를 추가하는 사람은 `max_epochs`를 바꿔 가며 Trainer를 만들어 봐야 한다.

## 7. 반복 방지
`CLAUDE.md` §4.5에 "ModelCheckpoint를 여러 개 쓸 때 state_key 충돌" 항목이 있으면 좋다 — 제안만 한다(사용자 관리 파일).
