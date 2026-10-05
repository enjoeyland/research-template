---
title: debug=smoke가 실제 실행의 체크포인트를 재개/오염시키던 문제, 테스트 헬퍼의 pkg_resources 의존
created: 2026-10-05
updated: 2026-10-05
status: 수정됨
commits: []
files: [scripts/sbatch/common/env.sh, scripts/sbatch/common/resume.sh, scripts/sbatch/template.sh, tests/helpers/package_available.py, tests/helpers/run_if.py, tests/test_sbatch_helpers.py]
related: []
---

# debug=smoke가 실제 실행의 체크포인트를 재개/오염시키던 문제, 테스트 헬퍼의 pkg_resources 의존

> **상태 (2026-10-05)**: 두 문제 모두 수정됨. 다른 프로젝트에서 이 template을 쓰다가 보고된 것을 재현해서 고쳤고, 회귀 테스트가 있다.

## 1. 증상
1. **`tests/helpers`가 Python 3.12 venv(setuptools 없음)에서 import 실패**: `ModuleNotFoundError: pkg_resources`.
2. **스윕 템플릿의 `prepare_resume`이 `debug=smoke`의 경로 격리를 몰라서**, 같은 실험 이름의 *실제* 실행 체크포인트를 재개하려다 실패.

## 2. 원인
| 가설 | 확인 방법 | 결과 |
|---|---|---|
| (1) `pkg_resources`는 setuptools 소속이고 Python 3.12 venv에는 기본 포함되지 않음 | `sys.modules["pkg_resources"]=None`으로 막고 `tests.helpers.run_if` import | ✅ 재현. 우리 venv들은 `setuptools`가 있어 드러나지 않았다 |
| (2) `checkpoint_dir()`은 항상 `${CKPT_ROOT}/runs/<exp>/checkpoints`인데 `debug=smoke`는 `logs/smoke/runs/<exp>/checkpoints`에 쓴다 | 실제 실행의 `seed0_resume.ckpt`를 만들어 두고 `EXTRA_ARGS=debug=smoke`로 `prepare_resume` 호출 | ✅ 재현 |

## 3. 근거 (2번을 재현하면 증상이 세 가지)
- smoke 실행이 실제 실행의 체크포인트로 재개하도록 `ckpt_path=logs/runs/<exp>/checkpoints/seed0_resume.ckpt`를 받는다(보고된 것).
- smoke 실행이 **실제 실행의 체크포인트 폴더에 `wandb_id_seed0.txt`를 새로 쓴다**(실험 폴더 오염, `CHECKPOINT_DIR`이 lustre이면 lustre 쪽).
- 학습 후 "체크포인트가 생겼나" 검사(`seed_ckpt_exists`)도 실제 폴더를 봐서 smoke 결과와 무관하게 오탐/미탐이 난다.
- 부수적으로 `env.sh`는 `.env`의 `CHECKPOINT_DIR`만 읽고 환경변수는 무시했다(파이썬 쪽은 환경변수가 우선).

## 4. 수정
1. `tests/helpers`: `pkg_resources` → `importlib.metadata`(표준 라이브러리).
2. `env.sh`: `is_debug_run`/`is_smoke_run` 추가(런처가 `EXTRA_ARGS`로 넘기는 hydra override에서 `debug=...` 판단), `checkpoint_dir()`은 smoke면 `logs/smoke/runs/<exp>/checkpoints`, `CHECKPOINT_DIR` 환경변수가 `.env`보다 우선.
3. `resume.sh`: 디버그 실행이면 `prepare_resume`이 재개와 wandb id 파일 기록을 모두 건너뛴다(`RESUME_ARGS=()`).
4. `template.sh`: `EXTRA_ARGS`를 train.py로 전달(`EXTRA_ARGS="debug=smoke" LOGGER=csv ./scripts/...`), smoke 외 debug 설정(콜백 꺼짐)에서는 체크포인트 확인을 건너뜀.

## 5. 검증
- `tests/test_sbatch_helpers.py`(실제 bash 호출): 일반 실행은 재개, smoke는 재개 안 함 + 실제 폴더에 파일을 쓰지 않음, `checkpoint_dir` 경로. **옛 스크립트로 되돌리면 smoke 관련 2개가 실패**한다.
- 끝까지: 실제 실행(`seed*_resume.ckpt` 생성) 뒤 같은 실험 이름으로 `EXTRA_ARGS=debug=smoke` 스윕을 돌리니 `no resume`, 성공, 실제 폴더 불변, `logs/smoke/...`에 체크포인트 생성.
- `pkg_resources`를 막은 환경에서 헬퍼 import 성공.

## 6. 영향 범위 / 남은 위험
- 템플릿에서 복사해 간 프로젝트의 스윕 스크립트는 자체 수정이 필요하다(보고한 프로젝트는 런처에서 디버그면 재개를 건너뛰도록 막았다고 함). `env.sh`/`resume.sh`를 가져가면 같은 효과.
- `EXTRA_ARGS` 문자열은 공백으로 나뉘므로 값에 공백이 든 override는 안 된다.
- `debug=smoke` 외의 debug 설정이 `paths.log_dir`를 바꾸면 `checkpoint_dir`이 따라가지 못한다(지금 설정들은 안 바꾼다).

## 7. 반복 방지
실행 격리를 만드는 설정(`debug=smoke`가 `paths.log_dir`를 바꿈)을 추가/변경하면 sbatch 헬퍼(`checkpoint_dir` 등)가 같은 경로를 보는지 `tests/test_sbatch_helpers.py`에서 확인한다. (`CLAUDE.md` 반영 여부는 사용자에게 제안.)
