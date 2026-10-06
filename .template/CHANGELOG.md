# Template 변경 이력

다운스트림에 영향이 있는 변경 **묶음마다 한 번** 버전을 올린다(규칙: [MAINTAINING.md](MAINTAINING.md)). 프로젝트는 `make template-status`로 자기 버전(`.template-version`)과
최신을 비교한다. 각 버전의 태그는 `template-vN`이다. 항목마다 **다운스트림에서 할 일**과 **영향 경로**를 적는다.

## v7 (2026-10-06) — 테스트 정리
- 변경: template 전용 테스트(`test_template_status.py`: `make rename`, 버전·changelog 일관성, template-status)를 `tests/`에서 `.template/tests/`로 옮겼다(`make rename`이 폴더째 지우므로 프로젝트에는 안 간다).
  `make test`와 CI는 `tests src`(+ `.template/tests`가 있으면 그것도)를 돌린다. `pytest . .template/tests`는 하위 경로를 중복으로 보고 빼 버려서 폴더를 따로 나열한다.
- 삭제: `tests/helpers/{run_if,package_available}.py`에서 쓰이지 않는 플래그(tpu, fairscale, deepspeed, neptune, comet, mlflow, skip_windows, min_torch, max_torch, min_python).
- 변경: `pyproject.toml`의 pytest 설정에 `pythonpath = ["."]`, `minversion = "7.0"`(`tests/` 밖의 테스트가 `src`를 import할 수 있게).
- 다운스트림에서 할 일: `tests/test_template_status.py`가 있으면 지운다. `tests/helpers/`를 가져오고 `pyproject.toml`의 pytest 설정, `Makefile`의 `test`/`test-full`, `.github/workflows/test.yml`의 pytest 줄을 반영한다.
- 영향 경로: `tests/helpers/`, `pyproject.toml`, `Makefile`, `.github/workflows/test.yml`.

## v6 (2026-10-06) — CLAUDE.md에 research/ 규칙
- 변경: `CLAUDE.md` §2에 `research/`는 사용자가 직접 쓰는 폴더이고 AI는 읽기만 한다는 규칙을 추가했다(v5에서 폴더만 만들고 AI 지침이 빠져 있었다).
- 다운스트림에서 할 일: v5의 `research/`를 쓰는 프로젝트는 `CLAUDE.md`에 같은 항목을 옮긴다.
- 변경: `research/README.md`와 템플릿에서 저자/AI 역할을 가르는 표현("사용자가 쓰고 AI는 읽기만", 1인칭 "내 말로")을 뺐다. 이 글은 다른 사람과 논의하거나 논문에 쓰는 문서라서 README에
  "사용자"/"AI"가 나오지 않는다. AI에게 주는 지침은 `CLAUDE.md` §2에만 둔다.
- 영향 경로: `CLAUDE.md`, `research/`.

## v5 (2026-10-06) — 사용자 소유 폴더 research/
- 추가: `research/`(사용자가 직접 쓰는 논의·논문 정리 폴더; AI는 읽기만, 요청이 있을 때만 수정). `YYMMDD_<주제>/`마다 `YYMMDD_<주제>.md` + `YYMMDD_<주제>.pptx` + `figures/`(파일 이름에 주제를 넣어
  따로 내려받아도 알 수 있게), `_TEMPLATE/YYMMDD_topic.md`(글 템플릿), `research/paper/`는 gitignore(별도 repo).
- 다운스트림에서 할 일: 선택. 쓰려면 `research/`와 `.gitignore`의 `research/paper/` 줄을 가져온다.
- 영향 경로: `research/`, `.gitignore`, `docs/adr/adr-template-structure.md`(구조).

## v4 (2026-10-06) — template 관리 구조
- 추가: `.template/`(template 전용: 이 변경 이력, 유지 규칙, 다운스트림 현황), `.template-version`(프로젝트의 버전 표식), `make template-status` / `make template-mark-synced`
  (`src/template_status.py`).
- 변경: `make rename`이 `.template/`을 삭제한다(template repo 자신에서는 거부, `FORCE=1`로만 허용). ADR의 변경 이력은 이 파일로 옮겼다.
- 다운스트림에서 할 일: `.template-version`을 만든다(`version:`에 자기가 복사한 시점의 버전, 모르면 가장 가까운 값), `src/template_status.py`와 Makefile의 `template-*` 타깃을 가져온다.
- 영향 경로: `Makefile`, `src/template_status.py`, `.template-version`.

## v3 (2026-10-06) — sbatch 제출 전 검사, study 구조
- 추가: `scripts/sbatch/common/preflight.sh`: 제출 직전 `sbatch --test-only`로 예상 시작 시각 출력(`SLURM_CHECK_START=1`, gpu48 기본), `TEST_ONLY=1`(제출 없이 확인),
  동시 N개 이상 + 시간 제한 초과 경고(gpu48은 5개 / 3시간), `NODELIST=`(train에만, 프로필 파티션 검사). `maybe_submit.sh`가 이를 호출하고 프로필마다 설정을 초기화한다.
- 변경: `csv`/`tensorboard` 로거에 `version: seed${seed}`(같은 실험 폴더를 쓰는 seed가 동시에 시작하면 같은 `metrics.csv`에 써서 `ValueError`가 나던 문제).
- 변경: study/analysis의 실행 셸(`run.sh`)은 코드 옆 폴더(`src/studies/<주제>/`, 반복하면 `src/analysis/<주제>/`)에 둔다. `scripts/`는 학습 스윕 전용.
  `scripts/sbatch/template_study.sh` 추가, `template.sh`와 함께 repo 루트를 어느 깊이에서든 `.project-root`로 찾는 스니펫을 쓴다.
- 삭제: `data/.gitkeep`, `third_party/.gitkeep`, `docs/{proposals,experiments}/.gitkeep`.
- 다운스트림에서 할 일: `scripts/sbatch/`(preflight.sh, maybe_submit.sh, template*.sh, profiles/gpu48.sh) 가져오기, `configs/logger/{csv,tensorboard}.yaml`의 `version` 반영,
  study를 `scripts/studies/`에 두었다면 코드 옆으로 옮기기, 템플릿으로 만든 런처는 새 루트 탐색 스니펫으로 교체.
- 영향 경로: `scripts/sbatch/`, `configs/logger/`, `src/studies/`, `src/analysis/`, `scripts/README.md`, `CLAUDE.md`(§1 gpu48 규칙, §3 커밋 규칙).

## v2 (2026-10-05) — third_party 사용 방식, debug 실행 격리
- 추가: `third_party/README.md`의 사용 방식 3가지(우리 구조에 맞춘 어댑터 / loss와 평가가 얽힌 경우의 어댑터 패턴 / 독립 실행)와 독립 실행 규칙(`#SBATCH` 직접 쓰기 금지, repo 전용 venv,
  산출물은 `logs/runs/<exp>/`), `src/models/README.md`의 "패턴 2: 어댑터".
- 수정: `debug=smoke` 실행이 같은 이름의 실제 실험 체크포인트를 재개하려 하고 그 폴더에 `wandb_id_seed*.txt`를 쓰던 문제. `env.sh`(`checkpoint_dir`이 smoke 격리 경로를 따름,
  `is_debug_run`/`is_smoke_run`, `CHECKPOINT_DIR` 환경변수가 `.env`보다 우선), `resume.sh`(디버그 실행은 재개 안 함), `template.sh`(`EXTRA_ARGS`).
- 수정: `tests/helpers`가 `pkg_resources`(setuptools, Python 3.12 venv에는 없음) 대신 `importlib.metadata`를 쓴다.
- 다운스트림에서 할 일: `scripts/sbatch/common/{env,resume}.sh`, `template.sh`, `tests/helpers/`, `tests/test_sbatch_helpers.py` 가져오기. 런처가 디버그를 다른 변수로 넘기면 그 이름에 맞춘다.
- 영향 경로: `scripts/sbatch/common/`, `tests/helpers/`, `third_party/README.md`, `src/models/README.md`.

## v1 (2026-10-04) — 연구 프로젝트용 template로 개편 (PR #1)
- 원본(MNIST 중심 lightning-hydra-template)을 Medical-CausalInference와 JointDLM에서 검증된 규칙으로 개편: 실행 산출물을 실험 하나 = 폴더 하나(`logs/runs/<exp>/{train,eval,analyze,checkpoints}`),
  `ModelOutput` → `CompositeLoss`/`TaskMetrics`(`preds_key`/`target_key`), `MetricTrends` 콜백(`val/<name>_best`, `val/overfit_gap`),
  SLURM 스윕 인프라(`scripts/sbatch/`: 스윕, `train → analyze` 의존성, 중간 재개 + wandb run id), `PROJECT_NAME`(`.env`, `make rename`), `/lustre` vs `/scratch2` 저장 규칙, `data/` 심볼릭 링크,
  `third_party/`, `CLAUDE.md`와 `docs/{proposals,experiments,implementation,adr}`, 최소 CI.
- 제거: MNIST, notebooks, `setup.py`, pre-commit, 안 쓰는 로거, release-drafter/dependabot/codecov.
- 다운스트림에서 할 일: 구조가 크게 바뀌었으므로 파일 단위 병합 대신 `docs/adr/adr-template-structure.md`를 기준으로 영역별로 옮긴다.
- 영향 경로: 전체.
