# research-template 개편 계획서

작성: 2026-10-04 · 상태: **제안 (아직 아무것도 수정하지 않음)**

## 0. 배경과 목표

- 출처 1: `Medical-CausalInference/medical_concept_causal_graph` (이하 **MCCG**) — template을 fork해 처음으로 잘 쓴 프로젝트. 학습 인프라, SLURM 스윕, 운영 규칙이 이미 검증됨.
- 출처 2: `JointDLM` — 시작한 지 얼마 안 된 두 번째 프로젝트. 이미 MCCG의 `scripts/sbatch/`를 **복사해 갔고**(일부만, `resume.sh`/`gpu24-cuda118.sh`/`gpu48bio.sh`는 빼고), `CLAUDE.md`·`.notes/gotchas.md`·`docs/` 구조·`studies/`·`debug=smoke`·`results.csv`를 독자적으로 다시 만들었다. → **복사본이 갈라지기 시작했다는 것 자체가 template에 올려야 할 항목의 증거.**
- 결정 사항(사용자 확정): MNIST 제거 / SLURM 인프라는 그대로 복사 / 문서는 일반화한 스켈레톤 / skills는 목록만 문서화 / 코드 → SLURM → 문서 순으로 단계별.
- **`third_party/`는 앞으로의 모든 연구에 계속 있을 폴더로 template에 상설 포함한다.** (JointDLM: `third_party/{FlexMDM,mdlm}` git submodule. 표기는 JointDLM와 같은 `third_party`(언더스코어)로 통일 제안 — Python 식별자로 쓰기 쉽고, MCCG의 `baselines/`와 용도가 겹치는 "외부 코드" 슬롯을 하나로 합친다.)

## 1. 현재 template에서 지울 것

| 대상 | 이유 | 비고 |
|---|---|---|
| `src/data/mnist_datamodule.py`, `src/models/mnist_module.py`, `src/models/components/simple_dense_net.py` | MNIST 제거 | §3 "toy 예제" 결정과 연동 |
| `configs/data/mnist.yaml`, `configs/model/mnist.yaml`, `configs/hparams_search/mnist_optuna.yaml`, `configs/experiment/example.yaml` | MNIST 의존 | `hparams_search`는 이름만 바꿔 `optuna_example.yaml`로 남기는 안도 가능 |
| `tests/test_datamodules.py` (MNIST 전제), `tests/test_sweeps.py`/`test_train.py`/`test_eval.py`의 MNIST 전제 부분 | 예제 교체 | toy 예제로 수정 |
| `README.md` (41KB, ashleve 원본 튜토리얼) | 원본 홍보/튜토리얼. 우리 구조와 달라짐 | 짧은 프로젝트 README 스켈레톤으로 교체 (§2 `README.md`) |
| `setup.py` | `train_command` 엔트리포인트용. 두 프로젝트 모두 안 씀(MCCG는 이미 삭제) | |
| `scripts/schedule.sh` | 원본 예시 스크립트. SLURM 스윕 인프라로 대체 | |
| `.github/workflows/release-drafter.yml`, `.github/release-drafter.yml`, `.github/dependabot.yml`, `.github/codecov.yml` | 비공개 연구 repo에 불필요 | **확인 필요**(§5-3) |
| `configs/logger/{aim,comet,neptune,mlflow}.yaml` | 안 쓰는 로거 | **확인 필요**(§5-4). 남길 것: `wandb`, `csv`, `tensorboard`, `many_loggers` |
| `configs/callbacks/rich_progress_bar.yaml` | sbatch 로그에 안 남는 문제(MCCG 2026-08-14 결정) → `progress_bar.yaml`(TQDM)로 교체 | |
| `.pre-commit-config.yaml`의 black / isort / docformatter / mdformat | 두 프로젝트 모두 수동 스타일 선호, 포맷터가 긴 줄을 깨뜨림(MCCG에서 이미 비활성화) | interrogate는 유지하되 MCCG 설정(`--ignore-regex=^[a-z_]` 등) 채택 |
| `environment.yaml`의 `name: myenv`, README의 conda 중심 설명 | 클러스터 venv(`/scratch2/<user>/venvs/…`) 중심으로 | conda는 대안으로만 |

**지우지 않는 것**: `configs/{debug,extras,hydra,paths,trainer,callbacks,logger}` 골격, `src/utils/*`, `.project-root`, `.env.example`, `Makefile`, `tests/helpers`, `logs/`, `data/` (`.gitkeep`).

## 2. 추가할 것 — 목표 폴더 구조

`★` = 새로 추가, `✎` = 기존 파일 수정, `·` = 그대로.

```
research-template/
├── README.md                          ✎ 짧게: 목적, 구조표, 새 프로젝트 시작 체크리스트, 실행 예시
├── CLAUDE.md                          ★ AI 규칙 + gotchas 통합 한 파일(.notes/ 폴더는 두지 않음). MCCG CLAUDE.md + gotchas.md 기반, 프로젝트 특화만 치환
├── .env.example                       ✎ DATA_DIR, CHECKPOINT_DIR, VENV, (WANDB_API_KEY) 주석
├── .gitignore                         ✎ /data/* + !.gitkeep, checkpoints, logs/, outputs/, wandb/, .env
├── .gitmodules                        ★ (빈 파일 또는 없음; third_party에 submodule 추가 시 생성)
├── .project-root                      ·
├── Makefile                           ✎ install / analyze / venv-activate-hint 추가
├── pyproject.toml                     ·
├── requirements.txt                   ✎ cu128 핀 방식(주석으로 설명) + analysis 의존성(pandas, matplotlib, scikit-learn)
├── environment.yaml                   ✎ 이름 placeholder, venv 우선 안내
├── docs/
│   ├── README.md                      ★ 문서 색인 + 파일명 규칙 + 표기 규칙([실측]/[제안]/[해석]/[미확인])
│   ├── CONTEXT.md                     ★ 도메인 용어집 템플릿 (Language / Avoid 형식)
│   ├── adr/README.md                  ★ ADR 규칙 + 템플릿 (YYMMDD-adr-주제.md)
│   ├── proposals/.gitkeep             ★ 제안/설계/예상 (실측 금지)
│   ├── experiments/.gitkeep           ★ 실행 기록/결과/해석
│   ├── papers/.gitkeep                ★ 참고 논문 PDF (로컬 보관, gitignore)
│   ├── figures/.gitkeep               ★
│   └── 261004-template-plan.md        ← 이 문서 (개편 완료 후 adr로 이동 또는 삭제)
├── third_party/                       ★ **상설 폴더 — 외부 코드(git submodule) 전용**
│   └── README.md                      ★ submodule 추가/갱신법, `--recurse-submodules`, import 경로 패턴,
│                                         "원본 수정 금지, 필요하면 fork 후 submodule URL 교체" 규칙
├── configs/
│   ├── train.yaml, eval.yaml          ✎ experiment_name, seed 기본값, `experiment/train` 그룹을 eval에도 선언
│   ├── analyze.yaml, compare.yaml     ★ (post-hoc 분석 / 다중 실험 비교 엔트리포인트용, 최소 골격)
│   ├── paths/default.yaml             ✎ run_id, ckpt_dir (CHECKPOINT_DIR env로 리다이렉트)
│   ├── hydra/default.yaml             ✎ 실행 폴더 = runs/${experiment_name}, 로그 파일에 seed 포함
│   ├── callbacks/
│   │   ├── default.yaml               ✎ filename "seed${seed}_epoch_…", monitor는 ${model.metrics.monitor_metric}
│   │   ├── model_checkpoint_last.yaml ★ MCCG에서 이식 (save_last가 안 도는 문제 우회, enable_version_counter=False)
│   │   ├── early_stopping.yaml        ✎ patience 현실화
│   │   └── progress_bar.yaml          ★ TQDM (rich_progress_bar 대체)
│   ├── metrics/classification_task.yaml ★ toy 예제용
│   ├── data/toy.yaml, model/toy.yaml  ★ §3 결정 시
│   ├── debug/smoke.yaml               ★ JointDLM에서 이식·일반화: 소량 step + 결과를 logs/smoke/로 격리
│   ├── experiment/train/<model>/<YYMMDD_topic>/<YYMMDD-name>.yaml  ★ 실험 1개 = 1파일 (README.md에 규칙)
│   ├── experiment/compare/.gitkeep    ★
│   └── (그 외 debug/extras/logger/trainer/local 은 유지)
├── src/
│   ├── train.py, eval.py              ✎ seed is not None, link_checkpoints_dir, weights_only=False
│   ├── analyze.py                     ★ 골격(체크포인트 로드 → 분석 → logs/analyze/runs/<exp>/ 저장)
│   ├── compare_experiments.py         ★ 골격
│   ├── data/                          ★ toy_datamodule.py (+ components/)
│   ├── models/                        ★ toy_module.py (+ components/)
│   ├── metrics/                       ★ metric_base.py (TaskMetrics/MetricGroup) + classification 최소셋
│   ├── analysis/README.md             ★ 재사용 분석 = src/analysis/<YYMMDD_topic>/, 만들기 전에 기존 것 먼저 확인
│   ├── studies/README.md              ★ 일회성 검증/진단 = src/studies/<YYMMDD_topic>/ (로그는 logs/studies/)
│   └── utils/
│       ├── utils.py                   ✎ link_checkpoints_dir() 추가
│       ├── callbacks.py               ★ NamedLastModelCheckpoint
│       ├── results.py                 ★ JointDLM의 results.csv 기록기를 일반화(컬럼 고정 대신 dict→append, fcntl 잠금) — §5-5
│       └── (instantiators, logging_utils, pylogger, rich_utils 그대로)
├── scripts/                           ★ **shell 전용** (.py 금지)
│   ├── README.md                      ★ 규칙: 스윕 실행기만, 일회성 검증은 studies/
│   └── sbatch/                        ★ MCCG에서 그대로 복사 (resume.sh, gpu24-cuda118, gpu48bio 포함)
│       ├── README.md                  ★ 출처/범위/갱신 방법 (JointDLM이 이미 쓰는 형식)
│       ├── template.sh                ★ 새 스윕 시작용 (복사해서 축만 수정)
│       ├── check_node_health.sh
│       ├── common/{env,sweep,jobs_per_gpu,launch_sweep,maybe_submit,resume}.sh
│       └── profiles/{cpu,gpu24,gpu24-cuda118,gpu4090,gpu48,gpu48bio,gpu96}.sh
├── results/                           ★ 정리된 결과물(results.csv 등) — 커밋 대상. 원본 로그는 logs/
│   └── .gitkeep
├── tests/                             ✎ conftest(절대 batch 수), test_metrics(reset 규칙 검증), toy 기반으로 교체
├── data/.gitkeep, logs/.gitkeep, configs/local/.gitkeep   ·
└── .github/                           ✎ test.yml, PULL_REQUEST_TEMPLATE.md만 유지
```

### 2-A. 학습 코드/설정 (코드 단계, MCCG diff에서 이식)
1. 실험 폴더 규칙: `experiment_name` + `paths.run_id` + `paths.ckpt_dir`, `hydra.run.dir=…/runs/${experiment_name}`, 파일명에 `seed${seed}_` (시드별 폴더 안 나눔).
2. `CHECKPOINT_DIR` env 리다이렉트 + `link_checkpoints_dir()` (dangling symlink 레이스 수정 포함).
3. `NamedLastModelCheckpoint` + `model_checkpoint_last.yaml` + `progress_bar.yaml`.
4. `weights_only=False` (train fit/test, eval) — torch≥2.6 재개/테스트 실패 방지.
5. `seed is not None` 체크, eval.yaml에 `experiment/train` 그룹·`seed` 선언, tests/conftest 절대 batch 수.
6. `src/metrics/metric_base.py`(+ 최소 `classification.py`) + `tests/test_metrics.py` — 구현 완료 (수동 `.reset()` 금지 규칙을 테스트로 고정)
7. `analyze.py`/`compare_experiments.py`는 도메인 로직 없이 엔트리포인트 골격만.

### 2-B. SLURM (그대로 복사, 사용자 결정)
- MCCG `scripts/sbatch/` 전체를 복사한다. 단 아래 두 가지만 **최소 수정**(복사 후 확인 요청):
  - `common/env.sh`의 `VENV` 기본값 `/scratch2/khmin1104/venvs/medical_ccg` → `.env`의 `VENV`(없으면 `/scratch2/$USER/venvs/<repo 폴더명>`)로. 프로젝트 이름이 박혀 있으면 새 프로젝트마다 틀린 venv를 쓰게 된다.
  - `profiles/gpu24.sh`의 `SLURM_EXCLUDE` 불량 노드 목록은 클러스터 공통 정보이므로 유지.
- 한계: 노드 목록은 template·MCCG·JointDLM에 3벌이 생긴다. `scripts/sbatch/README.md`에 "기준(source of truth) = template, 갱신일" 한 줄을 남겨 드리프트를 추적한다.

### 2-C. 문서 / AI 규칙 (일반화 스켈레톤, 한국어)
- **CLAUDE.md** (두 프로젝트 공통분만):
  1. 불확실하면 먼저 물어볼 것
  2. 무거운 실행은 로그인 노드 금지 → srun/sbatch (프로필 사용법, `--output`은 공유 경로, 제출 후 `squeue` 확인, `scancel`은 job ID로)
  3. "그려줘" = 실제 plot 파일
  4. 파일 위치 규칙(`scripts/`=shell, `src/analysis/`, `studies/`, `logs/studies/`, `docs/proposals`⇄`experiments` 분리)
  5. ADR 작성 기준
  6. 스모크 테스트는 `debug=smoke logger=csv`, wandb는 실제 실험에서만
- **gotchas는 CLAUDE.md §1~§3·§5로 통합**(`.notes/` 없음): 범용 항목만 — GPU 확인, wandb 끄기, 스윕 스크립트 컨벤션(같은 config → 같은 .sh, 라운드마다 덮어쓰기), 파일 위치, metrics `.reset()` 금지, 체크포인트 로드 시 hparam 기본값 주의, 실험 진행 4단계(진단→설계→판정기준 선기재→결과 기록), 결과 보고 6항목. 프로젝트 특화 항목(MILK10k 분할, 특정 venv 경로, 특정 날짜 사건)은 제외.
- **docs/**: 색인·표기 규칙·`CONTEXT.md`/`adr/README.md` 템플릿.
- **skills**: 복사하지 않음. README의 "Skills" 절에 목록과 설치법만 기록(`ce-ideate`, `grill-with-docs`, `domain-modeling`, `improve-codebase-architecture`, `setup-matt-pocock-skills`; MCCG `skills-lock.json` 참조). `baseline-adapt`는 MCCG 전용이라 제외.

## 3. toy 예제 (MNIST 제거의 부작용 처리)

MNIST를 지우면 `tests/`의 train/eval/sweep smoke test와 `make train`이 돌 대상이 없어진다. 제안: **다운로드 없는 합성 분류 데이터 + 작은 MLP**(`toy_datamodule.py`, `toy_module.py`, 각각 약 50줄)를 둔다. 새 프로젝트에서는 이 두 파일과 `configs/{data,model}/toy.yaml`만 지우고 자기 것으로 교체. (이 결정은 §5-1에서 확정 필요.)

## 4. 진행 순서

| 단계 | 내용 | 검증 |
|---|---|---|
| P0 | `git switch -c overhaul` (main 보호) | — |
| P1 코드/설정 | §1 삭제 + §2-A 이식 + toy 예제 + tests 교체 | **CPU srun** 으로 `pytest -k "not slow"` (로그인 노드 직접 실행 금지) + `debug=smoke logger=csv` 1회 |
| P2 SLURM | §2-B 복사·수정, `template.sh`로 toy 스윕 1회(`LOCAL=0`, gpu24 profile) | `squeue` RUNNING 확인, 로그 `config=` 출력, train→analyze 의존 연결 |
| P3 문서 | §2-C + README + `third_party/README.md` | 링크/경로 점검 |
| P4 마무리 | 이 계획서를 `docs/adr/261004-adr-template-v2.md`로 승격 또는 삭제, 단계별 커밋 | — |
| P5 (선택, 별도 세션) | JointDLM/MCCG를 template 기준으로 재동기화할지 결정 | — |

커밋은 P1/P2/P3 단계별로 나눈다.

## 5. 확인이 필요한 결정

1. **toy 예제**: 합성 toy 예제를 남길까(권장) / 완전 빈 스켈레톤으로 둘까?
2. **`third_party` 표기**: JointDLM 방식 `third_party/`(권장)로 통일할까? MCCG의 `baselines/`는 그대로 두고 template에는 `third_party/`만 둘까?
3. ~~**`.github/`**~~ → **결정됨(삭제 완료)**: release-drafter·dependabot·codecov를 지울까(권장), 남길까? 연구 repo가 GitHub Actions를 실제로 돌리는지도 알려주세요.
4. ~~**로거 설정**~~ → **결정됨(삭제 완료)**: aim/comet/neptune/mlflow 제거(권장)? 아니면 전부 유지?
5. **`results.py`**: JointDLM의 `results/results.csv` 방식(행 단위 append, 잠금)을 일반화해서 template에 넣을까? 전 프로젝트 공통 요구인지, JointDLM 고유인지 판단이 필요합니다.
6. **`env.sh`의 `VENV` 기본값**: "그대로 복사"에서 유일하게 벗어나는 수정입니다. 허용할까요?
7. **JointDLM 현황**: 작업 트리에 미커밋 변경(M/D 다수)이 있습니다. template 작업과 무관하게 건드리지 않습니다. 확인만 부탁드립니다.

## 6. 진행 상황과 결정 기록

브랜치 `overhaul` (main 보호). 갱신: 2026-10-04.

### 6-1. 완료

| 커밋 | 내용 |
|---|---|
| `58f3083` | MNIST(데이터모듈/모델/net/configs/hparams_search/example 실험/`test_datamodules.py`), `setup.py`, `scripts/schedule.sh`, `.github`의 release-drafter·dependabot·codecov, 로거 설정 aim·comet·neptune·mlflow, pre-commit의 black·isort·docformatter·mdformat·nbqa-black·nbqa-isort 삭제 |
| `4908416` | `notebooks/`는 `58f3083`에서 삭제, 이 커밋에서 Jupyter 관련 pre-commit 훅(nbstripout, nbqa-flake8), `.gitignore`의 Jupyter/IPython 항목, codespell의 `*.ipynb` 제외 삭제 |
| (미커밋) | `CLAUDE.md` 신규 — MCCG의 `CLAUDE.md` + `.notes/gotchas.md`를 한 파일로 통합(§2-C 참조). `.notes/` 폴더는 두지 않음 |

### 6-2. 결정 사항 (대화에서 확정)

- MNIST 제거, SLURM 인프라는 그대로 복사, 문서는 일반화한 스켈레톤, skills는 목록만 문서화, 코드 → SLURM → 문서 순.
- `notebooks/`는 쓰지 않으므로 Jupyter 관련 설정까지 함께 제거.
- `.github` 정리, 로거 정리는 권장안대로 진행(§5-3, §5-4 해소).
- `.notes/gotchas.md`는 따로 두지 않고 `CLAUDE.md`에 통합. 가져올 때 프로젝트 종속 부분(venv 경로, MILK10k 분할, 예제 명령, 날짜별 사건)만 치환하고 나머지는 원문 유지. 프로젝트별 내용은 `CLAUDE.md` §9에 모음.
- 교체가 필요한 항목은 삭제 단계에서 일부러 남김: `train.yaml`/`eval.yaml`의 mnist 참조, `tests/test_sweeps.py`, `rich_progress_bar.yaml`(→ `progress_bar.yaml`), 원본 `README.md`, `environment.yaml`의 `name: myenv`. → 현재 `train.py`/`eval.py`는 실행되지 않는 상태(P1에서 해소).

### 6-3. debug 설정 조사 결과 (smoke test는 어떤 설정을 쓰나)

- MCCG: `configs/debug/*`를 **쓰지 않음**. smoke test는 `+trainer.fast_dev_run=true logger=csv trainer=gpu`. template의 `debug/{default,fdr,limit,overfit,profiler}.yaml`은 방치 상태.
- JointDLM: 자체 제작 `debug/smoke.yaml`(100 step, 결과를 `logs/smoke/`로 격리, GPU 허용)을 `debug=smoke logger=csv`로 사용.
- `debug/default.yaml`은 callbacks·logger를 끄고 `accelerator: cpu`로 고정해서 GPU 방침과 맞지 않음.
- → template에는 `debug/smoke.yaml`(JointDLM 방식 일반화)을 추가하고, Lightning 표준 debug 설정은 유지하는 안. 아직 사용자 확정 전.

### 6-4. 남은 작업 (다음 단계)

- P1: §2-A 이식 + toy 예제 + tests 교체 + `debug/smoke.yaml` → 먼저 §5-1(toy 예제) 확정 필요
- P2: `scripts/sbatch/` 복사 (§5-6 확정 필요)
- P3: `docs/` 스켈레톤, README, `third_party/README.md`
- `CLAUDE.md` 개선 후보: JointDLM의 "로그인 서버에서 python 실행 전부 금지" 규칙(MCCG보다 엄격)을 §1에 합칠지

### 6-5. 추가 결정 (2026-10-04, 구현 중 사용자 요청)

- `docs/TODO.md`, `docs/ideation/` 삭제 (ideation은 proposals와 중복).
- `studies/` → `src/studies/` (일회성 검증도 `src` 아래 두어 import/rootutils 규칙을 `src/analysis/`와 통일). 로그는 계속 `logs/studies/`.
- `configs/experiment/compare/` 이름 유지: `analysis`는 `src/analysis`(재사용 분석 코드)·`analyze.py`와 겹쳐 혼동되고, 이 그룹의 실제 역할은 "여러 실험을 묶어 비교"이므로 `compare`가 정확. `analyze.py`는 `experiment/train`을 공유.
- `configs/experiment/train/` 계층: `<model>/<YYMMDD_topic>/<YYMMDD-name>.yaml` (MCCG 방식). 파일 이름 = `experiment_name`. 규칙은 `configs/experiment/README.md`.
- CLAUDE.md의 smoke test를 `debug=smoke logger=csv`로 갱신 (`fast_dev_run`은 더 가벼운 대안으로 병기).
- `docs/papers/` 추가: 논문 PDF 로컬 보관(`docs/papers/*` gitignore, `.gitkeep`만 추적).
- `third_party/README.md`: repo마다 자기 venv(`/scratch2/$USER/venvs/<repo>`)를 만들어 쓰는 것을 기본 규칙으로 명시.

### 6-6. 구현 중 발견한 문제

- 공유 venv `/scratch2/khmin1104/venvs/medical_ccg`가 깨져 있음 (`antlr4`, `pluggy`의 `__init__.py` 없음, 2026-10-04 19:10 수정 흔적). 수정하지 않고 검증용 venv `/scratch2/khmin1104/venvs/research-template`(CPU torch)를 새로 만들어 테스트했다.
- 복사한 `scripts/sbatch/common/env.sh`는 `.env`가 없으면 `set -e`+`pipefail` 때문에 조용히 종료됨 → `|| true` 추가. (MCCG 원본에도 같은 잠재 버그가 있으나 `.env`가 항상 있어 드러나지 않음.)
- `template.sh`는 sbatch가 스크립트를 spool로 복사하므로 `BASH_SOURCE` 대신 `SLURM_SUBMIT_DIR`로 `common/`을 찾도록 수정. `LOGGER=csv` 환경변수 지원 추가.
- 검증: `pytest`(느린 테스트 포함) train/eval/resume/ddp_sim 통과, `sh` 패키지가 없어 sweep 테스트 7개는 skip. toy 실험 `train.py` + `debug=smoke` 통과.

### 6-7. loss / metric 인터페이스 (2026-10-04)

MCCG 조사 결과: loss는 (a) `ConceptModelOutput` dataclass, (b) 580줄 forward 안의 중첩 클로저(`_loss_cat` 등) + 문자열 플래그(`continuous_loss`, `repr_indep_loss_type`), (c) components의 제각각 시그니처 함수로 섞여 있었고, metric은 `outputs` 자리에 `None`을 넘기고 `preds=`/`target=` 고정 kwargs로 받아 입력이 다른 지표마다 group 서브클래스와 `on_step` override가 필요했다.

결정: `src/losses/`(metrics와 대칭, config 그룹 `losses`) + `ModelOutput` dataclass(+`extras`). 경계 규칙은 "outputs = 모델만 만들 수 있는 값, 결정적 전처리 = loss/metric 소유, 필요한 필드는 `requires`/`preds_key`로 선언". 규약은 `src/models/README.md`, `src/losses/README.md`, `src/metrics/README.md`(요약: CLAUDE.md §3). 구현: `src/losses/`, `src/utils/model_output.py`, `configs/losses/ce.yaml`, metric의 `preds_key`/`target_key`/`name_prefix`, `tests/test_losses.py`. 아직 하지 않은 것: MCCG 쪽 이식(템플릿 스켈레톤만).

### 6-8. 제외하기로 한 것 (2026-10-04, 사용자 판단)

- `results.py`(JointDLM식 results.csv 기록기): 제외.
- `compare_experiments.py` + `configs/compare.yaml` + `configs/experiment/compare/`: 제외하고 템플릿에서 삭제. (MCCG의 비교 스크립트는 domain-shift 평가 결과에 묶여 있어 일반화할 가치가 낮다고 판단.) 6-5의 "compare 이름 유지" 결정은 이 결정으로 대체됨.

### 6-9. pre-commit 전체 제거 (2026-10-04)

MCCG는 포맷터만 껐을 뿐 나머지 훅은 켜 둔 채였지만 `.git/hooks/pre-commit`이 설치돼 있지 않아 실제로는 한 번도 돌지 않았고(JointDLM은 가져가지도 않음), 그래서 `.pre-commit-config.yaml`을 통째로 제거. 함께 제거: Makefile `format`, requirements/environment의 `pre-commit`, `.github/workflows/code-quality-*.yaml`(pre-commit을 돌리는 CI), PR 템플릿의 pre-commit 체크 항목, README 체크리스트의 `pre-commit install`. 코드 스타일은 수동 관리(CLAUDE.md의 "기존 코드 스타일을 따른다" 방침).

### 6-10. 저장소 구분, 캐시 기본 위치, wandb project 이름 (2026-10-04)

- 저장 위치 구분: `/lustre/<user>/` = 장기 보관(체크포인트 등), `/scratch2/<user>/` = 지워져도 되는 것(cache, tmp, venvs). CLAUDE.md §2와 `.env.example`에 기록.
- 사전학습 모델/데이터셋 캐시 기본 위치: `src/cache_env.py`의 `set_cache_defaults()`가 `train/eval/analyze` 시작 직후(rootutils가 `.env`를 읽은 뒤, torch/transformers가 import되기 전) 설정되지 않은 변수만 `/scratch2/$USER/cache/...`로 채운다(`HF_HOME`, `TORCH_HOME`, `TORCH_EXTENSIONS_DIR`, `TRITON_CACHE_DIR`, `XDG_CACHE_HOME`, `WANDB_CACHE_DIR`, `WANDB_DATA_DIR`, `MPLCONFIGDIR`, `PIP_CACHE_DIR`). `CACHE_DIR`로 루트를 바꾸고 개별 변수는 `.env`가 우선. 참고: 기존 `/scratch2/khmin1104/cache`에는 이미 HF 허브 형식 캐시(`datasets--*` 등)가 루트에 평평하게 들어 있어, 새 기본값(`cache/huggingface/`)은 그것을 재사용하지 않는다.
- wandb project 이름: 고정 문자열 대신 `WANDB_PROJECT`(.env) > repo 폴더 이름(`${basename:${paths.root_dir}}` resolver) > CLI `logger.wandb.project=`.

### 6-11. 산출물 구조: experiment-first, `results/` 삭제 (2026-10-04)

문제: `results/`는 "정리된 결과물, 커밋 대상"이라고만 정의돼 있고 실제 산출물은 전부 `logs/`에 생겨 역할이 애매했다. 또 task-first 구조(`logs/train/runs/<exp>`, `logs/analyze/runs/<exp>`)라 같은 실험의 학습과 분석 결과가 다른 폴더로 갈라져, 로그·체크포인트·결과를 한곳에서 볼 수 없었다.

결정: `logs/runs/<experiment_name>/{train,eval,analyze,checkpoints}`(`paths.exp_dir`). `hydra.run.dir=${paths.exp_dir}/${task_name}`, `ckpt_dir`은 task와 무관(`${CHECKPOINT_DIR|log_dir}/runs/<exp>/checkpoints`)이라 `analyze.yaml`의 ckpt 경로 override가 필요 없어졌고, `link_checkpoints_dir`은 `<exp_dir>/checkpoints`에 symlink를 둔다. `debug=smoke`는 `paths.log_dir=logs/smoke`로 같은 트리를 격리(ckpt는 `CHECKPOINT_DIR`로 새지 않음). `results/` 삭제. 그림은 한 실험이면 `logs/runs/<exp>/analyze/`, 일회성/여러 실험이면 `logs/studies/<YYMMDD_topic>/`, 문서에 실을 것만 `docs/figures/`로 복사(승격). `logs/` 이름은 유지(`runs/`·`outputs/`로의 개명은 slurm/studies/smoke 경로까지 바꾸게 되어 보류).
