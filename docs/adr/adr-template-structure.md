# ADR: research-template의 폴더 구조와 규칙

## Status
승인됨. (날짜 없는 파일명은 이 문서가 템플릿 자체의 기준 문서이기 때문이다. 이후 변경은 이 문서를 고치고 아래 "변경 이력"에 적는다.)

## Context
새 연구 프로젝트를 만들 때마다 구조를 처음부터 다시 정하고, 이전 프로젝트에서 검증된 것(실행 폴더 규칙, SLURM 스윕, 재개, 규칙 문서)을
손으로 복사해 왔다. 복사본은 곧 갈라졌다 (JointDLM은 Medical-CausalInference의 `scripts/sbatch/`를 일부만 복사했고, `CLAUDE.md`·`docs/`·`studies/`를
다시 만들었다). 이 템플릿은 두 프로젝트에서 검증된 것을 한 곳에 모아, 새 프로젝트를 복사하자마자 같은 규칙으로 시작하게 한다.
(ashleve/lightning-hydra-template를 fork.)

## Decision

### 1. 폴더 구조

```
research-template/
├── CLAUDE.md              AI 작업 규칙 + 반복된 실수(gotchas)를 한 파일에. §8에 프로젝트별 메모
├── README.md              새 프로젝트 시작 체크리스트, 구조 표
├── .env.example           CHECKPOINT_DIR, VENV, WANDB_*, 캐시 경로 (데이터는 data/ 링크, DATA_DIR 없음)
├── configs/               Hydra 설정
│   ├── train.yaml, eval.yaml, analyze.yaml      엔트리포인트별 기본 조합
│   ├── experiment/train/<model>/<YYMMDD_topic>/<YYMMDD-name>.yaml   실험 1개 = 파일 1개
│   ├── data/ model/ losses/ metrics/ callbacks/ logger/ trainer/ paths/ hydra/ extras/ debug/ hparams_search/
├── src/
│   ├── train.py, eval.py, analyze.py            Hydra 엔트리포인트
│   ├── data/            LightningDataModule (+ components/)
│   ├── models/          LightningModule (+ components/: 인코더·head 같은 구조 부품)
│   ├── losses/          LossTerm, CompositeLoss, PassBlend, FieldTerm
│   ├── metrics/         TaskMetrics, MetricGroup, 분류 지표, FieldMean
│   ├── utils/           콜백(MetricTrends 등), ModelOutput, instantiators, 로깅 유틸
│   ├── analysis/<YYMMDD_topic>/   재사용 가능한 사후 분석 코드
│   └── studies/<YYMMDD_topic>/    일회성 검증/진단 코드
├── scripts/               shell 전용. sbatch/ = SLURM 스윕 인프라 (common/, profiles/, template.sh)
├── data/                  무거운 데이터셋의 심볼릭 링크만 (gitignore, README와 .gitkeep만 추적)
├── third_party/           외부 코드(git submodule) 전용. repo마다 자기 venv
├── docs/                  proposals/ experiments/ implementation/ adr/ figures/ papers/(PDF, gitignore) + CONTEXT.md, README.md
├── logs/                  실행 산출물 (gitignore). runs/<experiment>/{train,eval,analyze,checkpoints}가 한곳에,
│                          slurm/ smoke/ studies/ 는 따로. results/ 폴더는 없다
└── tests/                 pytest (설정 조합, 학습/평가/재개, metrics, losses, trends, utils)
```

### 2. 규칙

**실행과 산출물**
- **한 실험 = 폴더 하나(experiment-first)**: `logs/runs/<experiment_name>/` 아래에 `train/`(hydra config, 로그, csv metrics), `eval/`, `analyze/`(csv·png),
  `checkpoints/seed<N>_epoch_XXX.ckpt`가 모인다. seed는 폴더가 아니라 파일명에 들어간다. `experiment_name`은 실험 config 파일 이름(확장자 제외)과 같게 둔다.
  `CHECKPOINT_DIR`(lustre)가 있으면 체크포인트는 `$CHECKPOINT_DIR/runs/<experiment_name>/checkpoints`에 쌓이고 `logs/runs/<exp>/checkpoints`가 그쪽을 가리키는
  symlink가 되어, 한 폴더에서 로그·체크포인트·결과가 다 보인다. `debug=smoke`는 `logs/smoke/`로 같은 트리를 격리한다.
- **그림/표**: 한 실험의 분석 결과는 `logs/runs/<exp>/analyze/`, 여러 실험에 걸치거나 일회성인 것은 `logs/studies/<YYMMDD_topic>/`(코드는 `src/studies/`).
  문서에 실을 것만 `docs/figures/`로 복사(승격)한다. `logs/`는 gitignore이므로 문서가 `logs/`를 직접 링크하지 않는다.
- 실험 config 경로는 `experiment/train/<model>/<YYMMDD_topic>/<YYMMDD-name>.yaml`. `<model>`은 같은 `configs/model/*.yaml`을 쓰는 계열이고, 스윕
  스크립트 하나(`scripts/<model>_train.sh`)가 한 `<model>`을 담당한다.
- `scripts/`는 shell 전용(sbatch 스윕 실행기). 분석은 `src/analysis/`, 일회성은 `src/studies/`, 로그는 `logs/studies/`. 새로 만들기 전에 기존 파일부터 확인한다.
- 학습 후 후속 조치는 스윕 스크립트의 `run_analyze()`에 넣어 `--dependency=afterok`로 자동 연결한다. 죽은 학습은 같은 명령으로 재제출하면
  `callbacks=default_resumable`의 `seed<N>_resume.ckpt`와 저장된 wandb id로 이어서 돈다.
- 스모크 테스트는 `debug=smoke logger=csv` (결과는 `logs/smoke/`로 격리), wandb는 실제 실험에서만.

**모델 / loss / metric 인터페이스** (각 폴더 README: `src/models/`, `src/losses/`, `src/metrics/`)
- `forward`는 스텝당 `ModelOutput`(`logits`, `target`, `preds`, `extras`)을 한 번만 만든다. loss와 metric이 같은 객체를 읽는다.
- outputs에는 모델만 만들 수 있는 값(logits, forward 중 샘플한 마스크, 중간 feature)만 담고, 결정적 전처리는 loss/metric이 소유한다. 그래야 config에서
  loss/metric을 바꿔 끼워도 모델을 건드리지 않는다.
- loss는 `LossTerm`(읽는 필드를 `requires`로 선언) + `CompositeLoss`(가중합, 항별 값을 `loss_<항>`으로 반환). 가중치와 항 구현은 config에서 정한다.
  metric은 `preds_key`/`target_key`/`name_prefix`로 읽을 필드를 config에서 정한다. 이중 pass는 `PassBlend`, 모델이 직접 계산하는 항은 `FieldTerm`,
  진단값은 loss가 아니라 `FieldMean` metric.
- `val/<name>_best`, `val/overfit_gap` 곡선은 `MetricTrends` 콜백이 이미 로깅된 `train/*`·`val/*`에서 만든다(같은 epoch끼리 짝, sanity-check 격리, resume 유지).

**저장 위치**
- `/lustre/<user>/` = 장기 보관(체크포인트, 남길 결과), `/scratch2/<user>/` = 지워져도 되는 것(cache, tmp, venvs).
- 사전학습 모델/데이터셋 캐시는 `~/.cache`가 아니라 `/scratch2/$USER/cache`로 간다. 코드가 아니라 `.env.example`의 `HUGGINGFACE_HUB_CACHE`와 `TMPDIR`로 정하고(`${USER}` 확장),
  `train/eval/analyze`가 `.env`를 먼저 읽으므로 라이브러리 import 전에 적용된다.
- 무거운 데이터셋은 repo 안에 복사하지 않고 `data/<dataset>` 심볼릭 링크. datamodule은 `${paths.data_dir}/<dataset>`를 `require_data_path()`로 연다.
- 외부 코드는 `third_party/`의 git submodule이고, 원본은 직접 수정하지 않는다. 각 repo는 자기 venv(`/scratch2/$USER/venvs/<repo>`)를 만든다.

**문서**
- `proposals/`(설계, 예상, 판정 기준)와 `experiments/`(실행 기록, 결과, 해석)를 분리한다. 실험 결과는 `proposals/`에 쓰지 않는다.
- `implementation/`은 코드 수정의 이야기(증상, 원인과 기각한 가설, 근거, 수정, 검증)를 주제별 파일로 남긴다. 코드 주석에는 불변 조건/함정만 1~3줄 쓰고 길면 이 문서를 가리킨다.
  한 줄 요약은 git 커밋 메시지, 성능 숫자와 해석은 `experiments/`.
- 파일명 `YYMMDD_주제.md`. 굵직한 결정은 `adr/`에 (이 문서처럼 템플릿 기준 문서는 날짜 없이).

**검증 도구**
- 포맷터·pre-commit·CI 품질 워크플로는 두지 않는다(두 프로젝트 모두 훅이 설치되지 않아 실제로는 돌지 않았다). 테스트는 `pytest`(`make test`).

### 3. 사용법 (최상위 `README.md`는 프로젝트 README로 교체되므로 여기가 기준)

**새 프로젝트 시작 체크리스트**
1. 이 repo를 복사해서 새 repo를 만든다(GitHub "Use this template" 또는 `git clone` 후 remote 교체).
2. 프로젝트 이름을 정한다: `make rename NAME=<project-name>`(README 제목, `environment.yaml`, `.env.example`의 `PROJECT_NAME`을 바꾼다).
   그다음 `cp .env.example .env`. `PROJECT_NAME` 한 줄이 체크포인트 폴더(`CHECKPOINT_DIR`), sbatch venv 이름, wandb project 기본값을 정한다.
3. `/scratch2/$USER/venvs/<PROJECT_NAME>`에 venv를 만들고 `requirements.txt`를 설치한다(GPU 클러스터에서는 torch를 먼저, 헤더 설명 참고).
4. 예제(toy)를 내 것으로 교체한다: `src/data/toy_datamodule.py`, `src/models/toy_module.py`, `configs/{data,model,metrics,losses}/toy*.yaml`,
   `configs/experiment/train/toy/`, `configs/hparams_search/toy_optuna.yaml`. 모델 config에는 `metrics`(`monitor_metric`/`monitor_mode` 포함)와 `loss`를 유지한다.
5. `CLAUDE.md` §8(프로젝트별 메모)과 `docs/CONTEXT.md`를 채운다. 외부 코드는 `third_party/`에 submodule로 추가한다.
6. `make test`.

**실행** (GPU 클러스터에서는 로그인 노드에서 직접 돌리지 말고 `srun`/`sbatch`, CLAUDE.md §1)
```bash
python src/train.py experiment/train=toy/261004_example/261004-toy-example                      # 학습 + 테스트
python src/train.py experiment/train=toy/261004_example/261004-toy-example debug=smoke logger=csv  # smoke test (logs/smoke/)
python src/eval.py  experiment/train=toy/261004_example/261004-toy-example ckpt_path=<ckpt>      # 평가
./scripts/sbatch/template.sh                                                                     # 스윕 예시 (scripts/<model>_train.sh로 복사해서 사용)
make test                                                                                        # 느린 테스트 제외
```

**Skills (선택)**: 저장소에 복사하지 않고 필요할 때 설치한다(`npx skills add <source>`).

| skill | 출처 | 용도 |
|---|---|---|
| `ce-ideate`, `ce-ideate-with-docs` | `everyinc/compound-engineering-plugin` | 아이디어 발산/정리 |
| `domain-modeling`, `grill-with-docs`, `improve-codebase-architecture`, `setup-matt-pocock-skills` | `mattpocock/skills` | 용어/설계 정리, 구조 개선 |

### 4. 의도적으로 넣지 않은 것
MNIST 예제(합성 toy 예제로 대체), notebooks, `setup.py`, 사용하지 않는 로거(aim/comet/neptune/mlflow), release-drafter/dependabot/codecov,
JointDLM식 `results.py`, 실험 비교 스크립트(`compare_experiments.py`), skills 복사본(목록만 README에 기록).

## Consequences
- 새 프로젝트는 복사 직후 같은 실행 폴더·스윕·재개·loss/metric 구조·문서 규칙을 갖고, 프로젝트별 차이는 `CLAUDE.md` §8과 `docs/CONTEXT.md`에만 쓴다.
- 대가: 구조 규칙이 많아 처음 읽는 비용이 있고, 템플릿이 갱신되면 기존 프로젝트에는 수동으로 반영해야 한다(`scripts/sbatch/README.md`에 template이 기준임을 명시).
- 모델-loss 결합(loss 선택이 head 출력 의미를 바꾸는 경우), loss와 metric이 공유하는 상수, 표본 가중치 항은 아직 구조가 규칙으로 다루지 않는다
  (`docs/experiments/261004_mccg-port-simulation.md` §6).
- `scripts/sbatch/`의 SLURM 불량 노드 목록과 파티션은 클러스터 고유 정보이므로 다른 클러스터에서는 `profiles/`를 고쳐야 한다.

## 변경 이력
- 2026-10-04: 최초 작성.
- 2026-10-05: 사용법(새 프로젝트 체크리스트, 실행 명령, skills)을 이 문서에 추가 — 최상위 README가 프로젝트 README로 교체되기 때문.
- 2026-10-05: `docs/implementation/`(코드 수정 이력) 추가. 캐시 위치를 `src/cache_env.py` 대신 `.env.example`로 이동.
- 2026-10-04: 모델/loss/metric 규약을 `src/models|losses|metrics/README.md`로 옮기고 CLAUDE.md에서는 삭제(이후 섹션 번호가 한 칸씩 당겨짐: 문서화 §3, 실험 진행 §4, 보고 §5, 그려줘 §6, 파일 관리 §7, 프로젝트별 메모 §8).
- 2026-10-04: 실행 산출물을 task-first(`logs/<task>/runs/<exp>`)에서 experiment-first(`logs/runs/<exp>/{train,eval,analyze,checkpoints}`)로 변경, `results/` 폴더 삭제, 그림 승격 규칙 추가.
