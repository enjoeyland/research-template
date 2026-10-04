# research-template

Lightning + Hydra 기반 연구 프로젝트 템플릿. [ashleve/lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template)를
fork해서, 실제 연구 프로젝트(Medical-CausalInference, JointDLM)에서 검증된 것들을 반영했다.
**새 연구 프로젝트를 빠르게 세팅**하는 것이 목적이다. AI 어시스턴트 작업 규칙은 [CLAUDE.md](CLAUDE.md).

## 새 프로젝트 시작 체크리스트

1. 이 repo를 복사해서 새 repo를 만든다 (GitHub "Use this template" 또는 `git clone` 후 remote 교체).
2. `cp .env.example .env` → `CHECKPOINT_DIR`, `VENV` 등 채우기.
3. venv 만들고 `requirements.txt` 설치 (헤더 설명 참고, GPU 클러스터에서는 torch를 먼저 설치).
4. 예제(toy)를 내 것으로 교체:
   `src/data/toy_datamodule.py`, `src/models/toy_module.py`, `configs/{data,model}/toy.yaml`,
   `configs/experiment/train/toy/`, `configs/hparams_search/toy_optuna.yaml`.
   모델 config에는 `metrics.monitor_metric` / `metrics.monitor_mode`를 유지한다 (체크포인트/early stopping이 읽는다).
5. `CLAUDE.md` §9(프로젝트별 메모)와 `docs/CONTEXT.md`를 채운다. 외부 코드는 `third_party/`에 submodule로 추가한다.
6. `pre-commit install`, `make test`.

## 구조

| Path | 내용 |
|---|---|
| `configs/` | Hydra 설정 (train / eval / analyze, data, model, trainer, callbacks, logger, paths, debug) |
| `configs/experiment/train/` | 실험 1개 = yaml 1개, `<topic>/<YYMMDD_round>/<YYMMDD-name>.yaml` ([규칙](configs/experiment/README.md)) |
| `configs/experiment/compare/` | 여러 실험을 묶어 비교하는 설정 (`configs/compare.yaml`용) |
| `src/train.py`, `eval.py`, `analyze.py` | Hydra 엔트리포인트 |
| `src/data/`, `src/models/` | LightningDataModule / LightningModule (toy 예제 포함) |
| `src/analysis/<YYMMDD_topic>/` | 재사용 가능한 사후 분석 코드 |
| `src/studies/<YYMMDD_topic>/` | 일회성 검증/진단 코드 (로그는 `logs/studies/...`) |
| `scripts/` | **shell 전용**. `scripts/sbatch/`는 SLURM 스윕 인프라 ([README](scripts/sbatch/README.md)) |
| `third_party/` | 외부 코드(git submodule) 전용 ([README](third_party/README.md)) |
| `results/` | 정리된 결과물(표 등). 원본 로그는 `logs/`(gitignore) |
| `docs/` | `proposals/` 설계, `experiments/` 결과, `adr/` 결정, `figures/` ([색인](docs/README.md)) |
| `tests/` | pytest (설정 조합, 학습/평가 smoke test) |

체크포인트는 `logs/train/runs/<experiment_name>/checkpoints/seed<N>_epoch_XXX.ckpt` (`.env`의 `CHECKPOINT_DIR`이 있으면
그쪽). 한 실험 = 폴더 하나, seed는 파일명에 들어간다.

## 실행

GPU 클러스터에서는 로그인 노드에서 직접 돌리지 말고 `srun`/`sbatch`로 실행한다 (CLAUDE.md §1).

```bash
python src/train.py experiment/train=toy/261004_example/261004-toy-example   # 학습 + 테스트
python src/train.py experiment/train=toy/261004_example/261004-toy-example debug=smoke logger=csv   # smoke test (logs/smoke/)
python src/eval.py  experiment/train=toy/261004_example/261004-toy-example ckpt_path=<ckpt>   # 평가
./scripts/sbatch/template.sh                                                # 스윕 예시 (복사해서 사용)
make test                                                                    # 느린 테스트 제외
```

## Skills (선택)

저장소에 복사하지 않고 필요할 때 설치한다 (`npx skills add <source>`, 설치 목록은 `skills-lock.json`):

| skill | 출처 | 용도 |
|---|---|---|
| `ce-ideate`, `ce-ideate-with-docs` | `everyinc/compound-engineering-plugin` | 아이디어 발산/정리 |
| `domain-modeling`, `grill-with-docs`, `improve-codebase-architecture`, `setup-matt-pocock-skills` | `mattpocock/skills` | 용어/설계 정리, 구조 개선 |

## 원본 template

기반: ashleve/lightning-hydra-template (MIT). 원본의 상세 튜토리얼(Hydra 설정 조합, 로거, 테스트 등)은 원 저장소 README를 참고.
