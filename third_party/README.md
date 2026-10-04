# third_party/

이 프로젝트가 참고하거나 가져다 쓰는 **외부 코드**를 git submodule로 둔다. 연구가 바뀌어도 계속 두는 폴더다.

## 규칙

- 원본을 직접 수정하지 않는다. 고쳐야 하면 GitHub에서 fork한 뒤 submodule URL을 fork로 바꾸고, 바꾼 내용(패치 목록)을 아래 "현재 목록" 표에 적는다.
- **우리가 쓰는 코드(어댑터, 실행 스크립트, 결과 수집)는 submodule 안이 아니라 우리 repo에 둔다**: 어댑터는 `src/models/`, 실행은 `scripts/<repo>_run.sh`,
  결과 수집은 `src/analysis/`. submodule 안에 스크립트를 커밋하면 원본을 갱신할 때 fork와 충돌한다.
- 이 폴더의 파일은 우리 테스트 대상이 아니다.

## 실행도 "한 실험"이다 (폴더를 더 만들지 않는다)

third_party를 돌린 것도 다른 실험과 똑같이 `logs/runs/<experiment_name>/`(`train/`, `analyze/`, `checkpoints/`)에 모은다. `third_party/` 하위 폴더는 만들지 않는다.

- 어떤 repo인지는 실험 config 경로의 `<model>` 자리로 구분한다: `configs/experiment/train/<repo>/<YYMMDD_topic>/<YYMMDD-name>.yaml`.
- 어떤 버전을 썼는지는 실행 로그 첫 줄에 남긴다: `echo "third_party/<repo> @ $(git -C third_party/<repo> rev-parse HEAD)"`.

## 어떻게 가져올까: 세 가지 중에서 고른다

| 상황 | 방식 | 어디서 돌고 무엇이 우리 구조와 통일되나 |
|---|---|---|
| loss가 모델과 깔끔히 분리돼 있고 metric이 배치 단위 | **우리 구조에 맞춰 어댑터** (`ModelOutput` → `CompositeLoss`, `TaskMetrics`) | 우리 `train.py`, 체크포인트, wandb, 스윕 전부 그대로 |
| **loss가 모델 안에서 노이즈 샘플링 등과 얽혀 있고, 평가가 생성(샘플링) 기반** | **어댑터 패턴** ([src/models/README.md](../src/models/README.md)의 "패턴 2": `loss(net, batch)`와 `evaluate(net, cfg)`만 제공) | 우리 `train.py`와 스윕. 평가는 `eval`/`analyze` 단계에서 호출 |
| 자체 학습 루프, config 체계, 데이터 파이프라인이 있어 우리 쪽에 끼우기 어렵다 | **독립 실행** (그 repo를 그대로 돌림) | 우리 sbatch 인프라, 산출물만 `logs/runs/<exp>/`로 모음 |

앞의 두 방식은 third_party를 **우리 venv에서 import**하므로 그 repo의 의존성(torch 버전 등)이 우리 venv와 맞아야 한다. 맞지 않으면 세 번째 방식으로 간다.
어느 쪽인지 모르겠으면 사용자에게 먼저 물어본다.

### 독립 실행: 우리 sbatch 인프라를 그대로 쓴다 (통일성이 깨지는 곳이라 규칙이 많다)

repo마다 `#SBATCH` 헤더와 경로를 직접 쓴 스크립트를 따로 만들면 불량 노드 목록, 파티션, 경로, 로그 위치가 스크립트마다 어긋난다.
**`#SBATCH` 헤더를 직접 쓰지 말고** `scripts/sbatch/template.sh`를 `scripts/<repo>_run.sh`로 복사해서 `run_one()`만 바꾼다 (프로필, `SLURM_EXCLUDE`,
`--dependency`, 재제출은 `common/*.sh`가 처리한다).

```bash
# scripts/<repo>_run.sh 의 run_one() 안
: "${VENV:=/scratch2/${USER}/venvs/<repo>}"      # (파일 맨 위, launch_sweep 를 source 하기 전에) repo 전용 venv
...
exp_name="<YYMMDD-name>"
echo "third_party/<repo> @ $(git -C third_party/<repo> rev-parse HEAD)"
unset SLURM_PROCID                                 # DeiT/BEiT 계열 등: SLURM 단일 GPU job을 distributed 실행으로 오인하는 repo가 많다
cd third_party/<repo>
python <그 repo의 학습 진입점> ... \
  --output_dir "${REPO_ROOT}/logs/runs/${exp_name}/train" \
  --ckpt_dir  "$(checkpoint_dir "${exp_name}")"    # 인자 이름은 repo마다 다름. 산출물은 우리 실험 폴더로 보낸다
```

- 로그와 체크포인트, 결과를 submodule 안(`third_party/<repo>/logs`, `outputs`)에 쌓지 않는다. 위처럼 `logs/runs/<exp>/`로 보낸다.
- 환경 땜질(`unset SLURM_PROCID`, 캐시 경로 등)은 스크립트마다 흩어 놓지 말고 같은 형태로 쓴다. 같은 땜질이 세 곳 이상에서 반복되면 `scripts/sbatch/common/`에
  함수로 올리자고 제안한다.
- 결과(지표)는 repo마다 형식이 다르므로, 비교에 쓸 값은 `src/analysis/`의 수집 스크립트가 `logs/runs/<exp>/`에서 읽어 한 표로 모은다. submodule 안에 추출 스크립트를 두지 않는다.

## 환경: 독립 실행하는 repo는 repo마다 venv를 따로 만든다

외부 repo는 우리 프로젝트와 의존성(torch, numpy 등 버전)이 다르고 서로도 충돌할 수 있으므로, **독립 실행하는 third_party repo는 자기 venv를 만들어서 실행한다.**
우리 프로젝트 venv에 그 repo의 requirements를 섞어 설치하지 않는다.

```bash
python -m venv /scratch2/$USER/venvs/<repo>      # 이름은 third_party 폴더 이름과 같게
source /scratch2/$USER/venvs/<repo>/bin/activate
pip install -U pip
pip install -r third_party/<repo>/requirements.txt   # 또는 그 repo의 README 설치법
```

- venv는 repo 안이 아니라 `/scratch2/$USER/venvs/`에 둔다 (공유 스토리지, git 대상 아님).
- 어댑터로 우리 venv에서 import하는 경우에만, 그 repo가 필요로 하는 패키지를 **우리 `requirements.txt`에 명시적으로 추가**한다(의존성이 맞는지 먼저 확인).
- sbatch로 돌릴 때는 `VENV=/scratch2/$USER/venvs/<repo>`를 지정한다 (`scripts/sbatch/common/env.sh`가 `$VENV`를 받는다).
- 새 venv를 만들면 아래 "현재 목록" 표에 venv 이름과 설치 방법을 적는다.

## 추가 / 갱신

```bash
git submodule add https://github.com/<owner>/<repo>.git third_party/<repo>
git submodule update --init --recursive           # clone 직후
git clone --recurse-submodules <this-repo-url>    # 처음부터 받을 때
git -C third_party/<repo> checkout <commit> && git add third_party/<repo>   # 버전 고정
```

## import 예시 (어댑터에서)

```python
import sys
sys.path.insert(0, "third_party/<repo>")   # 패키지로 설치되지 않는 코드
import <module>                             # 무거운 import는 그 모델을 실제로 쓸 때(함수 안)만 한다
```

## 현재 목록

| 폴더 | 출처 | 용도 | 방식 (어댑터 / 독립 실행) | venv | 패치(fork한 경우) |
|---|---|---|---|---|---|
