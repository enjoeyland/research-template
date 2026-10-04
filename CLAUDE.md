# CLAUDE.md

이 파일은 AI 어시스턴트가 이 저장소에서 작업할 때 지킬 규칙과, 실제로 두 번 이상 반복된 실수(gotchas)를
함께 담는다. 코드를 "고치기" 전에, 그리고 뭔가 직접 실행하기 전에 읽을 것. 같은 실수를 또 하면 이 파일에
추가한다 (§8).

## 0. 요청이 불확실하거나 미정인 부분이 있을 때

코딩 요청을 처리하다가 확실하지 않거나 아직 정해지지 않은 부분(예: 어떤 방식으로 구현할지, 어떤 파일/구조를
건드릴지, 기존 로직을 바꿔도 되는지 등)이 있으면 **마음대로 판단해서 그냥 진행하지 말고**, 먼저 사용자에게
상황을 설명하고 물어볼 것. 요청 자체가 명확하지 않을 때도 마찬가지로 다시 물어보고 설명할 것.

이유: 마음대로 짜놓은 다음에 그게 잘못된 방향이었다는 걸 나중에 발견하는 게 훨씬 힘들고, 사용자를 화나게
만든다. 확인 한번 하고 넘어가는 비용이 잘못 짜놓은 걸 나중에 찾아서 되돌리는 비용보다 훨씬 싸다.

## 1. AI가 혼자 (스모크테스트/스윕/분석) 실행할 때 지킬 것 — 대부분의 사고가 여기서 남

한 번의 실행에 아래 4가지가 다 걸려있다. 하나라도 빠뜨리면 사고 난다:

- **venv**: `.env`의 `VENV`(예: `/scratch2/<user>/venvs/<project>/bin/python`)를 쓸 것 — conda 아님, 셸의
  `which python`은 무관한 환경일 수 있다. `train.py`/`eval.py`/`analyze.py` 밖에서는 `.env`가 자동 로드
  안 되니 임시 스크립트는 `set -a; source .env; set +a` 먼저 할 것.
- **GPU 필수**: 로그인/dev 셸엔 GPU 없음(`torch.cuda.is_available()` == False). login 서버 CPU로 절대
  우회하지 말 것 — 이 프로젝트 모델뿐 아니라 GPU 쓸 수 있는 모든 무거운 python 실행에 적용(TF 등도 동일).
  `scripts/sbatch/profiles/*.sh`(`gpu24`/`gpu4090`/`gpu48`/`gpu96`)로 `srun`/`sbatch`. 할당받았다고 실제로
  GPU 쓰는지 확신하지 말 것 — `torch.cuda.is_available()`/`list_physical_devices`류로 확인.
  GPU가 필요 없는 일반 스크립트(데이터 생성, 분석 등)는 `cpu.sh` 프로필로 제출한다.
- **스모크테스트 = `debug=smoke logger=csv`** (`configs/debug/smoke.yaml`): 1 epoch, train 20 / val·test 5 batch만
  돌고, early stopping을 끄고, 체크포인트·결과를 `logs/smoke/`로 격리해서 실제 실험 폴더(`logs/train/runs/`)를
  오염시키지 않는다. 체크포인트 저장→로드→test 경로까지 실제로 지나간다. 더 가벼운 확인(1 batch, 체크포인트 저장
  안 함)이 필요하면 `+trainer.fast_dev_run=true`. 둘 다 GPU 할당에서(`trainer=gpu`) 돌릴 것.
- **wandb**: 스모크테스트/디버깅/버그재현이면 `logger=csv`. 사용자가 명시적으로 실제 실험을
  요청했을 때만(`sbatch` 제출 등) `logger=wandb`.
- **데이터 분할**: 새 스윕은 저장소에 고정된 공유 split 파일 기준으로 돌린다(예: `split/<dataset>/..._5fold_seed42.csv`).
  즉석 KFold/예전 split 데이터모듈로 새 실행을 시작하지 말 것(예전 로그 보존용). 어떤 split이 기준인지는
  이 파일 아래 "프로젝트별 메모"(§9)에 적는다.
- **스윕 스크립트 컨벤션**: 같은 `configs/model/*.yaml` → 같은 `.sh` 하나만. 새 라운드로 넘어가면 그 스크립트의
  `CONFIG_DIR`/`EXPERIMENTS`를 덮어쓸 것 — 모드 토글/`case`문 추가 금지(git 히스토리가 예전 버전 보관).
  스윕 축은 CV **fold**(`FOLDS=(0 1 2 3 4)`), seed-replicate 아님 — `seed="${fold}" data.fold="${fold}"` 명시.
  **예외가 필요해 보여도 AI 혼자 판단해서** `_tmp_*.sh` **조용히 만들지 말 것 — 먼저 사용자에게 물어볼 것.**
  (안 물어보고 만들었다가 "delete once submitted"라 스스로 적어놓고도 정리 안 한 적 있음)
  **다른 세션이 지금 같은 캐노니컬 스크립트로 job을 돌리고 있어도 충돌을 고려하지 말 것 — 캐노니컬
  스크립트의 `JOB_NAME`/`CONFIG_DIR`/`EXPERIMENT_NAME`을 그냥 덮어쓰고 새로 제출할 것.**
- **제출 후 확인**: `squeue`로 실제 RUNNING인지, slurm 로그에 `config=`가 찍히는지 본 뒤에 "돌고 있다"고
  보고한다. `scancel`은 job ID로만 한다. sbatch `--output`은 세션 로컬 `/tmp`가 아니라 클러스터 공유 경로
  (이 repo 안, `logs/slurm/...`)로 잡는다 — 컴퓨트 노드의 `/tmp`는 로그인 노드와 다른 디스크다.
- **불량 노드**: 프로필의 `SLURM_EXCLUDE`(CUDA 초기화가 실패하는 노드 목록)를 반드시 `--exclude`에 넘길 것.
  새 불량 노드를 발견하면 `scripts/sbatch/check_node_health.sh`로 확인한 뒤 프로필에 근거와 함께 추가한다.

**실행 예제:**

```bash
# (1) 스모크테스트 — GPU 할당 + debug=smoke + wandb 끔
source scripts/sbatch/profiles/gpu24.sh
srun --partition="$SLURM_PARTITION" --qos="$SLURM_QOS" --gres="$SLURM_GRES" \
  --exclude="$SLURM_EXCLUDE" --time=00:15:00 \
  "$VENV/bin/python" src/train.py \
  experiment/train=<topic>/<YYMMDD_round>/<YYMMDD-name> \
  debug=smoke trainer=gpu logger=csv seed=42 data.fold=0

# (2) 실제 5-fold 스윕 — 캐노니컬 스크립트 하나, JOB_NAME/CONFIG_DIR/EXPERIMENTS만 새 라운드로 덮어써서 실행
./scripts/<model>_train.sh
```

## 2. 파일 위치: `scripts/`는 shell 전용, 분석/검증 코드는 `src/analysis/`·`src/studies/`, 로그는 `logs/`

- `scripts/*.sh` — sbatch 스윕 실행기만 있는 폴더(전부 `.sh`). 여기에 `.py` 파일 만들지 말 것.
- fold-pooling/OOF metric 계산 같은 사후 분석 Python 스크립트는 `src/analysis/`에 만들 것. 새 분석 스크립트가
  필요해 보여도 먼저 `src/analysis/`를 훑어서 비슷한 게 이미 있는지 확인할 것 — 없는 줄 알고 새로 만들었다가
  나중에 기존 것과 거의 똑같아서 다시 합친 적 있음(기존 파일에 옵션만 추가하는 걸로 정리됨).
- **학습 후 후속 조치(평가/분석)는 손으로 따로 sbatch 하지 말고, 스윕 스크립트의
  `run_analyze()`(`--dependency=afterok:<train_job>`으로 이미 자동 연결됨, `maybe_submit.sh`)에
  라운드 종류를 분기해서 넣을 것.** 새 실험 축을 추가하면 그 라운드에 맞는 후속 조치를 `run_analyze()`
  안에 조건 분기로 추가할 것 — 기존 analyze가 안 맞으면 그냥 두지 말고 라운드에 맞는 걸로 바꿀 것.
- **실험 검증 코드는 재사용 가능하면 `src/analysis/<YYMMDD_topic>/`, 일회성이면
  `src/studies/<YYMMDD_topic>/`에 둔다**(§5.4 "결과 기록" 단계에서 나오는 코드가 여기 해당).
- **스모크테스트/실행 로그 원본(`*.log`)은 `src/studies/<YYMMDD_topic>/`에 같이 두지 말고
  `logs/studies/<YYMMDD_topic>/`에 저장할 것** — `logs/`는 `.gitignore` 대상이라 소스 코드와 같은 디렉토리에
  두면 로그만 영구 미추적 상태로 섞여 있게 된다. `src/studies/` 쪽 README/스크립트 주석에서 로그를 언급할 때는
  실제 위치(`logs/studies/...`)를 명시할 것.
- 문서는 용도별로 나눈다: 제안/설계/예상은 `docs/proposals/`, 실행 기록/결과/해석은 `docs/experiments/`,
  결정 기록은 `docs/adr/`. 파일명은 `YYMMDD_주제.md`. **`proposals/`에는 실험 결과(실측 수치, 예상과의 비교)를
  쓰지 않고 `experiments/`에만 쓴다.**
- 외부 코드는 `third_party/`의 git submodule로 둔다. 원본을 직접 수정하지 말 것.

## 3. `src/metrics/*`: 수동 `.reset()` 호출 금지

- `TaskMetrics`는 메트릭 *객체 자체*를 `self.log_dict(..., on_epoch=True)`로 로깅 → Lightning이
  epoch마다 자동으로 리셋(sanity-check도 자동 격리). 수동 리셋을 추가하면 이중 리셋 경고만 날 뿐.
- `on_step`이 `self` 대신 계산된 값을 반환하도록 "단순화"하면 자동 리셋 경로가 깨져서 매 epoch 값이
  누적 평균이 되어버림(검증됨: epoch0 전부정답+epoch1 전부오답 → 0.0 아니라 0.5로 보고).
- 순수 Python(`Trainer` 없이) 직접 호출은 원래 리셋 안 됨 — 이걸로 "버그"라 오판하지 말 것, 실제
  `lightning.Trainer`로 검증할 것(`tests/test_metrics.py`). 이미 두 번(리셋 직접 추가 / 순수-Python
  테스트로 "확인") 잘못됨.

## 4. ADR (Architecture Decision Record) 작성

`docs/adr/`에 프로젝트의 굵직한 아키텍처/방법론 결정을 기록한다. 규칙과 템플릿은 `docs/adr/README.md` 참고.

굵직한 결정 — 알고리즘 선택, 모델 백본 교체, 파이프라인 구조 변경처럼 되돌리기 어렵거나 프로젝트 방향에
영향이 큰 결정 — 을 내리게 되면, 그 결정이 확정된 직후 `docs/adr/YYMMDD-adr-주제.md` 형식으로 ADR을
작성(또는 제안)할 것. 실험 config 조정, 하이퍼파라미터 튜닝처럼 작은 결정은 대상이 아니다. 애매하면 작성
여부를 사용자에게 먼저 물어본다.

## 5. 실험 진행 형식

§6이 "끝난 뒤 어떻게 **보고**하나"라면, 이건 "실험을 **어떤 형식으로 진행**하나"다. 원인을 찾는 실험이든
개선을 검증하는 실험이든 아래 네 단계를 순서대로 밟고, 각 단계의 결과물을 문서에 남길 것.

### 5.1 원인 진단

무엇이 문제인지 **측정한 숫자로** 먼저 확정한다. 가설이 그럴듯하다는 건 증거가 아니다.
원인 후보가 여럿이면 각각의 기여도를 분해해서 잰다.

### 5.2 실험 설계

한 arm에 변경 하나. 조합을 보고 싶으면 단독 arm도 같이 넣는다. 도달 목표가 되는 기준선과
문제가 재현되는 상태를 **양 끝으로 같이** 돌려서, 개선이 무엇 대비 개선인지 명확하게 한다.

### 5.3 판정 기준

결과를 **보기 전에** "어떤 관찰이 나오면 어떻게 해석하고 다음에 뭘 할지"를 표로 써둔다.
결과를 본 뒤에 기준을 만들면 원하는 결론에 맞춰 읽게 된다.

| 관찰 | 해석 | 다음 |
|---|---|---|
| 수정이 기준선 수준 회복 | 가설한 메커니즘이 맞음 | 확정 |
| 일부만 회복 | 원인이 더 있음 | 나머지 분해 |
| 변화 없음 | 접근이 부족 | 방향 전환 |

### 5.4 결과 기록

성공/실패 여부와 **왜 그런지**를 문서에 남긴다. 음성 결과도 남겨야 다음 세션이 같은 걸 다시
하지 않는다. 이전 결론을 뒤집게 되면 지우지 말고 취소선 + 정정 링크. 검증 코드·로그를
어디 둘지는 §2 참조.

### 5.5 자주 틀리는 것

- **성능이 나쁜 것 ≠ 아이디어가 틀린 것** — NaN·발산은 버그 신호로 먼저 의심할 것. 구현이
  맞다는 걸 확인하기 전에 기능을 지우지 말 것.
- **체크포인트 로드** — 체크포인트에 없는 새 hparam은 경고 없이 현재 default로 채워진다.
  옛 체크포인트를 분석할 땐 그 시점 값을 명시적으로 넘길 것.
- **checkpoint 재개/평가** — torch>=2.6은 `torch.load` 기본값이 `weights_only=True`라 hparams에 든
  `omegaconf.ListConfig` 등을 못 읽는다. 우리 자신의 체크포인트는 `weights_only=False`로 읽는다.
- **`ModelCheckpoint`의 `save_last`** — 모니터 지표가 천장에 닿아 더 이상 개선되지 않으면 "last"도
  같이 멈춘다. 마지막 epoch 체크포인트가 필요하면 `configs/callbacks/model_checkpoint_last.yaml`을 쓴다.

## 6. 실험 결과 보고 형식

실험(학습/분석 잡)이 끝나고 사용자에게 결과를 알려줄 때는 아래 6가지를 **전부** 포함해서
설명할 것 — 결과 숫자만 던지지 말 것:

1. **실험을 하게 된 이유** — 어떤 질문/의심/발견 때문에 이 실험이 필요해졌는지.
2. **예상하는 목표** — 이 실험이 확인/증명하려는 게 정확히 뭔지(성공 시 어떤 패턴이 나와야
   "목표 달성"인지 미리 정의).
3. **어떤 실험을 했는지** — 어떤 config/데이터/조건으로 뭘 돌렸는지(비교군이 있으면 비교군도).
4. **결과** — 표로 정리. 숫자 나열만 하지 말고 표로 묶어서 한눈에 비교되게 할 것. 표에 나오는
   **각 지표(컬럼)가 정확히 뭘 측정한 값인지도 같이 설명**할 것 — 표만 던지고 숫자 이름만
   보여주면 안 됨.
5. **결과가 예상 목표와 같은지/다른지** — 목표를 달성했는지, 부분적으로만 맞는지, 아예
   예상과 다른(뜻밖의) 결과가 나왔는지 명시적으로 판단해줄 것 — 판단 없이 표만 던지고 끝내지
   말 것.
6. **다음으로 해야 할 것** — 이 결과를 보고 나서 자연스럽게 이어지는 다음 스텝(추가 검증,
   확정 못 한 부분, 후속 실험 등)을 제안할 것.

## 7. "그려줘" / "plot 그려줘" 요청

"그려줘", "그래프/막대그래프/plot 그려줘"라고 하면 **Artifact(HTML)가 아니라 실제 plot 파일**(matplotlib
등으로 렌더링한 PNG/PDF 등 이미지 파일)을 만들라는 뜻이다. Artifact HTML 페이지를 만들어서 publish하지
말 것 — 이 실수를 이미 두 번 반복했다. 결과 이미지는 Read 도구로 직접 보여주거나 파일 경로를 알려주면 된다.
사용자가 명시적으로 "아티팩트로 만들어줘"/"웹페이지로" 등을 요청할 때만 Artifact를 쓴다.

## 8. 이 파일 관리

- AI가 같은 실수를 두 번 이상 반복하면 여기에 추가한다(가장 위험한 것이 위로 오도록 §1~§3 안에 배치).
- 프로젝트 특화 내용은 아래 §9에만 쓴다. 위 §0~§8은 모든 연구 프로젝트에 공통이므로 template 갱신과
  함께 유지한다.

## 9. 프로젝트별 메모 (새 프로젝트에서 채울 것)

- 프로젝트 한 줄 소개:
- venv / `CHECKPOINT_DIR` 위치:
- 기준 데이터 split:
- 프로젝트 특화 gotchas:
