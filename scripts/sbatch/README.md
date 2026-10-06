# scripts/sbatch

SLURM 스윕 인프라. 이 template가 기준(source of truth)이며, 여러 프로젝트에 복사본이 생기므로 갱신은 여기서 하고 필요한 프로젝트에 반영한다.

venv는 `$VENV` > `.env`의 `VENV` > `/scratch2/$USER/venvs/<PROJECT_NAME>` 순서로 정한다(`common/env.sh`, `check_node_health.sh`, `resume.sh`).
`.env`의 `${USER}`·`${PROJECT_NAME}` 같은 참조는 `env.sh`가 `PROJECT_NAME`, `CHECKPOINT_DIR`, `VENV` 세 키에 한해 확장해서 읽는다(파일 전체를 source하지 않는다).

| Path | What |
|---|---|
| `common/env.sh` | repo 루트로 이동, venv 활성화, 체크포인트 경로 helper |
| `common/{sweep,jobs_per_gpu,maybe_submit,launch_sweep}.sh` | run_id 디코딩, GPU당 동시 실행 수, `--dependency=afterok`로 train → analyze 자동 제출 |
| `common/preflight.sh` | train job을 제출하기 **직전** 검사: 시작 예상 시각(`--test-only`), 동시 실행 경고, `NODELIST` 확인 |
| `common/resume.sh` | 중간 재개 + wandb run id |
| `profiles/*.sh` | `cpu`, `gpu24`, `gpu4090`, `gpu48`, `gpu96` (+ `gpu24-cuda118`: TF 전용, `gpu48bio`: 권한 확인 필요) |
| `check_node_health.sh` | `SLURM_EXCLUDE`에 넣을 불량 노드 재확인 |
| `template.sh` | 새 스윕 시작용 (복사해서 축만 수정) |

로그: `logs/slurm/<날짜>/<JOB_NAME>_<jobid>*.log`. `SLURM_EXCLUDE`(CUDA 초기화가 실패하는 노드)는 클러스터 공통
정보이므로 `profiles/*.sh`에 근거 주석과 함께 유지한다. 노드가 불량이면 근거를 갖춰 추가하고, 이후 정상으로
확인되면 근거와 함께 뺀다.

## 제출 전 검사 (`common/preflight.sh`)

train job을 제출하기 직전에 `maybe_submit.sh`가 아래를 확인한다. 프로필이 필요한 것만 켠다(프로필마다 초기화되므로 다른 프로필로 새지 않는다).

| 환경변수 / 프로필 설정 | 동작 |
|---|---|
| `SLURM_CHECK_START=1` (프로필) | 제출 전에 `sbatch --test-only`로 **예상 시작 시각**을 출력한다. 요청이 잘못됐으면 큐에 넣기 전에 여기서 실패한다. **gpu48은 GPU를 받기 어려우므로 기본으로 켜져 있다**: 대기 시간이 길면 gpu24/gpu96으로 바꾸는 것을 고려한다. |
| `TEST_ONLY=1` (환경변수, 모든 프로필) | 예상 시작 시각만 출력하고 **제출하지 않고 종료**한다: `TEST_ONLY=1 PROFILE=gpu48 ./scripts/<model>_train.sh` |
| `SLURM_WARN_CONCURRENCY=N` + `SLURM_WARN_AFTER=HH:MM:SS` (프로필) | 동시에 N개 이상 실행될 수 있고 시간 제한이 그보다 길면 **경고**한다. gpu48은 `4` / `03:00:00`: **동시 4개 이상이 3시간을 넘기면 중간에 멈출 수 있다.** `MAX_GPUS=3`으로 동시 수를 낮추거나 `TIME=03:00:00`으로 줄이고, 긴 런은 `callbacks=default_resumable`로 재개 가능하게 한다. 동시 수는 `--array`의 `%MAX_GPUS`에서 센다. |
| `NODELIST=node45` (환경변수) | train job을 그 노드(들)에서만 돌린다(호스트 RAM이 많이 필요한 job 등). 프로필의 파티션에 없는 노드면 경고한다. analyze job에는 적용하지 않는다. |
| `SLURM_EXCLUDE_EXTRA=nodeXX` (환경변수) | 프로필의 제외 목록에 **이번 제출에만** 노드를 더한다(공유 프로필을 한 번의 실패로 고치지 않기 위해). |

경고와 예상 시각은 제출을 막지 않는다(`TEST_ONLY=1`을 빼면 그대로 제출된다).
