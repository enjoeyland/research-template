# scripts/sbatch

SLURM 스윕 인프라. Medical-CausalInference(`medical_concept_causal_graph/scripts/sbatch/`, 2026-10-04 기준)에서
**그대로 복사**했다. 이 template가 기준(source of truth)이며, 여러 프로젝트에 복사본이 생기므로 갱신은 여기서
하고 필요한 프로젝트에 반영한다.

복사 후 바꾼 것은 하나뿐이다: venv 기본값 (`common/env.sh`, `check_node_health.sh`, `resume.sh`)을 특정 프로젝트의
`medical_ccg` 대신 `$VENV` > `.env`의 `VENV=` > `/scratch2/$USER/venvs/<repo 폴더명>` 순서로 정한다.

| Path | What |
|---|---|
| `common/env.sh` | repo 루트로 이동, venv 활성화, 체크포인트 경로 helper |
| `common/{sweep,jobs_per_gpu,maybe_submit,launch_sweep}.sh` | run_id 디코딩, GPU당 동시 실행 수, `--dependency=afterok`로 train → analyze 자동 제출 |
| `common/resume.sh` | 중간 재개 + wandb run id |
| `profiles/*.sh` | `cpu`, `gpu24`, `gpu4090`, `gpu48`, `gpu96` (+ `gpu24-cuda118`: TF 전용, `gpu48bio`: 권한 확인 필요) |
| `check_node_health.sh` | `SLURM_EXCLUDE`에 넣을 불량 노드 재확인 |
| `template.sh` | 새 스윕 시작용 (복사해서 축만 수정) |

로그: `logs/slurm/<날짜>/<JOB_NAME>_<jobid>*.log`. `SLURM_EXCLUDE`(CUDA 초기화가 실패하는 노드)는 클러스터 공통
정보이므로 `profiles/*.sh`에 근거 주석과 함께 유지한다. 노드가 불량이면 근거를 갖춰 추가하고, 이후 정상으로
확인되면 근거와 함께 뺀다.
