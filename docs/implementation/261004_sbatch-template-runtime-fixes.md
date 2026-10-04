---
title: SLURM 스윕 템플릿이 sbatch 안에서 도는지 검증하며 찾은 문제 4건
created: 2026-10-04
updated: 2026-10-05
status: 수정됨
commits: [f0ff9e9, bf93fd2]
files: [scripts/sbatch/common/env.sh, scripts/sbatch/template.sh, configs/analyze.yaml, configs/eval.yaml, tests/test_configs.py]
related: [261004_resume-checkpoint-state-key-collision.md]
---

# SLURM 스윕 템플릿이 sbatch 안에서 도는지 검증하며 찾은 문제 4건

> **상태 (2026-10-05)**: 4건 모두 수정. 2-fold 스윕이 train 배열 → resume 체크포인트 → 의존성으로 이어진 analyze까지 통과한다.

로그인 노드에서 `bash template.sh`로 제출하고 job 로그를 읽어 찾았다. 원인은 모두 "로컬에서는 되는데 sbatch 안에서/다른 조합에서는 안 되는" 부류다.

## 1. `.env`가 없으면 스크립트가 조용히 종료 (env.sh)
- 증상: job 로그가 `N_RUNS=...` 줄 직후 끊기고 아무 오류도 없음.
- 원인: `set -euo pipefail` 아래에서 `VENV="$(grep '^VENV=' .env | tail | cut)"`의 `grep`이 `.env`가 없어 비정상 종료 → 파이프 전체 실패 → `set -e`로 조용히 종료.
- 수정: 두 `grep` 파이프에 `|| true`. MCCG 원본은 `.env`가 항상 있어 드러나지 않았다.

## 2. sbatch 안에서 `common/`을 못 찾음 (template.sh)
- 증상: `.../job2390801/sbatch/common/launch_sweep.sh: No such file or directory`.
- 원인: sbatch는 스크립트를 slurm spool 디렉토리로 복사해 실행하므로 `BASH_SOURCE` 기준 경로가 깨진다.
- 수정: job 안에서는 `SLURM_SUBMIT_DIR`(repo 루트에서 제출) 기준으로 `scripts/sbatch`를 찾고, 밖에서는 파일 경로 기준.

## 3. `eval.yaml`/`analyze.yaml`에 없는 그룹을 실험 config가 override (configs)
- 증상: `Could not override 'callbacks'. No match in the defaults list.`
- 원인: 실험 config는 train/eval/analyze가 공유하는데, `override /callbacks: ...`를 쓰는 실험에 대해 eval/analyze의 defaults에 `callbacks` 그룹이 없었다.
- 수정: 두 파일에 `callbacks: null`, (`analyze`에는 `logger: null`도) 선언. `tests/test_configs.py::test_every_experiment_config_composes_for_train_eval_and_analyze`가 모든 실험 config를 세 엔트리포인트에서 조립해 본다.

## 4. analyze가 엉뚱한 폴더에 결과를 씀 (analyze.yaml)
- 증상: 분석 결과가 `logs/runs/<타임스탬프>/analyze`로 감(실험 폴더가 아님).
- 원인: `analyze.yaml`에서 `_self_`를 맨 뒤에 둬서 자기 `experiment_name: ${paths.run_id}`가 실험 config의 값을 덮어썼다(paths override를 위해 뒤에 뒀던 것이 이유가 없어진 뒤에도 남음).
- 수정: `_self_`를 앞으로. 위 테스트가 `cfg.experiment_name == 파일 이름`을 확인해 이 부류를 잡는다.

## 반복 방지
공유 설정(여러 엔트리포인트가 공유하는 defaults)을 바꾸면 모든 실험 config가 모든 엔트리포인트에서 조립되는지 테스트로 확인한다. 로컬 실행으로 sbatch를 대신 검증하지 않는다.
