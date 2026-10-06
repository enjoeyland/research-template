# 다운스트림 프로젝트 현황

이 template을 쓰거나 쓰다가 갈라진 프로젝트. 버전은 그 프로젝트의 `.template-version`(`version:`)이고, 아직 파일이 없으면 "미기록"이다.
새 소식(프로젝트에서 발견한 문제, 프로젝트에서 먼저 만든 기능)은 여기에 적고, template에 반영하면 [CHANGELOG.md](CHANGELOG.md)에 올린다.

| 프로젝트 | 경로 | template 버전 | 메모 |
|---|---|---|---|
| Medical-CausalInference (MCCG) | `/home/khmin1104/workspace/Medical-CausalInference` | 미기록 (template의 원조, fork) | `baselines/`의 sbatch 스크립트 19개가 `scripts/sbatch/`를 쓰지 않고 각자 `#SBATCH`, 절대 경로, 낡은 불량 노드 목록을 가짐 (third_party 통일 문제). `val/overfit_gap`이 한 epoch 밀려 있을 수 있음(v1에서 `MetricTrends`로 고침). `default_resumable`이 `max_epochs`가 resume 간격과 같을 때 `state_key` 충돌 가능(template은 `ResumeModelCheckpoint`로 수정) |
| JointDLM | `/home/khmin1104/workspace/JointDLM` | 미기록 (template 이전에 `scripts/sbatch/`를 일부만 복사, 2026-10-03) | loss가 모델 안에서 샘플링과 얽힌 생성 모델이라 `ModelOutput`/`CompositeLoss` 구조에 안 맞음 → "어댑터 패턴"(v2)의 근거. `debug=smoke`, `results.csv`, `docs/` 구조를 자체 구현 |
| HMM_Watermarking | `/home/khmin1104/workspace/HMM_Watermarking` | 미기록 | `maybe_submit.sh`에 `NODELIST`를 먼저 추가(v3에 반영). `gpu24.sh`에 `node02`(CUDA unknown error) 제외를 추가했으나 template에는 **미반영**. `scripts/studies/`와 `src/studies/`를 같은 이름으로 짝지어 둠(v3에서 `src/studies/<주제>/run.sh` 방식으로 변경) |

## 기록하는 법
1. 프로젝트에 `.template-version`을 만든다(template 것을 복사하고 `version:`을 그 프로젝트가 맞춘 버전으로).
2. 프로젝트에서 `make template-status`로 뒤처진 정도를 본다.
3. 여기 표의 "template 버전"을 갱신한다.
