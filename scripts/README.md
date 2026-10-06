# scripts/

**학습 스윕 실행기 전용(shell)** — 반복해서 도는 `scripts/<model>_train.sh`만 둔다. `.py` 파일을 만들지 말 것.
study나 분석의 실행 셸은 여기가 아니라 **그 코드 옆**(`src/studies/<YYMMDD_topic>/run.sh`, 반복해서 쓰면 `src/analysis/<YYMMDD_topic>/run.sh`)에 둔다.

- 인프라는 [`sbatch/`](sbatch/README.md). 새 스윕은 `sbatch/template.sh`를 `scripts/<model>_train.sh`로 복사해서 축만 고친다
  (study 실행기는 `sbatch/template_study.sh`).
- 같은 `configs/model/*.yaml` → 같은 `.sh` 하나. 새 실험 주제(`<YYMMDD_topic>`)는 그 파일의 `JOB_NAME`/`CONFIG_DIR`/`EXPERIMENTS`를 덮어쓴다 (CLAUDE.md §1).
