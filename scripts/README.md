# scripts/

**shell 전용** — sbatch 스윕 실행기만 둔다. `.py` 파일을 만들지 말 것 (분석은 `src/analysis/`, 일회성은 `src/studies/`).

- 인프라는 [`sbatch/`](sbatch/README.md). 새 스윕은 `sbatch/template.sh`를 `scripts/<model>_train.sh`로 복사해서 축만 고친다.
- 같은 `configs/model/*.yaml` → 같은 `.sh` 하나. 새 라운드는 그 파일의 `JOB_NAME`/`CONFIG_DIR`/`EXPERIMENTS`를 덮어쓴다 (CLAUDE.md §1).
