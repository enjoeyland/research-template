# src/studies/

일회성 검증/진단을 `<YYMMDD_topic>/` 폴더 하나에 둔다. **코드(`.py`), 실행 셸(`run.sh`), README가 한 폴더에 같이 있다.**

```
src/studies/<YYMMDD_topic>/
├── README.md     무엇을 확인하려 했는지, 실행 방법, 결론, 결과 문서(`docs/experiments/...`) 링크
├── main.py       (필요한 만큼의 .py)
└── run.sh        SLURM 실행 셸: scripts/sbatch/template_study.sh를 복사해서 STUDY, ITEMS, run_one만 고친다
```

- 실행 셸은 `scripts/`가 아니라 **study 폴더 안**에 둔다(`scripts/`에는 학습 스윕 실행기만). `#SBATCH` 헤더를 직접 쓰지 않고 템플릿이
  공용 인프라(프로필, 불량 노드 목록, `logs/slurm`, 재제출, `TEST_ONLY=1`)를 쓴다. 템플릿의 repo 루트 탐색은 어느 깊이에서든 `.project-root`를 찾는다.
- 실행 로그(`*.log`)와 **그려 본 그림/표(png, csv)** 는 여기에 두지 말고 `logs/studies/<YYMMDD_topic>/`에 저장한다(`logs/`는
  gitignore 대상). 문서에 실을 것만 `docs/figures/`로 복사(승격)해서 문서가 그 복사본을 링크한다.
- 새로 만들기 전에 `src/analysis/`에 비슷한 게 있는지 먼저 확인한다 (CLAUDE.md §2).
- **반복해서 쓰게 되면 `src/analysis/<YYMMDD_topic>/`로 승격한다**(`run.sh`를 포함해 폴더째 옮기고 README에 입력과 출력을 적는다).
