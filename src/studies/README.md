# src/studies/

일회성 검증/진단 코드를 `<YYMMDD_topic>/`로 둔다 (재사용 가능한 분석은 `src/analysis/<YYMMDD_topic>/`).

- 폴더마다 `README.md`: 무엇을 확인하려 했는지, 실행 방법, 결론, 결과 문서(`docs/experiments/...`) 링크.
- 실행 로그(`*.log`)는 여기에 두지 말고 `logs/studies/<YYMMDD_topic>/`에 저장한다 (`logs/`는 gitignore 대상).
- 새로 만들기 전에 `src/analysis/`에 비슷한 게 있는지 먼저 확인한다 (CLAUDE.md §2).
