# src/analysis/

학습이 끝난 뒤의 사후 분석 코드(fold/seed pooling, 지표 계산, 그림). 재사용 가능한 것만 두고, 일회성은
`src/studies/<YYMMDD_topic>/`에 둔다.

- 주제별로 `<YYMMDD_topic>/` 폴더를 만든다.
- 새로 만들기 전에 이 폴더를 먼저 훑어서 비슷한 게 있는지 확인한다. 있으면 옵션을 추가하는 쪽으로 합친다.
- 학습 직후 자동으로 돌릴 분석은 스윕 스크립트의 `run_analyze()`에 넣는다 (`scripts/sbatch/template.sh`).
- 결과는 `logs/analyze/runs/<experiment_name>/`에 쓰고, 정리된 표만 `results/`로 옮긴다.
