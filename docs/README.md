# 문서 색인

- 파일명: `YYMMDD_주제.md` (날짜_주제). ADR만 `YYMMDD-adr-주제.md`.
- 폴더:
  - `proposals/` — 제안, 설계, 예상, 문헌/코드 대조. **실험 결과(실측 수치, 예상과의 비교, 해석)는 쓰지 않는다.**
  - `experiments/` — 실행 기록, 결과, 해석. 판정 기준은 결과를 보기 전에 `proposals/`에 먼저 쓴다(CLAUDE.md §5).
  - `adr/` — 굵직한 결정 기록 ([규칙](adr/README.md)).
  - `papers/` — 참고 논문 PDF (로컬 보관, git에는 올리지 않음. 파일명 `<저자연도>_<제목요약>.pdf`).
  - `figures/` — 문서에 실린 그림(**승격된 복사본**). 시도해 본 그림은 `logs/studies/<YYMMDD_topic>/`에 두고, 문서에 실을 것만
    여기로 복사한다(`logs/`는 gitignore라 문서가 직접 링크하면 깨진다). 그림을 만든 코드 위치를 문서에 링크한다.
- 새 문서는 각 폴더의 `_TEMPLATE.md`를 복사해서 시작한다(공통 헤더: `title/created/updated/status/related`, 맨 위 "상태 (날짜)" 블록, "0. 결론 먼저").
  `proposals/_TEMPLATE.md`는 설계와 판정 기준, `experiments/_TEMPLATE.md`는 CLAUDE.md §6의 6항목이 뼈대다.
- 용어는 [CONTEXT.md](CONTEXT.md).
- 문장 표기: **[원문]** 직접 확인, **[스니펫]** 검색 요약만 확인, **[제안]** 설계·추정, **[실측]** 실행 결과,
  **[해석]**, **[미확인]**.

## 문서 목록

### proposals/

| 문서 | 내용 |
|---|---|

### experiments/

| 문서 | 내용 |
|---|---|
| [261004_mccg-port-simulation.md](experiments/261004_mccg-port-simulation.md) | MCCG SCM cascade를 `src/losses`·`src/metrics` 구조로 옮기는 시뮬레이션: 수치 동일성, 드러난 빈 곳(`PassBlend`/`FieldTerm`/`FieldMean` 추가), 한계 |
