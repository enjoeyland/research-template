---
title: <실험 제목>
created: YYYY-MM-DD
updated: YYYY-MM-DD
status: 진행 중          # 진행 중 | 확정 | 기각 | 대체됨
related: []              # 예: [../proposals/YYMMDD_xxx.md, ../adr/adr-xxx.md]
code: []                 # 예: [src/studies/YYMMDD_topic/, configs/experiment/train/<model>/<YYMMDD_topic>/]
---

# <실험 제목>

> **상태 (YYYY-MM-DD)**: 지금 어디까지 돌았고, **지금 말할 수 있는 것**이 무엇인지 2~3줄. 노이즈 안의 차이면 그렇게 쓴다.
> (갱신할 때마다 날짜와 함께 이 블록과 위의 `updated:`를 고친다.)

## 0. 결론 먼저

| 시점 | 진단 / 주장 | 상태 |
|---|---|---|
| YYYY-MM-DD | (한 줄) | ✅ 확정 / ⚠️ 부분 / ❌ 기각 |

이전 결론을 뒤집을 때는 지우지 말고 취소선(~~이전 주장~~)과 정정 링크를 남긴다. 음성 결과도 남긴다.

## 1. 실험을 하게 된 이유
어떤 질문/의심/발견 때문에 이 실험이 필요해졌나. 설계 문서가 있으면 링크(`../proposals/...`).

## 2. 예상하는 목표
무엇을 확인하려는 실험이고, 성공이면 어떤 패턴이 나와야 "목표 달성"인가. **결과를 보기 전에 정한** 판정 기준 표를 여기에 옮기거나 링크한다.

## 3. 한 실험
- config / 데이터 / 조건, 비교군, seed·fold
- 실행 명령 (`scripts/<model>_train.sh`의 `JOB_NAME`/`EXPERIMENTS`, job ID)
- 코드: `src/analysis/...` 또는 `src/studies/...`, 로그: `logs/runs/<experiment_name>/` 또는 `logs/studies/<YYMMDD_topic>/`

## 4. 결과

| 조건 | 지표 A | 지표 B | n(seed) |
|---|---|---|---|
| | | | |

**표의 각 지표가 무엇을 측정한 값인지**를 한 줄씩 쓴다(이름만 던지지 않는다). 그림은 `../figures/`로 **승격한 복사본**을 링크한다
(`logs/`는 gitignore라 직접 링크하면 깨진다).

## 5. 예상과 같은가
목표를 달성했나, 부분적인가, 예상과 다른 결과인가를 **명시적으로 판단**한다. 표만 던지고 끝내지 않는다.

## 6. 다음
이 결과에서 자연스럽게 이어지는 것(추가 검증, 확정 못 한 부분, 후속 실험).
