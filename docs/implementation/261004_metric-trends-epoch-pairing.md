---
title: val/overfit_gap이 한 epoch 밀려 있던 문제 (best/gap을 MetricTrends 콜백으로 이동)
created: 2026-10-04
updated: 2026-10-05
status: 수정됨
commits: [9b15f8e]
files: [src/utils/callbacks.py, src/metrics/metric_base.py, src/models/toy_module.py, configs/callbacks/metric_trends.yaml, tests/test_metric_trends.py]
related: [../experiments/261004_mccg-port-simulation.md]
---

# val/overfit_gap이 한 epoch 밀려 있던 문제 (best/gap을 MetricTrends 콜백으로 이동)

> **상태 (2026-10-05)**: 수정됨. `val/<name>_best`와 `val/overfit_gap`을 `TaskMetrics`가 아니라 `MetricTrends` 콜백이 만든다. 같은 epoch끼리 짝지어짐을 `Trainer`로 검증했다.
> MCCG의 `TaskMetrics`에는 같은 한 칸 밀림이 있다고 판단하지만 MCCG의 로그로 확인하지는 않았다(§6).

## 1. 증상
toy 3 epoch 로그(`metrics.csv`)에서 epoch 0의 `val/overfit_gap`이 비어 있고, epoch 1의 값이 `-0.255 = 0.745 - 1.0`이었다. 0.745는 **epoch 0의 train acc**, 1.0은 epoch 1의 val acc.
즉 val(N)이 train(N−1)과 짝지어지고 있었다.

## 2. 원인
| 가설 | 확인 방법 | 결과 |
|---|---|---|
| 지표 계산 부호 오류 | 부호는 `max`에서 train − val로 의도대로 | ❌ 값의 부호는 맞음 |
| `on_train_epoch_end`가 `on_validation_epoch_end`보다 먼저 실행된다는 가정(MCCG 주석)이 틀림 | 로그의 train 값 시점 비교, `Trainer`로 짠 시나리오 테스트 | ✅ 원인: Lightning은 validation을 epoch 끝에서 먼저 돌리고, 그 뒤에 `on_train_epoch_end`가 불린다 |

## 3. 근거
기존 구현은 `TaskMetrics.on_epoch_end("train")`에서 `_last_train_score`를 캐시하고 `on_epoch_end("valid")`에서 gap을 계산했다. validation 훅은 train 훅보다 먼저 불리므로
검증 시점에는 직전 epoch의 train 값이 캐시돼 있다. `MetricTrends`를 `on_train_epoch_end`에서 읽게 하니(validation이 끝난 뒤라 `callback_metrics`에 train(N), val(N)이 모두 있다)
스크립트 값과 일치했다.

## 4. 수정
- `TaskMetrics`에서 `best_score`, `_last_train_score`, `reset_best()`를 제거. 이제 `monitor_metric`/`monitor_mode`만 가진다.
- `MetricTrends(monitor, mode)` 콜백 추가: `on_train_epoch_end`에서 `trainer.callback_metrics`를 읽어 `val/<name>_best`(러닝 best)와 `val/overfit_gap`(양수 = 과적합)을 로깅.
- 부수 효과: sanity-check는 `on_train_epoch_end`에 도달하지 않아 best를 오염시키지 않고, best가 `state_dict`로 체크포인트에 들어가 resume 후에도 곡선이 이어진다. wandb에는 `define_metric(summary=...)`를 건다.
- wandb 웹에서 곡선으로 보는 것이 목적이라 summary만 남기는 안은 택하지 않았다.

## 5. 검증
`tests/test_metric_trends.py`(실제 `Trainer`): 스크립트된 train/val 값으로 (a) best가 러닝 max이고 sanity 값(99)을 무시, (b) gap이 같은 epoch의 train − val이며 epoch 0부터 존재, (c) `min` 모드에서 부호와 best, (d) `state_dict` 왕복. toy 4 epoch 실제 로그에서도 epoch 0의 gap = −0.255로 같은 epoch 짝.

## 6. 영향 범위 / 남은 위험
- MCCG의 `TaskMetrics`(`val/overfit_gap`)는 같은 순서 가정을 한다. 과거 wandb 곡선의 gap은 한 epoch 밀린 값일 수 있다 (MCCG 로그로는 확인하지 않았다).
- `check_val_every_n_epoch > 1`이면 val 값이 stale이라 콜백은 val이 갱신된 epoch만 곡선을 업데이트한다(코드에 처리, 테스트는 없음).

## 7. 반복 방지
Lightning 훅 순서(validation → `on_train_epoch_end`)를 가정하는 코드는 `Trainer`로 짠 테스트로 확인할 것(순수 Python 호출로 검증하지 않는다).
