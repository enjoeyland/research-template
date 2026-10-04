# src/metrics/

Metric은 `TaskMetrics`(= `MetricGroup`들의 묶음) 하나로 모델에 주입한다. 구성은 `configs/metrics/*.yaml`에서 하고, 모델 config가
`/metrics@metrics: <name>`으로 불러온다. 체크포인트/early stopping/`MetricTrends`가 읽는 `monitor_metric`·`monitor_mode`도 여기서 정한다.

| 파일 | 내용 |
|---|---|
| `metric_base.py` | `MetricHandler`(on_step / on_epoch_end), `MetricGroup`(train/valid/test ModuleDict), `TaskMetrics` |
| `classification.py` | torchmetrics 기반 handler: `MulticlassAccuracy`, `MulticlassF1Score` (새 지표는 이 패턴으로 추가) |
| `groups.py` | `ClassificationMetricGroup`, `FieldMeanGroup`(진단 스칼라) |
| `scalar.py` | `FieldMean`: `outputs[key]` 스칼라의 epoch 평균 |

## 수동 `.reset()` 호출 금지 (이미 두 번 틀렸다)

- `on_step`은 계산된 값이 아니라 **메트릭 객체 자신(`return self`)을 반환**하고, 모델은 그것을 `self.log_dict(..., on_epoch=True)`로 로깅한다.
  그러면 Lightning이 epoch 경계마다 계산하고 **자동으로 리셋**한다(sanity-check 배치도 자동으로 격리). 수동 리셋을 추가하면 이중 리셋 경고만 난다.
- `on_step`이 `self` 대신 계산된 값을 반환하도록 "단순화"하면 자동 리셋 경로가 깨져서 매 epoch 값이 누적 평균이 된다
  (검증됨: epoch0 전부 정답 + epoch1 전부 오답 → 0.0이 아니라 0.5로 보고).
- `Trainer` 없이 순수 Python으로 `on_step`/`on_epoch_end`를 직접 부르면 원래 리셋이 안 된다. 이걸로 "버그"라 오판하지 말고, 실제
  `lightning.Trainer`로 검증할 것 (`tests/test_metrics.py`). 이미 두 번(리셋을 직접 추가 / 순수 Python 테스트로 "확인") 잘못됐다.

## 읽을 필드는 config로 정한다

metric은 모델이 만든 `ModelOutput`(`src/utils/model_output.py`)을 `on_step(split, outputs, batch, ...)`로 받아 **읽을 필드를
`preds_key` / `target_key`로** 정한다. 고정 kwargs(`preds=`, `target=`)가 아니다.

- 입력이 다른 **두 번째 스트림**(예: 학습에만 있는 `gt_preds`)은 서브클래스나 `on_step` override가 아니라 **config에 group을 하나 더 선언**한다:

  ```yaml
  metric_groups:
    - _target_: src.metrics.ClassificationMetricGroup
      num_classes: 3
    - _target_: src.metrics.ClassificationMetricGroup
      num_classes: 3
      name_prefix: "gt/"          # 로깅 이름이 겹치지 않게
      preds_key: gt_preds
  ```

- **한 metric 객체에는 한 스트림만** 넣는다(epoch 동안 상태를 쌓기 때문에 섞으면 평균이 섞인다). 스트림이 다르면 객체를 따로 만든다.
- 필드가 없는 스텝(예: eval에는 없는 두 번째 pass)에서는 `on_step`이 `None`을 반환하고 아무것도 로깅하지 않는다.
- **전처리는 metric이 직접** 한다(예: AUROC가 확률이 필요하면 `logits`를 읽어 안에서 softmax). 모델이 metric마다 입력을 가공해 주지 않는다.
- **진단값**(entropy floor, gate temperature 등 loss가 아닌 측정값)은 `FieldMeanGroup`/`FieldMean`으로 로깅한다. loss dict에 섞지 않는다.

## `val/<name>_best`, `val/overfit_gap`은 여기서 만들지 않는다

이 곡선(wandb에서 그래프로 보려는 용도)은 `MetricTrends` 콜백(`src/utils/callbacks.py`, `configs/callbacks/metric_trends.yaml`)이 이미 로깅된
`train/*`·`val/*`에서 만든다. `on_train_epoch_end`에서 계산하므로 train(N)과 val(N)이 **같은 epoch끼리** 짝지어지고(validation 훅에서 읽으면
train(N-1)과 짝지어진다), sanity-check 값이 best에 섞이지 않으며, best가 체크포인트에 들어가 resume 후에도 곡선이 이어진다.
양수 = 과적합(max 지표는 train − val, min 지표는 val − train). 테스트: `tests/test_metric_trends.py`.

## 새 metric 추가

1. `classification.py`의 패턴(`_OutputsHandler` + torchmetrics 클래스)으로 handler를 만든다.
2. `groups.py`의 group에 등록하거나 새 group을 만든다.
3. `configs/metrics/*.yaml`의 `metric_groups`에 올린다.
4. `tests/test_metrics.py`에 **실제 `Trainer`를 거치는** 테스트를 추가한다(epoch 격리 확인).
