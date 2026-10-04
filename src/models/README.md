# src/models/

LightningModule을 둔다. 구조 부품(인코더, head, 마스킹 등 `nn.Module`)은 `components/`, loss는 `src/losses/`, metric은 `src/metrics/`.
참고 구현: `toy_module.py`.

## `forward`는 스텝당 `ModelOutput` 하나

`ModelOutput`(`src/utils/model_output.py`: `logits`, `target`, `preds`(기본 `logits.argmax`), `extras`)을 **한 번만** 만들고, loss와 metric이 **같은
객체**를 읽는다. 같은 값을 두 번 계산하지 않는다.

```python
outputs = self.forward(batch)                                  # 한 번만 계산
loss_dict = self.loss_fn(outputs, batch)                       # outputs를 읽기만 함 -> {"loss": ..., "loss_<항>": ...}
logged = {f"{split}/{k}": v for k, v in loss_dict.items()}
logged.update(self.metrics.on_step(split, outputs, batch, batch_idx))   # 메트릭 객체를 반환 -> log_dict(on_epoch=True)
self.log_dict(logged, on_step=False, on_epoch=True)
return loss_dict["loss"]
```

## 무엇을 outputs에 담나 (경계 규칙)

- **담는 것**: 모델만 만들 수 있는 값 — 네트워크 출력(`logits`), forward 중에 뽑은 확률적 값(샘플된 마스크, 노이즈), 비싸고 여러 곳이 쓰는 중간 feature,
  모델이 직접 계산하는 정규화 항(KL 등), 학습에만 있는 두 번째 pass의 출력(`extras["passes"]`, `extras["gt_preds"]`).
- **담지 않는 것**: softmax·argmax·logit adjustment처럼 싸고 결정적인 전처리. 그것은 쓰는 loss/metric이 직접 한다. 그래야 config에서 loss/metric을
  바꿔 끼울 때 모델을 안 건드린다.
- loss가 `requires`로 선언한 필드를 모델이 안 내놓으면 `CompositeLoss`가 어떤 항이 무엇을 요구하는지 알려 준다. 모델 README처럼 "이 모델이 내놓는
  필드" 목록을 모델 docstring에 적어 두면 loss를 고를 때 편하다.
- 학습/평가에서 내놓는 필드가 달라도 된다(예: eval에는 두 번째 pass가 없음). loss와 metric이 없는 필드를 조용히 건너뛰거나 명확히 오류를 낸다.

## 모델이 하지 않는 것

- `val/<name>_best`, `val/overfit_gap` 곡선: `MetricTrends` 콜백이 만든다(`src/metrics/README.md`). 모델에 `reset_best` 같은 호출이 필요 없다.
- metric 수동 `.reset()`: 하지 않는다(`src/metrics/README.md`).
- loss 항별 가중치/구현 선택: config(`configs/losses/`)에서 한다.

## 모델 config가 지켜야 할 것

- `metrics`(TaskMetrics, `monitor_metric`/`monitor_mode` 포함)와 `loss`(CompositeLoss)를 `defaults`로 불러온다: `/metrics@metrics:`, `/losses@loss:`.
  콜백(체크포인트, early stopping)이 `${model.metrics.monitor_metric}`을 읽으므로 모델이 그 지표를 실제로 로깅해야 한다.
