# src/losses/

loss는 **항(term) 하나 = `nn.Module` 하나**이고, 항들을 `CompositeLoss`가 가중합한다. 구성은 `configs/losses/*.yaml`에서 하고, 모델 config가
`/losses@loss: <name>`으로 불러온다. 모델 안에서는 `loss_dict = self.loss_fn(outputs, batch)` 한 줄이다.

| 파일 | 내용 |
|---|---|
| `base.py` | `LossTerm`: `forward(outputs, batch) -> scalar`, `requires = (...)`로 읽는 필드 선언 |
| `composite.py` | `CompositeLoss`: `{"loss": 가중합, "loss_<항>": 가중 전 값}`, `requires` 검사(매 호출), weight 0 항은 건너뜀 |
| `blend.py` | `PassBlend`(여러 forward pass의 가중 평균), `FieldTerm`(모델이 이미 계산한 스칼라를 항으로) |
| `classification.py` | `CrossEntropy` (새 항은 이 패턴으로) |

## 규칙

- **입력은 `ModelOutput`**(`src/utils/model_output.py`). `requires`에 적은 필드만 읽고, batch에서는 정답·표본 가중치 같은 것만 읽는다.
  필요한 필드를 모델이 안 내놓으면 어떤 항이 무엇을 요구하는지 알려 주는 `KeyError`가 난다.
- **경계 규칙**: outputs에는 모델만 만들 수 있는 값(logits, forward 중 샘플한 마스크/노이즈, 중간 feature → `extras`)만 담긴다.
  softmax, logit adjustment, 마진 계산, masked mean 같은 **결정적 전처리는 loss 항이 직접** 한다. 그래서 `_target_`만 바꿔 loss를
  교체해도 모델을 건드리지 않는다. 클래스 prior 같은 loss 고유 상태는 outputs가 아니라 **항의 생성자 인자(config)와 buffer**다.
- **확률적이거나 비싼 값을 항에서 다시 계산하지 않는다.** 마스크를 두 번 뽑으면 forward와 loss가 서로 다른 마스크를 보게 된다. forward에서 한 번
  계산해 `extras`에 둔다. 싼 결정적 파생값(argmax 등)의 중복 계산은 허용한다.
- **가중치와 구현은 config로** 바꾼다: `model.loss.terms.ce.weight=0.5`, 또는 항의 `_target_` 교체(문자열 플래그 + `assert` 대신).
  weight 0인 항은 계산도 요구도 하지 않는다(끄는 항은 모델이 필드를 안 내놔도 된다).
- **진단값은 loss가 아니다.** entropy floor 같은 측정값은 loss dict에 섞이면 pass 가중 평균에 끼어들어 어느 조건에서도 측정되지 않은 값이
  보고된다. `src/metrics`의 `FieldMeanGroup`으로 로깅한다.

## 이중 pass / 모델이 계산하는 항

- ground-truth pass와 self-predicted pass를 `w·L(gt) + (1−w)·L(self)`로 섞는 경우: 모델이 `outputs.extras["passes"] = {"gt": ModelOutput, "self": ModelOutput}`을
  내놓고, 항을 `PassBlend(term, weights={gt: w, self: 1-w})`로 감싼다. `passes`가 없으면(eval) 항을 그대로 적용한다. 중첩된 blend는 선형이라
  하나의 weight 집합으로 펼쳐 쓴다(`w·L(gt) + (1−w)((1−w2)·L(self) + w2·L(noexo))` = `{gt: w, self: (1−w)(1−w2), noexo: (1−w)w2}`).
- decoder 내부가 필요해 모델이 직접 계산하는 정규화 항(KL, orthogonality)은 `FieldTerm(key)`로 가중합에 참여시킨다.

## loss로 뺄지 모델 안에 둘지

항이 둘 이상이거나, 모델 사이에서 재사용하거나, 자체 상태/하이퍼파라미터가 있으면 이 폴더로 뺀다. 한 줄짜리 CE는 모델 안에 둬도 된다.

## 아직 규칙으로 다루지 못한 것

loss 선택이 모델 의미까지 바꾸는 경우(예: `gaussian_nll`은 head 출력을 (mean, logvar)로 재해석), loss와 metric이 같은 상수를 쓰는 경우,
표본 가중치 항. 자세한 내용: `docs/experiments/261004_mccg-port-simulation.md` §6.

## 새 항 추가

1. `LossTerm`을 상속하고 `requires`를 선언해서 `forward(outputs, batch)`를 쓴다.
2. `configs/losses/*.yaml`의 `terms`에 `weight`와 `term: {_target_: ...}`로 올린다.
3. `tests/test_losses.py`에 값·필드 요구를 확인하는 테스트를 추가한다.
