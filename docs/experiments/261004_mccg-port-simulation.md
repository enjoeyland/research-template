# MCCG 이식 시뮬레이션 (src/losses + src/metrics 구조)

코드: `src/studies/261004_mccg-port-sim/sim.py`. 실행: CPU `srun`, 6개 확인 모두 통과(`logs/studies/261004_mccg-port-sim/report.json`).

## 1. 왜 했나
새로 설계한 `ModelOutput` / `CompositeLoss` / metric 구조가 MCCG에서 실제로 맞는지는 설계만으로는 알 수 없었다. 가장 단순한 CBM이 아니라 **가장 복잡한 모델**(1824줄 `TabImgSCMCascadeModule`, 580줄 `forward` 안의 중첩 클로저로 loss 계산)이 구조에 들어가는지가 관건이다.

## 2. 예상 목표 (결과를 보기 전에 정한 기준)
| 관찰 | 해석 |
|---|---|
| MCCG와 같은 loss 값(train, eval)이 나오고, 항·스트림·진단값이 의도대로 로깅됨 | 구조가 MCCG의 loss 조립을 표현할 수 있음 |
| 일부 기능을 표현할 수 없음 | 구조에 빠진 부품이 있음 → 부품 추가 |
| 값이 다름 | 구조(또는 이식)에 오류 |

## 3. 한 일
SCM cascade의 축소판 `MiniCascade`(표 컬럼 4개 중 연속형 1개, 이미지 패치, GT pass + self pass)를 새 구조로 만들었다. MCCG의 실제 함수를 불러와 wrapping했고, MCCG forward의 조립 방식(`lambda × (w·gt + (1−w)·self)`)을 별도로 다시 구현한 기준값과 비교했다.

## 4. 결과
| 확인 | 결과 | 의미 |
|---|---|---|
| 학습 3~4 epoch: 항(`loss_image/cat/task/exogenous_kl`)·진단·gt 스트림·`val/macro_f1_best`·`val/overfit_gap` 로깅 | 통과 (36개 키, train/loss 4.5→3.3) | 이름 필터(`if k not in (...)`) 없이 항별 곡선이 나옴 |
| gt 스트림은 학습에만 존재 | `train/gt/acc` 있음, `val/gt/*`·`test/gt/*` 없음 | 필드가 없으면 조용히 로깅 안 함 |
| 진단값이 loss에 안 섞임 | `entropy_floor`는 `diag/*` metric으로만 로깅 | MCCG의 "진단 vs loss" 혼동 방지 |
| 수치 동일성 | train 4.915937 == 4.915937, eval 4.854240 == 4.854240 (1e-5 이내) | GT/self blend + 학습 전용 margin loss + eval CE 전환까지 동일 |
| `continuous_loss` 교체 | `_target_`만 바꿔 soft_ce 1.203 / mse 1.054 / gaussian_nll 1.044, 각각 MCCG 함수와 일치 | 문자열 플래그 + `assert` 불필요 |
| 모델이 필드를 안 내놓을 때 | `PassBlend: pass 'self' ... lacks ['exogenous_kl']` | 어느 pass의 어느 필드인지 알려 줌 (아래 §5-①) |
| weight 0 항 | 계산도 요구도 안 함 | 끄는 항은 모델이 필드를 안 내놔도 됨 |
| 모든 파라미터에 gradient | 통과 | `PassBlend`/`CompositeLoss`가 역전파를 막지 않음 |

## 5. 예상과 같은가
**대체로 같다.** 구조가 MCCG의 loss 조립을 표현했고 값도 일치했다. 다만 시뮬레이션이 **실제 설계의 빈 곳을 두 군데 드러냈고 고쳤다**:
1. `CompositeLoss`가 `requires`를 첫 호출에서만 검사해서, 이후 호출에서 필드가 빠지면 불친절한 `TypeError`가 났다 → 매 호출 검사로 변경. `PassBlend` 안에서 필드가 빠졌을 때는 pass와 필드를 이름으로 알려 주도록 수정(테스트 추가).
2. 이전 설계로는 표현할 수 없던 것: **이중 pass 가중 평균**, **모델이 직접 계산하는 정규화 항(KL, orthogonality)**, **진단 metric** → 범용 부품 `PassBlend`, `FieldTerm`, `FieldMean`/`FieldMeanGroup`을 template에 추가했다.

또 forward hook(`_capture_cat_logits`)으로 빼내던 `cat_logits`가 이제 `outputs.extras`에서 바로 읽히고, 두 번째 스트림(`gt_preds`)이 서브클래스·`on_step` override 없이 config로 선언된다.

## 6. 한계 (시뮬레이션이 확인하지 못한 것)
- **loss 선택이 모델 의미를 바꾸는 경우**: MCCG의 `continuous_loss="gaussian_nll"`은 head 출력의 의미를 (logit → mean, logvar)로 바꿔서 `forward`도 `continuous_head_to_dist`로 이를 알아야 한다. 이번에는 loss만 바꿨고, 이런 모델-loss 결합은 `_target_` 교체만으로는 안 풀린다. 공유 config 키(`${model.continuous_mode}`를 loss와 모델이 함께 읽음)가 필요하다.
- **loss와 metric이 같은 상수를 쓰는 경우**: concept accuracy의 logit adjustment가 loss의 prior와 같아야 한다. 각각 config에서 받으면 어긋날 수 있어서 한 곳에서 interpolation으로 공유해야 한다.
- **시뮬레이션 밖**: 표본 가중치(`sw`, 모든 항이 batch에서 읽음), 두 이미지용 `masked_patch_mse_split`, 적대적/CKA 계열 `repr_indep` 항(학습되는 파라미터를 가진 loss), `test_tab_mask_all`, GPU 성능(MCCG의 sync-free 마스킹, §23).
- **규모**: 컬럼 4개 장난감이고, 실제 1824줄 모듈을 옮긴 것이 아니다. 실제 이식은 `forward`를 forward + 항으로 쪼개고 `__init__`의 loss 가중치 하이퍼파라미터 약 30개를 config로 옮기는 작업이며, 이번에 하지 않았다.

## 7. 다음
1. 실제 모델 하나(예: 구조가 가장 단순한 `tabimg_posthoc`)를 MCCG 쪽에서 실제로 옮겨 값 동일성을 확인.
2. §6의 모델-loss 결합(공유 config 키)과 loss·metric 공유 상수 패턴을 `CLAUDE.md` §3.1에 규칙으로 추가.
3. 표본 가중치 항을 가진 term 하나를 시뮬레이션에 추가해 batch에서 읽는 경로를 확인.
