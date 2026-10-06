# 261004_mccg-port-sim

MCCG의 SCM cascade 모델(손실 8개, GT/self 이중 pass, 진단값, 두 번째 metric 스트림)을 template의 `ModelOutput` /
`CompositeLoss` / `TaskMetrics` 위에서 축소판으로 다시 만들어, 새 구조가 실제로 맞고 같은 값을 내는지 확인한다.
MCCG의 실제 함수(`pairwise_margin_loss`, `continuous_column_loss`, `soft_cross_entropy`, `reconcbm_concept_bins`)는
MCCG 체크아웃에서 파일 경로로 직접 불러와 term / metric으로 감싼다.

```bash
# srun/sbatch로 (CLAUDE.md §1). 로그/보고서: logs/studies/261004_mccg-port-sim/
python src/studies/261004_mccg-port-sim/sim.py
```

## 결과 (6개 확인 모두 통과)

- GT/self 이중 pass 가중 평균(`PassBlend`), 학습 전용 margin loss, 모델이 계산하는 정규화 항(`FieldTerm`), 진단값(`FieldMean`)을 포함한 loss 총합이
  MCCG 방식으로 따로 조립한 값과 train/eval 모두 1e-5 이내로 일치한다.
- `continuous_loss`(soft_ce / mse / gaussian_nll)는 `_target_`만 바꿔 교체되고 각각 MCCG 함수와 값이 일치한다.
- 학습에만 있는 두 번째 스트림(`gt_preds`)은 config의 group 하나로 선언되고 eval에서는 아무것도 로깅되지 않는다. 진단값은 loss에 섞이지 않는다.
- 모델이 필드를 안 내놓으면 어느 pass의 어느 필드인지 알려 주는 오류가 나고, 가중치 0인 항은 필드를 요구하지 않는다.

## 한계 (확인하지 못한 것)

- loss 선택이 모델 의미를 바꾸는 경우(`gaussian_nll`은 head 출력을 (mean, logvar)로 재해석): 모델도 그 사실을 알아야 해서 `_target_` 교체만으로는 안 풀린다.
- loss와 metric이 같은 상수(logit adjustment prior)를 쓰는 경우, 표본 가중치 항, 학습되는 파라미터를 가진 loss, GPU 성능.
- 컬럼 4개짜리 축소판이며 실제 1824줄 모듈을 옮긴 것이 아니다. MCCG 체크아웃 경로(`/home/khmin1104/workspace/Medical-CausalInference/...`)에 의존한다.
