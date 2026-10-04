# 261004_mccg-port-sim

MCCG의 SCM cascade 모델(손실 8개, GT/self 이중 pass, 진단값, 두 번째 metric 스트림)을 template의 `ModelOutput` /
`CompositeLoss` / `TaskMetrics` 위에서 축소판으로 다시 만들어, 새 구조가 실제로 맞고 같은 값을 내는지 확인한다.
MCCG의 실제 함수(`pairwise_margin_loss`, `continuous_column_loss`, `soft_cross_entropy`, `reconcbm_concept_bins`)는
MCCG 체크아웃에서 파일 경로로 직접 불러와 term / metric으로 감싼다.

```bash
# srun/sbatch로 (CLAUDE.md §1). 로그/보고서: logs/studies/261004_mccg-port-sim/
python src/studies/261004_mccg-port-sim/sim.py
```

결과와 해석: [docs/experiments/261004_mccg-port-simulation.md](../../../docs/experiments/261004_mccg-port-simulation.md)
