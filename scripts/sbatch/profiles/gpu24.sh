# ~24GB VRAM (RTX 3090 / A5000 / RTX 4090)
# Src: ImgEdit_BalModality/Trains/shell/run_train_array_24.sh
PROFILE_NAME=gpu24
SLURM_PARTITION=asus_a5000,gigabyte_a5000,suma_rtx4090,big_suma_rtx3090,base_suma_rtx3090,dell_rtx3090 
SLURM_QOS=big_qos
SLURM_GRES=gpu:1
SLURM_TIME=24:00:00
HYDRA_TRAINER=gpu
# cs-gpu-01 (part of the suma_rtx4090 partition, but a differently-named/managed node than the
# node31-40 fleet) fails CUDA init under concurrent job scheduling (2026-08-06: "CUDA unknown
# error" / "No supported gpu backend found" on 3+ concurrently-scheduled tasks landing there,
# reproduced twice; node31-40 unaffected) -- excluded until that's root-caused.
# node24 (dell_rtx3090, 8x RTX3090) shows the identical failure signature (2026-08-07: all 5
# tasks that landed there in one job, plus 1 more from a second concurrent job, all failed with
# "No supported gpu backend found"; other dell_rtx3090 nodes like node22 were fine) -- SLURM shows
# it as IDLE/healthy, so this looks like a real driver/hardware fault the scheduler doesn't know
# about, not a config quirk like cs-gpu-01. Excluded until root-caused.
# node05 (base_suma_rtx3090) and node23 (dell_rtx3090) show the same signature again (2026-08-10:
# a 25-task sweep had 17 fail with "CUDA unknown error", almost all landing on node05/node23; a
# same-size retry batch confirmed 100% failure on those two nodes vs. 100% success on node21 in
# the same batch). Excluded until root-caused.
# node23 RE-CHECKED 2026-09-07 via scripts/sbatch/check_node_health.sh: a real CUDA
# context init + matmul on node23 succeeded (RTX 3090, CUDA_OK) -- removed from the exclude list
# below. node05 was re-checked the same run and did NOT pass clean (SLURM allocated then revoked
# the job before it ran, its own prolog/health-check rejecting the node -- a stronger "still
# unhealthy" signal than a plain busy-timeout) -- stays excluded.
# node08 shows the same signature twice in one evening (2026-08-16: mask_recon 5-fold sweep's
# fold0 task failed with "CUDA unknown error" on node08 in back-to-back sweep submissions --
# optim-recipe-v1 and sparse-stage-remask-v1 -- while every other fold in both sweeps succeeded).
# Excluded until root-caused.
# node12 shows the same signature (2026-08-18: 3 concurrent fast_dev_run smoke-test tasks all
# landed on node12 and all 3 failed identically with "CUDA unknown error"; a same-batch retry with
# node12 excluded succeeded 3/3 elsewhere). Excluded until root-caused.
# node14 shows the same signature (2026-09-07: 2 independent srun analysis tasks (gate-saturation
# probe, seed3/seed4) both landed on node14 and both failed with "CUDA unknown error"; the same
# evening, a 10-task synthetic_mask_recon_train.sh array's one task that landed on node14 also
# failed identically -- a direct GPU health check via srun on the same exclude list landed on
# node21 and succeeded fine, ruling out a cluster-wide outage). Excluded until root-caused.
# node31 shows the same signature (2026-09-13: 3 separate occurrences same day -- a standalone
# srun diagnostic, then 4/10 tasks of a synthetic_mask_recon_train.sh array (tab-fused-direct-gated
# seed1/seed2, tab-fused-residual-gated seed0/seed1) all landed on node31 and all failed identically
# with "CUDA unknown error", while other tasks in the SAME array on node16/18/21 succeeded fine).
# Excluded until root-caused.
# node23 (2026-09-15): same "CUDA unknown error" signature -- 4/10 tasks of a
# synthetic_mask_recon_train.sh array (260915-...-posembed-noremask seed1/2/3/4) all landed on
# node23 and all failed identically, confirmed with check_node_health.sh
# (torch.cuda.is_available() itself fails there). Excluded until root-caused.
# node20/node09/node49: 2026-09-15 에 check_node_health.sh 의 REVOKED 를 근거로 잠시 추가했다가
# **같은 날 철회**했습니다. 철회 이유: 직후 확인한 클러스터 GPU 점유가 434개 중 422개 사용 중
# (여유 12개)로 포화 상태였고, REVOKED 는 노드 불건강이 아니라 **여유 GPU 부족으로 투기적
# allocation 이 회수된** 신호였을 가능성이 훨씬 큽니다. 위 항목들과 달리 "CUDA unknown error"
# 재현이 전혀 없었다는 점도 같은 방향입니다(node23 은 재현이 있어 유지). 포화가 아닌 시점에
# check_node_health.sh 로 다시 확인해서 그때도 실패하면 근거를 갖춰 다시 추가할 것.
# node40 (2026-09-16): "No CUDA GPUs are available" -- 260916-...-recon-necessity-neither 스모크
# 테스트가 여기 떨어져 실패, check_node_health.sh 로 재확인해도 FAIL(cuda not available).
# 근본 원인 밝혀질 때까지 제외.
SLURM_EXCLUDE=cs-gpu-01,node24,node05,node08,node12,node14,node31,node23,node40
