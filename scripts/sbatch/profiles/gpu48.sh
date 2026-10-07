# ~48GB VRAM (A6000 / RTX 6000 Ada)
PROFILE_NAME=gpu48
# gigabyte_a6000_bio 제외: 이 파티션은 AllowQos=etc_qos 라 아래 big_qos 를
# 거부한다(`Invalid qos specification`). 파티션 목록에 남겨두면 Slurm 이 하필 여기에 잡을
# 배정했을 때 영원히 (PartitionConfig) 로 대기한다 -- 실제로 두 번 당했다.
SLURM_PARTITION=gigabyte_a6000,suma_a6000,tyan_a6000,asus_6000ada
# base_qos: `sbatch --test-only` 기준 예상 시작이 big_qos(2026-10-09 07:15)보다 base_qos(2026-10-08 01:19)가 약 30시간 빨랐다.
# 1-GPU job이면 base_qos가 낫다. 다른 QOS가 필요하면 제출할 때 `QOS=big_qos ./scripts/<run>.sh`로 덮어쓴다.
SLURM_QOS=base_qos
SLURM_GRES=gpu:1
SLURM_TIME=24:00:00
HYDRA_TRAINER=gpu
# 48GB GPUs are comparatively hard to get: check the wait before queuing (preflight.sh prints `sbatch --test-only`'s estimated start
# time; `TEST_ONLY=1 PROFILE=gpu48 ./scripts/<run>.sh` shows it without submitting). The wait can be long enough that another
# profile (gpu24 / gpu96) is the better choice.
SLURM_CHECK_START=1
# With 5 or more allocations running at once, jobs that run past ~3 hours may be stopped midway: preflight.sh warns when a sweep
# would run that many at once with a longer time limit. Keep MAX_GPUS below 5 (4 or fewer at once) or use callbacks=default_resumable.
SLURM_WARN_CONCURRENCY=5
SLURM_WARN_AFTER=03:00:00
