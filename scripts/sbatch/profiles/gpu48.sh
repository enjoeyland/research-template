# ~48GB VRAM (A6000 / RTX 6000 Ada)
PROFILE_NAME=gpu48
# gigabyte_a6000_bio 제외: 이 파티션은 AllowQos=etc_qos 라 아래 big_qos 를
# 거부한다(`Invalid qos specification`). 파티션 목록에 남겨두면 Slurm 이 하필 여기에 잡을
# 배정했을 때 영원히 (PartitionConfig) 로 대기한다 -- 실제로 두 번 당했다.
SLURM_PARTITION=gigabyte_a6000,suma_a6000,tyan_a6000,asus_6000ada
SLURM_QOS=big_qos
SLURM_GRES=gpu:1
SLURM_TIME=24:00:00
HYDRA_TRAINER=gpu
