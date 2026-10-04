# ~96GB VRAM (RTX PRO 6000 Blackwell) — a cu128 venv is required
# Src: ImgEdit run_train_array_2pergpu.sh; suma_pro6000 retired → gigabyte_pro6000
PROFILE_NAME=gpu96
SLURM_PARTITION=asus_pro6000,gigabyte_pro6000
SLURM_QOS=pro6000_qos
SLURM_GRES=gpu:1
SLURM_TIME=24:00:00
HYDRA_TRAINER=gpu
