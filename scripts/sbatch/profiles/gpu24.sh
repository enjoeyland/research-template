# ~24GB VRAM (RTX 3090 / A5000 / RTX 4090)
PROFILE_NAME=gpu24
SLURM_PARTITION=asus_a5000,gigabyte_a5000,suma_rtx4090,big_suma_rtx3090,base_suma_rtx3090,dell_rtx3090 
SLURM_QOS=big_qos
SLURM_GRES=gpu:1
SLURM_TIME=24:00:00
HYDRA_TRAINER=gpu
# Nodes excluded because CUDA init fails there ("CUDA unknown error", "No supported gpu backend found" or "No CUDA GPUs
# are available") even though SLURM shows them healthy: cs-gpu-01, node05, node08, node12, node14, node23, node24,
# node31, node40. Each one was excluded on evidence -- several tasks of one job landing on it all failing identically
# while the same job succeeded on other nodes -- and can be re-checked with scripts/sbatch/check_node_health.sh.
# Remove a node only after it passes that check, and add one only with the same kind of evidence.
# A "REVOKED" result in check_node_health.sh is NOT such evidence: it can just mean the cluster was saturated and a
# speculative allocation was taken back. For a one-off exclusion use SLURM_EXCLUDE_EXTRA (maybe_submit.sh) instead of
# editing this shared list.
SLURM_EXCLUDE=cs-gpu-01,node24,node05,node08,node12,node14,node31,node23,node40
