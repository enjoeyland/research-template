# ~24GB VRAM (RTX 3090 / A5000 / RTX 4090), same partitions/exclude list as gpu24.sh --
# for code that needs CUDA 11.x on LD_LIBRARY_PATH (e.g. an old TensorFlow). This cluster's GPU nodes default to
# CUDA 12.8, under which such a framework can silently see no GPU (no error, just no GPU used): check that it really
# uses the allocated GPU.
PROFILE_NAME=gpu24-cuda118
SLURM_PARTITION=asus_a5000,gigabyte_a5000,suma_rtx4090,big_suma_rtx3090,base_suma_rtx3090,dell_rtx3090
SLURM_QOS=big_qos
SLURM_GRES=gpu:1
SLURM_TIME=24:00:00
CUDA_MODULE=11.8
# `module load cuda/${CUDA_MODULE}` before invoking the python that needs it -- the module itself comes from
# /opt/ohpc/pub/modulefiles (shared across all nodes), so this is not a per-node install; it only needs to actually
# run inside the job.
#
# Node health, not CUDA-11.8 availability, is what varies by node here -- same excluded nodes as gpu24.sh (see its
# comments for why they are excluded). Confirmed working with `module load cuda/11.8` on a non-excluded node of this
# partition list; reuse gpu24.sh's exclude list and update this comment if a node turns out bad for cuda/11.8 only.
SLURM_EXCLUDE=cs-gpu-01,node24,node05,node23,node08
