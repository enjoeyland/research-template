# ~24GB VRAM (RTX 3090 / A5000 / RTX 4090), same partitions/exclude list as gpu24.sh --
# for baselines/concept-quality's TensorFlow 2.7.0 (OIS/NIS), which needs CUDA 11.x on
# LD_LIBRARY_PATH (this cluster's GPU nodes default to CUDA 12.8, under which
# tf.config.list_physical_devices("GPU") silently returns [] -- no error, just no GPU used).
# See baselines/concept-quality/NOTES.md for the confirmed fix and an example srun command.
PROFILE_NAME=gpu24-cuda118
SLURM_PARTITION=asus_a5000,gigabyte_a5000,suma_rtx4090,big_suma_rtx3090,base_suma_rtx3090,dell_rtx3090
SLURM_QOS=big_qos
SLURM_GRES=gpu:1
SLURM_TIME=24:00:00
CUDA_MODULE=11.8
# `module load cuda/${CUDA_MODULE}` before invoking the concept-quality env's python -- the
# module itself comes from /opt/ohpc/pub/modulefiles (shared across all nodes), so this is not a
# per-node install; it only needs to actually run inside the job.
#
# Node health, not CUDA-11.8 availability, is what varies by node here -- same excluded nodes as
# gpu24.sh (cs-gpu-01/node24/node05/node23/node08 fail CUDA init regardless of module version, see
# gpu24.sh's comments for the incident history). Confirmed working 2026-08-17 with
# `module load cuda/11.8` on an unexcluded node from this partition list (TF saw the allocated
# GPU); no evidence yet that any specific *other* node in this list is bad for cuda/11.8
# specifically -- reuse gpu24.sh's exclude list and update this comment if one shows up.
SLURM_EXCLUDE=cs-gpu-01,node24,node05,node23,node08
