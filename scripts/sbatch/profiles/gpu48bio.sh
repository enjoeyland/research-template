# ***작동하지 않음 -- 이 계정(ys_ms / khmin1104)으로는 쓸 수 없다*** (2026-09-20 확인, 사용자도 "bio 는 내가 못 쓰나보네")
#
# gigabyte_a6000_bio 파티션: AllowQos=etc_qos 뿐, 노드는 node44 한 대(A6000 x 8).
# 그런데 이 계정에는 etc_qos 가 붙어 있지 않아서 **어떤 QOS 로도** 거부된다:
#     srun --partition=gigabyte_a6000_bio --qos=etc_qos ...   -> Invalid qos specification
#     (big_qos / normal / base_qos / debug_qos 도 전부 같은 에러)
# gpu48.sh 헤더가 "big_qos 를 거부한다" 고만 적어뒀는데 실제로는 etc_qos 도 안 된다 -- 코드로 우회할
# 수 없는 **계정 권한 문제**다. 이 프로필을 쓰려면 관리자가 ys_ms 계정에 etc_qos 를 붙여줘야 한다
# (`sacctmgr` 은 slurmdbd 다운으로 응답하지 않아 허용 QOS 목록을 직접 조회하지도 못했다).
#
# 권한이 생기기 전에는 gpu48.sh(4개 파티션, 작동)나 gpu24.sh 를 쓸 것.
PROFILE_NAME=gpu48bio
SLURM_PARTITION=gigabyte_a6000_bio
SLURM_QOS=etc_qos
SLURM_GRES=gpu:1
SLURM_TIME=24:00:00
HYDRA_TRAINER=gpu
