#!/bin/bash
# GPU node health check -- verifies whether specific nodes can actually initialize CUDA right now,
# independent of any particular training job. Built 2026-09-07 against gpu24.sh's SLURM_EXCLUDE
# list (grown to 7 nodes via a "fails once/twice -> excluded permanently, never re-checked" policy,
# see that file's own comments for each exclusion's evidence) -- this script lets a node be
# re-checked instead of staying excluded forever once a transient/since-fixed issue clears, rather
# than relying on the next unlucky sweep to rediscover whether it's still bad. Generalized
# 2026-09-08 to take PROFILE instead of hardcoding gpu24, so the same script works for any
# profiles/*.sh with a SLURM_EXCLUDE list (gpu4090.sh, gpu24-cuda118.sh, ...).
#
# For each node given (or, with no args, every node in ${PROFILE}.sh's SLURM_EXCLUDE):
#   1. `sinfo` its current SLURM state (idle/down/drained/allocated) -- DOWN/DRAIN is reported
#      directly without wasting an srun attempt (SLURM wouldn't schedule there anyway).
#   2. `srun --nodelist=<node>` a real CUDA init -- not just `torch.cuda.device_count()` (which can
#      report a device without actually initializing a context, per the "CUDA unknown error"
#      failures this checks for) but an actual context init + a real device tensor op
#      (`torch.randn(...).cuda()` + matmul + sync), the same call path training crashes on.
#   3. Bounded by an outer `timeout` (not just srun's own --time), since a healthy-but-currently-
#      BUSY node (SLURM state "allocated", not "down") would otherwise make srun queue and this
#      script hang waiting for it to free up -- that case is reported as INCONCLUSIVE (busy right
#      now, not proven bad), not FAIL.
#   4. PASS/FAIL/DOWN/INCONCLUSIVE summary per node at the end.
#
# Usage:
#   ./scripts/sbatch/check_node_health.sh                 # checks gpu24.sh's current SLURM_EXCLUDE list
#   ./scripts/sbatch/check_node_health.sh node24 node05   # checks just these nodes (still under gpu24's partition/qos)
#   PROFILE=gpu4090 ./scripts/sbatch/check_node_health.sh # checks gpu4090.sh's SLURM_EXCLUDE, under gpu4090's partition/qos
#
# This only CHECKS -- it never edits any profile file. Removing a PASSing node from a profile's
# SLURM_EXCLUDE is a separate, deliberate edit (per this repo's own convention of documenting each
# exclusion's evidence in a comment -- do the same for a removal).
set -uo pipefail  # not -e: one node's failure must not stop the rest

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_PROFILES_DIR="${_SCRIPT_DIR}/profiles"
_PY="${PYTHON:-${VENV:-/scratch2/${USER}/venvs/$(basename "$(pwd)")}/bin/python}"
_SRUN_TIMEOUT="90s"  # outer bound on queue-wait + run time per node, see header

: "${PROFILE:=gpu24}"
_PROFILE_FILE="${_PROFILES_DIR}/${PROFILE}.sh"
if [[ ! -f "${_PROFILE_FILE}" ]]; then
  echo "error: unknown profile '${PROFILE}' (expected ${_PROFILE_FILE})" >&2
  echo "available:" >&2
  ls -1 "${_PROFILES_DIR}"/*.sh | xargs -n1 basename | sed 's/\.sh$//' | sed 's/^/  /' >&2
  exit 1
fi
# shellcheck disable=SC1090
source "${_PROFILE_FILE}"
if [[ -z "${SLURM_GRES:-}" ]]; then
  echo "error: profile '${PROFILE}' has no GPU (SLURM_GRES is empty) -- pick a GPU profile" >&2
  exit 1
fi

if [[ $# -gt 0 ]]; then
  NODES=("$@")
else
  if [[ -z "${SLURM_EXCLUDE:-}" ]]; then
    echo "error: profile '${PROFILE}' (${_PROFILE_FILE}) has no SLURM_EXCLUDE -- pass node names explicitly" >&2
    exit 1
  fi
  IFS=',' read -r -a NODES <<< "${SLURM_EXCLUDE}"
fi

echo "profile=${PROFILE_NAME} partition=${SLURM_PARTITION} qos=${SLURM_QOS}"
echo "checking ${#NODES[@]} node(s): ${NODES[*]}"
echo

declare -A RESULT
for node in "${NODES[@]}"; do
  echo "=== ${node} ==="
  state="$(sinfo -h -N -n "${node}" -o "%T" 2>/dev/null | head -1)"
  if [[ -z "${state}" ]]; then
    echo "  sinfo: node not found (typo, or not visible in any partition from here)"
    RESULT["${node}"]="UNKNOWN (sinfo: not found)"
    echo
    continue
  fi
  echo "  sinfo state: ${state}"
  case "${state}" in
    down*|drain*|fail*|drng*)
      echo "  -> SLURM already marks this node '${state}' -- skipping srun (would just queue forever)"
      RESULT["${node}"]="DOWN (sinfo: ${state})"
      echo
      continue
      ;;
  esac

  out="$(timeout "${_SRUN_TIMEOUT}" srun --partition="${SLURM_PARTITION}" --qos="${SLURM_QOS}" --gres="${SLURM_GRES}" \
    --nodelist="${node}" --time=00:03:00 \
    "${_PY}" -c "
import torch
assert torch.cuda.is_available(), 'cuda not available'
x = torch.randn(1024, 1024, device='cuda')
y = x @ x
torch.cuda.synchronize()
print('CUDA_OK', torch.cuda.get_device_name(0))
" 2>&1)"
  rc=$?
  echo "${out}" | sed 's/^/  /'
  if grep -q "has been revoked" <<< "${out}"; then
    # 2026-09-07 (node05/08/12 all hit this): a PLAIN queue timeout would show the job still
    # PENDING when `timeout` kills it. "allocation ... revoked" means SLURM DID allocate the node,
    # then pulled it back BEFORE the job ran -- that's SLURM's own prolog/health-check (nhc)
    # rejecting the node, a stronger "still unhealthy" signal than "just busy right now", not a
    # mere INCONCLUSIVE-timeout.
    RESULT["${node}"]="REVOKED (SLURM's own health check pulled the allocation back -- likely still unhealthy, not just busy)"
  elif [[ ${rc} -eq 124 ]]; then
    RESULT["${node}"]="INCONCLUSIVE (queued the whole time, timed out -- node likely just busy with another job, not proven bad)"
  elif [[ ${rc} -eq 0 ]] && grep -q "CUDA_OK" <<< "${out}"; then
    RESULT["${node}"]="PASS"
  else
    RESULT["${node}"]="FAIL"
  fi
  echo
done

echo "=== summary ==="
for node in "${NODES[@]}"; do
  printf "  %-12s %s\n" "${node}" "${RESULT[${node}]:-UNKNOWN}"
done
echo
echo "PASS한 노드는 ${PROFILE}.sh의 SLURM_EXCLUDE에서 지워도 됨 -- 지울 때도 기존 컨벤션대로 이유/날짜를 주석으로 남길 것."
