#!/bin/bash
# One rna-age command as a Slurm job: `uv run --no-sync <command...>` in the repository.
# Submit through slurm/submit.sh. GPU jobs: there is no GRES, so the job picks the first card with no
# compute process (unless CUDA_DEVICE is set) and exits 3 when every card is busy.
set -euo pipefail
source "${REPO:?}/slurm/env.sh"
cd "$REPO"
echo "host=$(hostname) job=${SLURM_JOB_ID:-local} kind=${RNA_AGE_JOB_KIND:-cpu} commit=$(git rev-parse --short HEAD 2>/dev/null || echo ?) start=$(date -Is)"
if [ "${RNA_AGE_JOB_KIND:-cpu}" = gpu ]; then
  if [ -z "${CUDA_DEVICE:-}" ]; then
    busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | sort -u)
    while IFS=, read -r idx uuid; do
      uuid=$(echo "$uuid" | xargs)
      if ! grep -q "$uuid" <<<"$busy"; then CUDA_DEVICE=$(echo "$idx" | xargs); break; fi
    done < <(nvidia-smi --query-gpu=index,uuid --format=csv,noheader)
    [ -z "${CUDA_DEVICE:-}" ] && { echo "every GPU on $(hostname) is busy" >&2; exit 3; }
  fi
  export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv -i "$CUDA_DEVICE"
else
  export CUDA_VISIBLE_DEVICES=""
fi
uv run --no-sync "$@"
echo "end=$(date -Is)"
