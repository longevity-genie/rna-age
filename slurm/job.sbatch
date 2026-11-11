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
    # A job shows on its card only once it has loaded its data (minutes), so jobs starting together all saw card 0
    # free (2026-11-11: four jobs on one card, two out of memory). A node-local lock serialises the choice, and a
    # claim file per card, held while its job lives, marks a card taken before any process appears on it.
    claims=/tmp/rna-age-gpu-claims
    mkdir -p "$claims"
    exec 9>"$claims/.lock"
    flock 9
    busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | sort -u)
    while IFS=, read -r idx uuid; do
      idx=$(echo "$idx" | xargs); uuid=$(echo "$uuid" | xargs)
      grep -q "$uuid" <<<"$busy" && continue
      taken=""
      for c in "$claims/$idx".*; do
        [ -e "$c" ] || continue
        if squeue -h -j "${c##*.}" 2>/dev/null | grep -q .; then taken=1; else rm -f "$c"; fi
      done
      [ -n "$taken" ] && continue
      CUDA_DEVICE=$idx
      : >"$claims/$idx.${SLURM_JOB_ID:-$$}"
      trap 'rm -f "$claims/$CUDA_DEVICE.${SLURM_JOB_ID:-$$}"' EXIT
      break
    done < <(nvidia-smi --query-gpu=index,uuid --format=csv,noheader)
    flock -u 9
    [ -z "${CUDA_DEVICE:-}" ] && { echo "every GPU on $(hostname) is busy" >&2; exit 3; }
  fi
  export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
  if [ -n "${SLURM_JOB_ID:-}" ]; then  # the card into the job comment (cuda=-), for the shared GPU dashboard
    c=$(squeue -h -j "$SLURM_JOB_ID" -o %k 2>/dev/null || true)
    [ -n "$c" ] && scontrol update JobId="$SLURM_JOB_ID" Comment="${c/cuda=-/cuda=$CUDA_DEVICE}" 2>/dev/null || true
  fi
  nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv -i "$CUDA_DEVICE"
else
  export CUDA_VISIBLE_DEVICES=""
fi
uv run --no-sync "$@"
echo "end=$(date -Is)"
