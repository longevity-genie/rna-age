#!/bin/bash
# slurm/submit.sh cpu|gpu NAME COMMAND...   e.g.
#   slurm/submit.sh cpu parse-gtex rna-age parse gtex
#   CPUS=8 MEM=64G EXTRA=--tmp=80G RNA_AGE_LOCAL_TMP=/scratch/rna-age slurm/submit.sh cpu parse-archs4 rna-age parse archs4
# Builds sbatch flags from slurm/env.sh (and the git-ignored env.local.sh) and runs job.sbatch.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"
kind="${1:?cpu or gpu}"; name="${2:?job name}"; shift 2
[ $# -gt 0 ] || { echo "no command given" >&2; exit 2; }
mkdir -p "$LOG_DIR"
flags=(--job-name="rna-age-$name" --nodes=1 --ntasks=1 --output="$LOG_DIR/%x-%j.out" --time="${TIME:-$RNA_AGE_TIME}")
if [ "$kind" = gpu ]; then
  [ -n "$RNA_AGE_GPU_ACCOUNT" ] && flags+=(--account="$RNA_AGE_GPU_ACCOUNT")
  [ -n "$RNA_AGE_GPU_EXCLUDE" ] && flags+=(--exclude="$RNA_AGE_GPU_EXCLUDE")
  flags+=(--requeue)
  [ -n "$RNA_AGE_GPU_PARTITION" ] || { echo "set RNA_AGE_GPU_PARTITION in slurm/env.local.sh" >&2; exit 2; }
  flags+=(--partition="$RNA_AGE_GPU_PARTITION" --cpus-per-task="${CPUS:-$RNA_AGE_GPU_CPUS}" --mem="${MEM:-$RNA_AGE_GPU_MEM}")
  nodes="${NODE:-$RNA_AGE_GPU_NODES}"; [ -n "$nodes" ] && flags+=(--nodelist="$nodes")
else
  [ -n "$RNA_AGE_ACCOUNT" ] && flags+=(--account="$RNA_AGE_ACCOUNT")
  [ -n "$RNA_AGE_CPU_PARTITION" ] || { echo "set RNA_AGE_CPU_PARTITION in slurm/env.local.sh" >&2; exit 2; }
  flags+=(--partition="$RNA_AGE_CPU_PARTITION" --cpus-per-task="${CPUS:-$RNA_AGE_CPU_CPUS}" --mem="${MEM:-$RNA_AGE_CPU_MEM}")
fi
# EXTRA="--tmp=80G ..." passes further sbatch flags (node-local disk, dependencies, ...)
[ -n "${EXTRA:-}" ] && read -r -a extra <<<"$EXTRA" && flags+=("${extra[@]}")
sbatch "${flags[@]}" --export="ALL,REPO=$REPO,RNA_AGE_COMMIT=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null),RNA_AGE_JOB_KIND=$kind,CUDA_DEVICE=${CUDA_DEVICE:-}" "$HERE/job.sbatch" "$@"
