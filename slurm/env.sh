#!/bin/bash
# Cluster settings for rna-age's Slurm scripts. Everything that names a cluster (partitions, account,
# nodes) lives in slurm/env.local.sh, which is git-ignored, because this repository is public.
# Copy slurm/env.local.example.sh to slurm/env.local.sh on the cluster and fill it in.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$HERE/env.local.sh" ] && source "$HERE/env.local.sh"

export REPO="${REPO:-$(cd "$HERE/.." && pwd)}"
export RNA_AGE_CPU_PARTITION="${RNA_AGE_CPU_PARTITION:-}"
export RNA_AGE_GPU_PARTITION="${RNA_AGE_GPU_PARTITION:-}"
export RNA_AGE_ACCOUNT="${RNA_AGE_ACCOUNT:-}"
export RNA_AGE_GPU_NODES="${RNA_AGE_GPU_NODES:-}"       # optional --nodelist for GPU jobs
export RNA_AGE_GPU_EXCLUDE="${RNA_AGE_GPU_EXCLUDE:-}"   # GPU nodes never to use (--exclude)
export RNA_AGE_GPU_ACCOUNT="${RNA_AGE_GPU_ACCOUNT:-$RNA_AGE_ACCOUNT}"  # account allowed on the GPU partition
export RNA_AGE_CPU_CPUS="${RNA_AGE_CPU_CPUS:-4}" RNA_AGE_CPU_MEM="${RNA_AGE_CPU_MEM:-48G}"
export RNA_AGE_GPU_CPUS="${RNA_AGE_GPU_CPUS:-8}" RNA_AGE_GPU_MEM="${RNA_AGE_GPU_MEM:-96G}"
export RNA_AGE_TIME="${RNA_AGE_TIME:-04:00:00}"         # short jobs backfill sooner
export LOG_DIR="${LOG_DIR:-$REPO/slurm/logs}"
export PYTHONUNBUFFERED=1
for v in UV_CACHE_DIR UV_PYTHON_INSTALL_DIR XDG_CACHE_HOME HF_HOME TORCH_HOME; do
  [ -n "${!v:-}" ] && export "$v"
done
export PATH="$HOME/.local/bin:$PATH"
