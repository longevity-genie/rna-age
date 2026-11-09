#!/bin/bash
# Copy to slurm/env.local.sh (git-ignored) on the cluster and fill in. Read by slurm/env.sh.
export RNA_AGE_CPU_PARTITION=""   # CPU partition
export RNA_AGE_GPU_PARTITION=""   # GPU partition
export RNA_AGE_ACCOUNT=""         # Slurm account, if required
export RNA_AGE_GPU_NODES=""       # optional comma-separated GPU nodes
export RNA_AGE_GPU_EXCLUDE=""     # GPU nodes never to use
export RNA_AGE_GPU_ACCOUNT=""     # account allowed on the GPU partition, if different
# keep every cache off a small home quota
export UV_CACHE_DIR="" UV_PYTHON_INSTALL_DIR="" XDG_CACHE_HOME="" HF_HOME="" TORCH_HOME=""
