#!/bin/bash -l
# Build an isolated R environment for the tAge clocks (Gladyshev lab) with conda-forge R via micromamba.
# Usage: scripts/setup_tage_env.sh [ENV_PREFIX]   (default: $TAGE_ENV or ./data/envs/tage)
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREFIX="${1:-${TAGE_ENV:-$REPO/data/envs/tage}}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$(dirname "$PREFIX")/.mamba}"
if [ ! -x "$PREFIX/bin/Rscript" ]; then
  micromamba create -y -p "$PREFIX" -c conda-forge -c bioconda \
    r-base=4.4 r-devtools r-remotes r-reticulate r-data.table r-arrow r-matrix \
    bioconductor-biobase bioconductor-edger bioconductor-preprocesscore \
    python=3.12 scikit-learn joblib pandas numpy
fi
micromamba run -p "$PREFIX" Rscript -e '
  options(repos = c(CRAN = "https://cloud.r-project.org"), timeout = 600)
  remotes::install_github("Gladyshev-Lab/tAge", upgrade = "never", dependencies = TRUE)
  library(tAge); cat("tAge", as.character(packageVersion("tAge")), "installed\n")'
