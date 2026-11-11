#!/bin/bash
# RM10 step 2 (docs/ROADMAP.md): extract the six registered before/after series, retrain the plain and health MLPs
# with --external on the clean table, then the registered paired tests. Runs inside job.sbatch (uv env, repo cwd).
set -euo pipefail
rna-age external GSE65540,GSE274137,GSE192587,GSE66306,GSE137631,GSE107894
start=$(date +%s)
for s in 0 1 2; do rna-age probe --features hvg --model mlp --n-hvg 20000 --train-set all --seed "$s" --external; done
for s in 0 1 2; do rna-age train-health --rank-weight 1 --disease-head --seed "$s" --external; done
pick() {  # the runs of this job with external predictions, matching a name pattern, comma-separated
  for d in data/output/runs/*"$1"*; do
    [ -f "$d/predictions_external.parquet" ] && [ "$(stat -c %Y "$d")" -ge "$start" ] && basename "$d"
  done | paste -sd, -
}
plain=$(pick hvg20000-mlp-trainall); health=$(pick health-mlp-rank1-head)
echo "plain runs: $plain"; echo "health runs: $health"
rna-age biology-paired "$plain" --name plain
rna-age biology-paired "$health" --name health
