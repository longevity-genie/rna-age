"""Run tracking: ``data/output/runs/<YYYYMMDD-HHMM>-<name>/`` + the tracked ``experiments.parquet``.

Each run directory holds ``config.json``, ``metrics.parquet`` (long, from :func:`evaluation.evaluate`),
``predictions.parquet`` and logs; checkpoints live under ``data/interim/checkpoints/<run>/`` (path recorded in
the config). One row per run is appended to ``data/output/summary/experiments.parquet`` (tracked in git).
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

import polars as pl

from rna_age.paths import INTERIM, OUTPUT, REPO_ROOT
from rna_age.train.evaluation import evaluate, headline

RUNS = OUTPUT / "runs"
EXPERIMENTS = OUTPUT / "summary" / "experiments.parquet"


def new_run(name: str, config: dict) -> Path:
    run_id = f"{datetime.now().strftime('%Y%m%d-%H%M')}-{name}"
    d = RUNS / run_id
    d.mkdir(parents=True, exist_ok=True)
    commit = os.environ.get("RNA_AGE_COMMIT", "")
    if not commit:
        try:
            commit = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT
            ).stdout.strip()
        except FileNotFoundError:  # compute nodes have no git; submit.sh passes RNA_AGE_COMMIT
            commit = "unknown"
    config = {"run_id": run_id, "commit": commit, "checkpoint_dir": str(INTERIM / "checkpoints" / run_id), **config}
    (d / "config.json").write_text(json.dumps(config, indent=2, default=str))
    return d


def finish_run(run_dir: Path, predictions: pl.DataFrame, extra: dict | None = None) -> dict[str, float]:
    predictions.write_parquet(run_dir / "predictions.parquet", compression="zstd")
    metrics = evaluate(predictions)
    metrics.write_parquet(run_dir / "metrics.parquet")
    config = json.loads((run_dir / "config.json").read_text())
    row = {"run_id": config["run_id"], "model": config.get("model"), "features": config.get("features"),
           "train_set": config.get("train_set"), "commit": config.get("commit")}  # fmt: skip
    for split in ("val", "test"):
        h = headline(metrics, split)
        for k in ("exact_mae", "exact_median_ae", "exact_pearson_r", "exact_r2", "exact_bias_slope", "exact_n",
                  "binned_interval_hit", "binned_interval_dist", "binned_mae_mid"):  # fmt: skip
            row[f"{split}_{k}"] = h.get(k)
    for grouping, group in (("modality", "pseudobulk"), ("source", "gtex"), ("source", "archs4")):
        m = metrics.filter((pl.col("split") == "test") & (pl.col("grouping") == grouping) & (pl.col("group") == group))
        d = dict(zip(m["metric"].to_list(), m["value"].to_list(), strict=True))
        row[f"test_{group}_exact_mae"] = d.get("exact_mae")
        row[f"test_{group}_exact_r"] = d.get("exact_pearson_r")
    tr = headline(metrics, "transfer")
    for k in ("exact_mae", "exact_pearson_r", "exact_n", "exact_mean_error"):
        row[f"transfer_{k}"] = tr.get(k)
    for grp in ("census_pseudobulk", "terekhova2023"):
        m = metrics.filter(
            (pl.col("split") == "transfer") & (pl.col("grouping") == "source") & (pl.col("group") == grp)
        )
        d = dict(zip(m["metric"].to_list(), m["value"].to_list(), strict=True))
        row[f"transfer_{grp}_exact_mae"] = d.get("exact_mae")
        row[f"transfer_{grp}_exact_r"] = d.get("exact_pearson_r")
    row |= extra or {}
    new = pl.DataFrame([row])
    if EXPERIMENTS.exists():
        new = pl.concat([pl.read_parquet(EXPERIMENTS), new], how="diagonal_relaxed").unique("run_id", keep="last")
    EXPERIMENTS.parent.mkdir(parents=True, exist_ok=True)
    new.write_parquet(EXPERIMENTS)
    return row
