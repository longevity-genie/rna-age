"""Post-hoc linear calibration against regression to the mean, fitted on validation only.

On val exact-age samples fit pred = a·age + b (how predictions compress age), then map every prediction
through the inverse, pred' = (pred − b) / a. Chronological age is used only to fit (a, b) on val, never at
prediction time, so test numbers stay honest.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from rna_age.train.runs import RUNS, finish_run, new_run


def calibrate(run_id: str) -> dict:
    p = pl.read_parquet(RUNS / run_id / "predictions.parquet")
    v = p.filter((pl.col("split") == "val") & (pl.col("age_type") == "exact"))
    a, b = np.polyfit(v["age_years"].to_numpy(), v["pred"].to_numpy(), 1)
    out = p.with_columns(pred=(pl.col("pred") - b) / a)
    run_dir = new_run(f"{run_id.split('-', 2)[-1]}-cal", {"model": f"calibrated({run_id})", "features": "as base",
                                                          "calibration": {"a": a, "b": b}, "base_run": run_id,
                                                          "train_set": "as base"})  # fmt: skip
    return finish_run(run_dir, out)
