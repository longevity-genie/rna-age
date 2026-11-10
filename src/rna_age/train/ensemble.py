"""Average the predictions of finished runs (same samples) into a new run."""

from __future__ import annotations

import polars as pl

from rna_age.train.runs import RUNS, finish_run, new_run


def ensemble(run_ids: list[str], name: str) -> dict:
    preds = [pl.read_parquet(RUNS / r / "predictions.parquet") for r in run_ids]
    base = preds[0].drop("pred")
    joined = base
    for i, p in enumerate(preds):
        joined = joined.join(p.select("sample_id", pl.col("pred").alias(f"p{i}")), on="sample_id", how="inner")
    out = joined.with_columns(pred=pl.mean_horizontal([f"p{i}" for i in range(len(preds))])).drop(
        [f"p{i}" for i in range(len(preds))]
    )
    run_dir = new_run(name, {"model": "ensemble (mean)", "features": "+".join(run_ids), "members": run_ids,
                             "train_set": "as members"})  # fmt: skip
    return finish_run(run_dir, out)


def blend(run_a: str, run_b: str, name: str) -> dict:
    """Weighted mean of two runs, weight chosen on validation exact-age MAE (grid 0..1 step 0.05)."""
    import numpy as np

    a = pl.read_parquet(RUNS / run_a / "predictions.parquet")
    b = pl.read_parquet(RUNS / run_b / "predictions.parquet").select("sample_id", pl.col("pred").alias("pb"))
    j = a.join(b, on="sample_id", how="inner")
    v = j.filter((pl.col("split") == "val") & (pl.col("age_type") == "exact"))
    best = min(
        (float(np.mean(np.abs(w * v["pred"].to_numpy() + (1 - w) * v["pb"].to_numpy() - v["age_years"].to_numpy()))), w)
        for w in np.round(np.arange(0, 1.0001, 0.05), 2)
    )
    w = best[1]
    out = j.with_columns(pred=w * pl.col("pred") + (1 - w) * pl.col("pb")).drop("pb")
    run_dir = new_run(name, {"model": f"blend ({w:.2f}·A + {1 - w:.2f}·B, weight chosen on val)", "members": [run_a, run_b],
                             "features": f"{run_a}+{run_b}", "weight_a": w, "train_set": "as members"})  # fmt: skip
    return finish_run(run_dir, out, {"blend_weight_a": w})
