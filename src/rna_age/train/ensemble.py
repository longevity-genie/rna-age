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
