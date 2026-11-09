"""The fixed evaluation protocol (decided 2026-11-10 00:20 before the first model; do not change it).

Inputs: a predictions frame with ``sample_id, source, split, age_type, age_years, age_min, age_max, pred`` plus
grouping columns (``tissue``, ``is_healthy``, ``is_pseudobulk``, ``cohort``, ``in_tage_training``).

Metrics:
- exact ages: MAE, median AE, Pearson r, R², bias slope (slope of ``pred - age`` on ``age``; 0 = no regression
  to the mean, negative = old under-, young over-predicted), mean signed error, n.
- binned ages (GTEx): interval hit rate (``age_min <= pred < age_max + 1``) and mean distance to the interval
  (0 inside); plus the same exact-age metrics against the bin midpoint, labelled ``*_mid``.
- groupings: overall, per source, per tissue (n ≥ 20), per age decade, healthy vs disease, bulk vs pseudobulk.
- disease samples (``is_healthy == False``): mean (pred - age) should be positive; reported as ``age_gap``.
"""

from __future__ import annotations

import numpy as np
import polars as pl


def _exact(pred: np.ndarray, age: np.ndarray) -> dict[str, float]:
    if len(age) == 0:
        return {}
    err = pred - age
    out = {
        "n": float(len(age)),
        "mae": float(np.mean(np.abs(err))),
        "median_ae": float(np.median(np.abs(err))),
        "mean_error": float(np.mean(err)),
    }
    if len(age) > 2 and np.std(age) > 0 and np.std(pred) > 0:
        out["pearson_r"] = float(np.corrcoef(pred, age)[0, 1])
        out["r2"] = float(1 - np.sum(err**2) / np.sum((age - age.mean()) ** 2))
        out["bias_slope"] = float(np.polyfit(age, err, 1)[0])
    return out


def _binned(pred: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> dict[str, float]:
    if len(pred) == 0:
        return {}
    upper = hi + 1
    hit = (pred >= lo) & (pred < upper)
    dist = np.where(pred < lo, lo - pred, np.where(pred >= upper, pred - upper, 0.0))
    mid = (lo + upper) / 2
    out = {"n": float(len(pred)), "interval_hit": float(hit.mean()), "interval_dist": float(dist.mean())}
    out |= {f"{k}_mid": v for k, v in _exact(pred, mid).items() if k != "n"}
    return out


def metrics_for(df: pl.DataFrame) -> dict[str, float]:
    ex = df.filter(pl.col("age_type") == "exact")
    bi = df.filter(pl.col("age_type") == "binned")
    out = {f"exact_{k}": v for k, v in _exact(ex["pred"].to_numpy(), ex["age_years"].to_numpy()).items()}
    out |= {
        f"binned_{k}": v
        for k, v in _binned(bi["pred"].to_numpy(), bi["age_min"].to_numpy(), bi["age_max"].to_numpy()).items()
    }
    return out


def evaluate(pred: pl.DataFrame, min_group: int = 20) -> pl.DataFrame:
    """Long metrics table: ``split, grouping, group, metric, value``."""
    pred = pred.with_columns(
        decade=(pl.coalesce("age_years", (pl.col("age_min") + pl.col("age_max") + 1) / 2) // 10 * 10)
        .cast(pl.Int32)
        .cast(pl.String),
        health=pl.when(pl.col("is_healthy") == False)  # noqa: E712
        .then(pl.lit("disease"))
        .otherwise(pl.lit("healthy_or_unknown")),
        modality=pl.when(pl.col("is_pseudobulk")).then(pl.lit("pseudobulk")).otherwise(pl.lit("bulk")),
    )
    groupings = ["source", "tissue", "decade", "health", "modality", "cohort"]
    rows = []
    for split, sdf in pred.group_by("split"):
        split = split[0]
        rows += [(split, "all", "all", k, v) for k, v in metrics_for(sdf).items()]
        for g in groupings:
            if g not in sdf.columns:
                continue
            for key, gdf in sdf.group_by(g):
                if key[0] is None or gdf.height < min_group:
                    continue
                rows += [(split, g, str(key[0]), k, v) for k, v in metrics_for(gdf).items()]
        if "in_tage_training" in sdf.columns:
            clean = sdf.filter(~pl.col("in_tage_training").fill_null(False))
            rows += [(split, "not_in_tage_training", "all", k, v) for k, v in metrics_for(clean).items()]
    return pl.DataFrame(
        rows, schema={"split": pl.String, "grouping": pl.String, "group": pl.String, "metric": pl.String,
                      "value": pl.Float64}, orient="row",
    )  # fmt: skip


def headline(metrics: pl.DataFrame, split: str = "test") -> dict[str, float]:
    m = metrics.filter((pl.col("split") == split) & (pl.col("grouping") == "all"))
    return dict(zip(m["metric"].to_list(), m["value"].to_list(), strict=True))
