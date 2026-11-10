"""Disease age acceleration: age-adjusted residuals of disease vs healthy-or-unknown samples.

Raw residuals (pred − age) are dominated by regression to the mean, so the residual trend on age is fitted on
healthy-or-unknown held-out samples and removed; the adjusted gap of disease samples is then compared overall
and per disease label (n ≥ 30). Output: ``<run>/agegap.parquet`` and ``data/output/summary/agegap_<run>.parquet``.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from rna_age.paths import INTERIM, OUTPUT


def age_gap(run_id: str, min_n: int = 30) -> pl.DataFrame:
    p = pl.read_parquet(OUTPUT / "runs" / run_id / "predictions.parquet")
    m = pl.read_parquet(INTERIM / "samples_all.parquet", columns=["sample_id", "disease", "is_healthy"])
    p = p.drop("is_healthy").join(m, on="sample_id", how="left")  # current health labels
    p = p.filter(pl.col("split").is_in(["val", "test"]) & (pl.col("age_type") == "exact") & ~pl.col("is_pseudobulk"))
    healthy = p.filter(pl.col("is_healthy").fill_null(True))
    coef = np.polyfit(healthy["age_years"].to_numpy(), (healthy["pred"] - healthy["age_years"]).to_numpy(), 1)
    p = p.with_columns(
        gap=pl.col("pred") - pl.col("age_years") - (coef[0] * pl.col("age_years") + coef[1]),
        group=pl.when(pl.col("is_healthy").fill_null(True))
        .then(pl.lit("healthy_or_unknown"))
        .otherwise(pl.lit("disease")),
    )
    overall = p.group_by("group").agg(
        n=pl.len(), mean_gap=pl.col("gap").mean(), sem=pl.col("gap").std() / pl.len().sqrt(), studies=pl.col("study_id").n_unique()
    )  # fmt: skip
    per = (
        p.filter(pl.col("group") == "disease")
        .with_columns(label=pl.col("disease").str.to_lowercase().str.strip_chars())
        .group_by("label")
        .agg(
            n=pl.len(),
            mean_gap=pl.col("gap").mean(),
            sem=pl.col("gap").std() / pl.len().sqrt(),
            studies=pl.col("study_id").n_unique(),
        )  # fmt: skip
        .filter((pl.col("n") >= min_n) & (pl.col("studies") >= 2))
        .sort("mean_gap", descending=True)
        .rename({"label": "group"})
    )
    # within-study contrast: disease minus healthy/unknown mean gap in studies that have both (n >= 5 each)
    ws = (
        p.group_by("study_id", "group")
        .agg(pl.col("gap").mean(), pl.len())
        .filter(pl.col("len") >= 5)
        .pivot(on="group", index="study_id", values="gap")
        .drop_nulls()
    )
    within = pl.DataFrame()
    if ws.height and "disease" in ws.columns:
        d = (ws["disease"] - ws["healthy_or_unknown"]).to_numpy()
        within = pl.DataFrame({"group": ["within_study_disease_minus_control"], "n": [len(d)], "mean_gap": [float(d.mean())],
                               "sem": [float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else None],
                               "studies": [len(d)], "level": ["within_study"]})  # fmt: skip
    out = pl.concat([within, overall.with_columns(level=pl.lit("overall")), per.with_columns(level=pl.lit("disease_label"))],
                    how="diagonal_relaxed")  # fmt: skip
    out.write_parquet(OUTPUT / "runs" / run_id / "agegap.parquet")
    out.write_parquet(OUTPUT / "summary" / f"agegap_{run_id}.parquet")
    return out
