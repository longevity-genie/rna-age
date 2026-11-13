"""RM26's bar: does adding recount3 to training lower the 20k-HVG MLP's test error?

Two arms of three seeds each (control: the shared table; recount3: ``samples_all_recount3.parquet`` minus the cfRNA
study), on one gene axis. Each arm's prediction is the mean of its seeds per sample. On the old test samples
(``split == "test"``, exact ages, eligible in the shared table, predicted by every run compared):

- MAE per arm, and the paired difference arm − control with a 95 % CI from resampling test studies
  (:func:`rna_age.train.kron.paired_ci`; negative = the recount3 arm better);
- each arm against the headline run's predictions on the same samples (arm − headline, same CI).

``test_new`` (recount3 studies hashing to test) is scored apart, for the runs that predicted it (the recount3 arm).
Output: ``data/output/summary/rm26_compare.parquet``, one row per (set, comparison, source).
"""

from __future__ import annotations

import logging

import polars as pl

from rna_age.paths import OUTPUT
from rna_age.train.kron import paired_ci
from rna_age.train.runs import RUNS

log = logging.getLogger(__name__)

OUT = OUTPUT / "summary" / "rm26_compare.parquet"
META = ["sample_id", "source", "split", "study_id", "age_type", "age_years"]


def mean_pred(runs: list[str] | tuple[str, ...]) -> pl.DataFrame:
    """``sample_id, meta…, pred``: the per-sample mean over the runs' predictions (samples every run predicted)."""
    parts = [pl.read_parquet(RUNS / r / "predictions.parquet", columns=[*META, "pred"]) for r in runs]
    return mean_of(parts)


def mean_of(parts: list[pl.DataFrame]) -> pl.DataFrame:
    meta = parts[0].select(META)
    preds = [p.select("sample_id", pl.col("pred").alias(f"p{i}")) for i, p in enumerate(parts)]
    out = meta
    for p in preds:
        out = out.join(p, on="sample_id", how="inner")
    return out.with_columns(pred=pl.mean_horizontal([f"p{i}" for i in range(len(parts))])).drop(
        [f"p{i}" for i in range(len(parts))]
    )


def _mae(d: pl.DataFrame, col: str) -> float:
    return float((d[col] - d["age_years"]).abs().mean()) if d.height else float("nan")


def paired_rows(d: pl.DataFrame, a: str, b: str, set_name: str, n_boot: int, seed: int) -> list[dict]:
    """``d``: ``study_id, source, age_years`` and prediction columns ``a`` and ``b``; rows for a − b overall and
    per source."""
    d = d.with_columns(diff=(pl.col(a) - pl.col("age_years")).abs() - (pl.col(b) - pl.col("age_years")).abs())
    ci = paired_ci(d, n_boot, seed)
    rows = []
    for src, v in ci.items():
        sub = d if src == "all" else d.filter(pl.col("source") == src)
        rows.append({"set": set_name, "comparison": f"{a} - {b}", "source": src, "n": v["n"],
                     "studies": v["studies"], "mae_a": _mae(sub, a), "mae_b": _mae(sub, b), "diff": v["diff"],
                     "lo": v["lo"], "hi": v["hi"]})  # fmt: skip
    return rows


def compare_frames(
    control: pl.DataFrame,
    arm: pl.DataFrame,
    headline: pl.DataFrame | None,
    eligible_ids: pl.Series | None = None,
    n_boot: int = 2000,
    seed: int = 0,
) -> pl.DataFrame:
    """The comparison on seed-averaged frames (``META`` + ``pred``); ``eligible_ids`` restricts the test set."""
    test = arm.filter((pl.col("split") == "test") & (pl.col("age_type") == "exact")).select(
        "sample_id", "source", "study_id", "age_years", pl.col("pred").alias("arm")
    )
    if eligible_ids is not None:
        test = test.filter(pl.col("sample_id").is_in(eligible_ids.implode()))
    test = test.join(control.select("sample_id", pl.col("pred").alias("control")), on="sample_id", how="inner")
    if headline is not None:
        test = test.join(headline.select("sample_id", pl.col("pred").alias("headline")), on="sample_id", how="inner")
    rows = paired_rows(test, "arm", "control", "test", n_boot, seed)
    if headline is not None:
        rows += paired_rows(test, "arm", "headline", "test", n_boot, seed)
        rows += paired_rows(test, "control", "headline", "test", n_boot, seed)
    for name, frame in (("arm", arm), ("control", control)):
        new = frame.filter((pl.col("split") == "test_new") & (pl.col("age_type") == "exact"))
        if new.height:
            by_source = [(s[0], g) for s, g in new.sort("source").group_by("source", maintain_order=True)]
            for src, sub in [("all", new), *by_source]:
                rows.append({"set": "test_new", "comparison": name, "source": str(src), "n": sub.height,
                             "studies": sub["study_id"].n_unique(), "mae_a": _mae(sub, "pred"), "mae_b": None,
                             "diff": None, "lo": None, "hi": None})  # fmt: skip
    out = pl.DataFrame(rows, schema={"set": pl.String, "comparison": pl.String, "source": pl.String, "n": pl.Int64,
                                     "studies": pl.Int64, "mae_a": pl.Float64, "mae_b": pl.Float64,
                                     "diff": pl.Float64, "lo": pl.Float64, "hi": pl.Float64})  # fmt: skip
    return out.with_columns(
        beyond_ci=pl.when(pl.col("hi").is_not_null()).then(pl.col("hi") < 0),
        n_boot=pl.lit(n_boot),
        boot_seed=pl.lit(seed),
    )


def compare(
    control: tuple[str, ...] | list[str],
    arm: tuple[str, ...] | list[str],
    headline: str | None = "20261111-0747-ens-mlp3t2-plus-atlasft2",
    n_boot: int = 2000,
    seed: int = 0,
) -> pl.DataFrame:
    from rna_age.train.features import sample_table

    c, a = mean_pred(control), mean_pred(arm)
    h = pl.read_parquet(RUNS / headline / "predictions.parquet", columns=[*META, "pred"]) if headline else None
    eligible = sample_table("")["sample_id"]  # the shared table: the old test samples as they stand
    out = compare_frames(c, a, h, eligible, n_boot, seed).with_columns(
        control_runs=pl.lit(",".join(control)), arm_runs=pl.lit(",".join(arm)), headline_run=pl.lit(headline)
    )
    if n := out.filter((pl.col("set") == "test") & (pl.col("source") == "all"))["n"].unique().to_list():
        log.info("test samples compared: %s", n)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(OUT, compression="zstd")
    return out
