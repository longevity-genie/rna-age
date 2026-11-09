"""Figures (PNG + SVG) under ``data/output/figures/``: data overview and per-run evaluation plots."""

from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from rna_age.paths import INTERIM, OUTPUT  # noqa: E402
from rna_age.schema import age_mid  # noqa: E402

log = logging.getLogger(__name__)
FIG = OUTPUT / "figures"


def _save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("figure %s", FIG / f"{name}.png")


def data_overview() -> None:
    m = pl.read_parquet(INTERIM / "samples_all.parquet").filter(pl.col("split").is_not_null())
    m = m.with_columns(age=age_mid(m))
    sources = m.group_by("source").len().sort("len", descending=True)["source"].to_list()
    fig, axes = plt.subplots(1, len(sources), figsize=(3.2 * len(sources), 2.8), sharex=True)
    for ax, src in zip(np.atleast_1d(axes), sources, strict=True):
        a = m.filter(pl.col("source") == src)["age"].drop_nulls().to_numpy()
        ax.hist(a, bins=np.arange(0, 105, 5), color="#4C72B0")
        ax.set_title(f"{src}\nn={len(a):,}", fontsize=9)
        ax.set_xlabel("age (years)")
    np.atleast_1d(axes)[0].set_ylabel("samples")
    fig.suptitle("Eligible samples by source (GTEx: bin midpoints)", fontsize=10, y=1.08)
    _save(fig, "data_age_by_source")


def run_plots(run_id: str) -> None:
    d = OUTPUT / "runs" / run_id
    p = pl.read_parquet(d / "predictions.parquet")
    ex = p.filter((pl.col("split") == "test") & (pl.col("age_type") == "exact"))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    ax = axes[0]
    for src, color in (("archs4", "#4C72B0"), ("bgee", "#DD8452"), ("ncbi_geo", "#55A868")):
        s = ex.filter(pl.col("source") == src)
        if s.height:
            ax.scatter(s["age_years"], s["pred"], s=3, alpha=0.3, color=color, label=f"{src} (n={s.height:,})")
    ax.plot([0, 100], [0, 100], "k--", lw=0.8)
    ax.set_xlabel("chronological age")
    ax.set_ylabel("predicted age")
    ax.set_title("test, exact ages")
    ax.legend(fontsize=7, markerscale=3)
    tr = p.filter((pl.col("split") == "transfer") & (pl.col("age_type") == "exact"))
    ax = axes[1]
    if tr.height:
        for src, color in (("census_pseudobulk", "#8172B3"), ("terekhova2023", "#C44E52")):
            s = tr.filter(pl.col("source") == src)
            if s.height:
                ax.scatter(s["age_years"], s["pred"], s=4, alpha=0.4, color=color, label=f"{src} (n={s.height:,})")
        ax.plot([0, 100], [0, 100], "k--", lw=0.8)
        ax.legend(fontsize=7, markerscale=3)
    ax.set_title("bulk → pseudobulk transfer")
    ax.set_xlabel("chronological age")
    g = p.filter((pl.col("split") == "test") & (pl.col("source") == "gtex"))
    ax = axes[2]
    if g.height:
        mids = ((g["age_min"] + g["age_max"] + 1) / 2).to_numpy()
        bins = sorted(set(mids.tolist()))
        ax.boxplot([g["pred"].to_numpy()[mids == b] for b in bins], positions=bins, widths=6, showfliers=False)
        ax.plot([20, 80], [20, 80], "k--", lw=0.8)
        ax.set_xlim(15, 85)
    ax.set_title("GTEx test: prediction by age bin")
    ax.set_xlabel("bin midpoint")
    fig.suptitle(run_id, fontsize=9)
    _save(fig, f"run_{run_id}")

    met = pl.read_parquet(d / "metrics.parquet")
    t = met.filter((pl.col("split") == "test") & (pl.col("grouping") == "tissue") & (pl.col("metric") == "exact_mae"))
    t = t.sort("value").head(25)
    if t.height:
        fig, ax = plt.subplots(figsize=(6, 0.25 * t.height + 1))
        ax.barh(t["group"].str.slice(0, 40).to_list(), t["value"].to_list(), color="#4C72B0")
        ax.set_xlabel("test MAE (years)")
        ax.invert_yaxis()
        _save(fig, f"run_{run_id}_tissue_mae")
    dec = met.filter(
        (pl.col("split") == "test") & (pl.col("grouping") == "decade") & (pl.col("metric") == "exact_mean_error")
    )
    if dec.height:
        dec = dec.with_columns(pl.col("group").cast(pl.Int32)).sort("group")
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(dec["group"].to_list(), dec["value"].to_list(), width=8, color="#C44E52")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlabel("age decade")
        ax.set_ylabel("mean (pred − age)")
        ax.set_title("bias by decade (test)")
        _save(fig, f"run_{run_id}_decade_bias")


def compare_runs() -> None:
    e = OUTPUT / "summary" / "experiments.parquet"
    if not e.exists():
        return
    df = pl.read_parquet(e).drop_nulls("test_exact_mae").sort("test_exact_mae")
    fig, ax = plt.subplots(figsize=(7, 0.35 * df.height + 1))
    ax.barh(df["run_id"].to_list(), df["test_exact_mae"].to_list(), color="#4C72B0")
    ax.set_xlabel("test MAE, exact ages (years)")
    ax.invert_yaxis()
    _save(fig, "runs_test_mae")
