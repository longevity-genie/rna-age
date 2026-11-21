"""RM33 step: per-condition pooling of system age gaps across cohorts (desk #39's per-condition meta-analysis).

The single cohorts are underpowered on any scale (RM33 z-unit outcome: median 95 % CI 1.34 z with 22 controls). This
pools each counted prediction over every cohort that measured it, on the z scale, by random-effects meta-analysis
(DerSimonian-Laird), and scores the pooled estimate with the pre-registered rules (``sys_gap.verdict``, ±``Z_MARGIN``).

Per cohort one estimate per type or pair (``ESTIMATES``, fixed before pooling): the cohort's own clock where its
status is ``ok`` for every type involved, else the external clock where that is ``ok``; otherwise the cohort is left
out of that prediction. A pair's status is that of its two types (a pair row carries none of its own). The standard
error is the bootstrap 95 % CI's width / 3.92. A prediction is pooled only from ≥ 2 cohorts.
Rows go to ``data/output/summary/sys_gap_meta.parquet``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from rna_age.train.sys_gap import SUMMARY, Z_MARGIN, verdict


@dataclass(frozen=True)
class Pool:
    condition: str
    cohorts: tuple[tuple[str, tuple[str, ...]], ...]  # (cohort tag, clocks in order of preference)
    predictions: tuple[tuple[str, str], ...]  # (type or "a - b", kind)
    level: str = "coarse"


BULK_HEADLINE = "20261111-0747-ens-mlp3t2-plus-atlasft2"
POOLS = (
    Pool("Alzheimer disease",
         (("37a17b78", ("own",)), ("5e57cd50", ("own", "external:brain")), ("cff99df2", ("own", "external:brain"))),
         (("astrocyte", "accelerates"), ("astrocyte - OPC", "accelerates"), ("microglia", "accelerates"))),
    Pool("CMV seropositive",
         (("ff4235bb", ("own", "external:blood")), ("789ad837", ("own", "external:blood"))),
         (("CD8 T", "accelerates"), ("CD4 T", "accelerates"), ("CD8 T - monocyte", "accelerates"),
          ("monocyte", "no_acceleration"), ("B", "no_acceleration"))),
    Pool("sarcopenia", (("GSE111016", (BULK_HEADLINE,)), ("GSE167186", (BULK_HEADLINE,))),
         (("muscle", "accelerates"),), level="bulk"),
)  # fmt: skip


def type_status(rows: pl.DataFrame) -> dict[str, str]:
    return {t: s for t, s in zip(rows["type"].to_list(), rows["status"].to_list(), strict=True) if s}


def estimate(contrast: pl.DataFrame, cohort: str, clocks: tuple[str, ...], target: str, level: str) -> dict | None:
    """The cohort's estimate for ``target`` from the first clock whose types are all measured, else None."""
    for clock in clocks:
        rows = contrast.filter((pl.col("cohort") == cohort) & (pl.col("clock") == clock) & (pl.col("level") == level))
        if rows.is_empty():
            continue
        st = type_status(rows)
        r = rows.filter(pl.col("type") == target)
        if r.height != 1:  # absent, or ambiguous (subsets)
            continue
        r = r.row(0, named=True)
        if not all(st.get(x.strip(), "absent") == "ok" for x in target.split(" - ")):
            continue
        if not all(np.isfinite([r["delta_z"], r["lo_z"], r["hi_z"]])):
            continue
        return {"cohort": cohort, "clock": clock, "delta_z": r["delta_z"], "se": (r["hi_z"] - r["lo_z"]) / 3.92,
                "n_case": r["n_case"], "n_control": r["n_control"]}  # fmt: skip
    return None


def dersimonian_laird(y: np.ndarray, se: np.ndarray) -> dict[str, float]:
    """Random-effects pooled mean, its SE, tau², Cochran's Q and I²."""
    w = 1 / se**2
    fixed = float((w * y).sum() / w.sum())
    q = float((w * (y - fixed) ** 2).sum())
    df = len(y) - 1
    c = float(w.sum() - (w**2).sum() / w.sum())
    tau2 = max(0.0, (q - df) / c) if c > 0 else 0.0
    wr = 1 / (se**2 + tau2)
    mu = float((wr * y).sum() / wr.sum())
    return {"mu": mu, "se": float(np.sqrt(1 / wr.sum())), "tau2": tau2, "q": q,
            "i2": max(0.0, (q - df) / q) if q > 0 else 0.0}  # fmt: skip


def run(pools: tuple[Pool, ...] = POOLS, write: bool = True) -> pl.DataFrame:
    contrast = pl.read_parquet(SUMMARY / "sys_gap_contrast.parquet").with_columns(
        pl.col("level").fill_null("coarse"), pl.col("clock").fill_null("own"), pl.col("status").fill_null("")
    )
    rows = []
    for p in pools:
        for target, kind in p.predictions:
            ests = [e for c, clocks in p.cohorts if (e := estimate(contrast, c, clocks, target, p.level))]
            base = {"condition": p.condition, "type": target, "prediction": kind, "k": len(ests),
                    "cohorts": ", ".join(f"{e['cohort']}:{e['clock']}" for e in ests),
                    "n_case": sum(e["n_case"] for e in ests), "n_control": sum(e["n_control"] for e in ests)}  # fmt: skip
            if len(ests) < 2:
                rows.append({**base, "delta_z": np.nan, "lo_z": np.nan, "hi_z": np.nan, "lo90_z": np.nan,
                             "hi90_z": np.nan, "tau2": np.nan, "i2": np.nan, "verdict_z": "inconclusive"})  # fmt: skip
                continue
            m = dersimonian_laird(np.array([e["delta_z"] for e in ests]), np.array([e["se"] for e in ests]))
            ci = {"lo_z": m["mu"] - 1.96 * m["se"], "hi_z": m["mu"] + 1.96 * m["se"],
                  "lo90_z": m["mu"] - 1.645 * m["se"], "hi90_z": m["mu"] + 1.645 * m["se"]}  # fmt: skip
            rows.append({**base, "delta_z": m["mu"], **ci, "tau2": m["tau2"], "i2": m["i2"],
                         "verdict_z": verdict(kind, ci["lo_z"], ci["hi_z"], ci["lo90_z"], ci["hi90_z"], Z_MARGIN)})  # fmt: skip
    out = pl.DataFrame(rows)
    if write:
        out.write_parquet(SUMMARY / "sys_gap_meta.parquet", compression="zstd")
    return out
