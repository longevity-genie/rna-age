"""Rung 1 baselines: ridge / ElasticNet on log-CPM of HVGs (chosen on train only).

Training set (default ``healthy``): bulk, ``split == train``, ``is_healthy != False`` (healthy or unknown),
exact ages, plus GTEx with the bin midpoint as target when ``--gtex-midpoints`` (binned samples are still
*scored* by interval hits). Alpha (and l1_ratio) are chosen on the validation split's exact-age MAE.
Pseudobulk samples are never trained on here; they are predicted as the bulk→pseudobulk transfer test
(``split = "transfer"``).
"""

from __future__ import annotations

import logging
import time

import numpy as np
import polars as pl
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.preprocessing import StandardScaler

from rna_age.schema import age_mid
from rna_age.train.features import load_matrix, log_cpm, sample_table, select_hvg, shared_genes
from rna_age.train.runs import finish_run, new_run

log = logging.getLogger(__name__)

PRED_COLS = ["sample_id", "source", "split", "study_id", "tissue", "age_type", "age_years", "age_min", "age_max",
             "is_healthy", "is_pseudobulk", "cohort", "in_tage_training", "in_longevitybench"]  # fmt: skip


ADULT_AGE = 20.0


def horvath(age: np.ndarray) -> np.ndarray:
    """Horvath (2013) log-linear age transform: log below the adult age, linear above."""
    a = np.asarray(age, dtype=np.float64)
    return np.where(a <= ADULT_AGE, np.log(a + 1) - np.log(ADULT_AGE + 1), (a - ADULT_AGE) / (ADULT_AGE + 1))


def horvath_inv(f: np.ndarray) -> np.ndarray:
    f = np.asarray(f, dtype=np.float64)
    return np.where(f <= 0, (ADULT_AGE + 1) * np.exp(f) - 1, (ADULT_AGE + 1) * f + ADULT_AGE)


def run_baseline(
    model: str = "ridge",
    n_genes: int = 5000,
    gtex_midpoints: bool = True,
    hvg_sample: int = 20000,
    name: str | None = None,
    transform: str = "none",
) -> dict:
    t0 = time.time()
    s = sample_table()
    s = s.with_columns(target=age_mid(s))
    bulk = s.filter(~pl.col("is_pseudobulk"))
    pseudo = s.filter(pl.col("is_pseudobulk")).with_columns(split=pl.lit("transfer"))
    train_mask = (
        (pl.col("split") == "train")
        & pl.col("is_healthy").fill_null(True)  # healthy or unknown
        & ((pl.col("age_type") == "exact") | (pl.lit(gtex_midpoints) & (pl.col("source") == "gtex")))
    )
    genes = shared_genes()
    train = bulk.filter(train_mask)
    hv = train.sample(min(hvg_sample, train.height), seed=0)
    hvg = select_hvg(log_cpm(load_matrix(hv, genes)), genes, n_genes)
    log.info("HVGs: %d of %d shared genes (chosen on %d train samples)", len(hvg), len(genes), hv.height)
    # library size over the shared genes (consistent CPM denominator); features = log-CPM of HVGs
    allx = pl.concat([bulk, pseudo], how="vertical_relaxed")
    counts = load_matrix(allx, genes)
    lib = counts.sum(axis=1, keepdims=True)
    lib[lib == 0] = 1
    gi = np.array([genes.index(g) for g in hvg])
    X = np.log1p(counts[:, gi] / lib * 1e6).astype(np.float32)
    del counts
    split = allx["split"].to_numpy()
    is_train = allx.select(train_mask.fill_null(False))[:, 0].to_numpy() & ~allx["is_pseudobulk"].to_numpy()
    y_years = allx["target"].to_numpy()
    fwd, inv = (horvath, horvath_inv) if transform == "horvath" else (lambda a: a, lambda a: a)
    y = fwd(np.nan_to_num(y_years, nan=30.0))
    scaler = StandardScaler().fit(X[is_train])
    Xs = scaler.transform(X).astype(np.float32)
    val = (split == "val") & (allx["age_type"].to_numpy() == "exact")
    grid = (
        [{"alpha": a} for a in (1e2, 1e3, 1e4, 3e4, 1e5, 3e5, 1e6)]
        if model == "ridge"
        else [{"alpha": a, "l1_ratio": r} for a in (0.01, 0.05, 0.2) for r in (0.1, 0.5)]
    )
    best = None
    for params in grid:
        est = Ridge(**params) if model == "ridge" else ElasticNet(max_iter=3000, tol=1e-3, **params)
        est.fit(Xs[is_train], y[is_train])
        mae = float(np.mean(np.abs(inv(est.predict(Xs[val])) - y_years[val])))
        log.info("%s %s: val exact MAE %.2f", model, params, mae)
        if best is None or mae < best[0]:
            best = (mae, params, est)
    _, params, est = best
    pred = allx.select([c for c in PRED_COLS if c in allx.columns]).with_columns(
        pred=pl.Series(inv(est.predict(Xs)).astype(np.float64))
    )
    run_dir = new_run(
        name or f"{model}-hvg{n_genes}{'-gtexmid' if gtex_midpoints else ''}{'-horvath' if transform == 'horvath' else ''}",
        {"model": model, "features": f"logCPM HVG {n_genes}", "params": params, "target_transform": transform, "n_shared_genes": len(genes),
         "train_set": "bulk train, healthy-or-unknown, exact" + (" + GTEx midpoints" if gtex_midpoints else ""),
         "n_train": int(is_train.sum()), "hvg_sample": hv.height},
    )  # fmt: skip
    pl.DataFrame({"gene_id": hvg, "coef": est.coef_.astype(np.float64)}).write_parquet(run_dir / "coefficients.parquet")
    row = finish_run(run_dir, pred, {"seconds": time.time() - t0})
    log.info("run %s: %s", run_dir.name, {k: v for k, v in row.items() if k.startswith("test_")})
    return row
