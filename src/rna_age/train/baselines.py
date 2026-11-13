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
from rna_age.train.features import (
    gene_axis_label,
    load_matrix,
    log_cpm,
    sample_table,
    samples_path,
    select_hvg,
    shared_genes,
)
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


def hvg_design(
    n_genes: int,
    gtex_midpoints: bool = True,
    hvg_sample: int = 20000,
    with_pseudobulk: bool = False,
    samples_suffix: str = "",
    genes_with: tuple[str, ...] = (),
    exclude_studies: tuple[str, ...] = (),
) -> tuple[pl.DataFrame, np.ndarray, np.ndarray, list[str], list[str], int]:
    """Samples (bulk + pseudobulk), log-CPM of train-chosen HVGs, the train mask and the gene list (shared by every
    expression model so that all of them see the same samples, features and splits). The last three arguments are
    RM26's switches (:mod:`rna_age.train.features`): the table, the extra sources narrowing the gene axis, and studies
    kept out of training (and so out of the HVG choice)."""
    s = sample_table(samples_suffix)
    s = s.with_columns(target=age_mid(s))
    bulk = s.filter(~pl.col("is_pseudobulk"))
    pseudo = s.filter(pl.col("is_pseudobulk"))
    if not with_pseudobulk:  # pseudobulk only as a transfer test
        pseudo = pseudo.with_columns(split=pl.lit("transfer"))
    train_mask = (
        (pl.col("split") == "train")
        & pl.col("is_healthy").fill_null(True)  # healthy or unknown
        & ((pl.col("age_type") == "exact") | (pl.lit(gtex_midpoints) & (pl.col("source") == "gtex")))
    )
    if exclude_studies:
        train_mask = train_mask & ~pl.col("study_id").is_in(list(exclude_studies)).fill_null(False)
    genes = shared_genes(with_sources=genes_with)
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
    is_train = allx.select(train_mask.fill_null(False))[:, 0].to_numpy()
    if not with_pseudobulk:
        is_train &= ~allx["is_pseudobulk"].to_numpy()
    return allx, X, is_train, hvg, genes, hv.height


def run_baseline(
    model: str = "ridge",
    n_genes: int = 5000,
    gtex_midpoints: bool = True,
    hvg_sample: int = 20000,
    name: str | None = None,
    transform: str = "none",
    with_pseudobulk: bool = False,
    samples_suffix: str = "",
    genes_with: tuple[str, ...] = (),
    exclude_studies: tuple[str, ...] = (),
) -> dict:
    t0 = time.time()
    allx, X, is_train, hvg, genes, n_hv = hvg_design(
        n_genes, gtex_midpoints, hvg_sample, with_pseudobulk, samples_suffix, genes_with, exclude_studies
    )
    split = allx["split"].to_numpy()
    y_years = allx["target"].to_numpy()
    fwd, inv = (horvath, horvath_inv) if transform == "horvath" else (lambda a: a, lambda a: a)
    y = fwd(np.nan_to_num(y_years, nan=30.0))
    scaler = StandardScaler().fit(X[is_train])
    Xs = scaler.transform(X).astype(np.float32)
    val = (split == "val") & (allx["age_type"].to_numpy() == "exact")
    grid = (
        [{"alpha": a} for a in (1e2, 1e3, 1e4, 3e4, 1e5, 3e5, 1e6)]
        if model == "ridge"
        else [{"learning_rate": 0.1, "max_leaf_nodes": 63, "min_samples_leaf": 40}]
        if model == "hgb"
        else [{"alpha": a, "l1_ratio": r} for a in (0.01, 0.05, 0.2) for r in (0.1, 0.5)]
    )
    best = None
    for params in grid:
        if model == "ridge":
            est = Ridge(**params)
        elif model == "hgb":
            from sklearn.ensemble import HistGradientBoostingRegressor

            est = HistGradientBoostingRegressor(
                loss="absolute_error", max_iter=1500, early_stopping=True, validation_fraction=0.1,
                n_iter_no_change=30, random_state=0, **params,
            )  # fmt: skip
        else:
            est = ElasticNet(max_iter=3000, tol=1e-3, **params)
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
        name or f"{model}-hvg{n_genes}{'-gtexmid' if gtex_midpoints else ''}{'-horvath' if transform == 'horvath' else ''}{'-withpb' if with_pseudobulk else ''}"
        + (f"-genes-{'+'.join(genes_with)}" if genes_with else "")
        + (f"-table{samples_suffix.replace('_', '-')}" if samples_suffix else "")
        + (f"-excl{len(exclude_studies)}" if exclude_studies else ""),
        {"model": model, "features": f"logCPM HVG {n_genes}", "params": params, "target_transform": transform, "n_shared_genes": len(genes),
         "train_set": ("bulk + pseudobulk" if with_pseudobulk else "bulk") + " train, healthy-or-unknown, exact"
         + (" + GTEx midpoints" if gtex_midpoints else ""),
         "n_train": int(is_train.sum()), "hvg_sample": n_hv, "samples_table": samples_path(samples_suffix).name,
         "gene_axis": gene_axis_label(genes_with), "n_gene_axis": len(genes),
         "exclude_train_studies": list(exclude_studies)},
    )  # fmt: skip
    if hasattr(est, "coef_"):
        pl.DataFrame({"gene_id": hvg, "coef": est.coef_.astype(np.float64)}).write_parquet(
            run_dir / "coefficients.parquet"
        )
    row = finish_run(run_dir, pred, {"seconds": time.time() - t0})
    log.info("run %s: %s", run_dir.name, {k: v for k, v in row.items() if k.startswith("test_")})
    return row
