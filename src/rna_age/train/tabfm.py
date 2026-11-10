"""Tabular foundation models (in-context learners) as age regressors, on the same samples, splits and features as
the baselines.

- Features: ``hvg-pca`` = log-CPM of the train-chosen HVGs (as the MLP), standardised, PCA fitted on train rows
  (these models are pretrained on ≤ a few hundred features: TabPFN-3 uses at most 200 per estimator, TabICLv2 was
  pretrained on ≤ 100); ``atlas`` = BulkFormer-Atlas CLS (768-d), PCA'd the same way.
- Context: the "training set" of an in-context model is its prompt. Each study contributes at most
  ``cap_per_study`` samples (big ARCHS4 series do not dominate, as in fine-tuning), up to ``n_context`` rows.
  Healthy-or-unknown train samples with exact ages, plus GTEx at bin midpoints (as the baselines).
- No fitting of weights and no tuning on val: one configuration per run; val/test/transfer are scored as usual.

The model packages are not project dependencies: jobs run with ``uv run --with <package>`` so the project
environment (and jobs running from it) stays untouched.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import polars as pl

from rna_age.train.baselines import PRED_COLS, hvg_design
from rna_age.train.runs import finish_run, new_run

log = logging.getLogger(__name__)

MODELS = ("tabicl", "tabpfn")


def _estimator(model: str, seed: int):
    if model == "tabicl":
        from tabicl import TabICLRegressor

        return TabICLRegressor(random_state=seed)
    if model == "tabpfn":
        from tabpfn import TabPFNRegressor

        return TabPFNRegressor(device="cuda", random_state=seed, ignore_pretraining_limits=True)
    raise ValueError(f"unknown model {model!r}; expected one of {MODELS}")


def context_rows(allx: pl.DataFrame, is_train: np.ndarray, cap_per_study: int, n_context: int, seed: int) -> np.ndarray:
    """Indices of the prompt: train rows, at most ``cap_per_study`` per study, at most ``n_context`` in all."""
    idx = pl.DataFrame({"i": np.flatnonzero(is_train), "study": allx["study_id"].to_numpy()[is_train]})
    capped = (
        idx.sample(fraction=1.0, shuffle=True, seed=seed).group_by("study", maintain_order=True).head(cap_per_study)
    )
    rows = capped["i"].to_numpy()
    if len(rows) > n_context:
        rows = np.random.default_rng(seed).choice(rows, n_context, replace=False)
    return np.sort(rows)


def _features(kind: str, allx: pl.DataFrame, X: np.ndarray, is_train: np.ndarray, k: int, seed: int):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    if kind == "atlas":
        from rna_age.train.embed import load_embeddings

        emb = load_embeddings("atlas")
        cols = [c for c in emb.columns if c.startswith("e") and c[1:].isdigit()]
        joined = allx.select("sample_id").join(emb, on="sample_id", how="left")
        X = joined.select(cols).to_numpy().astype(np.float32)
    elif kind != "hvg-pca":
        raise ValueError(f"unknown features {kind!r}")
    have = ~np.isnan(X).any(axis=1)
    fit = is_train & have
    Xs = StandardScaler().fit(X[fit]).transform(np.where(have[:, None], X, 0)).astype(np.float32)
    pca = PCA(n_components=min(k, Xs.shape[1]), svd_solver="randomized", random_state=seed).fit(Xs[fit])
    log.info("%s: PCA %d components explain %.3f of train variance", kind, pca.n_components_,
             pca.explained_variance_ratio_.sum())  # fmt: skip
    return pca.transform(Xs).astype(np.float32), have


def run_tabfm(
    model: str = "tabicl",
    features: str = "hvg-pca",
    n_genes: int = 20000,
    n_components: int = 200,
    n_context: int = 10000,
    cap_per_study: int = 200,
    chunk: int = 2048,
    seed: int = 0,
    name: str | None = None,
) -> dict:
    t0 = time.time()
    allx, X, is_train, _, genes, n_hv = hvg_design(n_genes)
    Z, have = _features(features, allx, X, is_train, n_components, seed)
    del X
    y = allx["target"].to_numpy()
    ctx = context_rows(allx, is_train & have & ~np.isnan(y), cap_per_study, n_context, seed)
    n_studies = len(np.unique(allx["study_id"].to_numpy()[ctx].astype(str)))
    log.info("%s on %s: context %d rows from %d studies", model, features, len(ctx), n_studies)
    est = _estimator(model, seed).fit(Z[ctx], y[ctx])
    keep = have & (allx["split"].to_numpy() != "train")  # score val/test/transfer only
    out = np.full(allx.height, np.nan)
    rows = np.flatnonzero(keep)
    for i in range(0, len(rows), chunk):
        part = rows[i : i + chunk]
        out[part] = est.predict(Z[part])
        log.info("predicted %d / %d", min(i + chunk, len(rows)), len(rows))
    pred = (
        allx.select([c for c in PRED_COLS if c in allx.columns])
        .with_columns(pred=pl.Series(out))
        .filter(pl.Series(keep))
    )
    run_dir = new_run(
        name or f"tfm-{model}-{features}{n_components}-ctx{n_context // 1000}k",
        {"model": model, "features": f"{features} ({n_components} PCs of logCPM HVG {n_genes})"
         if features == "hvg-pca" else f"{features} ({n_components} PCs)",
         "n_context": int(len(ctx)), "n_context_studies": n_studies, "cap_per_study": cap_per_study, "seed": seed, "hvg_sample": n_hv,
         "n_shared_genes": len(genes), "context_set": "train, healthy-or-unknown, exact + GTEx midpoints",
         "tuning": "none (zero-shot in-context)"},
    )  # fmt: skip
    row = finish_run(run_dir, pred, {"seconds": time.time() - t0})
    log.info("run %s: %s", run_dir.name, {k: v for k, v in row.items() if k.startswith(("val_", "test_"))})
    return row
