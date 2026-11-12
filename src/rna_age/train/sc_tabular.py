"""Tabular foundation models as single-cell aging clocks (RM14, docs/DECISIONS.md D7).

On bulk, in-context tabular models lost to a plain network because 20k genes had to be squeezed into 200
components and there were ~47k training samples. Single-cell cohorts are the regime they were made for: the label is
the donor's age, so a cohort is a small table (a few hundred donors).

Per cohort and representation (``natural``: the donor's whole pseudobulk; ``class:<c>``: one cell class alone, for
classes with ≥ ``min_cells`` cells in ≥ 50 donors), donors × log-CPM; 5-fold cross-validation grouped by person.
Inside each training fold: the ``n_genes`` most variable genes, standardised on the fold. Models: ``ridge`` (RidgeCV),
``mlp`` (a small network), ``tabicl`` (TabICLv2, in context: ``uv run --with tabicl``). Out-of-fold predictions per
donor go to ``data/output/summary/sc_tabular_<cohort>.parquet``; :func:`compare` gives each model's MAE and r, and
the MAE difference against ridge with a 95 % CI from resampling donors (D7's bar: the CI excludes 0).
"""

from __future__ import annotations

import logging

import numpy as np
import polars as pl
from sklearn.model_selection import GroupKFold

from rna_age.paths import OUTPUT
from rna_age.train.composition import _cpm, donor_profiles

log = logging.getLogger(__name__)

MODELS = ("ridge", "mlp", "tabicl")


def _fit_predict(model: str, Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, seed: int) -> np.ndarray:
    if model == "ridge":
        from sklearn.linear_model import RidgeCV

        return RidgeCV(alphas=np.logspace(-1, 5, 13)).fit(Xtr, ytr).predict(Xte)
    if model == "mlp":
        from sklearn.neural_network import MLPRegressor

        m = MLPRegressor(hidden_layer_sizes=(128, 32), alpha=1e-2, early_stopping=True, max_iter=2000,
                         random_state=seed)  # fmt: skip
        return m.fit(Xtr, ytr).predict(Xte)
    if model == "tabicl":
        from rna_age.train.tabfm import _estimator

        return np.asarray(_estimator("tabicl", seed).fit(Xtr, ytr).predict(Xte), dtype=np.float64)
    raise ValueError(f"model must be one of {MODELS}, got {model!r}")


def fold_features(X: np.ndarray, tr: np.ndarray, te: np.ndarray, n_genes: int) -> tuple[np.ndarray, np.ndarray]:
    """The training fold's ``n_genes`` most variable columns, standardised on the training fold only."""
    keep = np.argsort(-X[tr].var(0))[: min(n_genes, X.shape[1])]
    mu, sd = X[tr][:, keep].mean(0), X[tr][:, keep].std(0) + 1e-6
    return (X[tr][:, keep] - mu) / sd, (X[te][:, keep] - mu) / sd


def cross_validate(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, models: tuple[str, ...], n_genes: int, seed: int = 0
) -> dict[str, np.ndarray]:
    """Out-of-fold predictions per model, folds grouped by ``groups``."""
    out = {m: np.full(len(y), np.nan) for m in models}
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        Xtr, Xte = fold_features(X, tr, te, n_genes)
        for m in models:
            out[m][te] = _fit_predict(m, Xtr, y[tr], Xte, seed)
    return out


def run(
    dataset_id: str,
    models: tuple[str, ...] = MODELS,
    n_genes: int = 500,
    min_cells: int = 20,
    mapping: str = "broad",
    seed: int = 0,
) -> pl.DataFrame:
    dp = donor_profiles(dataset_id, mapping)
    P, N, age, groups, donors, classes = (dp[k] for k in ("P", "N", "age", "groups", "donors", "classes"))
    reps: list[tuple[str, np.ndarray]] = [("natural", np.ones(len(donors), dtype=bool))]
    for c in classes:
        ok = N[:, dp["cix"][c]] >= min_cells
        if c != "other" and ok.sum() >= 50:
            reps.append((f"class:{c}", ok))
    rows = []
    for rep, ok in reps:
        X = np.log1p(_cpm(P[ok].sum(1) if rep == "natural" else P[ok, dp["cix"][rep[6:]]]))
        pred = cross_validate(X, age[ok], groups[ok], models, n_genes, seed)
        for m, p in pred.items():
            rows += [{"dataset_id": dataset_id, "representation": rep, "model": m, "donor": d, "age": float(a),
                      "pred": float(v)} for d, a, v in zip(np.array(donors)[ok], age[ok], p, strict=True)]  # fmt: skip
        log.info("%s %s: %s", dataset_id[:8], rep,
                 {m: round(float(np.abs(p - age[ok]).mean()), 2) for m, p in pred.items()})  # fmt: skip
    out = pl.DataFrame(rows)
    out.write_parquet(OUTPUT / "summary" / f"sc_tabular_{dataset_id[:8]}.parquet")
    return out


def compare(pred: pl.DataFrame, reference: str = "ridge", n_boot: int = 2000, seed: int = 0) -> pl.DataFrame:
    """Per representation and model: MAE, r, and MAE minus the reference's with a donor-bootstrap 95 % CI."""
    rng = np.random.default_rng(seed)
    rows = []
    for (rep,), d in pred.group_by("representation", maintain_order=True):
        ref = d.filter(pl.col("model") == reference).sort("donor")
        e_ref = (ref["pred"] - ref["age"]).abs().to_numpy()
        for (m,), dm in d.group_by("model", maintain_order=True):
            dm = dm.sort("donor")
            e = (dm["pred"] - dm["age"]).abs().to_numpy()
            diff = e - e_ref
            boots = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n_boot)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            rows.append({"representation": rep, "model": m, "n": len(e), "mae": float(e.mean()),
                         "r": float(np.corrcoef(dm["pred"], dm["age"])[0, 1]), "vs_ref": float(diff.mean()),
                         "lo": float(lo), "hi": float(hi)})  # fmt: skip
    return pl.DataFrame(rows)
