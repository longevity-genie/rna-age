"""RM20 metrics: Δ prediction on held-out genes, baselines, and age-shift agreement with the study's coefficients."""

from __future__ import annotations

import numpy as np
import polars as pl


def _rowcorr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    den = np.sqrt((a * a).sum(1) * (b * b).sum(1))
    return np.where(den > 0, (a * b).sum(1) / np.where(den > 0, den, 1), 0.0)


def deg_recall(pred: np.ndarray, true: np.ndarray, k: int = 50) -> np.ndarray:
    """Per row: share of the true top-k genes by |value| among the predicted top-k by |value|."""
    k = min(k, true.shape[1])
    tp = np.argpartition(-np.abs(true), k - 1, axis=1)[:, :k]
    pp = np.argpartition(-np.abs(pred), k - 1, axis=1)[:, :k]
    return np.array([len(np.intersect1d(x, y)) / k for x, y in zip(tp, pp, strict=True)])


def delta_metrics(pred: np.ndarray, true: np.ndarray, ref: np.ndarray, k: int = 50) -> dict[str, float]:
    """``ref``: the mean Δ of the training knockdowns of each unit's context (the mean perturbed profile, as Δ).
    - pearson_delta, deg_recall50: against control (Δ as measured);
    - pearson_delta_pert, deg_recall50_pert: against the mean perturbed profile (the systematic shift removed)."""
    return {
        "pearson_delta": float(_rowcorr(pred, true).mean()),
        "deg_recall50": float(deg_recall(pred, true, k).mean()),
        "pearson_delta_pert": float(_rowcorr(pred - ref, true - ref).mean()),
        "deg_recall50_pert": float(deg_recall(pred - ref, true - ref, k).mean()),
        "mse_weighted_pert": float(
            (np.abs(true - ref) / (np.abs(true - ref).mean(1, keepdims=True) + 1e-8) * (pred - true) ** 2).mean()
        ),  # fmt: skip
    }


def shift_rho(units: pl.DataFrame, shift: np.ndarray, coef: pl.DataFrame) -> dict[str, float]:
    """Spearman of the per-gene mean shift (over donors) with the study's coefficient, per condition; and their mean.
    ``coef``: condition, gene, coef_mean."""
    from scipy.stats import spearmanr

    g = (
        units.select("gene", "condition").with_columns(shift=pl.Series(shift))
        .group_by("condition", "gene").agg(pl.col("shift").mean())
        .join(coef, on=["condition", "gene"])
    )  # fmt: skip
    out = {}
    for (cond,), d in sorted(g.group_by("condition"), key=lambda t: t[0][0]):
        out[f"rho:{cond}"] = float(spearmanr(d["shift"], d["coef_mean"])[0]) if d.height > 2 else float("nan")
        out[f"n:{cond}"] = float(d.height)
    rhos = [v for k, v in out.items() if k.startswith("rho:") and np.isfinite(v)]
    out["rho_mean"] = float(np.mean(rhos)) if rhos else float("nan")
    return out


def ridge_donor_condition(units_tr: pl.DataFrame, delta_tr: np.ndarray, units_te: pl.DataFrame, alpha: float = 1.0):
    """Ridge on one-hot(donor) + one-hot(condition) over training knockdowns: an additive context mean that reaches a
    held-out donor × condition."""
    from sklearn.linear_model import Ridge

    donors = sorted(set(units_tr["donor"]) | set(units_te["donor"]))
    conds = sorted(set(units_tr["condition"]) | set(units_te["condition"]))

    def onehot(u: pl.DataFrame) -> np.ndarray:
        X = np.zeros((u.height, len(donors) + len(conds)), dtype=np.float32)
        X[np.arange(u.height), [donors.index(d) for d in u["donor"]]] = 1
        X[np.arange(u.height), [len(donors) + conds.index(c) for c in u["condition"]]] = 1
        return X

    return Ridge(alpha=alpha).fit(onehot(units_tr), delta_tr).predict(onehot(units_te)).astype(np.float32)
