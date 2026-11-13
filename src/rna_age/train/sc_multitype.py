"""RM31: several cell types per donor at once, against the whole-sample single-cell clock.

Anton (2026-11-13): bulk clocks see composition implicitly; can a single-cell clock gain from seeing several cell
types of the same donor together? Per cohort, on the donors that have at least ``min_cells`` cells in each of the
``k`` best-covered classes, out-of-fold predictions on the D7 folds (grouped by person) for:

- ``whole``: the donor's whole-sample pseudobulk (the reference, rerun here on the same donors and folds);
- ``concat``: the class pseudobulks side by side, each class's own most variable genes picked and standardised on
  the training fold (``n_genes // k`` per class), plus the log composition fractions of all classes;
- ``mean-types``: the mean of the per-class clocks' predictions (no learned weights, so no inner CV needed);
- ``mean-all``: the mean of the per-class clocks and ``whole``.

The bar (fixed before the run): an arm beats ``whole`` (TabICLv2) beyond the paired bootstrap's 95 % CI, resampling
persons, in at least 4 of the 7 cohorts. Output: ``data/output/summary/sc_multitype.parquet`` (predictions) and
``sc_multitype_compare.parquet``.
"""

from __future__ import annotations

import logging

import numpy as np
import polars as pl
from sklearn.model_selection import GroupKFold

from rna_age.paths import OUTPUT
from rna_age.train.composition import _cpm, donor_profiles
from rna_age.train.fm_embed import paired_bootstrap
from rna_age.train.sc_tabular import _fit_predict, fold_features

log = logging.getLogger(__name__)

PRED = OUTPUT / "summary" / "sc_multitype.parquet"
COMPARE = OUTPUT / "summary" / "sc_multitype_compare.parquet"


def pick_classes(N: np.ndarray, classes: list[str], k: int, min_cells: int, min_donors: int = 50) -> list[int]:
    """Indices of up to ``k`` classes, best covered first, whose joint coverage keeps at least ``min_donors``."""
    order = np.argsort(-(N >= min_cells).sum(0), kind="stable")
    chosen: list[int] = []
    for c in order:
        if classes[c] == "other":
            continue
        trial = [*chosen, int(c)]
        if (N[:, trial] >= min_cells).all(1).sum() >= min_donors:
            chosen = trial
        if len(chosen) == k:
            break
    return chosen


def concat_features(
    blocks: list[np.ndarray], comp: np.ndarray, tr: np.ndarray, te: np.ndarray, n_per: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per block its own training-fold top-variance genes, standardised on the training fold; composition appended."""
    parts_tr, parts_te = [], []
    for B in blocks:
        a, b = fold_features(B, tr, te, n_per)
        parts_tr.append(a)
        parts_te.append(b)
    mu, sd = comp[tr].mean(0), comp[tr].std(0) + 1e-6
    parts_tr.append((comp[tr] - mu) / sd)
    parts_te.append((comp[te] - mu) / sd)
    return np.hstack(parts_tr), np.hstack(parts_te)


def oof(
    X_whole: np.ndarray,
    blocks: list[np.ndarray],
    comp: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model: str,
    n_genes: int,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Out-of-fold predictions of every arm and of each class clock (``type:i``), folds grouped by person."""
    k = len(blocks)
    out = {a: np.full(len(y), np.nan) for a in ("whole", "concat", *(f"type:{i}" for i in range(k)))}
    for tr, te in GroupKFold(n_splits=5).split(X_whole, y, groups):
        a, b = fold_features(X_whole, tr, te, n_genes)
        out["whole"][te] = _fit_predict(model, a, y[tr], b, seed)
        a, b = concat_features(blocks, comp, tr, te, max(1, n_genes // k))
        out["concat"][te] = _fit_predict(model, a, y[tr], b, seed)
        for i, B in enumerate(blocks):
            a, b = fold_features(B, tr, te, n_genes)
            out[f"type:{i}"][te] = _fit_predict(model, a, y[tr], b, seed)
    types = np.stack([out[f"type:{i}"] for i in range(k)])
    out["mean-types"] = types.mean(0)
    out["mean-all"] = np.vstack([types, out["whole"][None]]).mean(0)
    return out


def run_cohort(
    dataset_id: str, model: str = "tabicl", k: int = 5, min_cells: int = 20, n_genes: int = 2000, seed: int = 0
) -> pl.DataFrame:
    """One cohort's predictions; the broad blood classes, or, where they do not apply (brain), the cohort's own
    most frequent cell types. A cohort with fewer than two usable classes is skipped (empty frame)."""
    dp = donor_profiles(dataset_id)
    cls = pick_classes(dp["N"], dp["classes"], k, min_cells)
    if len(cls) < 2:
        dp = donor_profiles(dataset_id, mapping="cell_type")
        cls = pick_classes(dp["N"], dp["classes"], k, min_cells)
    P, N, age, groups, donors, classes = (dp[x] for x in ("P", "N", "age", "groups", "donors", "classes"))
    if len(cls) < 2:
        log.warning("%s: fewer than two cell classes cover enough donors; skipped", dataset_id[:8])
        return pl.DataFrame()
    ok = (N[:, cls] >= min_cells).all(1)
    log.info("%s: classes %s, %d of %d donors", dataset_id[:8], [classes[c] for c in cls], ok.sum(), len(ok))
    X_whole = np.log1p(_cpm(P[ok].sum(1)))
    blocks = [np.log1p(_cpm(P[ok, c])) for c in cls]
    frac = N[ok] / N[ok].sum(1, keepdims=True)
    comp = np.log(frac + 1e-3)
    pred = oof(X_whole, blocks, comp, age[ok], groups[ok], model, n_genes, seed)
    names = {f"type:{i}": f"type:{classes[c]}" for i, c in enumerate(cls)}
    rows = []
    for arm, p in pred.items():
        rows += [{"dataset_id": dataset_id, "arm": names.get(arm, arm), "model": model, "classes": ",".join(classes[c] for c in cls),
                  "donor": d, "group": int(g), "age": float(a), "pred": float(v)}
                 for d, g, a, v in zip(np.array(donors)[ok], groups[ok], age[ok], p, strict=True)]  # fmt: skip
    return pl.DataFrame(rows)


def compare(pred: pl.DataFrame, reference: str = "whole", n_boot: int = 2000) -> pl.DataFrame:
    """Per cohort, model and arm: MAE, and the paired difference to ``reference`` with its CI over persons."""
    rows = []
    for (ds, model), g in pred.group_by("dataset_id", "model", maintain_order=True):
        ref = g.filter(pl.col("arm") == reference).sort("donor")
        e_ref = (ref["pred"] - ref["age"]).abs().to_numpy()
        for (arm,), a in g.group_by("arm", maintain_order=True):
            a = a.sort("donor")
            e = (a["pred"] - a["age"]).abs().to_numpy()
            d, lo, hi = paired_bootstrap(e, e_ref, a["group"].to_numpy(), n_boot)
            rows.append({"dataset_id": ds, "model": model, "arm": arm, "n": a.height, "mae": float(e.mean()),
                         "ref_mae": float(e_ref.mean()), "diff": d, "lo": lo, "hi": hi,
                         "beats_ref": bool(hi < 0) if arm != reference else None})  # fmt: skip
    return pl.DataFrame(rows)


def run(
    cohorts: tuple[str, ...], models: tuple[str, ...] = ("ridge", "tabicl"), k: int = 5, seed: int = 0
) -> pl.DataFrame:
    PRED.parent.mkdir(parents=True, exist_ok=True)
    parts = []
    for c in cohorts:
        for m in models:
            part = run_cohort(c, m, k=k, seed=seed)
            if part.height:
                parts.append(part)
                pl.concat(parts).write_parquet(PRED, compression="zstd")  # after every cohort: a crash keeps the rest
    pred = pl.concat(parts)
    out = compare(pred)
    out.write_parquet(COMPARE, compression="zstd")
    return out
