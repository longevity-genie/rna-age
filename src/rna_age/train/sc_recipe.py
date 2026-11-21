"""RM35 step 1b: the recipe of single-cell clocks that work, on our cohorts and folds.

Step 1 showed the loss does not set the shrinkage; the information in the input does. Published clocks that work
(sc-ImmuAging, Li 2025; Muralidharan 2025, brain) differ from ours in what goes in: genes chosen for age, many
resampled pseudobulks per donor, and clocks per cell type. Arms (``arm``), each read by ``ridge`` and ``tabicl``:

- ``agesel``: the donor's whole pseudobulk, the ``n_genes`` genes with the largest |Pearson r| with age on the
  training fold (instead of the most variable ones).
- ``agesel_class``: per cell class with ≥ ``min_cells`` cells in ≥ 50 donors, the ``n_genes // n_classes`` genes with
  the largest |r| with age on that class's pseudobulk in the training fold; features side by side (RM31's concat, with
  genes chosen for age); a donor missing a class gets the fold mean (0 after standardising).
- ``stack``: a clock per class and one on the whole pseudobulk (most variable genes, as RM14), their out-of-fold
  predictions inside the training fold (inner 5-fold) fed to a non-negative linear stacker; a donor missing a class
  gets the whole-pseudobulk prediction in that slot.
- ``boot``: ``n_boot`` pseudobulks per donor, each the mean of ``boot_cells`` of its cached panel cells drawn with
  replacement (BootstrapCell, Buckley 2023); trained on every resample, a donor's prediction the mean over its
  resamples. ``boot1`` is its control: one mean over all the donor's cached cells, same genes and model.

Folds: RM14's (GroupKFold(5) over the cohort's donors in ``donor_profiles`` order), so ``agesel``, ``agesel_class``
and ``stack`` meet the pseudobulk references on identical folds; ``boot`` uses the donors with cached cells (all
but a few). Rows ``recipe-<arm>-<model>`` go to ``data/output/summary/sc_pretrain_{oof,clock}.parquet``.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import polars as pl
from sklearn.model_selection import GroupKFold

from rna_age.train.composition import _cpm
from rna_age.train.sc_pretrain import EVAL_COHORTS, SUMMARY, _merge_summary, cohort_cells
from rna_age.train.sc_tabular import _fit_predict, fold_features

log = logging.getLogger(__name__)

ARMS = ("agesel", "agesel_class", "stack", "boot", "boot1")


def age_genes(X: np.ndarray, y: np.ndarray, n: int) -> np.ndarray:
    """Columns of ``X`` with the largest |Pearson r| with ``y`` (constant columns last)."""
    xc = X - X.mean(0)
    sd = xc.std(0)
    r = np.zeros(X.shape[1])
    ok = sd > 0
    r[ok] = (xc[:, ok] * (y - y.mean())[:, None]).mean(0) / (sd[ok] * y.std())
    return np.argsort(-np.abs(r))[: min(n, X.shape[1])]


def standardise(Xtr: np.ndarray, Xte: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    return (Xtr - mu) / sd, (Xte - mu) / sd


def class_blocks(dp: dict, min_cells: int) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """(class, donors × genes log-CPM, has-class mask) for classes with ≥ ``min_cells`` cells in ≥ 50 donors."""
    out = []
    for c, j in dp["cix"].items():
        ok = dp["N"][:, j] >= min_cells
        if c != "other" and ok.sum() >= 50:
            out.append((c, np.log1p(_cpm(dp["P"][:, j])), ok))
    return out


def agesel_class_features(blocks, tr: np.ndarray, te: np.ndarray, y: np.ndarray, n_genes: int):
    k = max(1, n_genes // max(1, len(blocks)))
    ftr, fte = [], []
    for _, X, ok in blocks:
        rows = tr[ok[tr]]
        cols = age_genes(X[rows], y[rows], k)
        mu, sd = X[rows][:, cols].mean(0), X[rows][:, cols].std(0) + 1e-6
        a = np.where(ok[tr, None], (X[tr][:, cols] - mu) / sd, 0.0)
        b = np.where(ok[te, None], (X[te][:, cols] - mu) / sd, 0.0)
        ftr.append(a)
        fte.append(b)
    return np.hstack(ftr), np.hstack(fte)


def stack_predict(Xw: np.ndarray, blocks, tr: np.ndarray, te: np.ndarray, y: np.ndarray, groups: np.ndarray,
                  model: str, n_genes: int, seed: int) -> np.ndarray:  # fmt: skip
    """Non-negative linear stacker over the whole-pseudobulk clock and the per-class clocks (inner out-of-fold)."""
    from sklearn.linear_model import LinearRegression

    def level0(rows_fit: np.ndarray, rows_pred: np.ndarray) -> np.ndarray:
        cols = []
        a, b = fold_features(Xw, rows_fit, rows_pred, n_genes)
        whole = _fit_predict(model, a, y[rows_fit], b, seed)
        cols.append(whole)
        for _, X, ok in blocks:
            f = rows_fit[ok[rows_fit]]
            p = whole.copy()
            have = ok[rows_pred]
            if have.any() and len(f) >= 20:
                a, b = fold_features(X, f, rows_pred[have], n_genes)
                p[have] = _fit_predict(model, a, y[f], b, seed)
            cols.append(p)
        return np.column_stack(cols)

    Z = np.full((len(tr), 1 + len(blocks)), np.nan)
    for itr, ite in GroupKFold(n_splits=5).split(tr, y[tr], groups[tr]):
        Z[ite] = level0(tr[itr], tr[ite])
    st = LinearRegression(positive=True).fit(Z, y[tr])
    return st.predict(level0(tr, te))


def boot_units(x: np.ndarray, donor: np.ndarray, donors: np.ndarray, n_boot: int, cells: int,
               rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """``n_boot`` means of ``cells`` cells drawn with replacement per donor (n_boot 1, cells 0: one mean of all)."""
    xs, owner = [], []
    for k, d in enumerate(donors):
        idx = np.flatnonzero(donor == d)
        for _ in range(n_boot):
            pick = idx if cells <= 0 else rng.choice(idx, cells, replace=True)
            xs.append(x[pick].astype(np.float32).mean(0))
            owner.append(k)
    return np.stack(xs), np.array(owner)


def run_cohort(cohort: str, arm: str, models: tuple[str, ...] = ("ridge", "tabicl"), n_genes: int = 500,
               min_cells: int = 20, n_boot: int = 20, boot_cells: int = 100, seed: int = 0,
               pool: str = "default") -> pl.DataFrame:  # fmt: skip
    from rna_age.train.fm_embed import clock_folds

    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")
    dp, _ = clock_folds(cohort)
    donors, age, groups = np.array(dp["donors"]), dp["age"], dp["groups"]
    rng = np.random.default_rng(seed)
    if arm.startswith("boot"):
        cc = cohort_cells(cohort, pool, 200, seed)
        keep = np.isin(donors, list(set(cc["donor"][cc["in_all"].astype(bool)].tolist())))
        donors, age, groups = donors[keep], age[keep], groups[keep]
        flag = cc["in_all"].astype(bool)
        xcell, dcell = cc["x"][flag], cc["donor"][flag]
    else:
        Xw = np.log1p(_cpm(dp["P"].sum(1)))
        blocks = class_blocks(dp, min_cells)
    pred = {m: np.full(len(donors), np.nan) for m in models}
    for tr, te in GroupKFold(n_splits=5).split(donors, age, groups):
        for m in models:
            if arm == "agesel":
                cols = age_genes(Xw[tr], age[tr], n_genes)
                a, b = standardise(Xw[tr][:, cols], Xw[te][:, cols])
                pred[m][te] = _fit_predict(m, a, age[tr], b, seed)
            elif arm == "agesel_class":
                a, b = agesel_class_features(blocks, tr, te, age, n_genes)
                pred[m][te] = _fit_predict(m, a, age[tr], b, seed)
            elif arm == "stack":
                pred[m][te] = stack_predict(Xw, blocks, tr, te, age, groups, m, n_genes, seed)
            else:
                nb, nc = (n_boot, boot_cells) if arm == "boot" else (1, 0)
                Utr, otr = boot_units(xcell, dcell, donors[tr], nb, nc, rng)
                Ute, ote = boot_units(xcell, dcell, donors[te], nb, nc, rng)
                cols = np.argsort(-Utr.var(0))[:n_genes]
                a, b = standardise(Utr[:, cols], Ute[:, cols])
                p = _fit_predict(m, a, age[tr][otr], b, seed)
                pred[m][te] = np.bincount(ote, p, len(te)) / np.bincount(ote, minlength=len(te))
    rows = []
    for m, p in pred.items():
        rows += [{"cohort": cohort[:8], "representation": "natural", "model": f"recipe-{arm}-{m}", "donor": d,
                  "group": int(g), "age": float(a), "pred": float(v)}
                 for d, g, a, v in zip(donors, groups, age, p, strict=True)]  # fmt: skip
    return pl.DataFrame(rows)


def run(arm: str, cohorts: tuple[str, ...] = EVAL_COHORTS, models: tuple[str, ...] = ("ridge", "tabicl"),
        n_genes: int = 500, n_boot: int = 20, boot_cells: int = 100, seed: int = 0, write: bool = True,
        n_bootstrap_ci: int = 2000) -> pl.DataFrame:  # fmt: skip
    from rna_age.train.fm_embed import compare

    parts = []
    for c in cohorts:
        t0 = time.time()
        o = run_cohort(c, arm, models, n_genes, n_boot=n_boot, boot_cells=boot_cells, seed=seed)
        parts.append(o)
        log.info("%s %s: %s (%.0fs)", c[:8], arm,
                 {m: round(float((g["pred"] - g["age"]).abs().mean()), 2) for (m,), g in o.group_by("model")},
                 time.time() - t0)  # fmt: skip
    oof = pl.concat(parts)
    cmp = compare(oof, n_boot=n_bootstrap_ci)
    if write:
        run_id = f"rm35-recipe-{arm}"
        _merge_summary(oof.with_columns(run_id=pl.lit(run_id)), SUMMARY / "sc_pretrain_oof.parquet")
        _merge_summary(cmp.with_columns(run_id=pl.lit(run_id)), SUMMARY / "sc_pretrain_clock.parquet")
    return cmp
