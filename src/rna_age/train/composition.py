"""Composition vs within-cell-type aging signal, from donor × tissue × cell-type pseudobulk.

For one single-cell cohort (Census dataset with ``by_dataset_celltype`` pseudobulk), per donor:
- cell types are mapped to broad classes (``BROAD``);
- ``natural``: the donor's pseudobulk (sum over all cells);
- ``proportions``: the donor's broad-class cell fractions only (no expression);
- ``fixed_composition``: the donor's own per-class expression profiles (CPM) mixed at the cohort-mean fractions,
  so only within-class change can carry age;
- ``composition_only``: the cohort-mean per-class profiles mixed at the donor's own fractions, so only the
  composition can carry age;
- ``class:<c>``: each broad class's pseudobulk alone.
Each representation gets the same clock (ridge on log-CPM of HVGs chosen inside the training fold; 5-fold CV
grouped by donor), and we report MAE and Pearson r on held-out donors. Donors need ≥ ``min_cells`` cells in every
major class used for the counterfactuals.
Output: ``data/output/summary/composition_<dataset>.parquet``.
"""

from __future__ import annotations

import logging
import re

import numpy as np
import polars as pl
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold

from rna_age.paths import INPUT, OUTPUT
from rna_age.sources.bgee import stage_age

log = logging.getLogger(__name__)

BROAD = [
    ("CD8 T", r"cd8"),
    ("CD4 T", r"cd4|regulatory t|helper t|t follicular"),
    ("other T", r"\bt cell|gamma-delta|mucosal invariant|mait|thymocyte|t-cell"),
    ("NK", r"natural killer|\bnk\b"),
    ("B", r"\bb cell|plasmablast|plasma cell|b-cell"),
    ("monocyte", r"monocyte"),
    ("DC", r"dendritic"),
    ("platelet", r"platelet|megakaryocyte"),
    ("granulocyte", r"neutrophil|granulocyte|basophil|eosinophil"),
    ("progenitor", r"progenitor|stem cell"),
]


def broad_class(cell_type: str | None) -> str:
    s = (cell_type or "").lower()
    for name, rx in BROAD:
        if re.search(rx, s):
            return name
    return "other"


def _cpm(x: np.ndarray) -> np.ndarray:
    tot = x.sum(axis=-1, keepdims=True)
    tot[tot == 0] = 1
    return x / tot * 1e6


def _cv_clock(X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_hvg: int = 3000, seed: int = 0) -> dict:
    pred = np.zeros_like(y)
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        if X.shape[1] > n_hvg:
            v = X[tr].var(0)
            keep = np.argsort(-v)[:n_hvg]
        else:
            keep = np.arange(X.shape[1])
        mu, sd = X[tr][:, keep].mean(0), X[tr][:, keep].std(0) + 1e-6
        m = RidgeCV(alphas=np.logspace(-1, 5, 13)).fit((X[tr][:, keep] - mu) / sd, y[tr])
        pred[te] = m.predict((X[te][:, keep] - mu) / sd)
    return {"n": len(y), "mae": float(np.abs(pred - y).mean()), "pearson_r": float(np.corrcoef(pred, y)[0, 1])}


def analyse(
    dataset_id: str,
    min_cells: int = 20,
    major: tuple[str, ...] = ("CD4 T", "CD8 T", "NK", "B", "monocyte"),
    mapping: str = "broad",
    n_classes: int = 12,
    n_major: int = 5,
    null_seed: int | None = None,
):
    """``mapping="broad"``: blood classes (``BROAD``); ``"cell_type"``: the dataset's own labels (tissues), the
    ``n_classes`` most abundant kept (the rest pooled as "other"), the ``n_major`` most abundant as ``major``."""
    d = INPUT / "census_pseudobulk" / "by_dataset_celltype" / dataset_id
    g = pl.read_parquet(d / "groups.parquet")
    counts = pl.read_parquet(d / "counts.parquet")
    genes = counts["gene_id"].to_list()
    ages = [stage_age(s) for s in g["development_stage"].to_list()]
    g = g.with_columns(
        age=pl.Series([a[1] for a in ages], dtype=pl.Float64),
        cls=pl.Series([broad_class(c) for c in g["cell_type"].to_list()])
        if mapping == "broad"
        else pl.col("cell_type"),
        donor=pl.col("donor_id") + ":" + pl.col("tissue"),
    ).filter(pl.col("age").is_not_null())
    if mapping == "cell_type":
        top = g.group_by("cls").agg(pl.col("n_cells").sum()).sort("n_cells", descending=True)["cls"].to_list()
        g = g.with_columns(cls=pl.when(pl.col("cls").is_in(top[:n_classes])).then("cls").otherwise(pl.lit("other")))
        major = tuple(top[:n_major])
    M = counts.select(g["sample_id"].to_list()).to_numpy().T.astype(np.float64)  # groups × genes
    donors = sorted(g["donor"].unique().to_list())
    dix = {x: i for i, x in enumerate(donors)}
    classes = sorted(g["cls"].unique().to_list())
    cix = {c: i for i, c in enumerate(classes)}
    P = np.zeros((len(donors), len(classes), len(genes)))
    N = np.zeros((len(donors), len(classes)))
    for row, (don, cl, nc) in enumerate(
        zip(g["donor"].to_list(), g["cls"].to_list(), g["n_cells"].to_list(), strict=True)
    ):
        P[dix[don], cix[cl]] += M[row]
        N[dix[don], cix[cl]] += nc
    age = g.group_by("donor").agg(pl.col("age").first()).sort("donor")["age"].to_numpy()
    if null_seed is not None:  # null check: donors' ages shuffled, so every representation should give r ≈ 0
        age = np.random.default_rng(null_seed).permutation(age)
    groups = np.arange(len(donors))
    res = []

    def add(rep: str, X: np.ndarray, y: np.ndarray, grp: np.ndarray, note: str = ""):
        r = _cv_clock(X, y, grp)
        res.append({"dataset_id": dataset_id, "representation": rep, **r, "note": note})
        log.info("%s %s: n=%d MAE %.2f r %.3f", dataset_id[:8], rep, r["n"], r["mae"], r["pearson_r"])

    natural = P.sum(1)
    add("natural", np.log1p(_cpm(natural)), age, groups)
    frac = N / N.sum(1, keepdims=True)
    add("proportions", frac, age, groups, f"{len(classes)} broad classes")
    mj = [cix[c] for c in major if c in cix]
    ok = (N[:, mj] >= min_cells).all(1)
    log.info("%d / %d donors have ≥ %d cells in each of %s", ok.sum(), len(donors), min_cells, major)
    prof = _cpm(P[ok][:, mj])  # donors × major × genes (per-class CPM)
    f_d = N[ok][:, mj] / N[ok][:, mj].sum(1, keepdims=True)
    f_ref = f_d.mean(0)
    p_ref = prof.mean(0)
    fixed = np.einsum("c,dcg->dg", f_ref, prof)
    comp_only = np.einsum("dc,cg->dg", f_d, p_ref)
    nat_major = np.einsum("dc,dcg->dg", f_d, prof)
    a_ok, g_ok = age[ok], groups[ok]
    add("natural_major_classes", np.log1p(_cpm(nat_major)), a_ok, g_ok, "donor fractions × donor profiles")
    add("fixed_composition", np.log1p(_cpm(fixed)), a_ok, g_ok, "cohort-mean fractions × donor profiles")
    add("composition_only", np.log1p(_cpm(comp_only)), a_ok, g_ok, "donor fractions × cohort-mean profiles")
    for c in classes:
        k = cix[c]
        okc = N[:, k] >= min_cells
        if okc.sum() >= 50:
            add(f"class:{c}", np.log1p(_cpm(P[okc, k])), age[okc], groups[okc])
    out = pl.DataFrame(res)
    OUTPUT.joinpath("summary").mkdir(parents=True, exist_ok=True)
    out = out.with_columns(mapping=pl.lit(mapping))
    tag = dataset_id[:8] + ("" if mapping == "broad" else f"_{mapping}") + ("" if null_seed is None else "_null")
    out.write_parquet(OUTPUT / "summary" / f"composition_{tag}.parquet")
    frac_df = pl.DataFrame({"donor": donors, "age": age}).with_columns([pl.Series(c, frac[:, cix[c]]) for c in classes])
    frac_df.write_parquet(OUTPUT / "summary" / f"composition_fractions_{tag}.parquet")
    return out
