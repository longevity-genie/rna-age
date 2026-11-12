"""Score a perturbation screen with a single-cell clock (RM17): which knockdowns make CD4 T cells read older or
younger.

The clock: CD4 T-cell pseudobulk of every donor of one cohort (default OneK1K-size ``3faad104…``, 981 donors),
log-CPM on the genes the screen measures too, the ``n_genes`` most variable of them, standardised; ridge (and
optionally TabICLv2). The screen: the CD4 T-cell Perturb-seq pseudobulk (``cd4_perturbseq``), one profile per guide ×
donor × condition, the same transform. Absolute ages mean nothing there (cultured, stimulated cells; another
chemistry); the readout is relative: a profile's **shift** = its prediction minus the mean prediction of the
non-targeting guides of the same donor and condition. Per perturbed gene and condition: the mean shift over its guides
and donors, and an empirical p from the non-targeting guides' own shifts (the null of "a guide that does nothing").
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import polars as pl

from rna_age.paths import OUTPUT, download_dir
from rna_age.train.composition import _cpm, donor_profiles

log = logging.getLogger(__name__)

SCREEN = "GWCD4i.pseudobulk_merged.h5ad"


def _decode(a: np.ndarray) -> list[str]:
    return [x.decode() if isinstance(x, bytes) else str(x) for x in a]


def _obs_column(obs, name: str) -> list:
    """An AnnData obs column from h5py: plain, or categorical (codes + categories)."""
    col = obs[name]
    if hasattr(col, "keys") and "codes" in col:
        cats = _decode(col["categories"][()])
        return [cats[c] if c >= 0 else None for c in col["codes"][()]]
    v = col[()]
    return _decode(v) if v.dtype.kind in "SO" else v.tolist()


def read_screen(path: Path, columns: tuple[str, ...], genes: list[str], chunk: int = 2000):
    """(obs table, counts on ``genes`` as rows × genes float32) of an AnnData pseudobulk file, read with h5py, X dense
    or CSR; genes the file lacks are zero."""
    import h5py

    with h5py.File(path, "r") as h:
        obs = pl.DataFrame({c: _obs_column(h["obs"], c) for c in columns})
        var = h["var"]
        gid_key = next(k for k in ("gene_ids", "_index") if k in var)
        file_genes = [g.split(".")[0] for g in _obs_column(var, gid_key)]
        pos = {g: i for i, g in enumerate(file_genes)}
        cols = np.array([pos.get(g, -1) for g in genes])
        have = cols >= 0
        out = np.zeros((obs.height, len(genes)), dtype=np.float32)
        X = h["X"]
        if hasattr(X, "keys"):  # CSR
            indptr, indices, data = X["indptr"][()], X["indices"], X["data"]
            remap = np.full(len(file_genes), -1)
            remap[cols[have]] = np.flatnonzero(have)
            for a in range(0, obs.height, chunk):
                b = min(a + chunk, obs.height)
                lo, hi = indptr[a], indptr[b]
                idx, val = indices[lo:hi], data[lo:hi]
                rows = np.repeat(np.arange(a, b), np.diff(indptr[a : b + 1]))
                keep = remap[idx] >= 0
                out[rows[keep], remap[idx[keep]]] = val[keep]
        else:
            for a in range(0, obs.height, chunk):
                out[a : a + chunk][:, have] = X[a : a + chunk][:, cols[have]]
    return obs, out


def train_clock(genes: list[str], cohort: str, cell_class: str = "CD4 T", n_genes: int = 2000, min_cells: int = 20):
    """Fit the clock on every donor of ``cohort`` (``cell_class`` pseudobulk); returns predict(counts) → ages."""
    from sklearn.linear_model import RidgeCV

    dp = donor_profiles(cohort, "broad")
    k = dp["cix"][cell_class]
    ok = dp["N"][:, k] >= min_cells
    pos = {g: i for i, g in enumerate(dp["genes"])}
    shared = [g for g in genes if g in pos]
    sel = np.array([pos[g] for g in shared])
    X = np.log1p(_cpm(dp["P"][ok, k][:, sel]))
    keep = np.argsort(-X.var(0))[: min(n_genes, X.shape[1])]
    mu, sd = X[:, keep].mean(0), X[:, keep].std(0) + 1e-6
    model = RidgeCV(alphas=np.logspace(-1, 5, 13)).fit((X[:, keep] - mu) / sd, dp["age"][ok])
    gpos = {g: i for i, g in enumerate(genes)}
    screen_cols = np.array([gpos[g] for g in shared])
    gi = screen_cols[keep]
    log.info(
        "clock: %s %s, %d donors, %d shared genes, %d used", cohort[:8], cell_class, ok.sum(), len(shared), len(gi)
    )

    def predict(counts: np.ndarray) -> np.ndarray:
        x = np.log1p(_cpm(counts[:, screen_cols]))  # library size over the shared genes, as in training
        x = x[:, keep]
        return model.predict((x - mu) / sd)

    return predict, {"donors": int(ok.sum()), "shared_genes": len(shared), "genes_used": len(gi)}


def shifts(obs: pl.DataFrame, pred: np.ndarray) -> pl.DataFrame:
    """Each profile's prediction minus the mean of the non-targeting profiles of its donor and condition."""
    d = obs.with_columns(pred=pl.Series(pred))
    ntc = (
        d.filter(pl.col("guide_type").str.contains("(?i)non"))
        .group_by("donor_id", "culture_condition")
        .agg(pl.col("pred").mean().alias("ntc"))
    )
    return d.join(ntc, on=["donor_id", "culture_condition"]).with_columns(shift=pl.col("pred") - pl.col("ntc"))


def per_gene(s: pl.DataFrame) -> pl.DataFrame:
    """Per perturbed gene and condition: mean shift, n profiles, and an empirical two-sided p against the
    non-targeting guides' mean shifts (each non-targeting guide averaged over donors, as a target gene's guides)."""
    is_ntc = s["guide_type"].str.contains("(?i)non")
    null = s.filter(is_ntc).group_by("culture_condition", "guide_id").agg(pl.col("shift").mean().alias("m"))
    tgt = (
        s.filter(~is_ntc)
        .group_by("culture_condition", "perturbed_gene_name")
        .agg(pl.col("shift").mean().alias("shift"), pl.len().alias("n"), pl.col("guide_id").n_unique().alias("guides"))
    )
    out = []
    for (cond,), t in tgt.group_by("culture_condition"):
        nm = np.abs(null.filter(pl.col("culture_condition") == cond)["m"].to_numpy())
        nm.sort()
        a = np.abs(t["shift"].to_numpy())
        p = (1 + len(nm) - np.searchsorted(nm, a, side="left")) / (1 + len(nm))
        out.append(t.with_columns(p=pl.Series(p)))
    return pl.concat(out).sort("p")


def run(cohort: str = "3faad104-2ab8-4434-816d-474d8d2641db", n_genes: int = 2000, min_cells_profile: int = 20):
    path = download_dir("cd4_perturbseq") / SCREEN
    import h5py

    with h5py.File(path, "r") as h:
        var = h["var"]
        gid_key = next(k for k in ("gene_ids", "_index") if k in var)
        genes = [g.split(".")[0] for g in _obs_column(var, gid_key)]
    predict, info = train_clock(genes, cohort, n_genes=n_genes)
    cols = ("donor_id", "culture_condition", "guide_id", "perturbed_gene_name", "guide_type", "n_cells")
    obs, counts = read_screen(path, cols, genes)
    ok = (obs["n_cells"] >= min_cells_profile).to_numpy()
    obs, counts = obs.filter(pl.Series(ok)), counts[ok]
    s = shifts(obs, predict(counts))
    g = per_gene(s)
    s.write_parquet(OUTPUT / "summary" / "perturb_cd4_profiles.parquet")
    g.write_parquet(OUTPUT / "summary" / "perturb_cd4_genes.parquet")
    log.info("scored %d profiles; clock %s", s.height, info)
    return g, info


COEFFICIENTS = "aging_prediction_condition_comparison_regulator_coefficients.csv"


def check(genes: pl.DataFrame, coef: pl.DataFrame, top: float = 0.2) -> pl.DataFrame:
    """D8's check, before any hit is read: per condition, Spearman of the per-gene shifts against the paper's own
    aging coefficients (their CD4 T rows of the same condition), over all shared regulators and over the ``top``
    fraction with the largest |coefficient|. ``validated`` = p < 0.05 over all of them."""
    from scipy.stats import spearmanr

    coef = coef.filter(pl.col("celltype") != "K562").select(
        pl.col("celltype").alias("culture_condition"), pl.col("regulator").alias("perturbed_gene_name"), "coef_mean"
    )
    j = genes.join(coef, on=["culture_condition", "perturbed_gene_name"])
    rows = []
    for (cond,), d in sorted(j.group_by("culture_condition"), key=lambda t: t[0][0]):
        rho, p = spearmanr(d["shift"], d["coef_mean"])
        q = d.filter(pl.col("coef_mean").abs() >= pl.col("coef_mean").abs().quantile(1 - top))
        rho_t, p_t = spearmanr(q["shift"], q["coef_mean"])
        rows.append({"culture_condition": cond, "n": d.height, "rho": float(rho), "p": float(p),
                     "n_top": q.height, "rho_top": float(rho_t), "p_top": float(p_t), "validated": bool(p < 0.05)})  # fmt: skip
    return pl.DataFrame(rows)


def run_check() -> pl.DataFrame:
    g = pl.read_parquet(OUTPUT / "summary" / "perturb_cd4_genes.parquet")
    c = pl.read_csv(download_dir("cd4_perturbseq") / COEFFICIENTS)
    out = check(g, c)
    out.write_parquet(OUTPUT / "summary" / "perturb_cd4_check.parquet")
    return out
