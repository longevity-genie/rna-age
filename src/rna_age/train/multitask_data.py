"""RM20 data: one gene panel for a donor-age task and a knockdown-Δ task (CD4 T cells).

- **Panel**: the screen's genes shared with the clock cohort; log-CPM over all shared genes (as ``train_clock``), the
  ``n_genes`` most variable in the cohort's CD4 T pseudobulk, standardised with the cohort's mean and sd. The same
  transform is applied to the screen profiles. Panel choice and standardisation use no ages.
- **Age task**: CD4 T pseudobulk of every cohort donor with ≥ ``min_cells`` cells; CV groups by person.
- **Δ task**: per donor × condition (a *context*), the control = mean of its non-targeting profiles; per target gene ×
  context (a *unit*), the mean of its perturbed profiles (over guides) minus that control.
- **Target-gene features** (input of the learned gene embedding, so it reaches genes never trained on): the target's
  correlation with each panel gene over the cohort's donors (coexpression), and a one-hot of its own panel position
  when it is a panel gene.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from rna_age.train.composition import _cpm, donor_profiles
from rna_age.train.perturb import _obs_column

log = logging.getLogger(__name__)

OBS = ("donor_id", "culture_condition", "guide_id", "perturbed_gene_id", "perturbed_gene_name", "guide_type", "n_cells")


@dataclass
class Rm20Data:
    panel: list[str]  # panel gene ids (Ensembl, no version)
    X_age: np.ndarray  # donors × panel, standardised
    age: np.ndarray  # years
    groups: np.ndarray  # CV groups (person)
    contexts: list[str]  # "donor|condition"
    controls: np.ndarray  # contexts × panel (standardised)
    units: pl.DataFrame  # one row per target gene × context: gene, gene_id, context, ctx (index), condition, donor
    delta: np.ndarray  # units × panel: mean perturbed − control
    gene_feat: dict[str, np.ndarray] = field(default_factory=dict)  # target gene name → feature vector


def _screen_genes(path: Path) -> list[str]:
    import h5py

    with h5py.File(path, "r") as h:
        var = h["var"]
        key = next(k for k in ("gene_ids", "_index") if k in var)
        return [g.split(".")[0] for g in _obs_column(var, key)]


def read_obs(path: Path, columns: tuple[str, ...] = OBS) -> pl.DataFrame:
    import h5py

    with h5py.File(path, "r") as h:
        return pl.DataFrame({c: _obs_column(h["obs"], c) for c in columns})


def _blocks(rows: np.ndarray, chunk: int):
    """Sorted row indices → (first, last, rows) blocks of ≤ ``chunk`` rows."""
    for a in range(0, len(rows), chunk):
        r = rows[a : a + chunk]
        yield int(r[0]), int(r[-1]), r


def read_logcpm_rows(path: Path, rows: np.ndarray, shared_cols: np.ndarray, panel: np.ndarray, chunk: int = 2000):
    """log1p(CPM) of the given screen rows (sorted), library size over the file columns ``shared_cols``, returned on
    ``panel`` (indices into ``shared_cols``); rows × panel float32. X is CSR or dense."""
    import h5py

    out = np.zeros((len(rows), len(panel)), dtype=np.float32)
    with h5py.File(path, "r") as h:
        X = h["X"]
        n_file = X.attrs.get("shape", (0, 0))[1] if hasattr(X, "keys") else X.shape[1]
        remap = np.full(max(int(n_file), int(shared_cols.max()) + 1), -1)
        remap[shared_cols] = np.arange(len(shared_cols))
        done = 0
        if hasattr(X, "keys"):
            indptr, indices, data = X["indptr"][()], X["indices"], X["data"]
        for first, last, r in _blocks(rows, chunk):
            dense = np.zeros((len(r), len(shared_cols)), dtype=np.float32)
            if hasattr(X, "keys"):
                spans = [(first, last + 1)] if last + 1 - first <= 4 * len(r) else [(i, i + 1) for i in r]
                loc = np.full(last + 1 - first, -1)
                loc[r - first] = np.arange(len(r))
                for a, b in spans:
                    lo, hi = indptr[a], indptr[b]
                    idx, val = indices[lo:hi], data[lo:hi]
                    sel = loc[np.repeat(np.arange(a, b), np.diff(indptr[a : b + 1])) - first]
                    keep = (sel >= 0) & (remap[idx] >= 0)
                    dense[sel[keep], remap[idx[keep]]] = val[keep]
            else:
                block = X[first : last + 1][r - first]
                dense[:] = block[:, shared_cols]
            out[done : done + len(r)] = np.log1p(_cpm(dense))[:, panel]
            done += len(r)
            log.info("screen rows %d / %d", done, len(rows))
    return out


def cohort_panel(screen_genes: list[str], cohort: str, n_genes: int = 2000, min_cells: int = 20):
    """Cohort CD4 T log-CPM over the shared genes; the panel (indices into the shared genes), mu, sd, and the
    standardised full shared matrix (for coexpression)."""
    dp = donor_profiles(cohort, "broad")
    k = dp["cix"]["CD4 T"]
    ok = dp["N"][:, k] >= min_cells
    pos = {g: i for i, g in enumerate(dp["genes"])}
    shared = [g for g in screen_genes if g in pos]
    L = np.log1p(_cpm(dp["P"][ok, k][:, [pos[g] for g in shared]]))
    panel = np.argsort(-L.var(0))[: min(n_genes, L.shape[1])]
    mu, sd = L[:, panel].mean(0), L[:, panel].std(0) + 1e-6
    log.info("cohort %s: %d donors, %d shared genes, panel %d", cohort[:8], ok.sum(), len(shared), len(panel))
    return {"shared": shared, "panel": panel, "mu": mu, "sd": sd, "L": L, "age": dp["age"][ok],
            "groups": dp["groups"][ok]}  # fmt: skip


def gene_features(L: np.ndarray, shared: list[str], panel: np.ndarray, targets: dict[str, str]) -> dict[str, np.ndarray]:
    """Per target (name → Ensembl id): [corr with each panel gene over donors, one-hot of its own panel slot]."""
    Z = (L - L.mean(0)) / (L.std(0) + 1e-6)
    Zp = Z[:, panel]
    pos = {g: i for i, g in enumerate(shared)}
    slot = {int(p): i for i, p in enumerate(panel)}
    out = {}
    for name, gid in targets.items():
        f = np.zeros(2 * len(panel), dtype=np.float32)
        i = pos.get(gid)
        if i is not None:
            f[: len(panel)] = Z[:, i] @ Zp / len(Z)
            if i in slot:
                f[len(panel) + slot[i]] = 1.0
        out[name] = f
    return out


def build_units(obs: pl.DataFrame, Y: np.ndarray):
    """Controls per context and Δ per target gene × context from standardised profiles ``Y`` (rows of ``obs``)."""
    obs = obs.with_columns(context=pl.col("donor_id") + "|" + pl.col("culture_condition"))
    contexts = sorted(obs["context"].unique().to_list())
    cix = {c: i for i, c in enumerate(contexts)}
    ctx = np.array([cix[c] for c in obs["context"].to_list()])
    is_ntc = obs["guide_type"].str.contains("(?i)non").to_numpy()
    controls = np.zeros((len(contexts), Y.shape[1]), dtype=np.float32)
    for i in range(len(contexts)):
        m = is_ntc & (ctx == i)
        controls[i] = Y[m].mean(0) if m.any() else np.nan
    t = obs.with_columns(ctx=pl.Series(ctx), row=pl.int_range(obs.height)).filter(pl.Series(~is_ntc))
    t = t.filter(pl.col("perturbed_gene_name").is_not_null())
    keys = (t["perturbed_gene_name"] + "\t" + t["context"]).to_numpy()
    uk, inv = np.unique(keys, return_inverse=True)
    sums = np.zeros((len(uk), Y.shape[1]), dtype=np.float64)
    np.add.at(sums, inv, Y[t["row"].to_numpy()])
    cnt = np.bincount(inv, minlength=len(uk))
    first = np.zeros(len(uk), dtype=int)
    first[inv[::-1]] = np.arange(len(inv))[::-1]
    units = t[first].select(
        gene=pl.col("perturbed_gene_name"), gene_id=pl.col("perturbed_gene_id"), context="context", ctx="ctx",
        donor="donor_id", condition="culture_condition",
    ).with_columns(n_profiles=pl.Series(cnt))  # fmt: skip
    ok = ~np.isnan(controls[units["ctx"].to_numpy()]).any(1)
    delta = (sums / cnt[:, None]).astype(np.float32) - controls[units["ctx"].to_numpy()]
    return contexts, controls, units.filter(pl.Series(ok)), delta[ok]


def load(
    screen: Path,
    cohort: str,
    n_genes: int = 2000,
    min_cells_profile: int = 20,
    subset_genes: int | None = None,
    ntc_per_context: int | None = None,
    seed: int = 0,
    clip: float = 10.0,
) -> Rm20Data:
    """Everything RM20 trains on. ``subset_genes`` / ``ntc_per_context`` read only some target genes and controls (a
    smoke run on the login node)."""
    genes = _screen_genes(screen)
    cp = cohort_panel(genes, cohort, n_genes)
    obs = read_obs(screen)
    keep = (obs["n_cells"] >= min_cells_profile).to_numpy()
    if subset_genes is not None or ntc_per_context is not None:
        rng = np.random.default_rng(seed)
        is_ntc = obs["guide_type"].str.contains("(?i)non").to_numpy()
        if subset_genes is not None:
            names = np.array(sorted(obs.filter(pl.Series(keep & ~is_ntc))["perturbed_gene_name"].unique().to_list()))
            pick = set(rng.choice(names, min(subset_genes, len(names)), replace=False).tolist())
            keep &= is_ntc | obs["perturbed_gene_name"].is_in(list(pick)).to_numpy()
        if ntc_per_context is not None:
            ctx = (obs["donor_id"] + "|" + obs["culture_condition"]).to_numpy()
            drop = np.zeros(len(keep), dtype=bool)
            for c in np.unique(ctx):
                idx = np.flatnonzero(keep & is_ntc & (ctx == c))
                if len(idx) > ntc_per_context:
                    drop[rng.choice(idx, len(idx) - ntc_per_context, replace=False)] = True
            keep &= ~drop
    rows = np.flatnonzero(keep)
    gpos = {g: i for i, g in enumerate(genes)}
    shared_cols = np.array([gpos[g] for g in cp["shared"]])
    Y = read_logcpm_rows(screen, rows, shared_cols, cp["panel"])
    Y = np.clip((Y - cp["mu"]) / cp["sd"], -clip, clip).astype(np.float32)
    contexts, controls, units, delta = build_units(obs[rows], Y)
    Xa = np.clip((cp["L"][:, cp["panel"]] - cp["mu"]) / cp["sd"], -clip, clip).astype(np.float32)
    targets = dict(units.select("gene", "gene_id").unique("gene").iter_rows())
    feats = gene_features(cp["L"], cp["shared"], cp["panel"], targets)
    panel_ids = [cp["shared"][i] for i in cp["panel"]]
    log.info("RM20 data: %d donors, %d contexts, %d units, %d target genes", len(Xa), len(contexts), units.height,
             len(targets))  # fmt: skip
    return Rm20Data(panel_ids, Xa, cp["age"], cp["groups"], contexts, controls, units, delta, feats)
