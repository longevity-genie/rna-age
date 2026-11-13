"""RM28: our own single-cell pretraining (masked expression, plus perturbations), then donor age.

Architecture: a masked autoencoder over a cell's 2,000-gene vector (log1p CP10k on a fixed HVG panel), not a
gene-token transformer. The input is the masked vector concatenated with its 0/1 mask indicator (4,000 → 1,024), two
pre-LayerNorm residual MLP blocks, then a 256-d embedding (the exposed layer, :meth:`ScMAE.encode`) that a
residual MLP decoder maps back to 2,000 values. Why: 2,000 gene tokens per cell cost O(2,000²) attention per layer
and would make a pass over ~1M cells take hours on an A100-40GB; the MLP has ~15M parameters, its whole training
matrix fits on the card in fp16 (one epoch in seconds to minutes), and a fixed panel loses nothing a token model
would use here, since every stage reads the same 2,000 genes.

Perturbation objective (arm ``mask+perturb``), on the Marson CD4 Perturb-seq pseudobulk: guide profiles with
≥ ``min_cells`` cells summed per donor × condition × knocked-down gene (targeting) and per donor × condition
(non-targeting: the control profile). Input: the control profile on the panel and the knocked-down gene's token;
output: the knockdown profile on the panel (control + predicted Δ). The gene token is the encoder's own input
weights for that gene (``inp.weight[:, g]``) projected to 256-d, so a held-out gene still has a meaningful token and
the perturbation loss shapes the encoder itself; only knocked-down genes inside the panel are used. Loss: MSE
weighted per gene by 1 + |Δ| / sd(Δ of that gene over the training knockdowns) (DEG-weighted). 20 % of the knocked-down
genes (fixed by seed) are held out for validation and never enter a training batch. The perturbation head keeps the
form (control profile on the panel, gene token) → profile on the panel, so it can be scored later on other screens
(RM29).

Stages:
- :func:`build_pool`: the pretraining pool, i.e. healthy (``disease == normal``), adult (stage ≥ 18 y), primary cells of
  the Census h5ads that are not the 7 evaluation cohorts, donors of the evaluation cohorts removed by donor id too,
  ≤ ``cap`` cells per donor; the gene panel (the ``n_genes`` most variable genes on the pool, mean of per-dataset
  variances, among genes present in every evaluation cohort, every pool dataset and the Perturb-seq panel); the panel
  matrices of the pool cells (fp16 ``.npy``) and the Perturb-seq tensors (``perturb.npz``).
- :func:`pretrain`: per-epoch masked-reconstruction MSE on held-out pool donors against the mean-gene baseline (each
  masked gene predicted by its training mean); stops and records a failure if epoch 1 is not below the baseline.
- :func:`embed_cohorts`: per-donor means of the frozen cell embeddings for each evaluation cohort (RM22's cell sample:
  ≤ 200 cells per donor, class-stratified, plus ≤ 200 CD4 T cells in blood).
- :func:`clock`: ridge and TabICLv2 on those means with RM14's folds and donors (``fm_embed.clock_oof``), compared with
  the stored ``pseudobulk-tabicl-2000g`` predictions by a paired bootstrap (``fm_embed.compare``).
- :func:`finetune`: encoder + linear head on donor age (mean-pooled cells) inside each training fold.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator
from pathlib import Path

import h5py
import numpy as np
import polars as pl
import scipy.sparse as sp
import torch
from torch import nn

from rna_age.paths import DOWNLOAD, INTERIM, OUTPUT

log = logging.getLogger(__name__)

EVAL_COHORTS: tuple[str, ...] = (
    "3faad104-2ab8-4434-816d-474d8d2641db",
    "c838aec3-03ef-4398-b882-0e3912abfff0",
    "218acb0f-9f2f-4f76-b90b-15a4b7c7f629",
    "37a17b78-4864-4a42-b67b-31c00962795a",
    "5c97eeeb-7e52-44b3-b010-b832b1f5424c",
    "5e57cd50-8e42-42d6-940d-5c1660d06864",
    "terekhova2023",
)
TEREKHOVA = "terekhova2023"
TEREKHOVA_DIR = INTERIM / "terekhova2023" / "unpacked" / "all_pbmcs"
H5ADS = DOWNLOAD / "census_pseudobulk" / "h5ads"
PERTURB_H5AD = DOWNLOAD / "cd4_perturbseq" / "GWCD4i.pseudobulk_merged.h5ad"
ROOT = INTERIM / "sc_pretrain"
RUNS = OUTPUT / "runs"
SUMMARY = OUTPUT / "summary"
TARGET_SUM = 10_000
ADULT = 18.0
ARMS = ("mask", "mask+perturb")


# ------------------------------------------------------------------ pool


def pool_datasets(h5ad_dir: Path = H5ADS) -> list[str]:
    """Census h5ads present on disk that are not evaluation cohorts."""
    return sorted(p.stem for p in h5ad_dir.glob("*.h5ad") if p.stem not in EVAL_COHORTS)


def read_obs(h5ad: Path) -> pl.DataFrame:
    """row, donor_id, disease, development_stage, is_primary_data, tissue, cell_type of a Census h5ad."""
    from rna_age.sources.census_pseudobulk import _obs_column

    with h5py.File(h5ad, "r") as f:
        o = f["obs"]
        cols = {c: _obs_column(o, c) for c in ("donor_id", "disease", "development_stage", "tissue", "cell_type")}
        n = len(cols["donor_id"])
        primary = _obs_column(o, "is_primary_data") if "is_primary_data" in o else np.ones(n, dtype=bool)
    return pl.DataFrame({
        "row": np.arange(n, dtype=np.int64),
        **{k: [None if v is None else str(v) for v in a] for k, a in cols.items()},
        "is_primary_data": np.asarray(primary).astype(bool),
    })  # fmt: skip


def eligible_cells(obs: pl.DataFrame, min_age: float = ADULT) -> pl.DataFrame:
    """Primary cells of ``normal`` donors whose stage is at least ``min_age`` (exact, binned or censored), with
    ``age`` (exact years, else the bin's lower bound) and ``age_exact``."""
    from rna_age.sources.bgee import stage_age

    stages = {s: stage_age(s) for s in obs["development_stage"].unique().to_list()}
    age = [stages[s][1] if stages[s][1] is not None else stages[s][2] for s in obs["development_stage"].to_list()]
    exact = [stages[s][0] == "exact" for s in obs["development_stage"].to_list()]
    return obs.with_columns(age=pl.Series(age, dtype=pl.Float64), age_exact=pl.Series(exact)).filter(
        pl.col("is_primary_data")
        & (pl.col("disease") == "normal")
        & pl.col("donor_id").is_not_null()
        & (pl.col("age") >= min_age)
    )


def cap_per_donor(cells: pl.DataFrame, cap: int, seed: int = 0, key: str = "donor_id") -> pl.DataFrame:
    """≤ ``cap`` cells per donor, a fixed-seed random sample; sorted by row."""
    rng = np.random.default_rng(seed)
    keep = []
    for (_,), d in cells.sort("row").group_by(key, maintain_order=True):
        r = d["row"].to_numpy()
        keep.append(r if len(r) <= cap else np.sort(rng.choice(r, cap, replace=False)))
    rows = np.concatenate(keep) if keep else np.array([], dtype=np.int64)
    return cells.filter(pl.col("row").is_in(rows.tolist())).sort("row")


def eval_donors() -> pl.DataFrame:
    """cohort, donor_id of every evaluation cohort (Census: obs ``donor_id``; Terekhova: ``Donor_id`` and ``Tube_id``)."""
    from rna_age.sources.census_pseudobulk import _obs_column

    parts = []
    for c in EVAL_COHORTS:
        if c == TEREKHOVA:
            meta = pl.read_csv(TEREKHOVA_DIR / "all_pbmcs_metadata.csv", columns=["Donor_id", "Tube_id"],
                               infer_schema_length=10000)  # fmt: skip
            ids = set(meta["Donor_id"].drop_nulls().cast(pl.String)) | set(meta["Tube_id"].drop_nulls().cast(pl.String))
        else:
            with h5py.File(H5ADS / f"{c}.h5ad", "r") as f:
                ids = {str(x) for x in _obs_column(f["obs"], "donor_id") if x is not None}
        parts.append(pl.DataFrame({"cohort": c, "donor_id": sorted(ids)}))
    return pl.concat(parts)


def donor_overlap(pool: pl.DataFrame, evals: pl.DataFrame) -> pl.DataFrame:
    """Pool (dataset_id, donor_id) rows whose donor_id is also an evaluation cohort's donor id."""
    return pool.select("dataset_id", "donor_id").unique().join(evals, on="donor_id", how="inner")


def exclude_eval(pool: pl.DataFrame, evals: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(pool without evaluation datasets and without donors sharing an id with an evaluation donor, the overlap)."""
    pool = pool.filter(~pl.col("dataset_id").is_in(list(EVAL_COHORTS)))
    ov = donor_overlap(pool, evals)
    return pool.filter(~pl.col("donor_id").is_in(ov["donor_id"].implode())), ov


# ------------------------------------------------------------------ genes


def perturb_genes(path: Path = PERTURB_H5AD) -> pl.DataFrame:
    """gene_id (Ensembl), symbol of the Perturb-seq measured panel."""
    from rna_age.sources.census_pseudobulk import _obs_column

    with h5py.File(path, "r") as f:
        ids = [str(x).split(".")[0] for x in _obs_column(f["var"], "gene_ids")]
        names = [str(x) for x in _obs_column(f["var"], "gene_name")]
    return pl.DataFrame({"gene_id": ids, "symbol": names}).filter(pl.col("gene_id").str.starts_with("ENSG"))


def file_gene_ids(h5ad: Path, sym2ens: dict[str, str] | None = None) -> list[str | None]:
    """Ensembl ids (no version) of an h5ad's matrix columns; symbols (Terekhova) mapped through ``sym2ens``."""
    from rna_age.train.fm_embed import var_ids

    v = var_ids(h5ad)
    if sym2ens is None:
        return [x.split(".")[0] if x.startswith("ENSG") else None for x in v]
    return [sym2ens.get(x) for x in v]


def symbol_map(genes: pl.DataFrame) -> dict[str, str]:
    """symbol → Ensembl id, symbols that name one gene only."""
    u = genes.group_by("symbol").agg(pl.col("gene_id").first(), pl.len().alias("k")).filter(pl.col("k") == 1)
    return dict(zip(u["symbol"].to_list(), u["gene_id"].to_list(), strict=True))


def cohort_h5ad(cohort: str) -> Path:
    return TEREKHOVA_DIR / "all_pbmcs_rna.h5ad" if cohort == TEREKHOVA else H5ADS / f"{cohort}.h5ad"


def columns_for(ids: list[str | None], panel: list[str]) -> np.ndarray:
    """For each panel gene its column in a file (first occurrence), -1 if absent."""
    pos: dict[str, int] = {}
    for i, g in enumerate(ids):
        if g is not None and g not in pos:
            pos[g] = i
    return np.array([pos.get(g, -1) for g in panel], dtype=np.int64)


def log_cp10k(X: sp.csr_matrix) -> sp.csr_matrix:
    """log1p of counts per 10k (library size = the row's sum over all of the file's genes)."""
    lib = np.asarray(X.sum(axis=1)).ravel().astype(np.float64)
    lib[lib == 0] = 1
    out = sp.csr_matrix(sp.diags(TARGET_SUM / lib) @ X)
    out.data = np.log1p(out.data)
    return out


def panel_dense(X: sp.csr_matrix, cols: np.ndarray) -> np.ndarray:
    """Dense rows × panel (float32), zeros for panel genes the file lacks."""
    out = np.zeros((X.shape[0], len(cols)), dtype=np.float32)
    have = cols >= 0
    if have.any():
        out[:, have] = X[:, cols[have]].toarray()
    return out


def panel_rows(h5ad: Path, rows: np.ndarray, cols: np.ndarray, log_input: bool = False,
               window: int = 50_000) -> Iterator[tuple[np.ndarray, np.ndarray]]:  # fmt: skip
    """(rows, log1p CP10k on the panel) for ``rows`` of an h5ad, in the file's row order. ``log_input``: the matrix
    holds log1p CP10k already (Terekhova), turned back into CP10k before renormalising."""
    from rna_age.train.fm_embed import read_rows

    for r, X in read_rows(h5ad, rows, window=window):
        if log_input:
            X.data = np.expm1(X.data)
        yield r, panel_dense(log_cp10k(X), cols)


def choose_panel(stats: dict[str, tuple[np.ndarray, np.ndarray]], candidates: list[str], n: int) -> pl.DataFrame:
    """The ``n`` candidates with the largest mean (over datasets) of their per-dataset variance of log1p CP10k.
    ``stats``: dataset → (mean, variance) over ``candidates``."""
    var = np.mean([v for _, v in stats.values()], axis=0)
    mean = np.mean([m for m, _ in stats.values()], axis=0)
    order = np.argsort(-var, kind="stable")[: min(n, len(candidates))]
    return pl.DataFrame({"gene_id": [candidates[i] for i in order], "rank": np.arange(len(order), dtype=np.int32),
                         "hvg_var": var[order], "pool_mean": mean[order]})  # fmt: skip


def pool_dir(name: str = "default") -> Path:
    return ROOT / name


def build_pool(name: str = "default", cap: int = 2000, hvg_cap: int = 200, n_genes: int = 2000, seed: int = 0,
               max_cells_per_dataset: int | None = None, datasets: tuple[str, ...] | None = None,
               perturb_max_rows: int | None = None, min_cells_profile: int = 20) -> dict:  # fmt: skip
    """Pool table, gene panel, pool panel matrices and Perturb-seq tensors under ``data/interim/sc_pretrain/<name>``;
    for ``name == "default"`` the pool and panel also go to ``data/output/summary/sc_pretrain_{pool,panel}.parquet``.
    ``max_cells_per_dataset`` and ``perturb_max_rows`` are for smoke runs."""
    t0 = time.time()
    out = pool_dir(name)
    out.mkdir(parents=True, exist_ok=True)
    datasets = tuple(datasets) if datasets else tuple(pool_datasets())
    bad = set(datasets) & set(EVAL_COHORTS)
    if bad:
        raise ValueError(f"evaluation cohorts cannot be pretraining data: {sorted(bad)}")
    evals = eval_donors()
    parts, census = [], []
    for ds in datasets:
        obs = read_obs(H5ADS / f"{ds}.h5ad")
        e = eligible_cells(obs).with_columns(dataset_id=pl.lit(ds))
        census.append({"dataset_id": ds, "cells_file": obs.height, "cells_eligible": e.height,
                       "donors_eligible": e["donor_id"].n_unique()})  # fmt: skip
        log.info("%s: %d cells, %d eligible (%d donors)", ds[:8], obs.height, e.height, e["donor_id"].n_unique())
        if e.height:
            parts.append(e)
    pool, overlap = exclude_eval(pl.concat(parts), evals)
    if overlap.height:
        log.warning("donor ids shared with evaluation cohorts, removed from the pool: %s", overlap.to_dicts())
    capped = []
    for _, d in pool.group_by("dataset_id", maintain_order=True):
        c = cap_per_donor(d, cap, seed)
        if max_cells_per_dataset and c.height > max_cells_per_dataset:
            c = c.sample(max_cells_per_dataset, seed=seed).sort("row")
        capped.append(c)
    pool = pl.concat(capped)
    # genes: Perturb-seq ∩ every evaluation cohort ∩ every pool dataset
    pg = perturb_genes()
    sym2ens = symbol_map(pg)
    cand = set(pg["gene_id"])
    for c in EVAL_COHORTS:
        cand &= {g for g in file_gene_ids(cohort_h5ad(c), sym2ens if c == TEREKHOVA else None) if g}
    log.info("candidates after Perturb-seq ∩ evaluation cohorts: %d", len(cand))
    ids = {ds: file_gene_ids(H5ADS / f"{ds}.h5ad") for ds in pool["dataset_id"].unique().sort()}
    for v in ids.values():
        cand &= {g for g in v if g}
    candidates = sorted(cand)
    log.info("candidates after every pool dataset: %d", len(candidates))
    # HVG pass on ≤ hvg_cap cells per donor
    stats = {}
    for ds, v in ids.items():
        d = pool.filter(pl.col("dataset_id") == ds)
        sub = cap_per_donor(d, hvg_cap, seed + 1)
        cols = columns_for(v, candidates)
        s1, s2, n = np.zeros(len(candidates)), np.zeros(len(candidates)), 0
        for _, x in panel_rows(H5ADS / f"{ds}.h5ad", sub["row"].to_numpy(), cols):
            s1 += x.sum(0)
            s2 += (x.astype(np.float64) ** 2).sum(0)
            n += len(x)
        m = s1 / n
        stats[ds] = (m, s2 / n - m**2)
        log.info("HVG stats %s: %d cells (%.0fs)", ds[:8], n, time.time() - t0)
    panel = choose_panel(stats, candidates, n_genes).join(pg, on="gene_id", how="left")
    panel.write_parquet(out / "panel.parquet")
    genes = panel["gene_id"].to_list()
    # pool matrices
    pool = pool.with_columns(offset=pl.int_range(pl.len()).over("dataset_id").cast(pl.Int64))
    for ds, v in ids.items():
        d = pool.filter(pl.col("dataset_id") == ds)
        cols = columns_for(v, genes)
        mm = np.lib.format.open_memmap(out / f"cells_{ds[:8]}.npy.tmp", mode="w+", dtype=np.float16,
                                       shape=(d.height, len(genes)))  # fmt: skip
        k = 0
        for r, x in panel_rows(H5ADS / f"{ds}.h5ad", d["row"].to_numpy(), cols):
            mm[k : k + len(r)] = x.astype(np.float16)
            k += len(r)
        assert k == d.height, (ds, k, d.height)
        mm.flush()
        del mm
        (out / f"cells_{ds[:8]}.npy.tmp").rename(out / f"cells_{ds[:8]}.npy")
        log.info("pool matrix %s: %d cells (%.0fs)", ds[:8], d.height, time.time() - t0)
    pool.select("dataset_id", "row", "offset", "donor_id", "tissue", "cell_type", "development_stage", "age",
                "age_exact").write_parquet(out / "pool.parquet")  # fmt: skip
    build_perturb(genes, out / "perturb.npz", max_rows=perturb_max_rows, min_cells=min_cells_profile)
    donors = pool.group_by("dataset_id", "donor_id").agg(pl.len().alias("cells"), pl.col("age").first(),
                                                          pl.col("age_exact").first()).sort("dataset_id", "donor_id")  # fmt: skip
    overlap.write_parquet(out / "donor_overlap.parquet")
    pl.DataFrame(census).write_parquet(out / "datasets.parquet")
    if name == "default":
        SUMMARY.mkdir(parents=True, exist_ok=True)
        donors.write_parquet(SUMMARY / "sc_pretrain_pool.parquet")
        panel.write_parquet(SUMMARY / "sc_pretrain_panel.parquet")
    info = {"datasets": int(donors["dataset_id"].n_unique()), "donors": donors.height, "cells": pool.height,
            "genes": len(genes), "candidates": len(candidates), "overlap_donors": overlap.height,
            "seconds": round(time.time() - t0)}  # fmt: skip
    (out / "build.json").write_text(json.dumps(info | {"cap": cap, "seed": seed, "n_genes": n_genes}, indent=2))
    log.info("pool %s: %s", name, info)
    return info


# ------------------------------------------------------------------ perturbation data


def _read_rows_panel(path: Path, rows: np.ndarray, cols: np.ndarray, chunk: int = 5000) -> np.ndarray:
    """Dense counts (rows × panel, float32) of selected rows of a CSR h5ad, read in contiguous runs."""
    from rna_age.sources.census_pseudobulk import _var_ids

    rows = np.sort(rows)
    out = np.zeros((len(rows), len(cols)), dtype=np.float32)
    with h5py.File(path, "r") as f:
        X = f["X"]
        n_var = len(_var_ids(f, False))
        remap = np.full(n_var, -1, dtype=np.int64)
        have = cols >= 0
        remap[cols[have]] = np.flatnonzero(have)
        indptr = X["indptr"]
        breaks = np.flatnonzero(np.diff(rows) != 1) + 1
        k = 0
        for run in np.split(rows, breaks):
            for a in range(0, len(run), chunk):
                r = run[a : a + chunk]
                lo, hi = int(r[0]), int(r[-1]) + 1
                ptr = indptr[lo : hi + 1]
                idx, val = X["indices"][ptr[0] : ptr[-1]], X["data"][ptr[0] : ptr[-1]]
                which = np.repeat(np.arange(len(r)), np.diff(ptr))
                g = remap[idx]
                keep = g >= 0
                out[k + which[keep], g[keep]] = val[keep]
                k += len(r)
    return out


def build_perturb(panel: list[str], path: Path, max_rows: int | None = None, min_cells: int = 20) -> dict:
    """Control (non-targeting) and knockdown profiles on the panel, log1p CP10k, per donor × condition (× gene).
    Saved as ``perturb.npz``: ``ctx`` names, ``x_ctl`` (contexts × panel), ``x_kd``, ``kd_ctx``, ``kd_gene``
    (panel index of the knocked-down gene), ``kd_n_cells``, ``panel``."""
    from rna_age.sources.census_pseudobulk import _obs_column

    with h5py.File(PERTURB_H5AD, "r") as f:
        o = f["obs"]
        obs = pl.DataFrame({c: [str(x) for x in _obs_column(o, c)] for c in
                            ("donor_id", "culture_condition", "guide_type", "perturbed_gene_id")} |
                           {"n_cells": _obs_column(o, "n_cells").astype(np.float64),
                            "total_counts": _obs_column(o, "total_counts").astype(np.float64)})  # fmt: skip
        file_ids = [str(x).split(".")[0] for x in _obs_column(f["var"], "gene_ids")]
    pix = {g: i for i, g in enumerate(panel)}
    obs = obs.with_row_index("row").with_columns(
        ctx=pl.col("donor_id") + "|" + pl.col("culture_condition"),
        ntc=pl.col("guide_type").str.contains("(?i)non"),
        gene=pl.col("perturbed_gene_id").str.split(".").list.first().replace_strict(pix, default=-1, return_dtype=pl.Int64),
    ).filter(pl.col("n_cells") >= min_cells)  # fmt: skip
    obs = obs.filter(pl.col("ntc") | (pl.col("gene") >= 0))
    if max_rows:  # smoke: one context, its controls and its first targeting rows
        first = obs["ctx"][0]
        o1 = obs.filter(pl.col("ctx") == first)
        obs = pl.concat([o1.filter(pl.col("ntc")), o1.filter(~pl.col("ntc")).head(max_rows)]).sort("row")
    counts = _read_rows_panel(PERTURB_H5AD, obs["row"].to_numpy(), columns_for(file_ids, panel))
    obs = obs.sort("row").with_row_index("k")

    def pseudo(t: pl.DataFrame, keys: list[str]) -> tuple[pl.DataFrame, np.ndarray]:
        """Guide profiles summed per ``keys`` (counts and library sizes), then log1p CP10k."""
        grp = t.group_by(keys, maintain_order=True).agg(pl.col("k"), pl.col("total_counts").sum(),
                                                        pl.col("n_cells").sum())  # fmt: skip
        X = np.stack([counts[k].sum(0) for k in grp["k"].to_list()])
        return grp.drop("k"), np.log1p(X / grp["total_counts"].to_numpy()[:, None] * TARGET_SUM).astype(np.float32)

    ctl_tab, x_ctl = pseudo(obs.filter(pl.col("ntc")), ["ctx"])
    kd_tab, x_kd = pseudo(obs.filter(~pl.col("ntc") & pl.col("ctx").is_in(ctl_tab["ctx"].implode())), ["ctx", "gene"])
    cix = {c: i for i, c in enumerate(ctl_tab["ctx"].to_list())}
    np.savez(path, ctx=np.array(ctl_tab["ctx"].to_list()), x_ctl=x_ctl, x_kd=x_kd.astype(np.float16),
             kd_ctx=np.array([cix[c] for c in kd_tab["ctx"].to_list()], dtype=np.int64),
             kd_gene=kd_tab["gene"].to_numpy().astype(np.int64), kd_n_cells=kd_tab["n_cells"].to_numpy(),
             panel=np.array(panel))  # fmt: skip
    info = {"contexts": len(cix), "knockdowns": kd_tab.height, "genes": int(kd_tab["gene"].n_unique())}
    log.info("perturbation tensors: %s", info)
    return info


# ------------------------------------------------------------------ model


class ResBlock(nn.Module):
    def __init__(self, d: int, dropout: float):
        super().__init__()
        self.f = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.f(x)


class ScMAE(nn.Module):
    """Masked autoencoder on a panel vector, with a perturbation head. ``encode`` is the exposed embedding."""

    def __init__(self, n_genes: int, hidden: int = 1024, emb: int = 256, depth: int = 2, dropout: float = 0.1):
        super().__init__()
        self.n_genes = n_genes
        self.inp = nn.Linear(2 * n_genes, hidden)
        self.blocks = nn.Sequential(*[ResBlock(hidden, dropout) for _ in range(depth)])
        self.to_emb = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, emb))
        self.dec = nn.Sequential(nn.Linear(emb, hidden), nn.GELU(), ResBlock(hidden, dropout), nn.LayerNorm(hidden),
                                 nn.Linear(hidden, n_genes))  # fmt: skip
        self.gene_proj = nn.Linear(hidden, emb)
        self.pert_head = nn.Sequential(nn.LayerNorm(emb), nn.Linear(emb, hidden), nn.GELU(), nn.Linear(hidden, n_genes))

    def encode(self, x: torch.Tensor, m: torch.Tensor | None = None) -> torch.Tensor:
        if m is None:
            m = torch.zeros_like(x)
        m = m.to(x.dtype)
        return self.to_emb(self.blocks(self.inp(torch.cat([x * (1 - m), m], dim=-1))))

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        return self.dec(self.encode(x, m))

    def gene_token(self, g: torch.Tensor) -> torch.Tensor:
        """A gene's token: the encoder's input weights for its value, projected to the embedding width."""
        return self.gene_proj(self.inp.weight[:, g].T)

    def perturb(self, x_ctl: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """Predicted knockdown profile on the panel from a control profile and the knocked-down gene (panel index)."""
        return x_ctl + self.pert_head(self.encode(x_ctl) + self.gene_token(g))


def sample_mask(n: int, g: int, lo: float, hi: float, generator: torch.Generator | None = None,
                device: str | torch.device = "cpu") -> torch.Tensor:  # fmt: skip
    """Boolean mask, each row hiding a fraction drawn uniformly in [lo, hi] of its genes (exact count, random genes)."""
    rate = lo + (hi - lo) * torch.rand(n, 1, generator=generator, device=device)
    k = torch.clamp((rate * g).round(), min=1).long()
    ranks = torch.rand(n, g, generator=generator, device=device).argsort(dim=1).argsort(dim=1)
    return ranks < k


def masked_mse(pred: torch.Tensor, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """Mean squared error over masked positions only."""
    m = m.to(pred.dtype)
    return ((pred - x) ** 2 * m).sum() / m.sum().clamp(min=1)


def mean_gene_mse(gene_mean: torch.Tensor, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """The baseline: each masked gene predicted by its training mean."""
    return masked_mse(gene_mean.expand_as(x).to(x.dtype), x, m)


def early_stop_failed(epoch: int, val_mse: float, baseline_mse: float) -> bool:
    """The plan's stop rule: after epoch 1 the masked loss must be below the mean-gene baseline."""
    return epoch == 1 and not val_mse < baseline_mse


# ------------------------------------------------------------------ perturbation batches


def heldout_genes(genes: np.ndarray, frac: float = 0.2, seed: int = 0) -> np.ndarray:
    """The held-out ``frac`` of the distinct knocked-down genes (fixed by seed)."""
    u = np.unique(genes)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(u, max(1, round(frac * len(u))), replace=False)) if len(u) > 1 else np.array([], int)


def perturb_split(kd_gene: np.ndarray, frac: float = 0.2, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """(training example indices, validation example indices): validation = every example of a held-out gene."""
    held = np.isin(kd_gene, heldout_genes(kd_gene, frac, seed))
    return np.flatnonzero(~held), np.flatnonzero(held)


def perturb_batches(train_idx: np.ndarray, batch: int, rng: np.random.Generator) -> Iterator[np.ndarray]:
    """Endless batches of training examples (reshuffled each pass); only ``train_idx`` ever appears."""
    while True:
        p = rng.permutation(train_idx)
        for a in range(0, len(p), batch):
            yield p[a : a + batch]


def deg_weights(delta: torch.Tensor, sd: torch.Tensor) -> torch.Tensor:
    """Per example × gene weight 1 + |Δ| / sd(Δ of the gene over the training knockdowns)."""
    return 1 + delta.abs() / sd.clamp(min=1e-3)


def weighted_mse(pred: torch.Tensor, target: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return ((pred - target) ** 2 * w).sum() / w.sum()


class PerturbData:
    """The perturbation tensors on a device, split into training and held-out genes."""

    def __init__(self, path: Path, device, frac: float = 0.2, seed: int = 0):
        z = np.load(path)
        self.x_ctl = torch.from_numpy(z["x_ctl"].astype(np.float32)).to(device)
        self.x_kd = torch.from_numpy(z["x_kd"].astype(np.float32)).to(device)
        self.ctx = torch.from_numpy(z["kd_ctx"]).to(device)
        self.gene = torch.from_numpy(z["kd_gene"]).to(device)
        self.train_idx, self.val_idx = perturb_split(z["kd_gene"], frac, seed)
        delta = self.x_kd - self.x_ctl[self.ctx]
        tr = torch.from_numpy(self.train_idx).to(device)
        self.sd = delta[tr].std(0) if len(tr) > 1 else torch.ones(delta.shape[1], device=device)
        self.mean_delta = torch.stack([delta[tr][self.ctx[tr] == c].mean(0) if (self.ctx[tr] == c).any()
                                       else torch.zeros(delta.shape[1], device=device)
                                       for c in range(len(self.x_ctl))])  # fmt: skip

    def loss(self, model: ScMAE, idx: np.ndarray) -> torch.Tensor:
        i = torch.from_numpy(idx).to(self.ctx.device)
        c = self.ctx[i]
        target = self.x_kd[i]
        w = deg_weights(target - self.x_ctl[c], self.sd)
        return weighted_mse(model.perturb(self.x_ctl[c], self.gene[i]), target, w)

    @torch.no_grad()
    def evaluate(self, model: ScMAE, idx: np.ndarray) -> dict[str, float]:
        """Held-out genes: weighted MSE of the model, of 'no change' and of the context's mean training Δ, and the
        mean per-example Pearson r of predicted against observed Δ."""
        if not len(idx):
            return {}
        i = torch.from_numpy(idx).to(self.ctx.device)
        c = self.ctx[i]
        x0, target = self.x_ctl[c], self.x_kd[i]
        w = deg_weights(target - x0, self.sd)
        pred = model.perturb(x0, self.gene[i]).float()
        dp, do = pred - x0, target - x0
        dp, do = dp - dp.mean(1, keepdim=True), do - do.mean(1, keepdim=True)
        r = (dp * do).sum(1) / (dp.norm(dim=1) * do.norm(dim=1)).clamp(min=1e-8)
        return {"pert_val_mse": float(weighted_mse(pred, target, w)),
                "pert_val_mse_nochange": float(weighted_mse(x0, target, w)),
                "pert_val_mse_meandelta": float(weighted_mse(x0 + self.mean_delta[c], target, w)),
                "pert_val_delta_r": float(r.mean())}  # fmt: skip


# ------------------------------------------------------------------ pretraining


def load_pool_matrix(name: str, max_cells: int | None = None, seed: int = 0) -> tuple[pl.DataFrame, np.ndarray]:
    """(pool table in matrix order, cells × panel float16)."""
    d = pool_dir(name)
    pool = pl.read_parquet(d / "pool.parquet")
    parts, tabs = [], []
    for (ds,), t in pool.group_by("dataset_id", maintain_order=True):
        parts.append(np.load(d / f"cells_{ds[:8]}.npy", mmap_mode="r"))
        tabs.append(t.sort("offset"))
    pool = pl.concat(tabs)
    X = np.concatenate([np.asarray(p) for p in parts])
    if max_cells and len(X) > max_cells:
        keep = np.sort(np.random.default_rng(seed).choice(len(X), max_cells, replace=False))
        pool, X = pool[keep], X[keep]
    return pool, X


def split_donors(pool: pl.DataFrame, frac: float = 0.05, seed: int = 0) -> np.ndarray:
    """Boolean per cell: belongs to a validation donor (``frac`` of the pool's donors, at least one)."""
    key = (pl.col("dataset_id") + ":" + pl.col("donor_id")).alias("k")
    donors = np.array(sorted(pool.select(key)["k"].unique().to_list()))
    rng = np.random.default_rng(seed)
    val = rng.choice(donors, max(1, round(frac * len(donors))), replace=False)
    return pool.select(key)["k"].is_in(val.tolist()).to_numpy()


def _device(device: str | None) -> torch.device:
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        torch.cuda.init()
        return torch.device("cuda")
    return torch.device("cpu")


def pretrain(arm: str = "mask", epochs: int = 30, seed: int = 0, pool: str = "default", batch_size: int = 1024,
             lr: float = 1e-3, weight_decay: float = 0.01, mask_lo: float = 0.15, mask_hi: float = 0.30,
             perturb_weight: float = 1.0, perturb_batch: int = 256, perturb_holdout: float = 0.2, val_frac: float = 0.05,
             val_cells: int = 50_000, hidden: int = 1024, emb: int = 256, depth: int = 2, dropout: float = 0.1,
             max_cells: int | None = None, max_steps: int | None = None, device: str | None = None,
             name: str | None = None) -> dict:  # fmt: skip
    """Train one arm; checkpoints per epoch under the run's ``checkpoint_dir``, ``metrics.parquet`` and ``status`` in
    the run dir. Returns the run's status dict (``failed`` true when the epoch-1 baseline rule stopped it)."""
    from rna_age.train.runs import new_run

    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}")
    dev = _device(device)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    table, X = load_pool_matrix(pool, max_cells, seed)
    is_val = split_donors(table, val_frac, seed)
    tr_idx, va_idx = np.flatnonzero(~is_val), np.flatnonzero(is_val)
    if len(va_idx) > val_cells:
        va_idx = np.sort(rng.choice(va_idx, val_cells, replace=False))
    Xt = torch.from_numpy(X).to(dev)  # fp16, the whole pool on the card
    n_genes = X.shape[1]
    gene_mean = torch.zeros(n_genes, dtype=torch.float64, device=dev)
    for a in range(0, len(tr_idx), 65536):
        gene_mean += Xt[torch.from_numpy(tr_idx[a : a + 65536]).to(dev)].double().sum(0)
    gene_mean = (gene_mean / len(tr_idx)).float()
    config = {"model": "sc_pretrain", "arm": arm, "pool": pool, "epochs": epochs, "seed": seed,
              "batch_size": batch_size, "lr": lr, "weight_decay": weight_decay, "mask": [mask_lo, mask_hi],
              "perturb_weight": perturb_weight if arm == "mask+perturb" else 0.0, "perturb_batch": perturb_batch,
              "perturb_holdout": perturb_holdout, "hidden": hidden, "emb": emb, "depth": depth, "dropout": dropout,
              "n_genes": n_genes, "train_cells": len(tr_idx), "val_cells": len(va_idx), "max_cells": max_cells,
              "device": str(dev)}  # fmt: skip
    run_dir = new_run(name or f"scpre-{arm.replace('+', '-')}-s{seed}", config)
    cfg = json.loads((run_dir / "config.json").read_text())
    ckpt = Path(cfg["checkpoint_dir"])
    ckpt.mkdir(parents=True, exist_ok=True)
    log.info("run %s: %d train / %d val cells × %d genes on %s (%.0fs)", cfg["run_id"], len(tr_idx), len(va_idx),
             n_genes, dev, time.time() - t0)  # fmt: skip
    model = ScMAE(n_genes, hidden, emb, depth, dropout).to(dev)
    pert = PerturbData(pool_dir(pool) / "perturb.npz", dev, perturb_holdout, seed) if arm == "mask+perturb" else None
    if pert is not None:
        log.info("perturbation: %d training / %d held-out knockdowns", len(pert.train_idx), len(pert.val_idx))
        pbatches = perturb_batches(pert.train_idx, perturb_batch, rng)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = -(-len(tr_idx) // batch_size)
    if max_steps:
        steps_per_epoch = min(steps_per_epoch, max_steps)
    total = steps_per_epoch * epochs
    warm = min(1000, max(1, total // 20))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + np.cos(np.pi * min(1.0, s / max(1, total))))
    )
    amp = (
        dict(device_type="cuda", dtype=torch.bfloat16) if dev.type == "cuda" else dict(device_type="cpu", enabled=False)
    )
    gval = torch.Generator(device=dev).manual_seed(seed + 12345)
    va_t = torch.from_numpy(va_idx).to(dev)
    va_masks = [sample_mask(min(8192, len(va_idx) - a), n_genes, mask_lo, mask_hi, gval, dev)
                for a in range(0, len(va_idx), 8192)]  # fmt: skip
    rows, status = [], {"run_id": cfg["run_id"], "failed": False, "reason": None}
    best = np.inf
    for epoch in range(1, epochs + 1):
        model.train()
        te = time.time()
        perm = torch.from_numpy(rng.permutation(tr_idx)).to(dev)
        tot, totp = 0.0, 0.0
        for step in range(steps_per_epoch):
            x = Xt[perm[step * batch_size : (step + 1) * batch_size]].float()
            m = sample_mask(len(x), n_genes, mask_lo, mask_hi, device=dev)
            with torch.autocast(**amp):
                loss = masked_mse(model(x, m).float(), x, m)
                lp = pert.loss(model, next(pbatches)) if pert is not None else None
            total_loss = loss + perturb_weight * lp if lp is not None else loss
            opt.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss.detach())
            totp += float(lp.detach()) if lp is not None else 0.0
            if step % 200 == 0:
                log.info("epoch %d step %d/%d: masked %.4f%s", epoch, step, steps_per_epoch, float(loss.detach()),
                         f" perturb {float(lp.detach()):.4f}" if lp is not None else "")  # fmt: skip
        model.eval()
        with torch.no_grad(), torch.autocast(**amp):
            se, sb, nm = 0.0, 0.0, 0.0
            for k, a in enumerate(range(0, len(va_idx), 8192)):
                x = Xt[va_t[a : a + 8192]].float()
                m = va_masks[k]
                w = float(m.sum())
                se += float(masked_mse(model(x, m).float(), x, m)) * w
                sb += float(mean_gene_mse(gene_mean, x, m)) * w
                nm += w
        val, base = se / nm, sb / nm
        row = {"epoch": epoch, "train_masked_mse": tot / steps_per_epoch, "val_masked_mse": val,
               "val_mean_gene_mse": base, "below_baseline": val < base, "seconds": time.time() - te}  # fmt: skip
        if pert is not None:
            model.eval()
            row |= {"train_perturb_loss": totp / steps_per_epoch, **pert.evaluate(model, pert.val_idx)}
        rows.append(row)
        pl.DataFrame(rows).write_parquet(run_dir / "metrics.parquet")
        log.info("epoch %d: %s", epoch, {k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()})
        state = {"model": model.state_dict(), "config": cfg, "epoch": epoch, "gene_mean": gene_mean.cpu()}
        torch.save(state, ckpt / "last.pt")
        if val < best:
            best = val
            torch.save(state, ckpt / "best.pt")
        if early_stop_failed(epoch, val, base):
            status |= {"failed": True, "reason": f"epoch 1 masked MSE {val:.4f} not below mean-gene {base:.4f}"}
            log.error("stopped: %s", status["reason"])
            break
    status |= {"epochs_run": len(rows), "best_val_masked_mse": best, "final": rows[-1],
               "seconds": round(time.time() - t0)}  # fmt: skip
    (run_dir / "status.json").write_text(json.dumps(status, indent=2, default=float))
    return status


# ------------------------------------------------------------------ read-out


def run_config(run_id: str) -> dict:
    return json.loads((RUNS / run_id / "config.json").read_text())


def load_encoder(run_id: str, device=None, which: str = "best") -> tuple[ScMAE, dict]:
    cfg = run_config(run_id)
    state = torch.load(Path(cfg["checkpoint_dir"]) / f"{which}.pt", map_location=device or "cpu", weights_only=False)
    model = ScMAE(cfg["n_genes"], cfg["hidden"], cfg["emb"], cfg["depth"], cfg["dropout"])
    model.load_state_dict(state["model"])
    return model.to(device or "cpu").eval(), cfg


def panel_genes(pool: str = "default") -> pl.DataFrame:
    return pl.read_parquet(pool_dir(pool) / "panel.parquet").sort("rank")


def cohort_cells(cohort: str, pool: str = "default", cap: int = 200, seed: int = 0,
                 max_donors: int | None = None) -> dict:  # fmt: skip
    """RM22's per-donor cell sample of a cohort on the panel (log1p CP10k), cached under the pool directory:
    ``donor``, ``in_all``, ``in_cd4``, ``cls`` per cell and ``x`` (cells × panel, fp16)."""
    from rna_age.train.fm_embed import BLOOD, _cohort_cells, sample_cells

    tag = f"{cohort[:8]}_c{cap}_s{seed}" + (f"_d{max_donors}" if max_donors else "")
    path = pool_dir(pool) / "cohorts" / f"{tag}.npz"
    if path.exists():
        z = np.load(path, allow_pickle=False)
        return {k: z[k] for k in z.files}
    h5ad, cells, log_input = _cohort_cells(cohort)
    if max_donors:
        keep = sorted(cells["donor"].unique().to_list())[:max_donors]
        cells = cells.filter(pl.col("donor").is_in(keep))
    sel = sample_cells(cells, cap, cap if cohort in BLOOD else 0, seed)
    panel = panel_genes(pool)
    sym2ens = dict(zip(panel["symbol"].to_list(), panel["gene_id"].to_list(), strict=True))
    ids = file_gene_ids(h5ad, sym2ens if cohort == TEREKHOVA else None)
    cols = columns_for(ids, panel["gene_id"].to_list())
    if (cols < 0).any():
        raise ValueError(f"{cohort[:8]}: {(cols < 0).sum()} panel genes missing")
    xs, rs = [], []
    for r, x in panel_rows(h5ad, sel["row"].to_numpy(), cols, log_input=log_input):
        rs.append(r)
        xs.append(x.astype(np.float16))
    rows = np.concatenate(rs)
    sel = pl.DataFrame({"row": rows}).join(sel, on="row", how="left")
    out = {"row": rows, "donor": np.array(sel["donor"].to_list()), "cls": np.array(sel["cls"].to_list()),
           "in_all": sel["in_all"].to_numpy(), "in_cd4": sel["in_cd4"].to_numpy(), "x": np.concatenate(xs)}  # fmt: skip
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **out)
    tmp.rename(path)
    log.info("%s: %d cells of %d donors on the panel", cohort[:8], len(rows), len(set(out["donor"].tolist())))
    return out


@torch.no_grad()
def encode_cells(model: ScMAE, x: np.ndarray, device=None, batch: int = 8192) -> np.ndarray:
    dev = device or next(model.parameters()).device
    out = [model.encode(torch.from_numpy(x[a : a + batch].astype(np.float32)).to(dev)).float().cpu().numpy()
           for a in range(0, len(x), batch)]  # fmt: skip
    return np.concatenate(out) if out else np.zeros((0, model.to_emb[-1].out_features), np.float32)


def donor_means(e: np.ndarray, donors: np.ndarray, flags: dict[str, np.ndarray], prefix: str = "emb") -> pl.DataFrame:
    """One row per donor and subset: the mean embedding of the donor's cells flagged for that subset."""
    rows = []
    for subset, flag in flags.items():
        for d in sorted(set(donors[flag].tolist())):
            k = flag & (donors == d)
            rows.append({"donor": d, "subset": subset, "n_cells": int(k.sum()),
                         **{f"{prefix}_{j}": float(v) for j, v in enumerate(e[k].mean(0))}})  # fmt: skip
    return pl.DataFrame(rows)


def embed_cohorts(run_id: str, cohorts: tuple[str, ...] = EVAL_COHORTS, cap: int = 200, seed: int = 0,
                  max_donors: int | None = None, device: str | None = None) -> list[Path]:  # fmt: skip
    """Per donor (and donor × CD4 T in blood) mean frozen embeddings; ``data/output/runs/<run>/emb_<cohort>.parquet``."""
    from rna_age.train.fm_embed import CD4

    dev = _device(device)
    model, cfg = load_encoder(run_id, dev)
    paths = []
    for c in cohorts:
        t0 = time.time()
        cc = cohort_cells(c, cfg["pool"], cap, seed, max_donors)
        e = encode_cells(model, cc["x"], dev)
        out = donor_means(e, cc["donor"], {"all": cc["in_all"].astype(bool), CD4: cc["in_cd4"].astype(bool)})
        p = RUNS / run_id / f"emb_{c[:8]}.parquet"
        out.write_parquet(p, compression="zstd")
        paths.append(p)
        log.info("%s: %d donor rows (%.0fs)", c[:8], out.height, time.time() - t0)
    return paths


def _merge_summary(new: pl.DataFrame, path: Path, keys: tuple[str, ...] = ("run_id", "model")) -> pl.DataFrame:
    if path.exists():
        old = pl.read_parquet(path)
        new = pl.concat([old.join(new.select(keys).unique(), on=list(keys), how="anti"), new], how="vertical_relaxed")
    path.parent.mkdir(parents=True, exist_ok=True)
    new.write_parquet(path, compression="zstd")
    return new


def bar(cmp: pl.DataFrame) -> pl.DataFrame:
    """RM28's bar (1) per model: natural-representation cohorts where the model beats ``pseudobulk-tabicl-2000g``
    beyond the paired CI (needs ≥ 4 of 7)."""
    return (
        cmp.filter((pl.col("representation") == "natural") & (pl.col("ref") == "pseudobulk-tabicl-2000g"))
        .group_by("run_id", "model")
        .agg(
            pl.len().alias("cohorts"),
            pl.col("beats_ref").sum().alias("beats"),
            pl.col("diff").mean().alias("mean_diff"),
        )
        .with_columns(passes=pl.col("beats") >= 4)
        .sort("run_id", "model")
    )


def clock(run_id: str, cohorts: tuple[str, ...] = EVAL_COHORTS, models: tuple[str, ...] = ("ridge", "tabicl"),
          seed: int = 0, n_boot: int = 2000) -> pl.DataFrame:  # fmt: skip
    """Ridge/TabICLv2 on the donor-mean embeddings, compared with the pseudobulk clocks; adds this run's rows to
    ``data/output/summary/sc_pretrain_{oof,clock}.parquet``."""
    from rna_age.train.fm_embed import clock_oof, compare

    cfg = run_config(run_id)
    tag = f"scpre-{cfg['arm'].replace('+', '-')}-s{cfg['seed']}"
    oof = []
    for c in cohorts:
        emb = pl.read_parquet(RUNS / run_id / f"emb_{c[:8]}.parquet")
        oof.append(clock_oof(c, emb, ("emb",), lambda f, m: f"{tag}-{m}", models, seed))
    oof = pl.concat(oof).with_columns(run_id=pl.lit(run_id))
    oof.write_parquet(RUNS / run_id / "clock_oof.parquet")
    cmp = compare(oof.drop("run_id"), n_boot=n_boot).with_columns(run_id=pl.lit(run_id))
    _merge_summary(oof, SUMMARY / "sc_pretrain_oof.parquet")
    _merge_summary(cmp, SUMMARY / "sc_pretrain_clock.parquet")
    return cmp


# ------------------------------------------------------------------ fine-tune read-out


def _inner_split(groups: np.ndarray, frac: float, seed: int) -> np.ndarray:
    """Boolean: rows whose group is in the inner validation ``frac`` of the groups."""
    u = np.unique(groups)
    rng = np.random.default_rng(seed)
    return np.isin(groups, rng.choice(u, max(1, round(frac * len(u))), replace=False))


def finetune_fold(model0: ScMAE, x: torch.Tensor, cell_donor: np.ndarray, train: list[str], age_tr: np.ndarray,
                  val: list[str], age_val: np.ndarray, test: list[str], epochs: int, lr: float, head_lr: float,
                  donors_per_batch: int, seed: int) -> np.ndarray:  # fmt: skip
    """Copy of the encoder + a linear head on the mean cell embedding of each donor; trained on ``train`` donors,
    the epoch with the lowest MAE on ``val`` donors kept; returns predictions for ``test`` donors."""
    import copy

    dev = x.device
    torch.manual_seed(seed)
    enc = copy.deepcopy(model0).train()
    mu, sd = float(age_tr.mean()), float(age_tr.std() + 1e-6)
    head = nn.Linear(enc.to_emb[-1].out_features, 1).to(dev)
    nn.init.zeros_(head.weight)
    nn.init.zeros_(head.bias)
    idx = {d: torch.from_numpy(np.flatnonzero(cell_donor == d)).to(dev) for d in set(train) | set(val) | set(test)}
    opt = torch.optim.AdamW([{"params": enc.parameters(), "lr": lr}, {"params": head.parameters(), "lr": head_lr}],
                            weight_decay=0.01)  # fmt: skip
    rng = np.random.default_rng(seed)

    def predict(donors: list[str]) -> np.ndarray:
        enc.eval()
        with torch.no_grad():
            out = [float(head(enc.encode(x[idx[d]]).mean(0))) * sd + mu for d in donors]
        enc.train()
        return np.array(out)

    best, best_state = np.inf, None
    y = torch.tensor((age_tr - mu) / sd, dtype=torch.float32, device=dev)
    for _ in range(epochs):
        for b in np.array_split(rng.permutation(len(train)), max(1, len(train) // donors_per_batch)):
            z = torch.stack([enc.encode(x[idx[train[i]]]).mean(0) for i in b])
            loss = ((head(z).squeeze(1) - y[torch.from_numpy(b).to(dev)]) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        mae = float(np.abs(predict(val) - age_val).mean()) if val else 0.0
        if mae < best:
            best = mae
            best_state = (copy.deepcopy(enc.state_dict()), copy.deepcopy(head.state_dict()))
    enc.load_state_dict(best_state[0])
    head.load_state_dict(best_state[1])
    return predict(test)


def finetune(run_id: str, cohorts: tuple[str, ...] = EVAL_COHORTS, epochs: int = 20, lr: float = 1e-4,
             head_lr: float = 1e-3, donors_per_batch: int = 16, inner_val: float = 0.15, cap: int = 200, seed: int = 0,
             max_donors: int | None = None, device: str | None = None, n_boot: int = 2000) -> pl.DataFrame:  # fmt: skip
    """Fine-tune read-out on RM14's folds; rows ``scpre-<arm>-s<seed>-finetune`` added to the summaries."""
    from sklearn.model_selection import GroupKFold

    from rna_age.train.fm_embed import clock_folds, compare

    dev = _device(device)
    model0, cfg = load_encoder(run_id, dev)
    tag = f"scpre-{cfg['arm'].replace('+', '-')}-s{cfg['seed']}-finetune"
    rows = []
    for c in cohorts:
        cc = cohort_cells(c, cfg["pool"], cap, seed, max_donors)
        x = torch.from_numpy(cc["x"].astype(np.float32)).to(dev)
        dp, reps = clock_folds(c)
        donors_all, age_all, groups_all = np.array(dp["donors"]), dp["age"], dp["groups"]
        for rep, subset, ok in reps:
            flag = cc["in_all"].astype(bool) if subset == "all" else cc["in_cd4"].astype(bool)
            have = set(cc["donor"][flag].tolist())
            ok = ok & np.isin(donors_all, list(have))
            donors, age, groups = donors_all[ok], age_all[ok], groups_all[ok]
            cell_donor = np.where(flag, cc["donor"], "")
            pred = np.full(len(donors), np.nan)
            for k, (tr, te) in enumerate(GroupKFold(n_splits=5).split(donors, age, groups)):
                iv = _inner_split(groups[tr], inner_val, seed + k)
                fit, val = tr[~iv], tr[iv]
                pred[te] = finetune_fold(model0, x, cell_donor, donors[fit].tolist(), age[fit], donors[val].tolist(),
                                         age[val], donors[te].tolist(), epochs, lr, head_lr, donors_per_batch,
                                         seed + k)  # fmt: skip
            rows += [{"cohort": c[:8], "representation": rep, "model": tag, "donor": d, "group": int(g),
                      "age": float(a), "pred": float(p)} for d, g, a, p in zip(donors, groups, age, pred, strict=True)]  # fmt: skip
            log.info("%s %s finetune MAE %.2f (%d donors)", c[:8], rep, float(np.abs(pred - age).mean()), len(donors))
    oof = pl.DataFrame(rows).with_columns(run_id=pl.lit(run_id))
    oof.write_parquet(RUNS / run_id / "finetune_oof.parquet")
    cmp = compare(oof.drop("run_id"), n_boot=n_boot).with_columns(run_id=pl.lit(run_id))
    _merge_summary(oof, SUMMARY / "sc_pretrain_oof.parquet")
    _merge_summary(cmp, SUMMARY / "sc_pretrain_clock.parquet")
    return cmp
