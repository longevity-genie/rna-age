"""RM30: TabICLv2's architecture trained further on real expression episodes. This module is the data side.

An **episode** is one small table drawn from our corpus with a target: context rows carry labels, query rows are
predicted (in-context learning, as TabICL is pretrained on synthetic tables). Tasks:

- ``age``: regression on samples or donors with ages (bulk: exact ages and GTEx bin midpoints; single-cell donor
  pseudobulk: the donor's age). Rows of one bulk (study, tissue), one single-cell table (a cohort's donors, whole
  pseudobulk or one cell class), or, for bulk, several training studies of one tissue pooled.
- ``health``: classification, case (an aging-linked category, :data:`rna_age.train.health.AGING_LINKED`) vs control,
  rows of one bulk training study.
- ``masked_gene``: regression, one panel gene held out of the columns is the target (masked expression in tabular
  form); bulk groups and single-cell tables.
- ``perturbation``: regression on the CD4 T Perturb-seq screen. Rows: knockdowns (one perturbed gene, its guides
  averaged) of one donor × condition; target: the change (log-CPM, against the same donor × condition's
  non-targeting mean) of one readout gene; features ``profile`` (other genes' changes plus the descriptor) or
  ``descriptor`` (the knocked-down gene's own non-targeting expression and a hashed id embedding: what is known of a
  knockdown never run).

Columns: a random subset of a fixed **gene panel** (:func:`build_panel`: the most variable genes of the bulk
training split among the genes every source measures), standardised on the episode's context rows only.

**Leakage guards** (training mode): no bulk val/test study (and no study that has any val/test sample), none of the
seven single-cell evaluation cohorts (:data:`EVAL_COHORTS`), no held-out perturbed gene (20 % by a stable hash,
:func:`is_heldout_gene`). ``mode="eval"`` builds episodes from the held-out parts only.

Prepared tables (``rna-age expr-icl-prepare``) live under ``data/interim/expr_icl/``: ``panel.parquet``,
``bulk.parquet`` (bulk samples × panel log-CPM), ``sc.parquet`` (donor tables × panel log-CPM), ``screen.parquet``
(knockdown × donor × condition rows, panel changes). The model and training loop: :mod:`rna_age.train.expr_icl_model`;
the evaluations: :mod:`rna_age.train.expr_icl_eval`.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from rna_age.paths import INPUT, INTERIM, download_dir

log = logging.getLogger(__name__)

CACHE = INTERIM / "expr_icl"
TASKS = ("age", "health", "masked_gene", "perturbation")
REGRESSION_TASKS = ("age", "masked_gene", "perturbation")
EVAL_COHORTS = (
    "3faad104-2ab8-4434-816d-474d8d2641db",
    "c838aec3-03ef-4398-b882-0e3912abfff0",
    "218acb0f-9f2f-4f76-b90b-15a4b7c7f629",
    "37a17b78-4864-4a42-b67b-31c00962795a",
    "5c97eeeb-7e52-44b3-b010-b832b1f5424c",
    "5e57cd50-8e42-42d6-940d-5c1660d06864",
    "terekhova2023",
)
HELDOUT_FRACTION = 0.2
SCREEN_FILE = "GWCD4i.pseudobulk_merged.h5ad"
# RM28's gene panel, when it exists: the RM30 panel is narrowed to it so both read the same genes (optional)
RM28_PANEL = INTERIM / "sc_pretrain" / "genes.parquet"
CLIP = 10.0


def is_eval_cohort(dataset_id: str) -> bool:
    return dataset_id in EVAL_COHORTS or any(dataset_id[:8] == c[:8] for c in EVAL_COHORTS)


def _hash(text: str) -> int:
    return int(hashlib.sha1(text.encode()).hexdigest()[:12], 16)


def is_heldout_gene(gene: str, fraction: float = HELDOUT_FRACTION) -> bool:
    """A perturbed gene is held out of training when its stable hash falls in the first ``fraction``."""
    return (_hash(gene) % 10_000) < fraction * 10_000


def hash_embedding(gene: str, dim: int = 16) -> np.ndarray:
    """A fixed random vector per gene id (the knocked-down gene's identity, with no biology in it)."""
    return np.random.default_rng(_hash("emb:" + gene)).standard_normal(dim).astype(np.float32)


def descriptors(pert: pl.DataFrame, dim: int = 16) -> np.ndarray:
    """Per knockdown row: the knocked-down gene's own non-targeting log-CPM, then its hashed id embedding."""
    gid = pert["perturbed_gene_id"].to_list()
    emb = np.stack([hash_embedding(g, dim) for g in gid]) if gid else np.zeros((0, dim), np.float32)
    return np.concatenate([pert["kd_baseline"].to_numpy().astype(np.float32)[:, None], emb], axis=1)


# ---------------------------------------------------------------- panel


def screen_genes(path: Path | None = None) -> list[str]:
    import h5py

    from rna_age.train.perturb import _obs_column

    with h5py.File(path or download_dir("cd4_perturbseq") / SCREEN_FILE, "r") as h:
        var = h["var"]
        key = next(k for k in ("gene_ids", "_index") if k in var)
        return [g.split(".")[0] for g in _obs_column(var, key)]


def _celltype_dir(dataset_id: str) -> Path:
    d = INPUT / "census_pseudobulk" / "by_dataset_celltype" / dataset_id
    return d if d.exists() else INPUT / dataset_id / "by_dataset_celltype" / dataset_id


def _to_ensembl(genes: list[str]) -> list[str]:
    """Symbols (Terekhova's table) to Ensembl ids through ARCHS4's symbol table; Ensembl ids pass through."""
    if all(g.startswith("ENSG") for g in genes[:50]):
        return genes
    from rna_age.train.features import symbol_map

    smap = symbol_map()
    return [smap.get(g, g) for g in genes]


def candidate_genes(rm28_panel: Path | None = RM28_PANEL) -> list[str]:
    """Genes the bulk sources, the Census pseudobulk, Terekhova's cohort and the screen all measure (∩ RM28's panel
    when that file exists)."""
    from rna_age.train.features import gene_ids, shared_genes

    genes = set(shared_genes())
    genes &= set(gene_ids("census_pseudobulk"))
    tere = _celltype_dir("terekhova2023") / "counts.parquet"
    if tere.exists():
        genes &= set(_to_ensembl(pl.read_parquet(tere, columns=["gene_id"])["gene_id"].to_list()))
    genes &= set(screen_genes())
    if rm28_panel is not None and Path(rm28_panel).exists():
        genes &= set(pl.read_parquet(rm28_panel)["gene_id"].to_list())
        log.info("narrowed to RM28's panel %s", rm28_panel)
    return sorted(genes)


def bulk_meta() -> pl.DataFrame:
    """Bulk samples (no pseudobulk, no tumour) with split, target age (``age``: exact, or the bin midpoint for GTEx),
    and health group/category."""
    from rna_age.schema import age_mid
    from rna_age.train.health import health_table

    h = health_table().filter(~pl.col("is_pseudobulk") & ~pl.col("is_tumor").fill_null(False))
    ok = (pl.col("age_type") == "exact") | ((pl.col("source") == "gtex") & (pl.col("age_type") == "binned"))
    return h.with_columns(age=pl.when(ok).then(age_mid(h)).otherwise(None).cast(pl.Float64)).select(
        "sample_id", "source", "study_id", "tissue", "split", "age", "age_type", "group", "category", "is_healthy"
    )


def build_panel(
    n_genes: int = 1000, hvg_sample: int = 5000, seed: int = 0, rm28_panel: Path | None = RM28_PANEL,
    cache: Path = CACHE,
) -> pl.DataFrame:  # fmt: skip
    """The ``n_genes`` most variable genes (log-CPM, mean ≥ 1) of a sample of bulk training samples, among
    :func:`candidate_genes`. Written to ``panel.parquet``."""
    from rna_age.train.features import load_matrix, log_cpm, select_hvg, shared_genes

    cand = candidate_genes(rm28_panel)
    shared = shared_genes()
    meta = bulk_meta().filter((pl.col("split") == "train") & pl.col("age").is_not_null())
    hv = meta.sample(min(hvg_sample, meta.height), seed=seed)
    lc = log_cpm(load_matrix(hv, shared))
    pos = {g: i for i, g in enumerate(shared)}
    panel = select_hvg(lc[:, [pos[g] for g in cand]], cand, n_genes)
    out = pl.DataFrame({"gene_id": panel})
    cache.mkdir(parents=True, exist_ok=True)
    out.write_parquet(cache / "panel.parquet")
    log.info("panel: %d genes (of %d candidates; HVGs on %d bulk train samples)", len(panel), len(cand), hv.height)
    return out


def read_panel(cache: Path = CACHE) -> list[str]:
    return pl.read_parquet(cache / "panel.parquet")["gene_id"].to_list()


# ---------------------------------------------------------------- prepared tables


def _frame(ids: dict[str, list], X: np.ndarray, genes: list[str]) -> pl.DataFrame:
    return pl.DataFrame(ids).hstack(pl.DataFrame(X, schema=genes))


def build_bulk(genes: list[str], max_samples: int | None = None, chunk: int = 4000, seed: int = 0,
               cache: Path = CACHE) -> Path:  # fmt: skip
    """Every bulk sample (all splits) × panel log-CPM, the library size over the shared bulk genes (as
    :func:`rna_age.train.baselines.hvg_design`)."""
    from rna_age.train.features import load_matrix, shared_genes

    meta = bulk_meta()
    if max_samples:
        meta = meta.sample(min(max_samples, meta.height), seed=seed)
    shared = shared_genes()
    pos = {g: i for i, g in enumerate(shared)}
    pidx = np.array([pos[g] for g in genes])
    out = np.zeros((meta.height, len(genes)), dtype=np.float32)
    for a in range(0, meta.height, chunk):
        c = load_matrix(meta.slice(a, chunk), shared)
        lib = c.sum(1, keepdims=True)
        lib[lib == 0] = 1
        out[a : a + chunk] = np.log1p(c[:, pidx] / lib * 1e6)
        log.info("bulk panel: %d / %d", min(a + chunk, meta.height), meta.height)
    path = cache / "bulk.parquet"
    cache.mkdir(parents=True, exist_ok=True)
    _frame({"sample_id": meta["sample_id"].to_list()}, out, genes).write_parquet(path, compression="zstd")
    return path


def sc_datasets() -> list[str]:
    d = INPUT / "census_pseudobulk" / "by_dataset_celltype"
    out = sorted(p.name for p in d.iterdir()) if d.exists() else []
    return out + (["terekhova2023"] if _celltype_dir("terekhova2023").exists() else [])


def _panel_logcpm(counts: np.ndarray, src_genes: list[str], genes: list[str]) -> np.ndarray:
    """log-CPM on ``genes`` (library over all of ``src_genes``); genes the source lacks are NaN."""
    lib = counts.sum(-1, keepdims=True)
    lib[lib == 0] = 1
    pos = {g: i for i, g in enumerate(src_genes)}
    idx = np.array([pos.get(g, -1) for g in genes])
    out = np.full((counts.shape[0], len(genes)), np.nan, dtype=np.float32)
    out[:, idx >= 0] = np.log1p(counts[:, idx[idx >= 0]] / lib * 1e6)
    return out


def build_sc(genes: list[str], datasets: list[str] | None = None, min_cells: int = 20, min_donors: int = 30,
             cache: Path = CACHE) -> Path:  # fmt: skip
    """Single-cell donor tables on the panel: per cohort the donors' whole pseudobulk (``natural``) and each cell
    class with ≥ ``min_cells`` cells in ≥ ``min_donors`` donors (``class:<c>``), as :mod:`rna_age.train.sc_tabular`
    reads them. Evaluation cohorts are kept (flagged ``role="eval"``) for the held-out episodes; training never
    reads them."""
    from rna_age.train.composition import donor_profiles
    from rna_age.train.fm_embed import BLOOD

    frames = []
    for ds in datasets or sc_datasets():
        dp = donor_profiles(ds, "broad" if ds in BLOOD else "cell_type")
        src = _to_ensembl(dp["genes"])
        reps = [("natural", np.ones(len(dp["donors"]), dtype=bool), dp["P"].sum(1))]
        for c in dp["classes"]:
            k = dp["cix"][c]
            ok = dp["N"][:, k] >= min_cells
            if c != "other" and ok.sum() >= min_donors:
                reps.append((f"class:{c}", ok, dp["P"][:, k]))
        for rep, ok, M in reps:
            X = _panel_logcpm(M[ok], src, genes)
            ids = {"table": [f"{ds[:8]}:{rep}"] * int(ok.sum()), "dataset_id": [ds] * int(ok.sum()),
                   "donor": list(np.array(dp["donors"])[ok]), "cv_group": dp["groups"][ok].astype(np.int64).tolist(),
                   "age": dp["age"][ok].astype(np.float64).tolist(),
                   "role": ["eval" if is_eval_cohort(ds) else "train"] * int(ok.sum())}  # fmt: skip
            frames.append(_frame(ids, X, genes))
        log.info("sc %s: %d donors, %d tables", ds[:8], len(dp["donors"]), len(reps))
        del dp
    path = cache / "sc.parquet"
    cache.mkdir(parents=True, exist_ok=True)
    pl.concat(frames, how="vertical_relaxed").write_parquet(path, compression="zstd")
    return path


def _read_csr_rows(X, rows: np.ndarray, remap: np.ndarray, n_out: int, span: int = 2000) -> np.ndarray:
    """Rows ``rows`` (sorted) of an h5py CSR group as dense rows × ``n_out``; file column j lands in ``remap[j]``
    (−1: dropped). Reads contiguous spans of at most ``span`` file rows, so a sparse selection reads little."""
    indptr = X["indptr"][()]
    out = np.zeros((len(rows), n_out), dtype=np.float32)
    i = 0
    while i < len(rows):
        a = rows[i]
        j = int(np.searchsorted(rows, a + span, side="left")) - 1
        b = rows[j] + 1
        lo, hi = indptr[a], indptr[b]
        idx, val = X["indices"][lo:hi], X["data"][lo:hi]
        r = np.repeat(np.arange(a, b), np.diff(indptr[a : b + 1])) - a
        pos = np.full(b - a, -1)
        pos[rows[i : j + 1] - a] = np.arange(i, j + 1)
        m = (pos[r] >= 0) & (remap[idx] >= 0)
        out[pos[r[m]], remap[idx[m]]] = val[m]
        i = j + 1
    return out


def build_screen(genes: list[str], min_cells: int = 20, max_rows: int | None = None, path: Path | None = None,
                 cache: Path = CACHE) -> Path:  # fmt: skip
    """The CD4 T Perturb-seq pseudobulk as knockdown rows: per (perturbed gene, donor, condition) the mean over its
    guides (profiles with ≥ ``min_cells`` cells) of log-CPM minus the donor × condition's non-targeting mean, on the
    panel; plus ``kd_baseline``, the knocked-down gene's own non-targeting log-CPM there. ``max_rows`` (smoke runs):
    only the first donor × condition, its first ``max_rows`` targeting profiles."""
    import h5py

    from rna_age.train.perturb import _obs_column

    path = path or download_dir("cd4_perturbseq") / SCREEN_FILE
    cols = ("donor_id", "culture_condition", "guide_id", "perturbed_gene_id", "perturbed_gene_name", "guide_type",
            "n_cells", "total_counts")  # fmt: skip
    with h5py.File(path, "r") as h:
        obs = pl.DataFrame({c: _obs_column(h["obs"], c) for c in cols}).with_row_index("row")
        var = h["var"]
        key = next(k for k in ("gene_ids", "_index") if k in var)
        file_genes = [g.split(".")[0] for g in _obs_column(var, key)]
        obs = obs.filter(pl.col("n_cells") >= min_cells).with_columns(
            ntc=pl.col("guide_type").str.contains("(?i)non"),
            perturbed_gene_id=pl.col("perturbed_gene_id").str.split(".").list.first(),
        )
        if max_rows:
            first = obs.row(0, named=True)
            blk = obs.filter((pl.col("donor_id") == first["donor_id"])
                             & (pl.col("culture_condition") == first["culture_condition"]))  # fmt: skip
            obs = pl.concat([blk.filter(~pl.col("ntc")).head(max_rows), blk.filter(pl.col("ntc"))]).sort("row")
        gpos = {g: i for i, g in enumerate(genes)}
        remap_panel = np.array([gpos.get(g, -1) for g in file_genes])
        rows = obs["row"].to_numpy()
        Xp = _read_csr_rows(h["X"], rows, remap_panel, len(genes))
        ntc_rows = obs.filter(pl.col("ntc"))["row"].to_numpy()
        Xn = _read_csr_rows(h["X"], ntc_rows, np.arange(len(file_genes)), len(file_genes))
    lib = obs["total_counts"].to_numpy()[:, None]
    Xp = np.log1p(Xp / np.maximum(lib, 1) * 1e6)
    Xn = np.log1p(Xn / np.maximum(obs.filter(pl.col("ntc"))["total_counts"].to_numpy()[:, None], 1) * 1e6)
    ntc_obs = obs.filter(pl.col("ntc"))
    key_cols = ["donor_id", "culture_condition"]
    fpos = {g: i for i, g in enumerate(file_genes)}
    deltas, meta = [], []
    for (donor, cond), blk in obs.with_row_index("_i").group_by(key_cols, maintain_order=True):
        nb = ntc_obs.with_row_index("_n").filter((pl.col("donor_id") == donor) & (pl.col("culture_condition") == cond))
        if nb.height == 0:
            continue
        base_full = Xn[nb["_n"].to_numpy()].mean(0)
        base_panel = Xp[blk.filter(pl.col("ntc"))["_i"].to_numpy()].mean(0)
        tg = blk.filter(~pl.col("ntc"))
        for (gid, gname), g in tg.group_by("perturbed_gene_id", "perturbed_gene_name", maintain_order=True):
            deltas.append(Xp[g["_i"].to_numpy()].mean(0) - base_panel)
            meta.append({"perturbed_gene_id": gid, "perturbed_gene_name": gname, "donor_id": donor,
                         "culture_condition": cond, "n_guides": g.height, "n_cells": float(g["n_cells"].sum()),
                         "kd_baseline": float(base_full[fpos[gid]]) if gid in fpos else 0.0})  # fmt: skip
    out = pl.DataFrame(meta).hstack(pl.DataFrame(np.stack(deltas).astype(np.float32), schema=genes))
    p = cache / "screen.parquet"
    cache.mkdir(parents=True, exist_ok=True)
    out.write_parquet(p, compression="zstd")
    log.info("screen: %d knockdown rows (%d genes, %d donor × conditions)", out.height,
             out["perturbed_gene_id"].n_unique(), out.select(key_cols).n_unique())  # fmt: skip
    return p


# ---------------------------------------------------------------- corpus and episodes


@dataclass
class Corpus:
    """Panel matrices with their row tables. ``bulk``: sample_id, study_id, tissue, split, age, group, category;
    ``sc``: table, dataset_id, donor, cv_group, age; ``pert``: perturbed_gene_id, donor_id, culture_condition,
    kd_baseline. Any part may be empty (zero rows)."""

    genes: list[str]
    bulk: pl.DataFrame
    bulk_X: np.ndarray
    sc: pl.DataFrame
    sc_X: np.ndarray
    pert: pl.DataFrame
    pert_X: np.ndarray


def _empty(cols: dict[str, pl.DataType], P: int) -> tuple[pl.DataFrame, np.ndarray]:
    return pl.DataFrame(schema=cols), np.zeros((0, P), dtype=np.float32)


def load_corpus(cache: Path = CACHE, with_meta: pl.DataFrame | None = None,
                parts: tuple[str, ...] = ("bulk", "sc", "screen")) -> Corpus:  # fmt: skip
    """The prepared tables named in ``parts`` (missing ones are empty); bulk rows are joined to :func:`bulk_meta`."""
    genes = read_panel(cache)
    P = len(genes)
    if "bulk" in parts and (cache / "bulk.parquet").exists():
        b = pl.read_parquet(cache / "bulk.parquet")
        meta = with_meta if with_meta is not None else bulk_meta()
        b = b.join(meta, on="sample_id", how="inner")
        bulk, bulk_X = b.drop(genes), b.select(genes).to_numpy().astype(np.float32)
    else:
        bulk, bulk_X = _empty({"sample_id": pl.String, "study_id": pl.String, "tissue": pl.String,
                               "split": pl.String, "age": pl.Float64, "group": pl.String,
                               "category": pl.String}, P)  # fmt: skip
    if "sc" in parts and (cache / "sc.parquet").exists():
        s = pl.read_parquet(cache / "sc.parquet")
        sc, sc_X = s.drop(genes), s.select(genes).to_numpy().astype(np.float32)
    else:
        sc, sc_X = _empty({"table": pl.String, "dataset_id": pl.String, "donor": pl.String, "cv_group": pl.Int64,
                           "age": pl.Float64}, P)  # fmt: skip
    if "screen" in parts and (cache / "screen.parquet").exists():
        s = pl.read_parquet(cache / "screen.parquet")
        pert, pert_X = s.drop(genes), s.select(genes).to_numpy().astype(np.float32)
    else:
        pert, pert_X = _empty({"perturbed_gene_id": pl.String, "donor_id": pl.String,
                               "culture_condition": pl.String, "kd_baseline": pl.Float64}, P)  # fmt: skip
    log.info("corpus: %d panel genes; %d bulk, %d single-cell donor rows, %d knockdown rows", P, bulk.height,
             sc.height, pert.height)  # fmt: skip
    return Corpus(genes, bulk, bulk_X, sc, sc_X, pert, pert_X)


@dataclass
class Episode:
    """One table. ``X_*`` standardised on the context rows; regression targets standardised on the context
    (``y_mean``, ``y_sd`` undo it); classification targets 0/1."""

    X_ctx: np.ndarray
    y_ctx: np.ndarray
    X_qry: np.ndarray
    y_qry: np.ndarray
    task: str
    source: str
    y_mean: float = 0.0
    y_sd: float = 1.0
    meta: dict = field(default_factory=dict)

    @property
    def classification(self) -> bool:
        return self.task == "health"


@dataclass
class Group:
    name: str
    kind: str  # bulk | sc | pert
    rows: np.ndarray  # indices into the kind's matrix
    cols: np.ndarray  # usable panel columns (measured in every row)
    pooled: bool = False


Y_SD_FLOOR = 0.05  # log units: an expression target's scale never below this


def standardise(Xc: np.ndarray, Xq: np.ndarray, clip: float = CLIP) -> tuple[np.ndarray, np.ndarray]:
    """Columns centred and scaled on the context rows only, clipped to ±``clip``."""
    mu = Xc.mean(0)
    sd = Xc.std(0) + 1e-6
    return (np.clip((Xc - mu) / sd, -clip, clip).astype(np.float32),
            np.clip((Xq - mu) / sd, -clip, clip).astype(np.float32))  # fmt: skip


class EpisodeSampler:
    """Episodes from a :class:`Corpus`. ``mode="train"`` keeps out every held-out part (see the module doc);
    ``mode="eval"`` uses only them (bulk: ``eval_split`` studies)."""

    def __init__(
        self,
        corpus: Corpus,
        mode: str = "train",
        n_rows: tuple[int, int] = (64, 512),
        n_cols: tuple[int, int] = (100, 500),
        query_frac: tuple[float, float] = (0.2, 0.5),
        min_rows: int = 24,
        min_age_sd: float = 2.0,
        min_per_class: int = 5,
        pool_prob: float = 0.3,
        perturb_modes: tuple[str, ...] = ("profile", "descriptor"),
        hash_dim: int = 16,
        eval_split: str = "val",
        seed: int = 0,
    ):
        if mode not in ("train", "eval"):
            raise ValueError(mode)
        self.c, self.mode = corpus, mode
        self.n_rows, self.n_cols, self.query_frac = n_rows, n_cols, query_frac
        self.min_rows, self.pool_prob, self.perturb_modes = min_rows, pool_prob, perturb_modes
        self.rng = np.random.default_rng(seed)
        self.hash_dim = hash_dim
        P = len(corpus.genes)
        self.groups: dict[str, list[Group]] = {t: [] for t in TASKS}
        b = corpus.bulk
        if b.height:
            split = b["split"].to_numpy().astype(str)
            study = b["study_id"].fill_null("").to_numpy().astype(str)
            heldout_studies = set(study[split != "train"])
            if mode == "train":
                use = (split == "train") & ~np.isin(study, list(heldout_studies))
            else:
                use = split == eval_split
            age = b["age"].to_numpy().astype(np.float64)
            ctrl = (b["group"] == "control").to_numpy()
            from rna_age.train.health import AGING_LINKED

            case = ((b["group"] == "case") & b["category"].is_in(list(AGING_LINKED))).fill_null(False).to_numpy()
            tissue = b["tissue"].fill_null("?").to_numpy().astype(str)
            allc = np.arange(P)
            keys = np.char.add(np.char.add(study, "|"), tissue)
            for k in np.unique(keys[use]):
                r = np.flatnonzero(use & (keys == k))
                ra = r[ctrl[r] & ~np.isnan(age[r])]
                if len(ra) >= min_rows and np.nanstd(age[ra]) >= min_age_sd:
                    self.groups["age"].append(Group(f"bulk:{k}", "bulk", ra, allc))
                if len(r) >= min_rows:
                    self.groups["masked_gene"].append(Group(f"bulk:{k}", "bulk", r, allc))
            for t in np.unique(tissue[use]):  # several studies of one tissue pooled (between-study variation)
                r = np.flatnonzero(use & (tissue == t) & ctrl & ~np.isnan(age))
                if len(np.unique(study[r])) >= 3 and len(r) >= 2 * min_rows:
                    self.groups["age"].append(Group(f"bulk-pooled:{t}", "bulk", r, allc, pooled=True))
            for s in np.unique(study[use]):
                r = np.flatnonzero(use & (study == s))
                ca, co = r[case[r]], r[ctrl[r]]
                if len(ca) >= min_per_class and len(co) >= min_per_class:
                    self.groups["health"].append(Group(f"bulk:{s}", "bulk", np.concatenate([ca, co]), allc))
            self._case = case
            self._age_bulk = age
        s = corpus.sc
        if s.height:
            ds = s["dataset_id"].to_numpy().astype(str)
            ev = np.array([is_eval_cohort(d) for d in ds])
            use = ~ev if mode == "train" else ev
            tab = s["table"].to_numpy().astype(str)
            age = s["age"].to_numpy().astype(np.float64)
            for t in np.unique(tab[use]):
                r = np.flatnonzero(use & (tab == t))
                cols = np.flatnonzero(~np.isnan(corpus.sc_X[r]).any(0))
                ra = r[~np.isnan(age[r])]
                if len(ra) >= min_rows and np.std(age[ra]) >= min_age_sd:
                    self.groups["age"].append(Group(f"sc:{t}", "sc", ra, cols))
                if len(r) >= min_rows:
                    self.groups["masked_gene"].append(Group(f"sc:{t}", "sc", r, cols))
            self._age_sc = age
        p = corpus.pert
        if p.height:
            gid = p["perturbed_gene_id"].to_numpy().astype(str)
            held = np.array([is_heldout_gene(g) for g in gid])
            use = ~held if mode == "train" else held
            key = (p["donor_id"] + "|" + p["culture_condition"]).to_numpy().astype(str)
            allc = np.arange(P)
            for k in np.unique(key[use]):
                r = np.flatnonzero(use & (key == k))
                if len(r) >= min_rows:
                    self.groups["perturbation"].append(Group(f"pert:{k}", "pert", r, allc))
            self._descr = descriptors(p, hash_dim)
            self._pert_var = corpus.pert_X[~held].var(0) if (~held).any() else np.ones(P)
        self.check_guards()

    # ---- guards
    def check_guards(self) -> None:
        """Raise if a training group reads a held-out part (the tests call this on purpose-built corpora too)."""
        if self.mode != "train":
            return
        c = self.c
        for t, gs in self.groups.items():
            for g in gs:
                if g.kind == "bulk":
                    sp = c.bulk["split"].to_numpy()[g.rows]
                    if (sp != "train").any():
                        raise AssertionError(f"{t} {g.name}: non-train bulk rows")
                elif g.kind == "sc":
                    if any(is_eval_cohort(d) for d in c.sc["dataset_id"].to_numpy()[g.rows]):
                        raise AssertionError(f"{t} {g.name}: an evaluation cohort")
                elif g.kind == "pert":
                    if any(is_heldout_gene(x) for x in c.pert["perturbed_gene_id"].to_numpy()[g.rows]):
                        raise AssertionError(f"{t} {g.name}: a held-out perturbed gene")

    def counts(self) -> dict[str, int]:
        return {t: len(g) for t, g in self.groups.items()}

    def tasks(self) -> list[str]:
        return [t for t in TASKS if self.groups[t]]

    # ---- sampling
    def _X(self, kind: str) -> np.ndarray:
        return {"bulk": self.c.bulk_X, "sc": self.c.sc_X, "pert": self.c.pert_X}[kind]

    def _pick(self, task: str, n_total: int | None, k: int | None) -> Group:
        gs = self.groups[task]
        if self.mode == "train" and task == "age" and self.rng.random() >= self.pool_prob:
            gs = [g for g in gs if not g.pooled] or gs
        ok = [g for g in gs if (n_total is None or len(g.rows) >= n_total) and (k is None or len(g.cols) > k)]
        gs = ok or gs
        w = np.sqrt([len(g.rows) for g in gs])
        return gs[self.rng.choice(len(gs), p=w / w.sum())]

    def sample(self, task: str, n_total: int | None = None, n_q: int | None = None, k: int | None = None,
               group: Group | None = None, perturb_mode: str | None = None) -> Episode:  # fmt: skip
        """One episode of ``task``; ``n_total``/``n_q``/``k`` (and ``perturb_mode``) fix its shape (for batches of
        equal shape)."""
        rng = self.rng
        g = group or self._pick(task, n_total, k)
        n_avail = len(g.rows)
        n = min(n_total or int(rng.integers(self.n_rows[0], self.n_rows[1] + 1)), n_avail)
        nq = n_q or max(1, int(round(n * rng.uniform(*self.query_frac))))
        nq = min(nq, n - 2)
        kk = min(k or int(rng.integers(self.n_cols[0], self.n_cols[1] + 1)), len(g.cols) - 1)
        X = self._X(g.kind)
        for _ in range(20):
            rows = rng.choice(g.rows, n, replace=False)
            ctx, qry = rows[: n - nq], rows[n - nq :]
            if task != "health":
                break
            y = self._case[rows].astype(np.float32)
            yc = y[: n - nq]
            if 0 < yc.sum() < len(yc):
                break
        meta = {"group": g.name, "n_ctx": len(ctx), "n_qry": len(qry)}
        if task == "masked_gene":
            varies = g.cols[np.nan_to_num(X[ctx][:, g.cols]).std(0) > 0.05]  # a target that varies in the context
            target = int(rng.choice(varies if len(varies) else g.cols))
            feats = rng.choice(g.cols[g.cols != target], kk, replace=False)
            yc, yq = X[ctx, target], X[qry, target]
            Xc, Xq = X[ctx][:, feats], X[qry][:, feats]
            meta["target_gene"] = self.c.genes[target]
        elif task == "perturbation":
            pv = self._pert_var[g.cols]
            varies = np.nan_to_num(X[ctx][:, g.cols]).std(0) > Y_SD_FLOOR  # the readout must vary in the context
            if rng.random() < 0.5:  # a responsive readout half of the time
                top = g.cols[np.argsort(-pv)[: max(10, len(pv) // 10)]]
                top_v = top[np.isin(top, g.cols[varies])]
                target = int(rng.choice(top_v if len(top_v) else top))
            else:
                target = int(rng.choice(g.cols[varies] if varies.any() else g.cols))
            yc, yq = X[ctx, target], X[qry, target]
            mode = perturb_mode or str(rng.choice(self.perturb_modes))
            D = self._descr
            if mode == "descriptor":
                Xc, Xq = D[ctx], D[qry]
            else:
                feats = rng.choice(g.cols[g.cols != target], max(1, kk - D.shape[1]), replace=False)
                Xc = np.concatenate([X[ctx][:, feats], D[ctx]], axis=1)
                Xq = np.concatenate([X[qry][:, feats], D[qry]], axis=1)
            meta |= {"readout": self.c.genes[target], "features": mode}
        else:
            feats = rng.choice(g.cols, kk, replace=False)
            Xc, Xq = X[ctx][:, feats], X[qry][:, feats]
            if task == "age":
                a = self._age_bulk if g.kind == "bulk" else self._age_sc
                yc, yq = a[ctx], a[qry]
            else:
                yc, yq = self._case[ctx].astype(np.float32), self._case[qry].astype(np.float32)
        return self.episode(Xc, yc, Xq, yq, task, g.kind, meta)

    @staticmethod
    def episode(Xc, yc, Xq, yq, task: str, source: str, meta: dict | None = None) -> Episode:
        Xc, Xq = standardise(np.nan_to_num(Xc), np.nan_to_num(Xq))
        if task == "health":
            return Episode(Xc, yc.astype(np.float32), Xq, yq.astype(np.float32), task, source, meta=meta or {})
        # an expression target (masked gene, perturbation readout) is scaled by at least Y_SD_FLOOR log units: a
        # near-constant context otherwise turns a small query difference into a huge target (RM30's first run: a
        # perturbation loss spike to 119 at step 100, held-out skill -10.5)
        floor = Y_SD_FLOOR if task in ("masked_gene", "perturbation") else 1e-6
        mu, sd = float(np.mean(yc)), max(float(np.std(yc)), floor)
        return Episode(Xc, ((yc - mu) / sd).astype(np.float32), Xq, ((yq - mu) / sd).astype(np.float32), task,
                       source, mu, sd, meta or {})  # fmt: skip

    def sample_batch(self, task: str, size: int) -> list[Episode]:
        """``size`` episodes of one task and one shape (rows, query rows, columns)."""
        first = self.sample(task)
        n, nq, k = first.meta["n_ctx"] + first.meta["n_qry"], first.meta["n_qry"], first.X_ctx.shape[1]
        out = [first]
        while len(out) < size:
            e = self.sample(task, n_total=n, n_q=nq, k=k, perturb_mode=first.meta.get("features"))
            if e.X_ctx.shape == first.X_ctx.shape and e.X_qry.shape == first.X_qry.shape:
                out.append(e)
            else:  # the only groups left are smaller: shrink the batch rather than pad
                break
        return out

    def fixed_set(self, per_task: int, seed: int = 1) -> list[Episode]:
        """A reproducible list of ``per_task`` episodes per available task (the periodic quick evaluation)."""
        keep = self.rng
        self.rng = np.random.default_rng(seed)
        out = [self.sample(t) for t in self.tasks() for _ in range(per_task)]
        self.rng = keep
        return out
