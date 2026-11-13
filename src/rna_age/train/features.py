"""Expression features across sources: shared gene axis, log-CPM, HVGs chosen on the training split only.

- The gene axis is the intersection of the bulk training sources' gene tables (Ensembl IDs without version);
  pseudobulk sources keyed by symbol are mapped through ARCHS4's symbol table.
- log-CPM = log1p(counts / library size × 1e6), with the library size over the shared genes, so every source
  is normalised on the same denominator.
- HVGs: highest variance of log-CPM among genes with mean log-CPM ≥ 1, computed on (a subsample of) the
  training samples only.

RM26 switches (all off by default, so default runs read and train exactly as before):
- :func:`set_samples_suffix` makes :func:`sample_table` read ``samples_all{suffix}.parquet`` (a table written by
  ``rna-age merge --suffix``) for the rest of the process.
- ``shared_genes(with_sources=(...))`` narrows the bulk gene axis to the genes the named extra sources measure
  (recount3: GENCODE v26), so arms with and without such a source share one axis.
- :func:`exclude_train_studies` drops named studies from a training mask (they are still predicted and scored).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import polars as pl

from rna_age.paths import INPUT, INTERIM

log = logging.getLogger(__name__)

BULK_SOURCES = ("archs4", "gtex", "bgee")


_SAMPLES_SUFFIX = ""


def set_samples_suffix(suffix: str) -> None:
    """Every later :func:`sample_table` call reads ``samples_all{suffix}.parquet`` ("" = the shared table)."""
    global _SAMPLES_SUFFIX
    _SAMPLES_SUFFIX = suffix


def samples_path(suffix: str | None = None) -> Path:
    return INTERIM / f"samples_all{_SAMPLES_SUFFIX if suffix is None else suffix}.parquet"


def sample_table(suffix: str | None = None) -> pl.DataFrame:
    """Eligible, non-duplicate samples with split labels (pseudobulk included, flagged)."""
    m = pl.read_parquet(samples_path(suffix))
    return m.filter(pl.col("split").is_not_null())


def exclude_train_studies(train: np.ndarray, study_ids: pl.Series, studies: list[str] | tuple[str, ...]) -> np.ndarray:
    """``train`` with the rows of ``studies`` set False (no-op for an empty list)."""
    if not studies:
        return train
    return train & ~study_ids.is_in(list(studies)).fill_null(False).to_numpy()


def gene_ids(source: str) -> list[str]:
    return pl.read_parquet(INPUT / source / "genes.parquet", columns=["gene_id"])["gene_id"].to_list()


def symbol_map() -> dict[str, str]:
    g = pl.read_parquet(INPUT / "archs4" / "genes.parquet").drop_nulls(["symbol"])
    return dict(zip(g["symbol"].to_list(), g["gene_id"].to_list(), strict=True))


def shared_genes(sources=BULK_SOURCES, with_sources: tuple[str, ...] | list[str] = ()) -> list[str]:
    """Genes all ``sources`` measure (those with a gene table), intersected with every ``with_sources`` gene table
    (which must exist)."""
    sets = [set(gene_ids(s)) for s in sources if (INPUT / s / "genes.parquet").exists()]
    genes = set.intersection(*sets)
    if with_sources:
        n0 = len(genes)
        for s in with_sources:
            genes &= set(gene_ids(s))
        log.info("shared genes across %s: %d; ∩ %s: %d", ",".join(sources), n0, ",".join(with_sources), len(genes))
        return sorted(genes)
    genes = sorted(genes)
    log.info("shared genes across %s: %d", ",".join(sources), len(genes))
    return genes


def gene_axis_label(with_sources: tuple[str, ...] | list[str] = ()) -> str:
    return f"shared({','.join(BULK_SOURCES)})" + "".join(f" ∩ {s}" for s in with_sources)


def load_counts(source: str, sample_ids: list[str], genes: list[str]) -> np.ndarray:
    """samples × genes float32 raw counts of one source on the given gene axis (absent genes = 0)."""
    d = INPUT / source
    index: dict[str, str] = json.loads((d / "counts" / "index.json").read_text())
    src_genes = gene_ids(source)
    if not set(genes) & set(src_genes):  # symbol-keyed source
        smap = symbol_map()
        src_genes = [smap.get(g, g) for g in src_genes]
    pos = {g: i for i, g in enumerate(src_genes)}
    rows = np.array([pos.get(g, -1) for g in genes])
    have = rows >= 0
    out = np.zeros((len(sample_ids), len(genes)), dtype=np.float32)
    col_of = {s: i for i, s in enumerate(sample_ids)}
    by_part: dict[str, list[str]] = {}
    for s in sample_ids:
        by_part.setdefault(index[s], []).append(s)
    for part, ids in by_part.items():
        block = pl.read_parquet(d / "counts" / part, columns=ids).to_numpy().astype(np.float32)  # genes × ids
        sub = block[rows[have]]
        for j, s in enumerate(ids):
            out[col_of[s], have] = sub[:, j]
    return out


def load_matrix(samples: pl.DataFrame, genes: list[str]) -> np.ndarray:
    """Raw counts for a multi-source sample table, rows in the table's order."""
    out = np.zeros((samples.height, len(genes)), dtype=np.float32)
    samples = samples.with_row_index("_row")
    for (source,), sdf in samples.group_by("source"):
        ids = sdf["sample_id"].to_list()
        out[sdf["_row"].to_numpy()] = load_counts(source, ids, genes)
        log.info("loaded %s: %d samples", source, len(ids))
    return out


def log_cpm(counts: np.ndarray) -> np.ndarray:
    lib = counts.sum(axis=1, keepdims=True)
    lib[lib == 0] = 1
    return np.log1p(counts / lib * 1e6).astype(np.float32)


def select_hvg(train_logcpm: np.ndarray, genes: list[str], k: int, min_mean: float = 1.0) -> list[str]:
    mean = train_logcpm.mean(axis=0)
    var = train_logcpm.var(axis=0)
    var[mean < min_mean] = -1
    top = np.argsort(-var)[:k]
    return [genes[i] for i in sorted(top)]


def gene_lengths(genes: list[str]) -> np.ndarray:
    """Gene lengths (bp) from BulkFormer's released normalisation vocabulary; NaN where unknown."""
    from rna_age.paths import DATA

    v = pl.read_csv(DATA / "download" / "models" / "bulkformer" / "atlas" / "normalization_vocab.csv")
    lens = dict(zip(v["ensembl_gene"].to_list(), v["gene_length_bp"].to_list(), strict=True))
    return np.array([lens.get(g, np.nan) for g in genes], dtype=np.float64)


def log_tpm(counts: np.ndarray, lengths: np.ndarray, umi_rows: np.ndarray | None = None) -> np.ndarray:
    """log2(TPM + 1) over the genes with a known length (BulkFormer's normalisation, on our gene axis).

    ``umi_rows`` marks 3'/5'-tag UMI samples (10x pseudobulk): their counts are not proportional to gene length,
    so they are not divided by it (TPM reduces to CPM on the same gene set).
    """
    rate = counts / (lengths[None, :] / 1000.0)
    if umi_rows is not None and umi_rows.any():
        rate[umi_rows] = counts[umi_rows]
    tot = rate.sum(axis=1, keepdims=True)
    tot[tot == 0] = 1
    return np.log2(rate / tot * 1e6 + 1).astype(np.float32)
