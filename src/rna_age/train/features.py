"""Expression features across sources: shared gene axis, log-CPM, HVGs chosen on the training split only.

- The gene axis is the intersection of the bulk training sources' gene tables (Ensembl IDs without version);
  pseudobulk sources keyed by symbol are mapped through ARCHS4's symbol table.
- log-CPM = log1p(counts / library size × 1e6), with the library size over the shared genes, so every source
  is normalised on the same denominator.
- HVGs: highest variance of log-CPM among genes with mean log-CPM ≥ 1, computed on (a subsample of) the
  training samples only.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import polars as pl

from rna_age.paths import INPUT, INTERIM

log = logging.getLogger(__name__)

BULK_SOURCES = ("archs4", "gtex", "bgee")


def sample_table() -> pl.DataFrame:
    """Eligible, non-duplicate samples with split labels (pseudobulk included, flagged)."""
    m = pl.read_parquet(INTERIM / "samples_all.parquet")
    return m.filter(pl.col("split").is_not_null())


def gene_ids(source: str) -> list[str]:
    return pl.read_parquet(INPUT / source / "genes.parquet", columns=["gene_id"])["gene_id"].to_list()


def symbol_map() -> dict[str, str]:
    g = pl.read_parquet(INPUT / "archs4" / "genes.parquet").drop_nulls(["symbol"])
    return dict(zip(g["symbol"].to_list(), g["gene_id"].to_list(), strict=True))


def shared_genes(sources=BULK_SOURCES) -> list[str]:
    sets = [set(gene_ids(s)) for s in sources if (INPUT / s / "genes.parquet").exists()]
    genes = sorted(set.intersection(*sets))
    log.info("shared genes across %s: %d", ",".join(sources), len(genes))
    return genes


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
