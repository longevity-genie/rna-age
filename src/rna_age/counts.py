"""Count matrix storage: ``data/input/<source>/counts/part-NNNNN.parquet``.

Each part is a *gene-by-sample* block: column ``gene_id`` (Ensembl ID without version, the same row order in
every part of a source, listed in ``genes.parquet``) followed by up to ``CHUNK`` sample columns of raw integer
counts (UInt32). Reading a subset of samples is a column projection (``pl.scan_parquet(...).select(ids)``),
and no step needs the whole matrix in memory. :func:`read_samples` returns a samples × genes numpy array.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import polars as pl

log = logging.getLogger(__name__)

CHUNK = 1000


def write_parts(
    out_dir: Path,
    gene_ids: list[str],
    blocks: Iterable[tuple[list[str], np.ndarray]],
) -> list[Path]:
    """Write gene × sample blocks (``matrix.shape == (len(gene_ids), len(sample_ids))``) as parquet parts."""
    counts_dir = out_dir / "counts"
    counts_dir.mkdir(parents=True, exist_ok=True)
    for old in counts_dir.glob("part-*.parquet"):
        old.unlink()
    paths, index = [], {}
    for i, (sample_ids, matrix) in enumerate(blocks):
        if matrix.shape != (len(gene_ids), len(sample_ids)):
            raise ValueError(f"block {i}: matrix {matrix.shape} vs {len(gene_ids)} genes × {len(sample_ids)} samples")
        if matrix.size and (np.nanmin(matrix) < 0):
            raise ValueError(f"block {i}: negative counts")
        mat = np.rint(matrix).astype(np.uint32, copy=False)
        df = pl.DataFrame({"gene_id": gene_ids}).with_columns(
            [pl.Series(sid, mat[:, j], dtype=pl.UInt32) for j, sid in enumerate(sample_ids)]
        )
        path = counts_dir / f"part-{i:05d}.parquet"
        df.write_parquet(path, compression="zstd")
        paths.append(path)
        index.update({sid: path.name for sid in sample_ids})
        log.info("wrote %s (%d genes × %d samples)", path.name, len(gene_ids), len(sample_ids))
    (counts_dir / "index.json").write_text(json.dumps(index))
    return paths


def blocks_of(sample_ids: list[str], size: int = CHUNK) -> list[list[str]]:
    return [sample_ids[i : i + size] for i in range(0, len(sample_ids), size)]


def read_samples(source_dir: Path, sample_ids: list[str]) -> tuple[list[str], np.ndarray]:
    """Return (gene_ids, samples × genes uint32 array) for ``sample_ids`` of one source."""
    counts_dir = source_dir / "counts"
    index: dict[str, str] = json.loads((counts_dir / "index.json").read_text())
    by_part: dict[str, list[str]] = {}
    for sid in sample_ids:
        by_part.setdefault(index[sid], []).append(sid)
    genes = pl.read_parquet(source_dir / "genes.parquet", columns=["gene_id"])["gene_id"].to_list()
    columns: dict[str, np.ndarray] = {}
    for part, ids in by_part.items():
        df = pl.read_parquet(counts_dir / part, columns=ids)
        columns.update({c: df[c].to_numpy() for c in ids})
    return genes, np.stack([columns[s] for s in sample_ids])
