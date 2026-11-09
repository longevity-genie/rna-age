"""ARCHS4 human gene counts (Maayan lab): one HDF5 with all GEO/SRA human bulk RNA-seq samples.

The file is ``human_gene_v2.latest.h5`` on the Maayan lab S3; no versioned name exists for this release, so
the S3 ETag recorded in the sidecar pins what we fetched.

Parsing has two steps, both run by ``rna-age parse archs4`` (a Slurm job: the counts step slices the 64 GB
HDF5):

1. metadata → ``metadata_all.parquet`` (every sample, raw GEO strings) and ``samples.parquet`` (samples with a
   parsed donor age, all flags; ages from ``characteristics_ch1`` through :mod:`rna_age.ages`).
2. counts of the age-annotated eligible samples (``exclude_reason`` null) → ``counts/`` + ``genes.parquet``.
"""

from __future__ import annotations

import itertools
import json
import logging
import os
import re
import shutil
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import polars as pl

from rna_age.ages import find_age
from rna_age.annotate import add_flags
from rna_age.counts import write_parts
from rna_age.download import Remote
from rna_age.download import fetch as _fetch
from rna_age.paths import download_dir, input_dir
from rna_age.schema import conform

log = logging.getLogger(__name__)

NAME = "archs4"
URL = "https://s3.dev.maayanlab.cloud/archs4/files/human_gene_v2.latest.h5"
H5 = Remote(
    url=URL,
    name="human_gene_v2.latest.h5",
    size=64_432_723_113,
    checksum="s3etag:9b8fcd0486471467ef00169bc4ce1511-7681",
)
META_FIELDS = (
    "geo_accession",
    "series_id",
    "title",
    "source_name_ch1",
    "characteristics_ch1",
    "organism_ch1",
    "molecule_ch1",
    "library_strategy",
    "library_source",
    "library_selection",
    "instrument_model",
    "singlecellprobability",
    "readsaligned",
    "relation",
    "sample",
)


def fetch(connections: int = 8) -> None:
    _fetch(H5, download_dir(NAME), connections=connections)


def h5_path() -> Path:
    return download_dir(NAME) / H5.name


def _decode(arr: np.ndarray) -> list[str]:
    if arr.dtype.kind in ("S", "O"):
        return [x.decode("utf-8", "replace") if isinstance(x, bytes) else str(x) for x in arr]
    return [str(x) for x in arr]


def read_metadata(path: Path | None = None) -> pl.DataFrame:
    """All per-sample metadata fields present in the file, as strings (plus numeric ones as numbers)."""
    path = path or h5_path()
    cols: dict[str, list] = {}
    with h5py.File(path, "r") as h:
        grp = h["meta/samples"]
        available = list(grp.keys())
        log.info("ARCHS4 meta/samples fields: %s", ", ".join(available))
        for name in available:
            ds = grp[name]
            if ds.ndim != 1:
                continue
            t0 = time.time()
            arr = ds[()]
            cols[name] = arr.tolist() if arr.dtype.kind in "fiu" else _decode(arr)
            log.info("read meta/samples/%s (%d) in %.1fs", name, len(arr), time.time() - t0)
    return pl.DataFrame(cols, strict=False)


def split_items(text: str) -> list[tuple[str, str]]:
    """Split a GEO characteristics string into (key, value) pairs.

    Items are separated by tabs or by commas; a comma only separates items when what follows looks like
    ``key:`` (so ``treatment: IL-4, IL-13`` stays one value).
    """
    out: list[tuple[str, str]] = []
    for part in re.split(r"\t|,(?=\s*[^,:]{1,80}:)", text):
        part = part.strip()
        if not part:
            continue
        key, colon, value = part.partition(":")
        if colon:
            out.append((key.strip(), value.strip()))
        elif out:
            k, v = out[-1]
            out[-1] = (k, f"{v}, {part}")
        else:
            out.append(("", part))
    return out


def parse_metadata(meta: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Samples whose characteristics carry a donor age (schema-conformed, with flags), plus an audit table."""
    m = meta.with_columns(
        text=pl.concat_str(
            [pl.col(c).fill_null("") for c in ("title", "source_name_ch1", "characteristics_ch1") if c in meta.columns],
            separator=" | ",
        ).str.to_lowercase()
    )
    cand = m.filter(pl.col("characteristics_ch1").str.contains(r"(?i)\bage"))
    log.info("ARCHS4: %d samples, %d mention 'age' in characteristics", m.height, cand.height)
    rows = []
    for gsm, chars in cand.select("geo_accession", "characteristics_ch1").iter_rows():
        items = split_items(chars)
        a = find_age(items)
        rows.append(
            {
                "geo_accession": gsm,
                "fields": [{"key": k, "value": v} for k, v in items],
                "age_type": a.age_type if a else None,
                "age_years": a.age_years if a else None,
                "age_min": a.age_min if a else None,
                "age_max": a.age_max if a else None,
                "age_raw": a.raw if a else None,
                "age_key_kind": a.key_kind if a else None,
            }
        )
    parsed = pl.DataFrame(
        rows,
        schema={
            "geo_accession": pl.String,
            "fields": pl.List(pl.Struct({"key": pl.String, "value": pl.String})),
            "age_type": pl.String,
            "age_years": pl.Float64,
            "age_min": pl.Float64,
            "age_max": pl.Float64,
            "age_raw": pl.String,
            "age_key_kind": pl.String,
        },
    ).filter(pl.col("age_type").is_not_null())
    df = cand.join(parsed, on="geo_accession", how="inner")
    log.info("ARCHS4: %d samples with a parsed donor age", df.height)
    organism = pl.col("organism_ch1") if "organism_ch1" in df.columns else pl.lit("Homo sapiens")
    sc = (
        (pl.col("singlecellprobability").cast(pl.Float64, strict=False) > 0.5)
        if "singlecellprobability" in df.columns
        else pl.lit(False)
    )
    df = df.with_columns(
        exclude_reason=pl.when(~organism.str.contains("(?i)homo sapiens"))
        .then(pl.lit("non_human"))
        .when(sc)
        .then(pl.lit("single_cell"))
    )
    df = add_flags(df)
    series = pl.col("series_id").str.replace_all(r"\s+", "").str.split(",")
    srr = pl.col("relation").str.extract_all(r"SR[RX]\d+").list.join(",") if "relation" in df.columns else None
    df = df.with_columns(
        sample_id=pl.col("geo_accession"),
        source=pl.lit(NAME),
        study_id=series.list.first(),
        series_ids=series.list.join(","),
        donor_id=pl.when(pl.col("donor_id").is_not_null())
        .then(series.list.first() + ":" + pl.col("donor_id"))
        .otherwise(pl.col("geo_accession")),
        tissue=pl.coalesce("tissue", "source_name_ch1"),
        platform=pl.col("instrument_model") if "instrument_model" in df.columns else pl.lit(None),
        library_type=pl.concat_str(
            [pl.col(c) for c in ("library_strategy", "library_selection", "library_source") if c in df.columns],
            separator="/",
            ignore_nulls=True,
        ),
        assay=pl.col("library_strategy") if "library_strategy" in df.columns else pl.lit(None),
        srr_ids=srr if srr is not None else pl.lit(None),
        read_count_total=pl.col("readsaligned").cast(pl.Int64, strict=False)
        if "readsaligned" in df.columns
        else pl.lit(None),
    )
    keep_extra = df.select("sample_id", "age_key_kind", "characteristics_ch1", "title")
    return conform(df), keep_extra


_H5: h5py.File | None = None


def _init_worker(path: str) -> None:
    global _H5
    _H5 = h5py.File(path, "r", rdcc_nbytes=256 * 1024 * 1024)


def _runs(sel: np.ndarray, max_gap: int = 64, max_width: int = 2000) -> list[tuple[int, int]]:
    """Split sorted column indices into (start, end) index ranges of ``sel`` whose columns are close together."""
    out, a = [], 0
    for i in range(1, len(sel) + 1):
        if i == len(sel) or sel[i] - sel[i - 1] > max_gap or sel[i] - sel[a] >= max_width:
            out.append((a, i))
            a = i
    return out


def _read_block(sel: np.ndarray) -> np.ndarray:
    """genes × len(sel) counts for sorted sample indices ``sel``.

    h5py fancy indexing over scattered columns is slow and memory-hungry (~15 GB for 1,000 columns), so nearby
    columns are read as bounded dense windows (≤ 2,000 columns, ~0.5 GB) and isolated ones one at a time.
    """
    expr = _H5["data/expression"]
    t0 = time.time()
    out = np.empty((expr.shape[0], len(sel)), dtype=expr.dtype)
    for a, b in _runs(sel):
        lo, hi = int(sel[a]), int(sel[b - 1]) + 1
        if b - a == 1:
            out[:, a] = expr[:, lo]
        else:
            out[:, a:b] = expr[:, lo:hi][:, sel[a:b] - lo]
    log.info("read %d samples (span %d) in %.1fs", len(sel), int(sel[-1] - sel[0]) + 1, time.time() - t0)
    return out


def local_copy(path: Path) -> Path:
    """Copy the HDF5 to node-local scratch (random chunk reads on the shared filesystem are slow)."""
    tmp = os.environ.get("RNA_AGE_LOCAL_TMP")  # opt-in only: $TMPDIR may be tmpfs (counts as job memory)
    if not tmp:
        return path
    Path(tmp).mkdir(parents=True, exist_ok=True)
    dest = Path(tmp) / path.name
    if not (dest.exists() and dest.stat().st_size == path.stat().st_size):
        t0 = time.time()
        if shutil.disk_usage(tmp).free < path.stat().st_size * 1.1:
            log.warning("not enough space in %s for a local copy; reading %s in place", tmp, path)
            return path
        shutil.copyfile(path, dest)
        log.info("copied %s to %s in %.0fs", path.name, tmp, time.time() - t0)
    return dest


def extract_counts(samples: pl.DataFrame, out: Path, path: Path | None = None, batch: int = 1000) -> None:
    """Write raw counts of ``samples`` (column ``sample_id`` = GSM) from the HDF5 as parquet parts."""
    src = path or h5_path()
    path = local_copy(src)
    try:
        _extract(samples, out, path, batch)
    finally:
        if path != src:
            path.unlink(missing_ok=True)


def _extract(samples: pl.DataFrame, out: Path, path: Path, batch: int) -> None:
    with h5py.File(path, "r") as h:
        gsm = _decode(h["meta/samples/geo_accession"][()])
        pos = {g: i for i, g in enumerate(gsm)}
        gm = h["meta/genes"]
        log.info("ARCHS4 meta/genes fields: %s", ", ".join(gm.keys()))
        ens_key = next(k for k in ("ensembl_gene", "ensembl_gene_id", "gene_id", "ensembl") if k in gm)
        sym_key = next((k for k in ("symbol", "gene_symbol", "genes") if k in gm), None)
        biotype_key = next((k for k in ("biotype", "gene_biotype") if k in gm), None)
        genes = pl.DataFrame(
            {
                "gene_id": _decode(gm[ens_key][()]),
                "symbol": _decode(gm[sym_key][()]) if sym_key else None,
                "biotype": _decode(gm[biotype_key][()]) if biotype_key else None,
            }
        ).with_columns(pl.col("gene_id").str.replace(r"\.\d+$", ""))
        expr = h["data/expression"]
        log.info(
            "data/expression %s %s chunks=%s compression=%s", expr.shape, expr.dtype, expr.chunks, expr.compression
        )
        ids = samples["sample_id"].to_list()
        idx = np.array([pos[s] for s in ids])
        order = np.argsort(idx)
        ids_sorted, idx_sorted = [ids[i] for i in order], idx[order]
        # unique gene IDs: ARCHS4 may list an Ensembl ID more than once; keep the first occurrence
        dup = genes["gene_id"].is_duplicated().sum()
        first = (
            genes.with_row_index()
            .group_by("gene_id", maintain_order=True)
            .agg(pl.col("index").min())["index"]
            .to_numpy()
        )
        if dup:
            log.warning("ARCHS4: %d rows share an Ensembl ID; keeping the first of each", dup)
        gene_rows = np.sort(first)
        genes = genes[gene_rows]

    batches = [(ids_sorted[b : b + batch], idx_sorted[b : b + batch]) for b in range(0, len(ids_sorted), batch)]
    workers = int(os.environ.get("RNA_AGE_WORKERS", os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    log.info("ARCHS4: %d samples in %d batches, %d workers", len(ids_sorted), len(batches), workers)

    def blocks():
        # in-order results with a bounded window: at most 2 × workers blocks (~270 MB each) in memory
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(str(path),)) as pool:
            pending: deque = deque()
            it = iter(batches)
            for ids_b, sel in itertools.islice(it, 2 * workers):
                pending.append((ids_b, pool.submit(_read_block, sel)))
            while pending:
                ids_b, fut = pending.popleft()
                block = fut.result()
                nxt = next(it, None)
                if nxt is not None:
                    pending.append((nxt[0], pool.submit(_read_block, nxt[1])))
                yield ids_b, block[gene_rows]

    write_parts(out, genes["gene_id"].to_list(), blocks())
    genes.write_parquet(out / "genes.parquet")


def parse() -> None:
    out = input_dir(NAME)
    meta_all = out / "metadata_all.parquet"
    if meta_all.exists():
        meta = pl.read_parquet(meta_all)
    else:
        meta = read_metadata()
        meta.write_parquet(meta_all, compression="zstd")
    samples, audit = parse_metadata(meta)
    samples.write_parquet(out / "samples.parquet")
    audit.write_parquet(out / "age_audit.parquet")
    log.info(
        "ARCHS4 ages: %s",
        samples.group_by("age_type", "exclude_reason").len().sort("len", descending=True).to_dicts(),
    )
    eligible = samples.filter(pl.col("exclude_reason").is_null())
    log.info("ARCHS4: extracting counts for %d eligible age-annotated samples", eligible.height)
    extract_counts(eligible, out)
    (out / "README.json").write_text(
        json.dumps(
            {
                "source": NAME,
                "file": H5.name,
                "etag": H5.checksum,
                "samples_with_age": samples.height,
                "eligible_with_counts": eligible.height,
            },
            indent=2,
        )
    )
