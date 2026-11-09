"""GTEx v10 (public): RNA-SeQC gene read counts + sample and subject annotations from ``gs://adult-gtex``.

Public donor age is a 10-year bin (``"50-59"``); exact age is dbGaP-controlled and not used. Every sample is
``age_type = "binned"``; ``study_id`` and ``donor_id`` are the GTEx donor (``GTEX-1117F``), the unit to group by
in splits. Decisions:
- ``is_healthy`` is True for all samples (GTEx donors are non-diseased post-mortem or surgical donors);
  ``disease`` keeps the Hardy death-circumstance scale (``hardy:0`` = ventilator case ... ``hardy:4``).
- EBV-transformed lymphocytes are a cell line (``is_cell_line``, excluded). Cultured fibroblasts are primary
  cultures from the donor and stay eligible (``cell_type = "cultured fibroblasts"``), as in Fleischer 2018.
"""

from __future__ import annotations

import gzip
import json
import logging

import httpx
import numpy as np
import polars as pl

from rna_age.counts import write_parts
from rna_age.download import Remote, gcs_md5_hex
from rna_age.download import fetch as _fetch
from rna_age.paths import download_dir, input_dir
from rna_age.schema import conform

log = logging.getLogger(__name__)

NAME = "gtex"
BUCKET = "adult-gtex"
OBJECTS = (
    "bulk-gex/v10/rna-seq/GTEx_Analysis_v10_RNASeQCv2.4.2_gene_reads.gct.gz",
    "annotations/v10/metadata-files/GTEx_Analysis_v10_Annotations_SampleAttributesDS.txt",
    "annotations/v10/metadata-files/GTEx_Analysis_v10_Annotations_SubjectPhenotypesDS.txt",
    "annotations/v10/metadata-files/GTEx_Analysis_v10_Annotations_SampleAttributesDD.xlsx",
    "annotations/v10/metadata-files/GTEx_Analysis_v10_Annotations_SubjectPhenotypesDD.xlsx",
)
GCT, SAMPLE_ATTR, SUBJECT = (o.rsplit("/", 1)[1] for o in OBJECTS[:3])


def remotes() -> list[Remote]:
    """Look each object up in the GCS JSON API to get its size and md5."""
    out = []
    with httpx.Client(timeout=60, follow_redirects=True) as c:
        for obj in OBJECTS:
            meta = (
                c.get(f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o/{obj.replace('/', '%2F')}")
                .raise_for_status()
                .json()
            )
            out.append(
                Remote(
                    url=f"https://storage.googleapis.com/{BUCKET}/{obj}",
                    name=obj.rsplit("/", 1)[1],
                    size=int(meta["size"]),
                    checksum="md5:" + gcs_md5_hex(meta["md5Hash"]),
                )
            )
    return out


def fetch(connections: int = 8) -> None:
    for r in remotes():
        _fetch(r, download_dir(NAME), connections=connections)


def parse_samples() -> pl.DataFrame:
    d = download_dir(NAME)
    attr = pl.read_csv(d / SAMPLE_ATTR, separator="\t", infer_schema_length=0, quote_char=None)
    subj = pl.read_csv(d / SUBJECT, separator="\t", infer_schema_length=0)
    with gzip.open(d / GCT, "rt") as f:
        f.readline(), f.readline()
        header = f.readline().rstrip("\n").split("\t")
    in_matrix = pl.DataFrame({"SAMPID": header[2:]})
    df = (
        in_matrix.join(attr, on="SAMPID", how="left")
        .with_columns(SUBJID=pl.col("SAMPID").str.extract(r"^([^-]+-[^-]+)"))
        .join(subj, on="SUBJID", how="left")
    )
    bounds = pl.col("AGE").str.extract_groups(r"^(\d+)-(\d+)$")
    lcl = pl.col("SMTSD").str.contains("(?i)EBV-transformed")
    return conform(
        df.with_columns(
            sample_id=pl.col("SAMPID"),
            source=pl.lit(NAME),
            study_id=pl.col("SUBJID"),
            donor_id=pl.col("SUBJID"),
            age_min=bounds.struct.field("1").cast(pl.Float64),
            age_max=bounds.struct.field("2").cast(pl.Float64),
            age_type=pl.when(pl.col("AGE").is_not_null()).then(pl.lit("binned")),
            age_raw=pl.col("AGE"),
            sex=pl.col("SEX").replace_strict({"1": "male", "2": "female"}, default=None),
            tissue=pl.col("SMTSD"),
            cell_type=pl.when(pl.col("SMTSD").str.contains("(?i)fibroblasts"))
            .then(pl.lit("cultured fibroblasts"))
            .when(lcl)
            .then(pl.lit("EBV-transformed lymphocytes")),
            disease=pl.when(pl.col("DTHHRDY").is_not_null() & (pl.col("DTHHRDY") != "")).then(
                pl.lit("hardy:") + pl.col("DTHHRDY")
            ),
            is_healthy=pl.lit(True),
            is_tumor=pl.lit(False),
            is_cell_line=lcl,
            platform=pl.lit("Illumina"),
            read_count_total=pl.col("SMRDTTL").cast(pl.Float64, strict=False).cast(pl.Int64),
            library_type=pl.col("SMGEBTCHT"),
            exclude_reason=pl.when(lcl).then(pl.lit("cell_line")),
        )
    )


def _read_gct(path, chunk_rows: int = 2000):
    """Yield (gene_ids, symbols, float32 block genes × samples) from a GCT, a few thousand rows at a time."""
    with gzip.open(path, "rt") as f:
        f.readline(), f.readline()
        f.readline()
        ids, syms, rows = [], [], []
        for line in f:
            name, sym, values = line.rstrip("\n").split("\t", 2)
            ids.append(name)
            syms.append(sym)
            rows.append(np.fromstring(values, sep="\t", dtype=np.float32))
            if len(rows) == chunk_rows:
                yield ids, syms, np.vstack(rows)
                ids, syms, rows = [], [], []
        if rows:
            yield ids, syms, np.vstack(rows)


def parse() -> None:
    out = input_dir(NAME)
    samples = parse_samples()
    log.info("GTEx: %d samples, %d donors", samples.height, samples["donor_id"].n_unique())

    gene_ids, symbols, blocks = [], [], []
    for ids, syms, block in _read_gct(download_dir(NAME) / GCT):
        gene_ids += ids
        symbols += syms
        blocks.append(block.astype(np.uint32) if np.all(block == np.rint(block)) else np.rint(block).astype(np.uint32))
        log.info("read %d genes", len(gene_ids))
    matrix = np.vstack(blocks)  # genes × samples, uint32 (~6 GB for 74k × 20k)
    del blocks

    genes = pl.DataFrame({"gene_id_versioned": gene_ids, "symbol": symbols}).with_columns(
        gene_id=pl.col("gene_id_versioned").str.replace(r"\.\d+(_PAR_Y)?$", "${1}")
    )
    keep = ~genes["gene_id"].str.ends_with("_PAR_Y")
    keep_idx = np.flatnonzero(keep.to_numpy())
    genes = genes.filter(keep)
    if genes["gene_id"].n_unique() != genes.height:
        raise ValueError("duplicate gene IDs after stripping versions")
    matrix = matrix[keep_idx]

    with gzip.open(download_dir(NAME) / GCT, "rt") as f:
        f.readline(), f.readline()
        header = f.readline().rstrip("\n").split("\t")[2:]
    col = {s: i for i, s in enumerate(header)}
    ids = samples["sample_id"].to_list()

    def blocks_iter():
        for i in range(0, len(ids), 1000):
            chunk = ids[i : i + 1000]
            yield chunk, matrix[:, [col[s] for s in chunk]]

    write_parts(out, genes["gene_id"].to_list(), blocks_iter())
    totals = matrix.sum(axis=0, dtype=np.int64)
    samples = samples.with_columns(read_count_total=pl.Series([int(totals[col[s]]) for s in ids], dtype=pl.Int64))
    genes.select("gene_id", "symbol", "gene_id_versioned").with_columns(biotype=pl.lit(None, pl.String)).write_parquet(
        out / "genes.parquet"
    )
    samples.write_parquet(out / "samples.parquet")
    (out / "README.json").write_text(
        json.dumps({"source": NAME, "release": "v10", "samples": samples.height, "genes": genes.height}, indent=2)
    )
    log.info("GTEx parsed: %d samples × %d genes → %s", samples.height, genes.height, out)
