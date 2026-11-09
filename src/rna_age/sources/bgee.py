"""Bgee 15.2 human bulk RNA-seq: curated healthy wild-type libraries with developmental-stage annotation.

Bgee maps adult donors to stages such as ``"54-year-old stage (human)"`` (exact age) or
``"fifth decade stage (human)"`` (binned 40-49). We take every non-GTEx experiment that has at least one
library with an exact-year or decade stage (GTEx, ``SRP012682``, is its own source with the same bins).
Read counts are Bgee's kallisto estimates (fractional), rounded to integers.
"""

from __future__ import annotations

import gzip
import json
import logging
import re
import tarfile

import httpx
import numpy as np
import polars as pl

from rna_age.counts import blocks_of, write_parts
from rna_age.download import Remote
from rna_age.download import fetch as _fetch
from rna_age.paths import download_dir, input_dir
from rna_age.schema import conform

log = logging.getLogger(__name__)

NAME = "bgee"
RELEASE = "15.2"
BASE = "https://www.bgee.org/ftp/bgee_v15_2/download/processed_expr_values/rna_seq/Homo_sapiens"
LIBRARIES_TAR = "Homo_sapiens_RNA-Seq_experiments_libraries.tar.gz"
GTEX_EXPERIMENT = "SRP012682"
ORDINALS = {
    "first": 0, "second": 10, "third": 20, "fourth": 30, "fifth": 40, "sixth": 50, "seventh": 60,
    "eighth": 70, "ninth": 80, "tenth": 90,
}  # fmt: skip


def stage_age(stage: str | None) -> tuple[str | None, float | None, float | None, float | None]:
    """(age_type, age_years, age_min, age_max) from a Bgee HsapDv stage name."""
    if not stage:
        return None, None, None, None
    s = stage.lower()
    if m := re.match(r"^(\d+)-year-old", s):
        y = float(m.group(1))
        return "exact", y, y, y
    if m := re.match(r"^(\d+)-month-old", s):
        y = float(m.group(1)) / 12
        return "exact", y, y, y
    if m := re.match(r"^(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth) decade", s):
        lo = float(ORDINALS[m.group(1)])
        return "binned", None, lo, lo + 9
    if re.search(r"(\d+)(?:th|st|nd|rd) decade", s):
        lo = float(re.search(r"(\d+)(?:th|st|nd|rd) decade", s).group(1)) * 10 - 10
        return "binned", None, lo, lo + 9
    if re.match(r"^(\d+) years? and over|^80 year-old and over", s):
        return "censored", None, float(re.match(r"^(\d+)", s).group(1)), None
    return None, None, None, None


def read_libraries() -> pl.DataFrame:
    tar = download_dir(NAME) / LIBRARIES_TAR
    with tarfile.open(tar) as t:
        member = next(m for m in t.getmembers() if m.name.endswith("_libraries.tsv"))
        raw = t.extractfile(member).read().decode()
    lines = raw.splitlines()
    header = lines[0].rstrip("\t").split("\t")
    rows = [line.split("\t") for line in lines[1:] if line.strip()]
    width = max(len(r) for r in rows)
    names = header + [f"extra_{i}" for i in range(width - len(header))]
    df = pl.DataFrame([r + [""] * (width - len(r)) for r in rows], schema=names, orient="row")
    return df.with_columns(pl.all().str.strip_chars('"'))


def _experiments_with_age(libs: pl.DataFrame) -> list[str]:
    ages = libs["Stage name"].map_elements(lambda s: stage_age(s)[0], return_dtype=pl.String)
    return sorted(
        libs.filter(ages.is_not_null() & (pl.col("Experiment ID") != GTEX_EXPERIMENT))["Experiment ID"].unique()
    )


def fetch(connections: int = 4) -> None:
    d = download_dir(NAME)
    _fetch(Remote(url=f"{BASE}/{LIBRARIES_TAR}", name=LIBRARIES_TAR), d, connections=1)
    experiments = _experiments_with_age(read_libraries())
    log.info("Bgee: %d non-GTEx experiments with age-annotated libraries", len(experiments))
    for exp in experiments:
        name = f"Homo_sapiens_RNA-Seq_read_counts_TPM_{exp}.tsv.gz"
        try:
            _fetch(Remote(url=f"{BASE}/{name}", name=name), d, connections=min(connections, 4))
        except httpx.HTTPStatusError as e:
            log.warning("Bgee %s: %s", exp, e)


def parse() -> None:
    out = input_dir(NAME)
    libs = read_libraries()
    ages = [stage_age(s) for s in libs["Stage name"].to_list()]
    libs = libs.with_columns(
        age_type=pl.Series([a[0] for a in ages], dtype=pl.String),
        age_years=pl.Series([a[1] for a in ages], dtype=pl.Float64),
        age_min=pl.Series([a[2] for a in ages], dtype=pl.Float64),
        age_max=pl.Series([a[3] for a in ages], dtype=pl.Float64),
    ).filter(pl.col("age_type").is_not_null() & (pl.col("Experiment ID") != GTEX_EXPERIMENT))
    run_col = next((c for c in libs.columns if libs[c].str.contains(r"^[SED]RR\d+").any()), None)
    samples = conform(
        libs.with_columns(
            sample_id=pl.col("Library ID"),
            source=pl.lit(NAME),
            study_id=pl.col("Experiment ID"),
            donor_id=pl.lit(None, pl.String),
            age_raw=pl.col("Stage name"),
            sex=pl.col("Sex").replace_strict({"female": "female", "male": "male"}, default=None),
            tissue=pl.col("Anatomical entity name"),
            disease=pl.lit("normal"),
            is_healthy=pl.lit(True),
            is_tumor=pl.lit(False),
            is_cell_line=pl.lit(False),
            platform=pl.col("Platform ID"),
            library_type=pl.concat_str(["Protocol", "Library type"], separator="/"),
            assay=pl.lit("RNA-Seq"),
            srr_ids=pl.col(run_col) if run_col else pl.lit(None),
            exclude_reason=pl.lit(None, pl.String),
        )
    )
    # counts: one long file per experiment → gene × library blocks
    d = download_dir(NAME)
    mats: dict[str, dict[str, np.ndarray]] = {}
    gene_order: list[str] | None = None
    have = []
    for exp in samples["study_id"].unique().sort().to_list():
        path = d / f"Homo_sapiens_RNA-Seq_read_counts_TPM_{exp}.tsv.gz"
        if not path.exists():
            log.warning("Bgee %s: counts file missing, skipping", exp)
            continue
        with gzip.open(path, "rb") as f:
            long = pl.read_csv(
                f.read(), separator="\t", columns=["Library ID", "Gene ID", "Read count"], quote_char='"'
            )
        wide = long.pivot(on="Library ID", index="Gene ID", values="Read count").sort("Gene ID")
        genes = wide["Gene ID"].to_list()
        if gene_order is None:
            gene_order = genes
        elif genes != gene_order:
            wide = pl.DataFrame({"Gene ID": gene_order}).join(wide, on="Gene ID", how="left").fill_null(0)
        for lib in wide.columns[1:]:
            mats[lib] = wide[lib].to_numpy()
        have.append(exp)
        log.info("Bgee %s: %d libraries", exp, wide.width - 1)
    samples = samples.filter(pl.col("sample_id").is_in(list(mats)))
    ids = samples["sample_id"].to_list()
    totals = {s: int(np.rint(mats[s]).sum()) for s in ids}
    samples = samples.with_columns(read_count_total=pl.Series([totals[s] for s in ids], dtype=pl.Int64))
    write_parts(out, gene_order, ((b, np.column_stack([mats[s] for s in b])) for b in blocks_of(ids)))
    pl.DataFrame({"gene_id": gene_order}).with_columns(
        symbol=pl.lit(None, pl.String), biotype=pl.lit(None, pl.String)
    ).write_parquet(out / "genes.parquet")
    samples.write_parquet(out / "samples.parquet")
    (out / "README.json").write_text(
        json.dumps({"source": NAME, "release": RELEASE, "experiments": have, "samples": samples.height}, indent=2)
    )
    log.info("Bgee parsed: %d libraries from %d experiments", samples.height, len(have))
