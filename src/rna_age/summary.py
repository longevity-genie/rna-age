"""``rna-age summary``: small parquet tables under ``data/output/summary/`` (tracked in git).

- ``by_source.parquet``: samples per source × age_type, eligible ones, studies, donors.
- ``age_hist.parquet``: eligible samples per source × 10-year bin of the age point estimate.
- ``by_tissue.parquet``: eligible samples per source × tissue.
- ``health.parquet``: samples per source × is_healthy × is_tumor × is_cell_line × exclude_reason.
- ``duplicates.parquet``: samples present in more than one source (by GSM; after ``rna-age merge`` also by
  SRA experiment, with the source the kept copy is in).
- ``splits.parquet`` / ``cohorts.parquet`` (after ``rna-age merge``): split sizes by source, named cohorts.
- ``downloads.parquet``: every downloaded file with size, checksum and fetch time.

"Eligible" = has an age, ``exclude_reason`` is null and not a tumour (cancer is out of scope for now).
"""

from __future__ import annotations

import json
import logging

import polars as pl

from rna_age.paths import DOWNLOAD, INPUT, INTERIM, OUTPUT
from rna_age.schema import age_mid

log = logging.getLogger(__name__)


def load_samples() -> pl.DataFrame:
    paths = sorted(INPUT.glob("*/samples.parquet"))
    if not paths:
        raise FileNotFoundError(f"no {INPUT}/*/samples.parquet yet; run `rna-age parse <source>` first")
    return pl.concat([pl.read_parquet(p) for p in paths], how="vertical_relaxed")


def eligible(df: pl.DataFrame) -> pl.Expr:
    return pl.col("age_type").is_not_null() & pl.col("exclude_reason").is_null() & ~pl.col("is_tumor").fill_null(False)


def write_summary() -> dict[str, pl.DataFrame]:
    out = OUTPUT / "summary"
    out.mkdir(parents=True, exist_ok=True)
    df = load_samples()
    df = df.with_columns(eligible=eligible(df), age_mid=age_mid(df))
    el = df.filter("eligible")
    tables: dict[str, pl.DataFrame] = {}
    tables["by_source"] = (
        df.group_by("source", "age_type")
        .agg(
            samples=pl.len(),
            eligible=pl.col("eligible").sum(),
            studies=pl.col("study_id").filter("eligible").n_unique(),
            donors=pl.col("donor_id").filter("eligible").drop_nulls().n_unique(),
            healthy_eligible=(pl.col("eligible") & pl.col("is_healthy").fill_null(False)).sum(),
            pseudobulk=pl.col("is_pseudobulk").sum(),
        )
        .sort("source", "age_type", nulls_last=True)
    )
    tables["age_hist"] = (
        el.with_columns(age_bin=(pl.col("age_mid") // 10 * 10).cast(pl.Int32))
        .group_by("source", "age_type", "age_bin")
        .len("samples")
        .sort("source", "age_type", "age_bin")
    )
    tables["by_tissue"] = (
        el.group_by("source", "tissue")
        .agg(samples=pl.len(), studies=pl.col("study_id").n_unique(), age_median=pl.col("age_mid").median())
        .sort("source", "samples", descending=[False, True])
    )
    tables["health"] = (
        df.group_by("source", "is_healthy", "is_tumor", "is_cell_line", "exclude_reason")
        .agg(samples=pl.len(), with_age=pl.col("age_type").is_not_null().sum())
        .sort("source", "samples", descending=[False, True])
    )
    gsm = df.filter(pl.col("sample_id").str.starts_with("GSM"))
    tables["duplicates"] = (
        gsm.group_by("sample_id")
        .agg(sources=pl.col("source").unique().sort().str.join(","), n=pl.col("source").n_unique())
        .filter(pl.col("n") > 1)
        .group_by("sources")
        .agg(samples=pl.len())
        .sort("samples", descending=True)
    )
    merged = INTERIM / "samples_all.parquet"
    if merged.exists():
        m = pl.read_parquet(merged)
        tables["splits"] = (
            m.filter(pl.col("split").is_not_null())
            .group_by("source", "is_pseudobulk", "age_type", "split")
            .agg(samples=pl.len(), groups=pl.col("group").n_unique(), age_median=pl.col("age_mid").median())
            .sort("source", "age_type", "split")
        )
        tables["duplicates"] = (
            m.filter(pl.col("duplicate_of").is_not_null())
            .with_columns(kept_in=pl.col("duplicate_of").str.split(":").list.first())
            .group_by("source", "kept_in")
            .agg(samples=pl.len())
            .sort("samples", descending=True)
        )
        tables["cohorts"] = (
            m.filter(pl.col("cohort").is_not_null())
            .group_by("source", "cohort")
            .agg(samples=pl.len(), eligible=pl.col("eligible").sum())
            .sort("source", "samples", descending=[False, True])
        )
    rows = []
    for meta in sorted(DOWNLOAD.rglob("*.meta.json")):
        m = json.loads(meta.read_text())
        rows.append({"source": meta.relative_to(DOWNLOAD).parts[0], **m})
    tables["downloads"] = pl.DataFrame(rows) if rows else pl.DataFrame()
    for name, t in tables.items():
        t.write_parquet(out / f"{name}.parquet", compression="zstd")
    with pl.Config(tbl_rows=60, tbl_cols=12, fmt_str_lengths=40):
        print(tables["by_source"])
        print(tables["duplicates"])
    log.info("summary written to %s", out)
    return tables
