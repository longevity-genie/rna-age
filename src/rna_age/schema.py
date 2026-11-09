"""The shared ``samples.parquet`` schema (one row per sample, same columns for every source).

Age columns:
- ``age_type = "exact"``: ``age_years`` set; ``age_min = age_max = age_years``.
- ``age_type = "binned"``: ``age_years`` null; ``[age_min, age_max]`` is the closed interval of completed years
  (GTEx ``"50-59"`` → 50, 59; the midpoint of that bin is 55.0, see :func:`age_mid`).
- ``age_type = "censored"``: ``age_min`` set (e.g. ``">89"`` → 90), ``age_max`` null, ``age_years`` null.
``age_raw`` keeps the string the age was parsed from.
"""

from __future__ import annotations

import polars as pl

SAMPLE_SCHEMA: dict[str, pl.DataType] = {
    "sample_id": pl.String,  # GSM / GTEx SAMPID / TCGA aliquot barcode
    "source": pl.String,  # gtex | archs4 | ncbi_geo | bgee | geo_cohorts | census_pseudobulk | terekhova2023
    "study_id": pl.String,  # GSE / TCGA project / GTEx donor (grouping unit for splits)
    "donor_id": pl.String,
    "age_years": pl.Float64,
    "age_min": pl.Float64,
    "age_max": pl.Float64,
    "age_type": pl.String,  # exact | binned | censored
    "age_raw": pl.String,
    "sex": pl.String,  # female | male | null
    "tissue": pl.String,
    "cell_type": pl.String,
    "disease": pl.String,
    "is_healthy": pl.Boolean,
    "is_tumor": pl.Boolean,
    "is_cell_line": pl.Boolean,
    "platform": pl.String,
    "read_count_total": pl.Int64,
    "library_type": pl.String,
    "cohort": pl.String,  # named curated cohort this sample also belongs to (e.g. "fleischer2018"), else null
    "series_ids": pl.String,  # all GSEs of a GEO sample, comma-separated (study_id is the first)
    "srr_ids": pl.String,  # SRA runs, comma-separated, for cross-source deduplication
    "exclude_reason": pl.String,  # null when eligible for age training; else why not (cell_line, fetal, tumor, ...)
    # pseudobulk sources (census_pseudobulk, terekhova2023); null / False for bulk
    "is_pseudobulk": pl.Boolean,
    "assay": pl.String,  # e.g. "10x 3' v3"; for bulk the library strategy when known
    "suspension_type": pl.String,  # cell | nucleus
    "n_cells": pl.Int64,
    "n_umis": pl.Int64,
}

AGE_TYPES = ("exact", "binned", "censored")


def conform(df: pl.DataFrame) -> pl.DataFrame:
    """Add missing schema columns as nulls, cast, and order; extra columns are dropped."""
    exprs = []
    for name, dtype in SAMPLE_SCHEMA.items():
        if name in df.columns:
            exprs.append(pl.col(name).cast(dtype, strict=False))
        else:
            exprs.append(pl.lit(None, dtype=dtype).alias(name))
    out = df.select(exprs).with_columns(pl.col("is_pseudobulk").fill_null(False))
    bad = out.filter(pl.col("age_type").is_not_null() & ~pl.col("age_type").is_in(AGE_TYPES))
    if bad.height:
        raise ValueError(f"unknown age_type values: {bad['age_type'].unique().to_list()}")
    return out


def age_mid(df: pl.DataFrame | pl.LazyFrame) -> pl.Expr:
    """Best point estimate of age: exact age, else bin midpoint, else the censoring bound."""
    return (
        pl.when(pl.col("age_type") == "exact")
        .then(pl.col("age_years"))
        .when(pl.col("age_type") == "binned")
        .then((pl.col("age_min") + pl.col("age_max") + 1) / 2)
        .otherwise(pl.col("age_min"))
        .alias("age_mid")
    )
