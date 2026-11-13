"""``rna-age merge``: one harmonised sample table and study-grouped splits in ``data/interim/``.

- ``samples_all.parquet``: every ``data/input/*/samples.parquet`` row plus
  ``eligible`` (has age, no exclusion, not tumour), ``duplicate_of`` (``source:sample_id`` of the copy that is
  kept when the same GEO sample / SRA experiment appears in several bulk sources; ARCHS4 > NCBI GEO > Bgee),
  ``cohort`` filled from :mod:`rna_age.sources.geo_cohorts`, and ``split``.
- ``splits.parquet``: ``source, sample_id, group, split`` for eligible, non-duplicate samples.

Splits: the grouping unit is ``study_id`` (GSE / Bgee experiment / Census dataset), except GTEx where it is
the donor. A group goes wholly to train / val / test by a stable hash of ``source:group`` (70/15/15), so adding
sources never reshuffles existing assignments, and no donor or series is on both sides. Pseudobulk sources get
the same treatment but are reported separately (domain shift).

Sources added after the test set was fixed (:data:`NEW_SOURCES`, recount3 since RM26) hash by the same rule, but a
group an older source already has (a GEO series recount3 maps through ARCHS4) takes that source's split, and their
samples that land in test get split ``test_new``: scored as unseen new studies, never mixed into ``test``, so every
``split == "test"`` filter (evaluation, figures, clocks) still reads exactly the old test samples.
"""

from __future__ import annotations

import hashlib
import logging

import polars as pl

from rna_age.paths import INPUT, INTERIM
from rna_age.schema import age_mid
from rna_age.summary import eligible, load_samples

log = logging.getLogger(__name__)

# Human training data of the tAge clocks (Tyshkovskiy et al., Nature 2026, Supp Table 1c): GEO series (plus MESA,
# dbGaP, not public). Samples from these series flatter tAge in an evaluation; flag them.
TAGE_TRAINING_SERIES = (
    "GSE36192", "GSE190125", "GSE113957", "GSE134080", "GSE75337", "GSE123658", "GSE226189", "GSE5086",
    "GSE183701", "GSE221178", "GSE53890", "GSE103232", "GSE40645", "GSE17612", "GSE21935", "GSE674",
)  # fmt: skip

PRIORITY = {"archs4": 0, "ncbi_geo": 1, "bgee": 2, "recount3": 3}
SPLIT_FRACTIONS = (0.70, 0.15)  # train, val; rest test
# sources added after the test set was fixed; their test samples are split TEST_NEW (see the module docstring)
NEW_SOURCES = ("recount3",)
TEST_NEW = "test_new"


def _split_of(key: str) -> str:
    h = int(hashlib.sha1(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    if h < SPLIT_FRACTIONS[0]:
        return "train"
    if h < SPLIT_FRACTIONS[0] + SPLIT_FRACTIONS[1]:
        return "val"
    return "test"


def assign_splits(df: pl.DataFrame, use: pl.Expr) -> pl.DataFrame:
    """``split`` per row (null where ``use`` is false), by ``source, group`` (module docstring)."""
    groups = df.filter(use).select("source", "group").unique()
    groups = groups.with_columns(
        split=pl.Series([_split_of(f"{s}:{g}") for s, g in groups.iter_rows()], dtype=pl.String)
    )
    is_new = pl.col("source").is_in(NEW_SOURCES)
    known = (
        groups.filter(~is_new)
        .with_columns(_prio=pl.col("source").replace_strict(PRIORITY, default=9, return_dtype=pl.Int32))
        .sort("_prio", "source")
        .unique("group", keep="first", maintain_order=True)
        .select("group", _known="split")
    )
    groups = (
        groups.join(known, on="group", how="left")
        .with_columns(split=pl.when(is_new).then(pl.coalesce("_known", "split")).otherwise("split"))
        .with_columns(split=pl.when(is_new & (pl.col("split") == "test")).then(pl.lit(TEST_NEW)).otherwise("split"))
        .drop("_known")
    )
    df = df.drop("split", strict=False).join(groups, on=["source", "group"], how="left")
    return df.with_columns(split=pl.when(use).then(pl.col("split")))


def _duplicates(df: pl.DataFrame) -> pl.DataFrame:
    """``duplicate_of`` for bulk samples present in several sources (by GSM, then by SRX/SRR overlap)."""
    df = df.with_columns(_prio=pl.col("source").replace_strict(PRIORITY, default=9, return_dtype=pl.Int32))
    # same GSM in several sources
    by_gsm = (
        df.filter(pl.col("sample_id").str.starts_with("GSM"))
        .sort("_prio")
        .group_by("sample_id", maintain_order=True)
        .agg(keep=pl.concat_str(pl.col("source").first(), pl.lit(":"), pl.col("sample_id").first()))
    )
    df = df.join(by_gsm, on="sample_id", how="left").with_columns(
        duplicate_of=pl.when(pl.col("keep") != pl.concat_str("source", pl.lit(":"), "sample_id")).then("keep")
    )
    # Bgee libraries are SRX/ERX ids: match them to GEO samples' SRA relations
    rel = (
        df.filter(pl.col("srr_ids").is_not_null() & pl.col("sample_id").str.starts_with("GSM"))
        .sort("_prio")
        .select(
            pl.col("srr_ids").str.split(",").alias("acc"),
            pl.concat_str("source", pl.lit(":"), "sample_id").alias("geo_key"),
        )
        .explode("acc")
        .unique("acc", keep="first")
    )
    df = df.join(rel, left_on="sample_id", right_on="acc", how="left").with_columns(
        duplicate_of=pl.coalesce("duplicate_of", pl.when(pl.col("source") == "bgee").then("geo_key"))
    )
    return df.drop("_prio", "keep", "geo_key")


def _cohorts(df: pl.DataFrame) -> pl.DataFrame:
    d = INPUT / "geo_cohorts"
    if not (d / "cohort_series.parquet").exists():
        return df
    series = pl.read_parquet(d / "cohort_series.parquet")
    shok = (
        pl.read_parquet(d / "shokhirev2021_samples.parquet")
        .select(sample_id=pl.col("gsm"), shokhirev_age=pl.col("age_years"), in_shokhirev2021=pl.lit(True))
        .unique("sample_id")
    )
    df = df.join(series.rename({"gse": "study_id", "cohort": "_cohort"}), on="study_id", how="left")
    df = df.join(shok, on="sample_id", how="left").with_columns(
        cohort=pl.coalesce(
            "cohort",
            "_cohort",
            pl.when(pl.col("in_shokhirev2021")).then(pl.lit("shokhirev2021")),
        ),
        in_shokhirev2021=pl.col("in_shokhirev2021").fill_null(False),
    )
    return df.drop("_cohort")


def merge(suffix: str = "") -> pl.DataFrame:
    """Write ``samples_all{suffix}.parquet`` and ``splits{suffix}.parquet`` (a suffix leaves the shared files alone)."""
    INTERIM.mkdir(parents=True, exist_ok=True)
    df = load_samples()
    df = df.with_columns(eligible=eligible(df), age_mid=age_mid(df))
    df = _duplicates(df)
    df = _cohorts(df)
    df = df.with_columns(
        in_tage_training=pl.col("series_ids")
        .fill_null(pl.col("study_id"))
        .str.split(",")
        .list.eval(pl.element().is_in(TAGE_TRAINING_SERIES))
        .list.any()
        .fill_null(False)
    )
    lb = INPUT / "longevitybench" / "gtex_tasks.parquet"
    lb_ids = pl.read_parquet(lb)["sample_id"].unique().to_list() if lb.exists() else []
    df = df.with_columns(in_longevitybench=(pl.col("source") == "gtex") & pl.col("sample_id").is_in(lb_ids))
    df = df.with_columns(
        group=pl.when(pl.col("source") == "gtex").then(pl.col("donor_id")).otherwise(pl.col("study_id"))
    )
    use = pl.col("eligible") & pl.col("duplicate_of").is_null()
    df = assign_splits(df, use)
    df.write_parquet(INTERIM / f"samples_all{suffix}.parquet", compression="zstd")
    splits = df.filter(use).select("source", "sample_id", "group", "split", "age_type", "age_mid", "is_pseudobulk")
    splits.write_parquet(INTERIM / f"splits{suffix}.parquet", compression="zstd")
    shok = (
        df.filter(pl.col("in_shokhirev2021") & pl.col("age_years").is_not_null())
        if "in_shokhirev2021" in df.columns
        else None
    )
    if shok is not None and shok.height:
        agree = (shok["age_years"] - shok["shokhirev_age"]).abs() <= 1
        log.info("age audit vs Shokhirev 2021 curation: %d/%d samples within 1 year", agree.sum(), shok.height)
    log.info(
        "merged: %d samples, %d eligible non-duplicate; splits %s",
        df.height, splits.height, splits.group_by("split").len().sort("split").to_dicts(),
    )  # fmt: skip
    return df
