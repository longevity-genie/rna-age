"""Samples outside the age table, scored by trained models (RM10's before/after tests).

``extract(series)`` writes every sample of the given GEO series from the local ARCHS4 file to
``data/input/external/`` (``samples.parquet``, ``genes.parquet``, ``counts/``), in the counts layout of every
source, whatever their age annotation: a before/after comparison within a person needs no age. These samples
never enter the age table, a split or any training.

A trainer that is given ``external=True`` predicts them with its own genes, normalisation and scaling and writes
``predictions_external.parquet`` (``sample_id``, ``pred``) beside its run's predictions.
"""

from __future__ import annotations

import logging

import numpy as np
import polars as pl

from rna_age.paths import INPUT

log = logging.getLogger(__name__)

EXTERNAL = INPUT / "external"


def extract(series: list[str]) -> pl.DataFrame:
    """All ARCHS4 samples of ``series`` (first-listed or any listed series id) to ``data/input/external/``."""
    from rna_age.sources import archs4

    meta = pl.read_parquet(INPUT / "archs4" / "metadata_all.parquet")
    ids = pl.col("series_id").str.replace_all(r"\s+", "").str.split(",")
    sel = meta.filter(ids.list.eval(pl.element().is_in(series)).list.any())
    samples = sel.select(
        sample_id=pl.col("geo_accession"),
        study_id=ids.list.eval(pl.element().filter(pl.element().is_in(series))).list.first(),
        title=pl.col("title"),
        source_name=pl.col("source_name_ch1"),
        characteristics=pl.col("characteristics_ch1"),
    )
    missing = sorted(set(series) - set(samples["study_id"].unique().to_list()))
    if missing:
        log.warning("external: no ARCHS4 samples for %s", ", ".join(missing))
    EXTERNAL.mkdir(parents=True, exist_ok=True)
    archs4.extract_counts(samples.select("sample_id"), EXTERNAL)
    samples.write_parquet(EXTERNAL / "samples.parquet", compression="zstd")
    log.info("external: %d samples of %d series", samples.height, samples["study_id"].n_unique())
    return samples


def available() -> bool:
    return (EXTERNAL / "samples.parquet").exists() and (EXTERNAL / "counts" / "index.json").exists()


def log_cpm_matrix(genes: list[str]) -> tuple[list[str], np.ndarray]:
    """External sample ids and their log-CPM on ``genes`` (library size over ``genes``, as in training)."""
    from rna_age.train.features import load_counts, log_cpm

    ids = pl.read_parquet(EXTERNAL / "samples.parquet", columns=["sample_id"])["sample_id"].to_list()
    return ids, log_cpm(load_counts("external", ids, genes))
