"""Whole-series single-cell exclusion (RM12, D4): plate single-cell series score below 0.5 sample by sample."""

import polars as pl

from rna_age.sources.archs4 import single_cell_series


def test_series_median_decides_and_counts_all_samples():
    meta = pl.DataFrame(
        {
            "series_id": ["GSE1"] * 5 + ["GSE2,GSE9"] * 4 + ["GSE3"] * 3,
            "singlecellprobability": ["0.45", "0.44", "0.46", "0.9", "0.3"]  # plate sc: median 0.45
            + ["0.01", "0.02", "0.6", "0.03"]  # bulk with one odd sample
            + ["0.41", "0.39", "0.40"],  # at the threshold: median 0.40, not above
        }
    )
    assert single_cell_series(meta) == {"GSE1"}
    assert single_cell_series(meta, 0.3) == {"GSE1", "GSE3"}


def test_no_probability_column_excludes_nothing():
    assert single_cell_series(pl.DataFrame({"series_id": ["GSE1"]})) == set()
