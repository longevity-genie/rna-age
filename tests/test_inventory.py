import polars as pl
import pytest

from rna_age.sources.inventory import census_donors, classify, sra_flags, stage_years


@pytest.mark.parametrize(
    "stage,cls,lo",
    [
        ("54-year-old stage", "exact", 54.0),
        ("sixth decade stage", "decade", 50.0),
        ("60-79 year-old stage", "wide_bin", 60.0),
        ("15-19 year-old", "decade", 15.0),
        ("90 year-old and over stage", "censored", 90.0),
        ("80 year-old and over stage", "censored", 80.0),
        ("adult stage", None, None),
        ("prime adult stage", None, None),
        ("13th week post-fertilization stage", None, None),
        ("Carnegie stage 16", None, None),
        ("unknown", None, None),
    ],
)
def test_stage_years(stage, cls, lo):
    c, low, _ = stage_years(stage)
    assert c == cls
    assert low == lo


def test_classify():
    assert classify("binned", 50, 59, None) == "decade"
    assert classify("binned", 20, 39, None) == "wide_bin"
    assert classify(None, None, None, None) is None


def test_census_donors_flags():
    obs = pl.DataFrame(
        {
            "dataset_id": ["d1"] * 5,
            "donor_id": ["a", "a", "b", "c", "d"],
            "development_stage": ["54-year-old stage"] * 2 + ["adult stage", "30-year-old stage", "70-year-old stage"],
            "is_primary_data": [True, False, True, True, True],
            "disease": ["normal", "normal", "normal", "lung adenocarcinoma", "normal"],
            "tissue_general": ["blood"] * 5,
            "tissue_type": ["tissue"] * 4 + ["organoid"],
            "assay": ["10x 3' v3"] * 5,
            "suspension_type": ["cell"] * 5,
            "n_cells": [150, 500, 1000, 1000, 1000],
        }
    )
    ds = pl.DataFrame({"dataset_id": ["d1"], "collection_id": ["c1"]})
    d = census_donors(obs, ds).sort("donor_id")
    assert d["n_primary"].to_list() == [150, 1000, 1000, 1000]
    assert d["usable"].to_list() == [True, False, True, True]
    # a: < 200 primary cells; c: tumour; d: organoid
    assert d["ok"].to_list() == [False, False, False, False]


def test_sra_flags_age_and_health():
    df = pl.DataFrame(
        {
            "fields": [
                [{"key": "age", "value": "54 years"}, {"key": "disease state", "value": "healthy control"}],
                [{"key": "age", "value": "50-59"}, {"key": "cell line", "value": "K562"}],
                [{"key": "tissue", "value": "fetal liver"}],
            ]
        },
        schema={"fields": pl.List(pl.Struct({"key": pl.String, "value": pl.String}))},
    )
    out = sra_flags(df)
    assert out["age_class"].to_list() == ["exact", "decade", None]
    assert out["age_lo"].to_list()[:2] == [54.0, 50.0]
    assert out["healthy"].to_list() == [True, False, False]
    # no disease stated counts as not diseased (most SRA samples state nothing)
    assert out["not_diseased"].to_list() == [True, True, True]
    assert out["culture"].to_list() == [False, True, False]
    assert out["prenatal"].to_list() == [False, False, True]
