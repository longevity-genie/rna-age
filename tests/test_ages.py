import pytest

from rna_age.ages import find_age, is_age_key, parse_age_field, split_characteristics


@pytest.mark.parametrize(
    "key,value,age_type,years,lo,hi",
    [
        ("age", "54", "exact", 54, 54, 54),
        ("age (years)", "54", "exact", 54, 54, 54),
        ("Age", "54 years", "exact", 54, 54, 54),
        ("age", "54 yrs", "exact", 54, 54, 54),
        ("age", "54y", "exact", 54, 54, 54),
        ("age", "54Y", "exact", 54, 54, 54),
        ("age", "54 years old", "exact", 54, 54, 54),
        ("age", "54.5", "exact", 54.5, 54.5, 54.5),
        ("donor age", "71", "exact", 71, 71, 71),
        ("patient_age", "33", "exact", 33, 33, 33),
        ("age_years", "40", "exact", 40, 40, 40),
        ("age", "6 months", "exact", 0.5, 0.5, 0.5),
        ("age (months)", "18", "exact", 1.5, 1.5, 1.5),
        ("age", "2 weeks", "exact", 2 / (365.25 / 7), None, None),
        ("age", "10 days", "exact", 10 / 365.25, None, None),
        ("age (days)", "365.25", "exact", 1.0, 1.0, 1.0),
        ("age", "5 years 6 months", "exact", 5.5, 5.5, 5.5),
        ("age", "50-59", "binned", None, 50, 59),
        ("age", "50 - 59 years", "binned", None, 50, 59),
        ("age", "50 to 59", "binned", None, 50, 59),
        ("age group", "60-69", "binned", None, 60, 69),
        ("age", "30s", "binned", None, 30, 39),
        ("age", ">89", "censored", None, 90, None),
        ("age", "90+", "censored", None, 90, None),
        ("age", ">=90", "censored", None, 90, None),
        ("age", "≥90", "censored", None, 90, None),
        ("age", "90 or older", "censored", None, 90, None),
        ("age", "<1", "binned", None, 0, 0),
        ("age at diagnosis", "62", "exact", 62, 62, 62),
        ("age_at_death", "81", "exact", 81, 81, 81),
        ("age at time of collection", "45", "exact", 45, 45, 45),
        ("Age (yrs)", "aged 37", "exact", 37, 37, 37),
        ("age", "54 years of age", "exact", 54, 54, 54),
    ],
)
def test_parses(key, value, age_type, years, lo, hi):
    a = parse_age_field(key, value)
    assert a is not None, (key, value)
    assert a.age_type == age_type
    if years is not None:
        assert a.age_years == pytest.approx(years)
    if lo is not None:
        assert a.age_min == pytest.approx(lo)
    if hi is not None:
        assert a.age_max == pytest.approx(hi)
    assert a.raw == f"{key}: {value}"


@pytest.mark.parametrize(
    "key,value",
    [
        ("gestational age", "20 weeks"),
        ("gestational age (weeks)", "38"),
        ("passage", "5"),
        ("age", "NA"),
        ("age", "n/a"),
        ("age", "unknown"),
        ("age", "adult"),
        ("age", "young"),
        ("age", "Not available"),
        ("age", "20 weeks gestation"),
        ("age", "E14.5"),
        ("age", "12 pcw"),
        ("age", "fetal"),
        ("age", "3 dpi"),
        ("age", "24 hours"),
        ("age", "P5"),
        ("days post infection", "3"),
        ("time point", "day 7"),
        ("age at onset", "30"),
        ("developmental stage", "54-year-old human stage"),
        ("cell passage", "12"),
        ("dosage", "10"),
        ("stage", "II"),
        ("age", "150"),
        ("age", "60-50"),
        ("mother age", "31"),
        ("predicted age", "40"),
        ("tissue", "liver"),
        ("age", ""),
    ],
)
def test_rejects(key, value):
    assert parse_age_field(key, value) is None


@pytest.mark.parametrize("key", ["age", "Age", "AGE", "age (years)", "donor_age", "Age_at_diagnosis", "age(yrs)"])
def test_age_keys(key):
    assert is_age_key(key)


@pytest.mark.parametrize("key", ["stage", "passage", "gestational age", "dosage", "lineage", "age of onset"])
def test_non_age_keys(key):
    assert not is_age_key(key)


def test_find_age_prefers_exact_and_sampling():
    fields = [("age group", "50-59"), ("age at diagnosis", "51"), ("age at sampling", "53")]
    a = find_age(fields)
    assert a.age_type == "exact" and a.age_years == 53 and a.key_kind == "sampling"


def test_find_age_none():
    assert find_age([("tissue", "liver"), ("passage", "4")]) is None


def test_split_characteristics():
    assert split_characteristics("tissue: liver\tage: 54\tmale") == [("tissue", "liver"), ("age", "54"), ("", "male")]
    assert split_characteristics("a: 1,b: 2", sep=",") == [("a", "1"), ("b", "2")]


def test_archs4_split_items():
    from rna_age.sources.archs4 import split_items

    assert split_items("tissue: liver,age: 54,treatment: IL-4, IL-13") == [
        ("tissue", "liver"),
        ("age", "54"),
        ("treatment", "IL-4, IL-13"),
    ]
    assert split_items("cell type: T cells\tage (years): 33") == [("cell type", "T cells"), ("age (years)", "33")]


def test_archs4_runs():
    import numpy as np

    from rna_age.sources.archs4 import _runs

    sel = np.array([1, 2, 3, 200, 201, 5000])
    assert _runs(sel, max_gap=64, max_width=2000) == [(0, 3), (3, 5), (5, 6)]
    assert _runs(np.arange(0, 5000, 10), max_gap=64, max_width=2000)[0] == (0, 200)
