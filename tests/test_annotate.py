import polars as pl

from rna_age.annotate import add_flags


def _frame(rows):
    return pl.DataFrame(
        {
            "fields": [[{"key": k, "value": v} for k, v in r[0]] for r in rows],
            "text": [r[1].lower() for r in rows],
        },
        schema={"fields": pl.List(pl.Struct({"key": pl.String, "value": pl.String})), "text": pl.String},
    )


def test_flags():
    df = add_flags(
        _frame(
            [
                ([("tissue", "liver"), ("Sex", "F"), ("disease state", "healthy")], "liver | tissue: liver"),
                ([("cell line", "HeLa")], "hela cells"),
                ([("tissue", "tumor"), ("disease", "breast cancer")], "breast tumor"),
                ([("tissue", "adjacent normal"), ("disease", "breast cancer")], "adjacent normal breast, cancer"),
                ([("cell type", "iPSC-derived neurons")], "ipsc-derived neurons"),
                ([("tissue", "placenta")], "placenta"),
                ([("cell line", "none"), ("gender", "male"), ("diagnosis", "COPD")], "lung copd"),
            ]
        )
    )
    assert df["sex"].to_list() == ["female", None, None, None, None, None, "male"]
    assert df["exclude_reason"].to_list() == [
        None,
        "cell_line",
        "tumor",
        "cancer_patient",
        "stem_or_organoid",
        "prenatal",
        None,
    ]
    assert df["is_healthy"].to_list() == [True, None, False, False, None, None, False]
    assert df["is_tumor"].to_list() == [False, False, True, False, False, False, False]


def test_health_labels():
    df = add_flags(
        _frame(
            [
                ([("disease state", "Ctrl")], "x"),
                ([("group", "HC")], "x"),
                ([("group", "discovery")], "x"),
                ([("diagnosis", "non-demented")], "x"),
                ([("disease", "MDD")], "x"),
            ]
        )
    )
    assert df["is_healthy"].to_list() == [True, True, None, True, False]
    assert df["disease"].to_list()[2] is None
