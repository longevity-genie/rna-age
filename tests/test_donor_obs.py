import polars as pl

from rna_age.sources.census_pseudobulk import donor_columns


def test_donor_columns_keeps_per_donor_constants_only():
    obs = pl.DataFrame(
        {
            "donor_id": ["a", "a", "b", "b"],
            "batch": ["x", "x", "y", "y"],  # per donor, varies: kept
            "cell_type": ["T", "B", "T", "B"],  # varies within donor: dropped
            "assay": ["10x", "10x", "10x", "10x"],  # constant everywhere: dropped
        }
    )
    out = donor_columns(obs)
    assert out.columns == ["donor_id", "batch"]
    assert out.rows() == [("a", "x"), ("b", "y")]


def test_donor_mixed_lists_batches_within_a_donor():
    from rna_age.sources.census_pseudobulk import donor_mixed

    obs = pl.DataFrame({"donor_id": ["a", "a", "a", "b"], "cohort": ["1", "2", "2", "1"], "x": ["u", "u", "u", "v"]})
    out = donor_mixed(obs)
    assert out.filter(pl.col("donor_id") == "a").select("value", "n_cells").rows() == [("1", 1), ("2", 2)]
    assert set(out["column"]) == {"cohort"}
