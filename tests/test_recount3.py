"""recount3 source (RM26): coverage → counts, runs → experiments, ages, study mapping, and the split guard."""

from __future__ import annotations

import gzip

import numpy as np
import polars as pl
import pytest

from rna_age.merge import NEW_SOURCES, TEST_NEW, _split_of, assign_splits
from rna_age.paths import INTERIM
from rna_age.sources import recount3 as r3


def test_coverage_to_counts_divides_by_mapped_length_and_rounds():
    # two runs: single-end 100 bp, paired-end 2 × 75 bp (STAR's average mapped length counts both mates)
    cov = np.array([[1000.0, 1500.0], [149.0, 76.0], [0.0, 0.0]])
    out = r3.coverage_to_counts(cov, np.array([100.0, 150.0]))
    assert out.tolist() == [[10.0, 10.0], [1.0, 1.0], [0.0, 0.0]]


def test_coverage_to_counts_rejects_missing_lengths():
    with pytest.raises(ValueError):
        r3.coverage_to_counts(np.ones((2, 2)), np.array([100.0, 0.0]))
    with pytest.raises(ValueError):
        r3.coverage_to_counts(np.ones((2, 2)), np.array([100.0]))


def test_sum_runs_sums_an_experiments_runs():
    counts = np.array([[1.0, 2.0, 4.0], [10.0, 20.0, 40.0]])
    exps, out = r3.sum_runs(counts, ["SRR1", "SRR2", "SRR3"], {"SRR1": "SRX9", "SRR2": "SRX8", "SRR3": "SRX9"})
    assert exps == ["SRX9", "SRX8"]
    assert out.tolist() == [[5.0, 2.0], [50.0, 20.0]]


def test_age_columns_exact_and_decade():
    df = pl.DataFrame(
        {"age_class": ["exact", "decade", None], "age_lo": [45.0, 50.0, None], "age_hi": [45.0, 59.0, None]}
    )
    out = r3.age_columns(df)
    assert out["age_type"].to_list() == ["exact", "binned", None]
    assert out["age_years"].to_list() == [45.0, None, None]
    assert out["age_min"].to_list() == [45.0, 50.0, None]
    assert out["age_max"].to_list() == [45.0, 59.0, None]


def test_map_studies():
    runs = pl.DataFrame(
        {
            "study": ["SRP1", "SRP1", "SRP2", "SRP3", "SRP3"],
            "experiment_acc": ["SRX1", "SRX2", "SRX3", "SRX4", "SRX5"],
        }
    )
    archs4 = pl.DataFrame({"srx": ["SRX1", "SRX4", "SRX5", None], "gse": ["GSE10", "GSE20", "GSE21", "GSE99"]})
    out = {r["study"]: r for r in r3.map_studies(runs, archs4).iter_rows(named=True)}
    assert (out["SRP1"]["mapped"], out["SRP1"]["study_id"]) == ("archs4", "GSE10")  # partly in ARCHS4 → its GSE
    assert (out["SRP2"]["mapped"], out["SRP2"]["study_id"]) == ("none", "SRP2")  # entirely new → the project
    assert out["SRP3"]["mapped"] == "ambiguous" and out["SRP3"]["study_id"] is None  # two series → dropped


def test_gene_rows_drop_par_y_and_versions():
    rows, ids = r3.gene_rows(["ENSG1.5", "ENSG2.1", "ENSG2.1_PAR_Y", "ENSG3.12"])
    assert rows.tolist() == [0, 1, 3]
    assert ids == ["ENSG1", "ENSG2", "ENSG3"]


def test_read_gene_sums(tmp_path):
    p = tmp_path / "sra.gene_sums.SRP1.G026.gz"
    with gzip.open(p, "wt") as f:
        f.write("##annotation=G026\n##date.generated=x\ngene_id\tSRR1\tSRR2\tSRR3\n")
        f.write("ENSG1.1\t100\t0\t5\nENSG2.3\t300\t50\t7\n")
    genes, cov = r3.read_gene_sums(p, ["SRR3", "SRR1"])
    assert genes == ["ENSG1.1", "ENSG2.3"]
    assert cov.tolist() == [[5.0, 100.0], [7.0, 300.0]]


def test_sra_fields_parse_attributes():
    df = r3.sra_fields(pl.DataFrame({"sample_attributes": ["age;;33|gender;;male|empty;;", None]}))
    assert df["fields"][0].to_list() == [{"key": "age", "value": "33"}, {"key": "gender", "value": "male"}]
    assert df["fields"][1].to_list() == []


# ------------------------------------------------------------------------------------------- split guard


def _table(seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for src, n in (("archs4", 400), ("gtex", 100), ("bgee", 60)):
        for g in range(n):
            group = f"GSE{g}" if src == "archs4" else f"{src}{g}"
            rows += [(src, f"{src}-{g}-{i}", group, bool(rng.random() > 0.1)) for i in range(3)]
    return pl.DataFrame(rows, schema=["source", "sample_id", "group", "use"], orient="row")


def test_adding_recount3_keeps_old_splits_and_the_main_test_set():
    old_df = _table()
    old = assign_splits(old_df, pl.col("use"))
    # recount3: 300 new projects plus 30 projects mapped to ARCHS4 series (some in test there)
    new_rows = [("recount3", f"SRX{g}-{i}", f"SRP{g}", True) for g in range(300) for i in range(2)]
    new_rows += [("recount3", f"SRXa{g}", f"GSE{g}", True) for g in range(30)]
    added = pl.concat([old_df, pl.DataFrame(new_rows, schema=old_df.schema, orient="row")])
    new = assign_splits(added, pl.col("use"))

    # every old row keeps its split
    joined = old.join(new, on=["source", "sample_id"], suffix="_new")
    assert joined.height == old.height
    assert joined.filter(pl.col("split").ne_missing(pl.col("split_new"))).height == 0
    # the main test set is exactly the old one
    old_test = set(old.filter(pl.col("split") == "test")["sample_id"])
    assert set(new.filter(pl.col("split") == "test")["sample_id"]) == old_test
    r = new.filter(pl.col("source").is_in(NEW_SOURCES))
    assert set(r["split"].unique()) <= {"train", "val", TEST_NEW}
    assert r.filter(pl.col("split") == TEST_NEW).height > 0  # unseen new studies exist for the separate read-out
    # a mapped group sits on the same side as its ARCHS4 series (test → test_new)
    a4 = dict(old.filter((pl.col("source") == "archs4") & pl.col("split").is_not_null())
              .select("group", "split").unique().iter_rows())  # fmt: skip
    for g, s in r.filter(pl.col("group").str.starts_with("GSE")).select("group", "split").unique().iter_rows():
        want = a4.get(g, _split_of(f"recount3:{g}"))  # an ARCHS4 series with no used sample: the normal hash
        assert s == (TEST_NEW if want == "test" else want)
    # an unmapped project hashes by the normal rule
    for g, s in r.filter(pl.col("group").str.starts_with("SRP")).select("group", "split").unique().iter_rows():
        h = _split_of(f"recount3:{g}")
        assert s == (TEST_NEW if h == "test" else h)


def test_real_merge_keeps_old_assignments():
    """On the shared data: the recount3 merge (``rna-age merge --suffix _recount3``) against the current one."""
    old_p, new_p = INTERIM / "samples_all.parquet", INTERIM / "samples_all_recount3.parquet"
    if not (old_p.exists() and new_p.exists()):
        pytest.skip("no merged tables to compare")
    cols = ["source", "sample_id", "split"]
    old = pl.read_parquet(old_p, columns=cols)
    new = pl.read_parquet(new_p, columns=cols)
    j = old.join(new.filter(~pl.col("source").is_in(NEW_SOURCES)), on=["source", "sample_id"], suffix="_new")
    assert j.height == old.height
    assert j.filter(pl.col("split").ne_missing(pl.col("split_new"))).height == 0
    old_test = set(old.filter(pl.col("split") == "test")["sample_id"])
    assert set(new.filter(pl.col("split") == "test")["sample_id"]) == old_test
    assert new.filter(pl.col("source").is_in(NEW_SOURCES) & (pl.col("split") == "test")).height == 0
