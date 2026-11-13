"""RM26 plumbing: the table switch (default unchanged), the gene-axis intersection, the training exclusion, recount3
loadable as a bulk source, and the comparison's study-bootstrap CI."""

from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest

from rna_age.train import features as F
from rna_age.train.rm26 import compare_frames, mean_of


@pytest.fixture
def data(tmp_path, monkeypatch):
    interim, inp = tmp_path / "interim", tmp_path / "input"
    interim.mkdir()
    for name, split in (("samples_all.parquet", "old"), ("samples_all_recount3.parquet", "new")):
        pl.DataFrame({"sample_id": ["a", "b"], "split": [split, None]}).write_parquet(interim / name)
    for src, genes in {"archs4": ["G1", "G2", "G3", "G4"], "gtex": ["G1", "G2", "G3", "G5"], "bgee": ["G1", "G2", "G3"],
                       "recount3": ["G2", "G3", "G9"]}.items():  # fmt: skip
        (inp / src / "counts").mkdir(parents=True)
        pl.DataFrame({"gene_id": genes}).write_parquet(inp / src / "genes.parquet")
    # recount3 counts: 3 genes × 2 samples in one part, as the parser writes them
    pl.DataFrame({"gene_id": ["G2", "G3", "G9"], "SRX1": [1, 2, 3], "SRX2": [4, 5, 6]}, schema_overrides={
        "SRX1": pl.UInt32, "SRX2": pl.UInt32}).write_parquet(inp / "recount3" / "counts" / "part-00000.parquet")  # fmt: skip
    (inp / "recount3" / "counts" / "index.json").write_text(json.dumps({"SRX1": "part-00000.parquet",
                                                                         "SRX2": "part-00000.parquet"}))  # fmt: skip
    monkeypatch.setattr(F, "INTERIM", interim)
    monkeypatch.setattr(F, "INPUT", inp)
    monkeypatch.setattr(F, "_SAMPLES_SUFFIX", "")
    return tmp_path


def test_sample_table_defaults_to_the_shared_table(data):
    assert F.samples_path().name == "samples_all.parquet"
    assert F.sample_table()["split"].to_list() == ["old"]  # null splits dropped, as before
    assert F.sample_table("_recount3")["split"].to_list() == ["new"]
    F.set_samples_suffix("_recount3")
    assert F.sample_table()["split"].to_list() == ["new"]
    assert F.sample_table("")["split"].to_list() == ["old"]  # an explicit "" still reads the shared table


def test_gene_axis_default_and_intersection(data):
    assert F.shared_genes() == ["G1", "G2", "G3"]
    assert F.shared_genes(with_sources=("recount3",)) == ["G2", "G3"]
    assert F.gene_axis_label(("recount3",)) == "shared(archs4,gtex,bgee) ∩ recount3"


def test_recount3_loads_through_load_matrix(data):
    s = pl.DataFrame({"sample_id": ["SRX2", "SRX1"], "source": ["recount3", "recount3"]})
    x = F.load_matrix(s, ["G3", "G1", "G2"])
    assert x.tolist() == [[5, 0, 4], [2, 0, 1]]  # G1 absent from recount3 -> 0


def test_exclude_train_studies_only_touches_named_studies():
    train = np.array([True, True, False, True, True])
    study = pl.Series(["SRP223445", "S1", "SRP223445", None, "S2"])
    assert F.exclude_train_studies(train, study, ()) is train
    assert F.exclude_train_studies(train, study, ("SRP223445",)).tolist() == [False, True, False, True, True]


def _frame(ids, split, study, age, pred, source="archs4"):
    n = len(ids)
    return pl.DataFrame({"sample_id": ids, "source": [source] * n, "split": [split] * n, "study_id": study,
                         "age_type": ["exact"] * n, "age_years": age, "pred": pred})  # fmt: skip


def test_mean_of_seeds_is_per_sample_mean():
    a = _frame(["x", "y"], "test", ["S1", "S1"], [10.0, 20.0], [12.0, 18.0])
    b = _frame(["y", "x"], "test", ["S1", "S1"], [20.0, 10.0], [22.0, 14.0])
    m = mean_of([a, b]).sort("sample_id")
    assert m["pred"].to_list() == [13.0, 20.0]


def test_compare_ci_on_a_synthetic_example():
    rng = np.random.default_rng(1)
    ids = [f"s{i}" for i in range(200)]
    study = [f"S{i // 10}" for i in range(200)]  # 20 studies of 10
    age = rng.uniform(20, 80, 200)
    control = _frame(ids, "test", study, age, age + rng.normal(0, 8, 200))
    arm_pred = age + rng.normal(0, 4, 200)  # clearly better
    arm = pl.concat([
        _frame(ids, "test", study, age, arm_pred),
        _frame(["n1", "n2"], "test_new", ["R1", "R1"], [30.0, 50.0], [33.0, 49.0], source="recount3"),
    ])  # fmt: skip
    head = _frame(ids, "test", study, age, age.copy())  # a perfect headline
    out = compare_frames(control, arm, head, eligible_ids=pl.Series(ids[:-10]), n_boot=500, seed=0)
    row = out.filter((pl.col("comparison") == "arm - control") & (pl.col("source") == "all")).row(0, named=True)
    assert row["n"] == 190 and row["studies"] == 19  # the last study is not eligible
    assert row["diff"] == pytest.approx(row["mae_a"] - row["mae_b"])
    assert row["lo"] < row["diff"] < row["hi"] < 0 and row["beyond_ci"]
    head_row = out.filter(pl.col("comparison") == "arm - headline").row(0, named=True)
    assert head_row["mae_b"] == 0 and head_row["diff"] > 0 and not head_row["beyond_ci"]
    new = out.filter((pl.col("set") == "test_new") & (pl.col("source") == "all")).row(0, named=True)
    assert new["comparison"] == "arm" and new["n"] == 2 and new["mae_a"] == pytest.approx(2.0)
    # the same seed gives the same interval
    again = compare_frames(control, arm, head, eligible_ids=pl.Series(ids[:-10]), n_boot=500, seed=0)
    assert again.equals(out)
