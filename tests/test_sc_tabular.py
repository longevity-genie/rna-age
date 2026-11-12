"""RM14: per-fold features never see the held-out donors; the comparison against ridge."""

import numpy as np
import polars as pl

from rna_age.train.sc_tabular import compare, cross_validate, fold_features


def test_fold_features_use_training_rows_only():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 6))
    X[15:, 0] *= 100  # huge variance only in the held-out rows
    X[:15, 3] *= 10  # the most variable column within the training rows
    tr, te = np.arange(15), np.arange(15, 20)
    Xtr, _ = fold_features(X, tr, te, 1)
    expect = (X[tr, 3] - X[tr, 3].mean()) / (X[tr, 3].std() + 1e-6)
    assert np.allclose(Xtr[:, 0], expect)  # column 3 chosen and scaled on the training rows, not column 0


def test_cross_validation_recovers_a_linear_age_signal():
    rng = np.random.default_rng(1)
    n = 120
    age = rng.uniform(20, 80, n)
    X = rng.normal(size=(n, 40))
    X[:, 0] += age / 3
    pred = cross_validate(X, age, np.arange(n), ("ridge",), n_genes=40)["ridge"]
    assert np.corrcoef(pred, age)[0, 1] > 0.8 and not np.isnan(pred).any()


def test_compare_reports_the_difference_to_the_reference():
    rows = []
    for d in range(50):
        rows.append({"representation": "natural", "model": "ridge", "donor": f"d{d}", "age": 50.0, "pred": 55.0})
        rows.append({"representation": "natural", "model": "tabicl", "donor": f"d{d}", "age": 50.0, "pred": 52.0})
    c = compare(pl.DataFrame(rows), n_boot=200)
    t = c.filter(pl.col("model") == "tabicl").row(0, named=True)
    assert t["vs_ref"] == -3.0 and t["hi"] < 0


def test_new_models_add_to_an_earlier_run(tmp_path):
    from rna_age.train.sc_tabular import merge

    path = tmp_path / "x.parquet"
    old = pl.DataFrame({"representation": ["natural"] * 2, "model": ["ridge", "tabicl"], "donor": ["a", "a"],
                        "age": [50.0, 50.0], "pred": [51.0, 52.0]})  # fmt: skip
    old.write_parquet(path)
    new = pl.DataFrame({"representation": ["natural"] * 2, "model": ["gbm", "ridge"], "donor": ["a", "a"],
                        "age": [50.0, 50.0], "pred": [49.0, 50.5]})  # fmt: skip
    m = merge(new, path).sort("model")
    assert m["model"].to_list() == ["gbm", "ridge", "tabicl"]
    assert m.filter(pl.col("model") == "ridge")["pred"].item() == 50.5  # the new run replaces its own models


def test_enet_and_gbm_recover_a_signal():
    rng = np.random.default_rng(2)
    n = 150
    age = rng.uniform(20, 80, n)
    X = rng.normal(size=(n, 30))
    X[:, 0] += age / 3
    pred = cross_validate(X, age, np.arange(n), ("enet", "gbm"), n_genes=30)
    assert np.corrcoef(pred["enet"], age)[0, 1] > 0.8 and np.corrcoef(pred["gbm"], age)[0, 1] > 0.7
