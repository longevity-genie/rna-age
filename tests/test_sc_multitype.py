import numpy as np
import polars as pl

from rna_age.train.sc_multitype import compare, concat_features, oof, pick_classes


def test_pick_classes_skips_other_and_keeps_coverage():
    N = np.array([[30, 30, 0, 50], [30, 30, 30, 50], [30, 0, 30, 50], [30, 30, 30, 50]] * 20, dtype=float)
    classes = ["A", "B", "C", "other"]
    cls = pick_classes(N, classes, k=3, min_cells=20, min_donors=40)
    assert 3 not in cls  # "other" never chosen
    assert cls[0] == 0  # best covered first
    assert (N[:, cls] >= 20).all(1).sum() >= 40


def test_concat_features_fold_only_statistics():
    rng = np.random.default_rng(0)
    blocks = [rng.normal(size=(20, 8)), rng.normal(size=(20, 8)) * 3]
    comp = rng.normal(size=(20, 3))
    tr, te = np.arange(15), np.arange(15, 20)
    a, b = concat_features(blocks, comp, tr, te, n_per=4)
    assert a.shape == (15, 4 + 4 + 3) and b.shape == (5, 11)
    assert np.allclose(a.mean(0), 0, atol=1e-6)  # standardised on the training fold


def test_oof_and_compare_shapes():
    rng = np.random.default_rng(1)
    n, k = 60, 2
    y = rng.uniform(20, 80, n)
    blocks = [np.c_[y[:, None] + rng.normal(size=(n, 1)), rng.normal(size=(n, 5))] for _ in range(k)]
    X = np.hstack(blocks)
    comp = rng.normal(size=(n, 3))
    groups = np.arange(n)
    pred = oof(X, blocks, comp, y, groups, "ridge", n_genes=6)
    assert set(pred) == {"whole", "concat", "type:0", "type:1", "mean-types", "mean-all"}
    assert all(np.isfinite(p).all() for p in pred.values())
    rows = [{"dataset_id": "d", "model": "ridge", "arm": arm, "donor": f"x{i}", "group": i, "age": float(y[i]),
             "pred": float(p[i])} for arm, p in pred.items() for i in range(n)]  # fmt: skip
    out = compare(pl.DataFrame(rows), n_boot=50)
    assert out.filter(pl.col("arm") == "whole")["diff"][0] == 0.0
    assert out.height == 6


def test_run_cohort_skips_without_two_classes(monkeypatch):
    from rna_age.train import sc_multitype as sm

    def fake(dataset_id, mapping="broad", n_classes=12):
        return {"P": np.zeros((60, 1, 4)), "N": np.full((60, 1), 30.0), "age": np.linspace(20, 80, 60),
                "groups": np.arange(60), "donors": [f"d{i}" for i in range(60)], "classes": ["A"], "cix": {"A": 0}}  # fmt: skip

    monkeypatch.setattr(sm, "donor_profiles", fake)
    assert sm.run_cohort("x", "ridge").height == 0
