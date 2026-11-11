"""The statistics behind RM10's pre-registered tests."""

import numpy as np
import polars as pl

from rna_age.train import biology
from rna_age.train.biology import REGISTRY, GapTest, gap_residual, run_test, spearman


def test_registry_names_are_unique_and_directions_valid():
    names = [t.name for t in REGISTRY]
    assert len(names) == len(set(names))
    assert all(t.direction in (-1, 0, 1) for t in REGISTRY)
    assert any(t.direction == 0 for t in REGISTRY)  # at least one negative control


def test_gap_residual_removes_age():
    age = np.linspace(20, 80, 50)
    pred = 0.8 * age + 10
    assert np.allclose(gap_residual(pred, age), 0, atol=1e-9)


def test_spearman_is_rank_based():
    x = np.array([1, 2, 3, 4, 5.0])
    assert np.isclose(spearman(x, x**3), 1.0) and np.isclose(spearman(x, -x), -1.0)


def _fake(monkeypatch, x, effect):
    rng = np.random.default_rng(0)
    n = len(x)
    age = rng.uniform(20, 80, n)
    pred = age + effect * x + rng.normal(0, 1, n)
    ids = [f"S{i}" for i in range(n)]
    monkeypatch.setattr(biology, "covariates", lambda t: pl.DataFrame({"sample_id": ids, "x": x.astype(float)}))
    return pl.DataFrame({"sample_id": ids, "pred": pred, "age_years": age})


def test_a_planted_effect_passes_in_its_direction_and_fails_the_other(monkeypatch):
    x = np.repeat([0, 1, 2, 3], 15)
    p = _fake(monkeypatch, x, effect=3.0)
    up = run_test(GapTest("t", "GSE0", "k", {}, +1, ""), p, n_perm=300)
    down = run_test(GapTest("t", "GSE0", "k", {}, -1, ""), p, n_perm=300)
    assert up["verdict"] == "as expected" and up["p"] < 0.01
    assert down["verdict"] == "opposite"


def test_negative_control_passes_on_no_effect_and_fails_on_one(monkeypatch):
    x = np.repeat([0, 1, 2], 20)
    assert run_test(GapTest("c", "GSE0", "k", {}, 0, ""), _fake(monkeypatch, x, 0.0), 300)["verdict"].startswith("as")
    assert (
        run_test(GapTest("c", "GSE0", "k", {}, 0, ""), _fake(monkeypatch, x, 5.0), 300)["verdict"]
        == "unexpected change"
    )


def test_paired_stat_detects_a_planted_drop_within_subjects():
    from rna_age.train.biology import paired_stat

    rng = np.random.default_rng(0)
    rows = []
    for s in range(12):
        base = rng.normal(50, 10)
        rows += [
            {"subject": f"p{s}", "after": False, "pred": base},
            {"subject": f"p{s}", "after": True, "pred": base - 3},
        ]
    d = pl.DataFrame(rows)
    down = paired_stat(d, -1, 500)
    assert down["design"] == "paired" and abs(down["delta"] + 3) < 1e-9 and down["p"] < 0.01
    assert paired_stat(d, +1, 500)["p"] > 0.5


def test_paired_stat_without_subjects_compares_groups():
    from rna_age.train.biology import paired_stat

    d = pl.DataFrame({"subject": [None] * 20, "after": [False] * 10 + [True] * 10,
                      "pred": [50.0] * 10 + [45.0] * 10}, schema_overrides={"subject": pl.String})  # fmt: skip
    s = paired_stat(d, -1, 500)
    assert s["design"] == "groups" and s["delta"] == -5.0 and s["p"] < 0.01
