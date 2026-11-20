import numpy as np
import polars as pl
import pytest

from rna_age.train import sys_gap, sys_gap_ext
from rna_age.train.sys_gap import CaseDef, Cohort, _merge_write
from rna_age.train.sys_gap_ext import (
    POOLS,
    Target,
    _measure,
    common_genes,
    gated,
    load_pool,
    pool_cohorts,
    run_target,
    training_set,
)

N_GENES = 60


def _write_cohort(root, name, ages, disease, genes, rng, accel=None):
    """A one-type cohort on disk: expression of genes G0-G9 drifts with age (+ ``accel`` years in some donors)."""
    d = root / "census_pseudobulk" / "by_dataset_celltype" / name
    d.mkdir(parents=True)
    base = np.random.default_rng(99).uniform(5, 50, N_GENES)  # same biology in every cohort
    eff = np.zeros(N_GENES)
    eff[:10] = 0.03
    accel = np.zeros(len(ages)) if accel is None else accel
    rows, counts = [], {"gene_id": genes}
    gi = [int(g[1:]) for g in genes]
    for k, (a, dis) in enumerate(zip(ages, disease, strict=True)):
        mu = base * np.exp(eff * (a + accel[k] - 50)) * 40
        sid = f"{name}-D{k}"
        rows.append({"donor_id": f"{name}-D{k}", "tissue": "blood", "cell_type": "CD8-positive, alpha-beta T cell",
                     "n_cells": 50, "sample_id": sid, "development_stage": f"{int(a)}-year-old stage",
                     "disease": dis, "sex": "male" if k % 2 else "female"})  # fmt: skip
        counts[sid] = rng.poisson(mu[gi]).tolist()
    pl.DataFrame(rows).write_parquet(d / "groups.parquet")
    pl.DataFrame(counts).write_parquet(d / "counts.parquet")


@pytest.fixture
def disk(tmp_path, monkeypatch):
    monkeypatch.setattr(sys_gap, "INPUT", tmp_path)
    rng = np.random.default_rng(0)
    all_genes = [f"G{i}" for i in range(N_GENES)]
    # pool: 160 healthy donors, 20 children (left out: < 18 y) and 20 sick donors (left out), genes in another order
    # plus pool-only genes are absent from the target
    pa = np.r_[rng.uniform(20, 90, 160), rng.uniform(1, 15, 20), rng.uniform(20, 90, 20)].round()
    pdis = ["normal"] * 180 + ["COVID-19"] * 20
    pool_genes = list(np.random.default_rng(1).permutation(all_genes))
    _write_cohort(tmp_path, "poolA", pa, pdis, pool_genes, rng)
    # target: 30 controls, 40 cases 10 years older in expression; G50-G59 missing (intersection = 50 genes)
    ta = rng.uniform(40, 80, 70).round()
    tdis = ["normal"] * 30 + ["COVID-19"] * 40
    accel = np.r_[np.zeros(30), np.full(40, 10.0)] + rng.normal(0, 6, 70)  # a stable personal gap: reliability
    _write_cohort(tmp_path, "tgt", ta, tdis, all_genes[:50][::-1], rng, accel)
    return pa, ta


TGT = Target("tgt", "blood", CaseDef("x", r"covid-19", pairs=(), predictions=(("CD8 T", "accelerates"),)),
             guards=False)  # fmt: skip


def test_external_clock_never_sees_target_donors(disk, monkeypatch):
    pa, ta = disk
    seen = []
    real = sys_gap_ext.fit_fold

    def spy(X, y, n):
        seen.append((X.shape, np.sort(y)))
        return real(X, y, n)

    monkeypatch.setattr(sys_gap_ext, "fit_fold", spy)
    pool = [load_pool("poolA")]
    assert sorted(pool[0].age.tolist()) == sorted(pa[:160].tolist())  # healthy adults only
    out = run_target(TGT, pool, n_genes=30, n_boot=200)
    ((shape, y),) = seen
    assert shape == (160, 50)  # pool donors only, common genes only
    assert np.allclose(y, np.sort(pa[:160]))
    with pytest.raises(ValueError):
        run_target(TGT, [load_pool("tgt")], n_genes=30)
    assert "tgt" not in pool_cohorts("blood", "tgt")
    assert pool_cohorts("brain", POOLS["brain"][0]) == POOLS["brain"][1:]
    # the planted 10 years are recovered in the target's own years scale
    r = {x["type"]: x for x in out["contrast"]}["CD8 T"]
    assert r["lo"] > 0 and abs(r["delta_years"] - 10) < 4
    (t,) = out["type"]
    assert t["status"] == "ok" and t["n_genes_common"] == 50 and t["n_controls"] == 30 and t["n_train"] == 160
    assert all(x["clock"] == "external:blood" for v in out.values() for x in v)
    p = out["predictions"][0]
    assert p["verdict"] == "hold" and p["counted"]


def test_bias_line_from_target_controls_only():
    rng = np.random.default_rng(2)
    n = 80
    age = rng.uniform(40, 80, n)
    ctrl = np.arange(n) < 30
    pred = 20 + 0.5 * age + rng.normal(0, 2, n)
    m = _measure(pred, pred + rng.normal(0, 0.5, n), pred + rng.normal(0, 0.5, n), age, ctrl, 10)
    b, a = np.polyfit(age[ctrl], pred[ctrl], 1)
    assert np.isclose(m["slope"], b) and np.isclose(m["intercept"], a)
    p2 = pred.copy()
    p2[~ctrl] += rng.normal(30, 10, (~ctrl).sum())  # cases moved: the line does not move
    m2 = _measure(p2, p2, p2, age, ctrl, 10)
    assert np.isclose(m2["slope"], m["slope"]) and np.isclose(m2["intercept"], m["intercept"])
    assert _measure(pred, pred, pred, age, ctrl & (np.arange(n) < 5), 10)["status"].startswith("unmeasured: <")
    flat = 60 - 0.2 * age + rng.normal(0, 2, n)
    assert _measure(flat, flat, flat, age, ctrl, 10)["status"] == "unmeasured: slope <= 0"


def test_gene_intersection_aligns_columns_by_id():
    assert common_genes([["a", "b", "c", "d"], ["d", "c", "a"], ["c", "a", "x"]]) == ["a", "c"]
    C1 = np.array([[1, 2, 3], [4, 5, 6]])
    C2 = np.array([[30, 10, 20]])  # same genes, other order: c, a, b

    def co(name, genes, C, age):
        return Cohort(name, [f"{name}{i}" for i in range(len(C))], np.arange(len(C)), np.array(age, float),
                      ["normal"] * len(C), ["female"] * len(C), None, None,
                      types={"B": (np.arange(len(C)), C, np.full(len(C), 50.0))}, genes=genes)  # fmt: skip

    pool = [co("p1", ["a", "b", "c"], C1, [30, 40]), co("p2", ["c", "a", "b"], C2, [50])]
    X, y, src = training_set(pool, "B", ["c", "a"])
    expect = sys_gap.log_cpm(np.array([[3, 1], [6, 4], [30, 10]]))
    assert np.allclose(X, expect) and y.tolist() == [30, 40, 50] and src == ["p1", "p1", "p2"]
    assert training_set(pool, "NK", ["a"])[0].shape == (0, 1)


def test_unmeasured_type_makes_counted_prediction_inconclusive():
    rows = [{"type": "CD8 T - monocyte", "counted": True, "verdict": "hold"},
            {"type": "CD8 T", "counted": True, "verdict": "hold"}]  # fmt: skip
    out = gated(rows, {"CD8 T": "ok", "monocyte": "unmeasured: slope <= 0"}, counted=True)
    assert [r["verdict"] for r in out] == ["inconclusive", "hold"]
    assert out[0]["verdict_ci"] == "hold" and out[0]["status"] == "ok / unmeasured: slope <= 0"
    assert not any(r["counted"] for r in gated(rows, {"CD8 T": "ok", "monocyte": "ok"}, counted=False))


def test_merge_keeps_own_rows_and_adds_clock(tmp_path, monkeypatch):
    monkeypatch.setattr(sys_gap, "SUMMARY", tmp_path)
    old = pl.DataFrame({"cohort": ["a", "a", "GSE1"], "level": ["coarse", "coarse", "bulk"], "x": [1.0, 2.0, 3.0],
                        "clock": [None, None, "m1"]})  # fmt: skip
    old.write_parquet(tmp_path / "sys_gap_type.parquet")
    new = pl.DataFrame({"cohort": ["a"], "level": ["coarse"], "x": [9.0], "clock": ["external:blood"]})
    out = _merge_write("type", new)
    assert out.height == 4
    kept = out.filter(pl.col("clock") != "external:blood")
    assert kept.drop("clock").equals(old.drop("clock")) and kept["clock"].to_list() == ["own", "own", "m1"]
    again = _merge_write("type", new.with_columns(x=pl.lit(7.0)))  # a rerun replaces only its own clock's rows
    assert again.height == 4 and again.filter(pl.col("clock") == "external:blood")["x"].to_list() == [7.0]


def test_symbols_to_ensembl_only_when_unique():
    from rna_age.train.sys_gap_ext import to_ensembl

    table = pl.DataFrame({"gene_id": ["E1", "E2", "E3", "E4"], "symbol": ["A", "B", "B", "C"]})
    assert to_ensembl(["A", "B", "C", "X", "A"], table) == ["E1", "B", "E4", "X", "A"]
