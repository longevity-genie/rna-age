"""RM30: episode sampler, leakage guards, the model wrapper and a training smoke, on synthetic tables."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from rna_age.train.expr_icl import (
    EVAL_COHORTS,
    TASKS,
    Corpus,
    EpisodeSampler,
    is_eval_cohort,
    is_heldout_gene,
    standardise,
)

P = 30
TRAIN_SC = "aaaa1111-0000-0000-0000-000000000000"


def make_corpus(seed: int = 0) -> Corpus:
    rng = np.random.default_rng(seed)
    genes = [f"ENSG{i:011d}" for i in range(P)]
    rows = []
    studies = [(f"S{i}", "train", "blood" if i % 2 else "liver") for i in range(6)]
    studies += [("V1", "val", "blood"), ("T1", "test", "blood"), ("M1", "train", "blood")]
    for s, split, tissue in studies:
        for j in range(40):
            case = s in ("S0", "V1") and j < 12
            rows.append({"sample_id": f"{s}_{j}", "study_id": s, "tissue": tissue, "split": split,
                         "age": float(rng.uniform(20, 80)), "group": "case" if case else "control",
                         "category": "metabolic" if case else None})  # fmt: skip
    # M1: one study with a sample in val (studies are split whole; a mixed one must never train)
    rows[-1]["split"] = "val"
    bulk = pl.DataFrame(rows)
    age = bulk["age"].to_numpy()
    bulk_X = (rng.normal(size=(bulk.height, P)) + np.outer(age / 30, np.linspace(-1, 1, P))).astype(np.float32)
    sc_rows = []
    for ds in (TRAIN_SC, EVAL_COHORTS[0], "terekhova2023"):
        for rep in ("natural", "class:CD4 T"):
            for j in range(40):
                sc_rows.append({"table": f"{ds[:8]}:{rep}", "dataset_id": ds, "donor": f"{ds[:4]}{j}", "cv_group": j,
                                "age": float(rng.uniform(20, 80))})  # fmt: skip
    sc = pl.DataFrame(sc_rows)
    sc_X = rng.normal(size=(sc.height, P)).astype(np.float32)
    sc_X[sc["table"].str.ends_with("CD4 T").to_numpy(), 0] = np.nan  # a gene one table lacks
    pert_rows = []
    for donor in ("D1", "D2"):
        for g in range(150):
            pert_rows.append({"perturbed_gene_id": f"ENSG9{g:010d}", "donor_id": donor, "culture_condition": "Rest",
                              "kd_baseline": float(rng.uniform(0, 5))})  # fmt: skip
    pert = pl.DataFrame(pert_rows)
    pert_X = rng.normal(scale=0.2, size=(pert.height, P)).astype(np.float32)
    return Corpus(genes, bulk, bulk_X, sc, sc_X, pert, pert_X)


def sampler(mode: str = "train", **kw) -> EpisodeSampler:
    kw = {"n_rows": (24, 36), "n_cols": (5, 12), "min_rows": 20, **kw}
    return EpisodeSampler(make_corpus(), mode, **kw)


def test_episode_shapes_per_task():
    s = sampler()
    assert set(s.tasks()) == set(TASKS)
    for t in TASKS:
        for _ in range(5):
            e = s.sample(t)
            assert e.X_ctx.shape[1] == e.X_qry.shape[1] >= 1
            assert len(e.y_ctx) == len(e.X_ctx) and len(e.y_qry) == len(e.X_qry) >= 1
            assert np.isfinite(e.X_ctx).all() and np.isfinite(e.X_qry).all()
            if t == "health":
                assert set(np.unique(e.y_ctx)) == {0.0, 1.0}  # both classes in the context
            else:
                assert abs(e.y_ctx.mean()) < 1e-4 and abs(e.y_ctx.std() - 1) < 1e-3
        for _ in range(5):
            batch = s.sample_batch(t, 3)
            assert len(batch) == 3 and len({(b.X_ctx.shape, b.X_qry.shape) for b in batch}) == 1


def test_masked_gene_target_is_not_among_the_features():
    s = sampler()
    c = s.c
    for _ in range(20):
        e = s.sample("masked_gene")
        g = e.meta["group"]
        assert g and e.meta["target_gene"] in c.genes
        assert e.X_ctx.shape[1] <= len(c.genes) - 1


def test_training_episodes_exclude_evaluation_cohorts_heldout_studies_and_genes():
    s = sampler()
    c = s.c
    for t, groups in s.groups.items():
        for g in groups:
            if g.kind == "bulk":
                assert set(c.bulk["split"].to_numpy()[g.rows]) == {"train"}, (t, g.name)
                assert not set(c.bulk["study_id"].to_numpy()[g.rows]) & {"V1", "T1", "M1"}, (t, g.name)
            elif g.kind == "sc":
                ds = set(c.sc["dataset_id"].to_numpy()[g.rows])
                assert ds == {TRAIN_SC}, (t, g.name)
            else:
                assert not any(is_heldout_gene(x) for x in c.pert["perturbed_gene_id"].to_numpy()[g.rows])


def test_eval_mode_uses_only_heldout_parts():
    s = sampler("eval")
    c = s.c
    for _t, groups in s.groups.items():
        for g in groups:
            if g.kind == "bulk":
                assert set(c.bulk["split"].to_numpy()[g.rows]) == {"val"}
            elif g.kind == "sc":
                assert all(is_eval_cohort(d) for d in c.sc["dataset_id"].to_numpy()[g.rows])
            else:
                assert all(is_heldout_gene(x) for x in c.pert["perturbed_gene_id"].to_numpy()[g.rows])
    assert s.groups["health"] and s.groups["perturbation"] and any(g.kind == "sc" for g in s.groups["age"])


def test_guard_raises_on_a_leaking_group():
    s = sampler()
    ev = sampler("eval")
    s.groups["age"].append(next(g for g in ev.groups["age"] if g.kind == "sc"))
    with pytest.raises(AssertionError, match="evaluation cohort"):
        s.check_guards()


def test_heldout_genes_are_stable_and_about_a_fifth():
    genes = [f"ENSG{i:011d}" for i in range(5000)]
    frac = np.mean([is_heldout_gene(g) for g in genes])
    assert 0.17 < frac < 0.23
    assert [is_heldout_gene(g) for g in genes[:50]] == [is_heldout_gene(g) for g in genes[:50]]
    assert all(is_eval_cohort(c) for c in EVAL_COHORTS) and not is_eval_cohort(TRAIN_SC)


def test_standardisation_uses_context_rows_only():
    rng = np.random.default_rng(0)
    Xc, Xq = rng.normal(5, 3, (50, 4)), rng.normal(5, 3, (10, 4))
    a_c, a_q = standardise(Xc, Xq)
    b_c, b_q = standardise(Xc, Xq * 100)  # queries do not move the scaling
    assert np.allclose(a_c, b_c)
    assert np.allclose(a_c.mean(0), 0, atol=1e-5) and np.allclose(a_c.std(0), 1, atol=1e-3)
    assert np.allclose(a_q, (Xq - Xc.mean(0)) / (Xc.std(0) + 1e-6), atol=1e-4)
    e = EpisodeSampler.episode(Xc, Xc[:, 0], Xq, Xq[:, 0] + 1000, "age", "bulk")
    assert abs(e.y_mean - Xc[:, 0].mean()) < 1e-6  # the target too: context only


# ---------------------------------------------------------------- model (needs the tabicl package)

TINY = {"num_quantiles": 19, "embed_dim": 16, "col_num_blocks": 1, "col_nhead": 2, "col_num_inds": 8,
        "row_num_blocks": 1, "row_nhead": 2, "row_num_cls": 2, "row_rope_interleaved": False, "icl_num_blocks": 1,
        "icl_nhead": 2, "ff_factor": 2, "bias_free_ln": True}  # fmt: skip


def tiny_model():
    pytest.importorskip("tabicl")
    import torch

    from rna_age.train.expr_icl_model import ExprICL

    torch.manual_seed(0)
    return ExprICL(TINY)


def test_wrapper_fit_predict_contract():
    from rna_age.train.expr_icl_model import ExprICLRegressor

    m = tiny_model()
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(60, 8)), rng.uniform(20, 80, 60)
    est = ExprICLRegressor(m, max_features=5, n_estimators=2)
    with pytest.raises(RuntimeError):
        est.predict(X[:5])
    assert est.fit(X[:50], y[:50]) is est
    p = est.predict(X[50:])
    assert p.shape == (10,) and np.isfinite(p).all()
    assert np.allclose(p, est.predict(X[50:]))
    assert len(est.cols_) == 5
    with pytest.raises(ValueError):
        est.predict(X[50:, :3])
    assert set(est.get_params()) >= {"model", "max_features", "n_estimators"}


def test_registered_wrapper_runs_through_sc_tabular_cross_validate():
    from rna_age.train import sc_tabular
    from rna_age.train.expr_icl_model import ExprICLRegressor

    m = tiny_model()
    rng = np.random.default_rng(1)
    X, y = rng.normal(size=(40, 12)), rng.uniform(20, 80, 40)
    sc_tabular.ESTIMATORS["expr-icl"] = lambda s: ExprICLRegressor(m, n_estimators=1, seed=s)
    try:
        out = sc_tabular.cross_validate(X, y, np.arange(40), ("expr-icl",), n_genes=6)
    finally:
        sc_tabular.ESTIMATORS.pop("expr-icl")
    assert np.isfinite(out["expr-icl"]).all()


def test_two_training_steps_on_cpu(tmp_path, monkeypatch):
    from rna_age.train import runs
    from rna_age.train.expr_icl_model import TrainConfig, load_trained, train

    m = tiny_model()
    monkeypatch.setattr(runs, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(runs, "INTERIM", tmp_path / "interim")
    cfg = TrainConfig(steps=2, warmup=1, micro_batch=2, accum=4, n_rows=(24, 36), n_cols=(5, 12), eval_every=1,
                      eval_episodes=2, ckpt_every=2, device="cpu")  # fmt: skip
    before = {k: v.clone() for k, v in m.state_dict().items()}
    run_dir = train(cfg, corpus=make_corpus(), model=m, name="t")
    met = pl.read_parquet(run_dir / "train_metrics.parquet")
    assert set(met.filter(pl.col("split") == "train")["step"].to_list()) == {1, 2}
    assert met.filter(pl.col("split") == "eval")["step"].unique().sort().to_list() == [0, 1, 2]
    assert any(not np.allclose(before[k], v) for k, v in m.state_dict().items())
    again = load_trained(run_dir.name)
    for k, v in m.state_dict().items():
        assert np.allclose(again.state_dict()[k], v)


def test_released_weights_reproduce_tabicl_regressor():
    """The loaded TabICLv2 checkpoint, run through our forward on TabICLRegressor's own preprocessed table, gives
    the regressor's predictions (one estimator, no shuffling): the weights and the forward are TabICL's."""
    pytest.importorskip("tabicl")
    import torch
    from tabicl import TabICLRegressor

    from rna_age.train.expr_icl_model import checkpoint_path, from_pretrained

    try:
        ckpt = checkpoint_path(allow_download=False)
    except Exception:
        pytest.skip("TabICLv2 checkpoint not in the Hugging Face cache")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 12)).astype(np.float32)
    y = 3 * X[:, 0] + X[:, 1] ** 2 + rng.normal(scale=0.1, size=80)
    ref = TabICLRegressor(n_estimators=1, norm_methods=["none"], feat_shuffle_method="none", device="cpu",
                          use_amp=False, random_state=0, model_path=ckpt).fit(X[:60], y[:60])  # fmt: skip
    expected = ref.predict(X[60:])
    model = from_pretrained(ckpt, classifier=False).eval()
    ours, theirs = model.reg.state_dict(), ref.model_.state_dict()
    assert ours.keys() == theirs.keys() and all(torch.equal(ours[k], theirs[k]) for k in ours)
    Xs, ys = next(iter(ref.ensemble_generator_.transform(ref.X_encoder_.transform(X[60:]), mode="both").values()))
    with torch.no_grad():
        q = model(torch.from_numpy(Xs).float(), torch.from_numpy(ys).float())
    got = q[0].mean(-1).numpy() * ref.y_scaler_.scale_[0] + ref.y_scaler_.mean_[0]
    assert np.allclose(got, expected, atol=1e-4)
    # and through our own wrapper (own preprocessing): close to TabICLRegressor, not identical
    from rna_age.train.expr_icl_model import ExprICLRegressor

    p = ExprICLRegressor(model, n_estimators=1).fit(X[:60], y[:60]).predict(X[60:])
    assert np.corrcoef(p, expected)[0, 1] > 0.9
