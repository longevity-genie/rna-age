"""RM28: masking and its loss, the mean-gene baseline and stop rule, pool exclusion, perturbation batches and per-donor
pooling (synthetic data)."""

import numpy as np
import polars as pl
import scipy.sparse as sp
import torch

from rna_age.train import sc_pretrain as scp


def test_mask_hides_15_to_30_percent_per_cell():
    g = torch.Generator().manual_seed(0)
    m = scp.sample_mask(500, 2000, 0.15, 0.30, g)
    frac = m.float().mean(1)
    assert m.dtype == torch.bool and frac.min() >= 0.149 and frac.max() <= 0.301
    assert frac.std() > 0.02  # the rate varies per cell
    assert not torch.equal(m[0], m[1])  # different genes per cell


def test_loss_counts_masked_positions_only():
    x = torch.zeros(2, 4)
    pred = torch.tensor([[1.0, 100.0, 0.0, 0.0], [0.0, 0.0, 2.0, 50.0]])
    m = torch.tensor([[True, False, False, False], [False, False, True, False]])
    assert scp.masked_mse(pred, x, m).item() == (1 + 4) / 2  # the 100 and 50 sit on visible positions
    # the model never sees masked values: changing them leaves the encoding unchanged
    model = scp.ScMAE(4, hidden=8, emb=3, depth=1, dropout=0.0).eval()
    x1 = torch.rand(2, 4)
    x2 = x1.clone()
    x2[m] += 5
    assert torch.allclose(model.encode(x1, m), model.encode(x2, m))


def test_mean_gene_baseline_and_stop_rule():
    x = torch.tensor([[1.0, 2.0], [3.0, 6.0]])
    mean = x.mean(0)  # [2, 4]
    m = torch.ones_like(x, dtype=torch.bool)
    assert scp.mean_gene_mse(mean, x, m).item() == (1 + 4 + 1 + 4) / 4
    assert scp.early_stop_failed(1, 0.5, 0.5)  # not below: failure
    assert scp.early_stop_failed(1, 0.6, 0.5)
    assert not scp.early_stop_failed(1, 0.4, 0.5)
    assert not scp.early_stop_failed(2, 0.6, 0.5)  # the rule is checked after epoch 1


def _obs():
    return pl.DataFrame({
        "row": np.arange(8, dtype=np.int64),
        "donor_id": ["a", "a", "b", "b", "c", "kid", "sick", "a"],
        "disease": ["normal"] * 6 + ["COVID-19", "normal"],
        "development_stage": ["40-year-old stage", "40-year-old stage", "sixth decade stage", "sixth decade stage",
                              "80 year-old and over stage", "11-year-old stage", "50-year-old stage", "unknown"],
        "tissue": ["blood"] * 8, "cell_type": ["T cell"] * 8,
        "is_primary_data": [True, True, True, False, True, True, True, True],
    })  # fmt: skip


def test_pool_keeps_healthy_adult_primary_cells():
    e = scp.eligible_cells(_obs())
    assert e["row"].to_list() == [0, 1, 2, 4]  # not: non-primary (3), child (5), disease (6), unknown stage (7)
    assert e.filter(pl.col("row") == 2)["age"].item() == 50.0 and not e.filter(pl.col("row") == 2)["age_exact"].item()


def test_cap_per_donor():
    cells = pl.DataFrame({"row": np.arange(100), "donor_id": ["a"] * 90 + ["b"] * 10})
    c = scp.cap_per_donor(cells, 20, seed=0)
    assert c.group_by("donor_id").len().sort("donor_id")["len"].to_list() == [20, 10]
    assert c.equals(scp.cap_per_donor(cells, 20, seed=0))


def test_evaluation_cohorts_and_their_donors_are_excluded():
    pool = pl.DataFrame({
        "dataset_id": ["pool1", "pool1", "pool2", scp.EVAL_COHORTS[0]],
        "donor_id": ["d1", "shared", "d2", "e1"], "row": [0, 1, 0, 0],
    })  # fmt: skip
    evals = pl.DataFrame({"cohort": [scp.EVAL_COHORTS[0], scp.TEREKHOVA], "donor_id": ["e1", "shared"]})
    kept, ov = scp.exclude_eval(pool, evals)
    assert kept["donor_id"].to_list() == ["d1", "d2"]
    assert ov["donor_id"].to_list() == ["shared"] and ov["cohort"].to_list() == [scp.TEREKHOVA]
    assert scp.donor_overlap(kept, evals).height == 0


def test_pool_datasets_skip_evaluation_cohorts(tmp_path):
    for name in [*scp.EVAL_COHORTS[:6], "pool-a", "pool-b"]:
        (tmp_path / f"{name}.h5ad").touch()
    assert scp.pool_datasets(tmp_path) == ["pool-a", "pool-b"]


def test_log_cp10k_and_panel_columns():
    X = sp.csr_matrix(np.array([[1.0, 3.0, 6.0], [0.0, 0.0, 0.0]]))
    L = scp.log_cp10k(X)
    assert np.allclose(L.toarray()[0], np.log1p(np.array([1, 3, 6]) / 10 * 10_000))
    cols = scp.columns_for(["G1", None, "G3", "G1"], ["G3", "G2", "G1"])
    assert cols.tolist() == [2, -1, 0]
    d = scp.panel_dense(L, cols)
    assert d.shape == (2, 3) and d[0, 1] == 0 and np.isclose(d[0, 0], np.log1p(6000))


def test_choose_panel_ranks_by_mean_variance():
    stats = {"a": (np.zeros(3), np.array([1.0, 0.0, 5.0])), "b": (np.zeros(3), np.array([3.0, 0.0, 0.0]))}
    p = scp.choose_panel(stats, ["g1", "g2", "g3"], 2)
    assert p["gene_id"].to_list() == ["g3", "g1"]  # means 2.5, 2.0


def test_heldout_genes_never_in_training_batches():
    kd_gene = np.repeat(np.arange(50), 4)  # 50 genes × 4 contexts
    tr, va = scp.perturb_split(kd_gene, 0.2, seed=1)
    held = set(kd_gene[va].tolist())
    assert len(held) == 10 and not held & set(kd_gene[tr].tolist())
    assert len(tr) + len(va) == len(kd_gene)
    it = scp.perturb_batches(tr, 7, np.random.default_rng(0))
    seen = np.concatenate([next(it) for _ in range(100)])
    assert set(seen.tolist()) <= set(tr.tolist()) and not set(kd_gene[seen].tolist()) & held
    assert np.array_equal(scp.perturb_split(kd_gene, 0.2, seed=1)[1], va)  # fixed by seed


def test_perturbation_head_and_deg_weights():
    model = scp.ScMAE(6, hidden=8, emb=4, depth=1, dropout=0.0)
    x = torch.rand(3, 6)
    out = model.perturb(x, torch.tensor([0, 5, 2]))
    assert out.shape == (3, 6)
    out.sum().backward()
    assert model.inp.weight.grad is not None and model.inp.weight.grad.abs().sum() > 0  # the token trains the encoder
    w = scp.deg_weights(torch.tensor([[0.0, 2.0]]), torch.tensor([1.0, 0.5]))
    assert w.tolist() == [[1.0, 5.0]]


def test_perturb_data_end_to_end(tmp_path):
    rng = np.random.default_rng(0)
    n_ctx, G = 2, 6
    kd_gene = np.tile(np.arange(G), n_ctx)
    kd_ctx = np.repeat(np.arange(n_ctx), G)
    x_ctl = rng.random((n_ctx, G)).astype(np.float32)
    np.savez(tmp_path / "p.npz", ctx=np.array(["d|a", "d|b"]), x_ctl=x_ctl,
             x_kd=(x_ctl[kd_ctx] + rng.normal(0, 0.1, (len(kd_ctx), G))).astype(np.float16), kd_ctx=kd_ctx,
             kd_gene=kd_gene, kd_n_cells=np.ones(len(kd_ctx)), panel=np.array([f"g{i}" for i in range(G)]))  # fmt: skip
    pd = scp.PerturbData(tmp_path / "p.npz", "cpu", 0.34, seed=0)
    assert not set(kd_gene[pd.train_idx]) & set(kd_gene[pd.val_idx])
    model = scp.ScMAE(G, hidden=8, emb=4, depth=1, dropout=0.0)
    assert pd.loss(model, pd.train_idx[:4]).item() > 0
    ev = pd.evaluate(model, pd.val_idx)
    assert set(ev) == {"pert_val_mse", "pert_val_mse_nochange", "pert_val_mse_meandelta", "pert_val_delta_r"}


def test_split_donors_keeps_donors_whole():
    pool = pl.DataFrame({"dataset_id": ["p"] * 100, "donor_id": [f"d{i % 20}" for i in range(100)]})
    v = scp.split_donors(pool, 0.1, seed=0)
    by = (
        pool.with_columns(v=pl.Series(v))
        .group_by("donor_id")
        .agg(pl.col("v").n_unique().alias("k"), pl.col("v").first())
    )
    assert (by["k"] == 1).all() and by["v"].sum() == 2


def test_donor_means_pool_per_donor_and_subset():
    e = np.array([[1.0, 0.0], [3.0, 2.0], [10.0, 10.0], [5.0, 5.0]])
    donors = np.array(["a", "a", "b", "a"])
    out = scp.donor_means(e, donors, {"all": np.array([True, True, True, False]),
                                      "CD4 T": np.array([False, True, False, True])})  # fmt: skip
    a_all = out.filter((pl.col("donor") == "a") & (pl.col("subset") == "all"))
    assert a_all["n_cells"].item() == 2 and a_all["emb_0"].item() == 2.0 and a_all["emb_1"].item() == 1.0
    a_cd4 = out.filter((pl.col("donor") == "a") & (pl.col("subset") == "CD4 T"))
    assert a_cd4["emb_0"].item() == 4.0 and a_cd4["emb_1"].item() == 3.5
    assert out.filter(pl.col("subset") == "CD4 T").height == 1  # b has no CD4 T cell


def test_tiny_pretraining_beats_the_mean_gene_baseline():
    """A few hundred steps on a low-rank synthetic expression matrix: masked MSE goes below predicting the mean."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    z = rng.normal(size=(2000, 3))
    x = torch.from_numpy(np.log1p(np.exp(z @ rng.normal(size=(3, 40))))).float()
    model = scp.ScMAE(40, hidden=64, emb=8, depth=1, dropout=0.0)
    opt = torch.optim.Adam(model.parameters(), 3e-3)
    for _ in range(300):
        b = x[torch.randint(0, 1500, (128,))]
        m = scp.sample_mask(len(b), 40, 0.15, 0.3)
        loss = scp.masked_mse(model(b, m), b, m)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    xv = x[1500:]
    m = scp.sample_mask(len(xv), 40, 0.15, 0.3, torch.Generator().manual_seed(1))
    with torch.no_grad():
        assert scp.masked_mse(model(xv, m), xv, m) < 0.8 * scp.mean_gene_mse(x[:1500].mean(0), xv, m)


def test_cellpool_mean_equals_head_on_mean_embedding_and_attn_starts_as_mean():
    import torch

    from rna_age.train.sc_pretrain import CellPool

    torch.manual_seed(0)
    z = torch.randn(30, 8)
    m = CellPool(8, "mean")
    torch.nn.init.normal_(m.head.weight)
    don, cell = m(z)
    assert cell.shape == (30,)
    assert torch.allclose(don, m.head(z.mean(0)).squeeze(), atol=1e-5)
    a = CellPool(8, "attn")
    a.head.load_state_dict(m.head.state_dict())
    assert torch.allclose(a.weights(z), torch.full((30,), 1 / 30), atol=1e-6)  # zero-initialised gate = mean
    assert torch.allclose(a(z)[0], don, atol=1e-5)


def test_finetune_fold_runs_with_cell_loss_and_attention():
    import numpy as np
    import torch

    from rna_age.train.sc_pretrain import ScMAE, finetune_fold

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    model = ScMAE(n_genes=12, hidden=16, emb=4)
    donors = [f"d{i}" for i in range(12)]
    cell_donor = np.repeat(donors, 5)
    x = torch.from_numpy(rng.normal(size=(60, 12)).astype(np.float32))
    age = rng.uniform(20, 80, 12)
    for pool, w in (("mean", 0.0), ("attn", 0.5)):
        p = finetune_fold(model, x, cell_donor, donors[:8], age[:8], donors[8:10], age[8:10], donors[10:], 2, 1e-3,
                          1e-2, 4, 0, pool, w)  # fmt: skip
        assert p.shape == (2,) and np.isfinite(p).all()
