"""The KronLSTM arm's tokenizers, its ablation encoding and the paired bar (D2, D3); the model on kron-fm's CPU
stand-in when kron_core is installed."""

import numpy as np
import polars as pl
import pytest

from rna_age.train.kron import (
    cpm,
    mask_id,
    mask_tokens,
    micro_batches,
    nonzero_medians,
    paired_ci,
    rank_dense,
    rank_tokens,
)


def test_nonzero_medians_ignore_zeros_and_unseen_genes():
    x = np.array([[0, 2, 0], [4, 4, 0], [6, 0, 0]], dtype=np.float32)
    assert nonzero_medians(x).tolist() == [5.0, 3.0, 1.0]


def test_rank_tokens_order_by_normalised_expression_and_pad_unexpressed():
    c = np.array([[10, 1, 0, 5]], dtype=np.float32)
    med = np.array([10, 0.1, 1, 1], dtype=np.float32)  # gene 1 is 10x its median, gene 0 at it
    tok = rank_tokens(c, med, 6)
    assert tok.tolist() == [[2, 4, 1, 0, 0, 0]]  # gene ids + 1; gene 2 unexpressed -> padding


def test_rank_tokens_ties_keep_gene_order():
    tok = rank_tokens(np.ones((1, 4), dtype=np.float32), np.ones(4, dtype=np.float32), 3)
    assert tok.tolist() == [[1, 2, 3]]


def test_rank_encoding_is_library_size_free():
    a = np.array([[10, 30, 5, 0]], dtype=np.float32)
    med = np.ones(4, dtype=np.float32)
    assert (rank_tokens(cpm(a), med, 4) == rank_tokens(cpm(a * 7), med, 4)).all()


def test_rank_dense_scores_by_position():
    d = rank_dense(np.array([[3, 1, 0, 0]], dtype=np.int32), 4)
    assert d.tolist() == [[0.75, 0.0, 1.0, 0.0]]


def test_paired_ci_signs_and_study_resampling():
    rng = np.random.default_rng(0)
    rows = [{"study_id": f"S{i // 10}", "source": "archs4" if i < 50 else "gtex", "diff": -1.0 + rng.normal(0, 0.1)}
            for i in range(100)]  # fmt: skip
    out = paired_ci(pl.DataFrame(rows), n_boot=200)
    assert out["all"]["hi"] < 0 and out["all"]["studies"] == 10
    assert set(out) == {"all", "archs4", "gtex"}


def test_model_runs_on_the_cpu_stand_in(monkeypatch):
    torch = pytest.importorskip("torch")
    from rna_age.train.kron import build_model, kron_fm_src

    try:
        kron_fm_src()
    except FileNotFoundError:
        pytest.skip("no kron-fm clone (RNA_AGE_KRON_FM)")
    import kron_core.blocks as blocks
    from kron_core.testing import CausalMixer

    monkeypatch.setattr(blocks, "kronlstm_mixer", CausalMixer)
    for values in (False, True):
        m = build_model(n_genes=50, d_model=32, n_layers=2, n_heads=2, dropout=0.0, values=values)
        ids = torch.tensor([[3, 7, 1, 0, 0], [5, 2, 9, 4, 8]])
        vals = torch.randn(2, 5) if values else None
        out = m(ids, vals)
        assert out.shape == (2,) and torch.isfinite(out).all()
        # padding is read by nobody: more padding leaves a row's prediction unchanged
        m.eval()
        one = m(ids[:1], None if vals is None else vals[:1])
        longer = torch.cat([ids[:1], torch.zeros(1, 3, dtype=ids.dtype)], 1)
        lv = None if vals is None else torch.cat([vals[:1], torch.randn(1, 3)], 1)
        assert torch.allclose(m(longer, lv), one, atol=1e-5)


def test_rank_tokens_ties_keep_gene_order_on_long_rows():
    c = np.random.default_rng(0).integers(1, 4, (3, 500)).astype(np.float32)  # three levels, many ties
    tok = rank_tokens(c, np.ones(500, dtype=np.float32), 500)
    for row, t in zip(c, tok, strict=True):
        for level in (3, 2, 1):
            ids = t[np.isin(t - 1, np.flatnonzero(row == level))]
            assert (np.diff(ids) > 0).all()  # within a level, gene order


def test_mask_tokens_masks_only_real_tokens_and_every_row():
    rng = np.random.default_rng(0)
    ids = np.array([[5, 3, 9, 0, 0], [7, 0, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=np.int32)
    m, tgt = mask_tokens(ids, 0.0, rng, mask_id(9))  # frac 0 still masks one token per non-empty row
    assert (tgt > 0).sum(1).tolist() == [1, 1, 0]
    assert ((m == 10) == (tgt > 0)).all() and (tgt[tgt > 0] == ids[tgt > 0]).all()
    big = np.tile(np.arange(1, 1001, dtype=np.int32), (20, 1))
    _, t2 = mask_tokens(big, 0.15, rng, mask_id(1000))
    assert 0.12 < (t2 > 0).mean() < 0.18


def test_padding_is_never_masked():
    ids = np.array([[4, 2, 0, 0, 0, 0]], dtype=np.int32)
    m, tgt = mask_tokens(ids, 1.0, np.random.default_rng(1), mask_id(4))
    assert tgt.tolist() == [[4, 2, 0, 0, 0, 0]]
    assert m.tolist() == [[5, 5, 0, 0, 0, 0]]  # padding stays padding


def test_pretrained_encoder_loads_into_an_age_model(monkeypatch):
    torch = pytest.importorskip("torch")
    from rna_age.train.kron import build_model, kron_fm_src

    try:
        kron_fm_src()
    except FileNotFoundError:
        pytest.skip("no kron-fm clone (RNA_AGE_KRON_FM)")
    import kron_core.blocks as blocks
    from kron_core.testing import CausalMixer

    monkeypatch.setattr(blocks, "kronlstm_mixer", CausalMixer)
    pre = build_model(n_genes=20, d_model=32, n_layers=2, n_heads=2, dropout=0.0, values=False)
    enc = {k: v for k, v in pre.state_dict().items() if not k.startswith(("head.", "norm."))}
    age = build_model(n_genes=20, d_model=32, n_layers=2, n_heads=2, dropout=0.0, values=False)
    missing, unexpected = age.load_state_dict(enc, strict=False)
    assert not unexpected and all(k.startswith(("head.", "norm.")) for k in missing)
    assert torch.equal(age.gene.weight, pre.gene.weight)
    h = age.encode(torch.tensor([[mask_id(20), 3, 0]]))  # the mask token embeds
    assert h.shape == (1, 3, 32)


def test_micro_batches_cover_the_batch_in_order():
    rows = np.arange(10)
    assert [p.tolist() for p in micro_batches(rows, 4)] == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]
    assert [p.tolist() for p in micro_batches(rows, 0)] == [rows.tolist()]


def test_accumulated_gradient_equals_the_full_batch_gradient():
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    net = torch.nn.Linear(5, 1)
    x, y = torch.randn(10, 5), torch.randn(10)
    rows = np.arange(10)

    def grad(micro):
        net.zero_grad()
        for part in micro_batches(rows, micro):
            loss = torch.nn.functional.huber_loss(net(x[part]).squeeze(-1), y[part], delta=5.0)
            (loss * len(part) / len(rows)).backward()
        return net.weight.grad.clone()

    assert torch.allclose(grad(0), grad(3), atol=1e-6)
