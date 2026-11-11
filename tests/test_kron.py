"""The KronLSTM arm's tokenizers, its ablation encoding and the paired bar (D2, D3); the model on kron-fm's CPU
stand-in when kron_core is installed."""

import numpy as np
import polars as pl
import pytest

from rna_age.train.kron import cpm, nonzero_medians, paired_ci, rank_dense, rank_tokens


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
