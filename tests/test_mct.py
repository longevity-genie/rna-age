"""RM37: MCT wrapper (checkpoint reader, bag packing, fine-tune) on synthetic data."""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from rna_age.train import mct

ORDER = {"CD4-positive T cells": (0, 6), "B cells": (6, 10), mct.OTHER: (10, 16)}


def small_encoder(n_genes=40, n_input=16, layers=3):
    return mct.MctEncoder(n_genes, n_input, emb=8, hidden=16, n_heads=2, n_layers=layers, dropout=0.0,
                          transformer_dropout=0.0)  # fmt: skip


def fake_checkpoint(tmp_path, enc):
    """A Lightning-style checkpoint whose config is a dataclass from a module that will not exist at load time."""
    name = "cell_cell_interactions.modeling.data_types"
    for i, part in enumerate(name.split(".")):
        sys.modules.setdefault(".".join(name.split(".")[: i + 1]), types.ModuleType(part))

    @dataclass
    class MulticellularGraphConfig:
        n_heads: int = 2
        dropout: float = 0.0
        transformer_dropout: float = 0.0
        lr: float = 1e-3

    MulticellularGraphConfig.__module__ = name
    MulticellularGraphConfig.__qualname__ = "MulticellularGraphConfig"
    sys.modules[name].MulticellularGraphConfig = MulticellularGraphConfig
    cfg = MulticellularGraphConfig(lr=np.float64(1e-3))
    sd = {f"model.encoder.{k}": v for k, v in enc.state_dict().items()}
    sd["model.gene_id_classifier.dense.weight"] = torch.zeros(8, 8)
    path = tmp_path / "m.ckpt"
    torch.save({"state_dict": sd, "hyper_parameters": {"config": cfg}, "epoch": 3}, path)
    for k in [k for k in sys.modules if k.startswith("cell_cell_interactions")]:
        del sys.modules[k]
    return path


def test_checkpoint_round_trip(tmp_path):
    torch.manual_seed(0)
    enc = small_encoder().eval()
    path = fake_checkpoint(tmp_path, enc)
    loaded, cfg = mct.load_encoder(path)
    loaded.eval()
    assert cfg["n_heads"] == 2 and float(cfg["lr"]) == pytest.approx(1e-3)
    assert len(loaded.encoder.layers) == 3
    x, tok = torch.rand(1, 5, 16), torch.tensor([[2, 3, 7, 9, 11]])
    assert torch.allclose(enc(x, tok), loaded(x, tok), atol=1e-6)
    stub = sys.modules["cell_cell_interactions.modeling.data_types"].MulticellularGraphConfig
    assert issubclass(stub, mct._Stub)  # rebuilt as an inert stub, not MCT's class


def test_checkpoint_refuses_unsafe_global(tmp_path):
    path = tmp_path / "bad.ckpt"
    torch.save({"state_dict": {}, "f": print}, path)
    with pytest.raises(ValueError, match="builtins.print"):
        mct.read_checkpoint(path)


def test_slot_type():
    idx = mct.Index({}, {}, ORDER, {"naive B cell": "B cells", "neuron": "Neurons"})
    assert idx.slot_type("B cells") == "B cells"
    assert idx.slot_type("naive B cell") == "B cells"
    assert idx.slot_type("neuron") == mct.OTHER  # mapped to a type with no slot
    assert idx.slot_type("anything") == mct.OTHER
    assert idx.slot_type("CD4+ T cells", mct.TEREKHOVA) == "CD4-positive T cells"


def test_pack_bag_fills_ranges_and_pads():
    names = list(ORDER)
    types_ = np.array([0] * 9 + [1] * 2)  # 9 CD4 cells (6 slots), 2 B cells (4 slots), no Other
    out = mct.pack_bag(types_, ORDER, names, np.random.default_rng(0))
    assert len(out) == 16
    assert len(set(out[0:6])) == 6 and set(out[0:6]) <= set(range(9))
    assert sorted(out[6:8]) == [9, 10] and (out[8:10] == -1).all()
    assert (out[10:16] == -1).all()


def make_cells(n_cells=30, n_genes=12, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.poisson(0.5, (n_cells, n_genes)).astype(np.float32)
    X[:, 0] = 0  # never expressed
    X[:, 1] = 0
    X[:3, 1] = 1  # expressed in 3 cells only
    X[:, 2:] += 1  # everywhere
    return mct.DonorCells(sp.csr_matrix(X), np.arange(10, 10 + n_genes), ["d0", "d1"],
                          [np.arange(0, 15), np.arange(15, 30)],
                          [np.array([0] * 10 + [1] * 5), np.array([0] * 8 + [2] * 7)], list(ORDER), ORDER)  # fmt: skip


def test_bag_tensor_gene_rule():
    dc = make_cells()
    rng = np.random.default_rng(0)
    x, tok = mct.bag_tensor(dc, 0, rng, max_genes=100, min_frac=0.2)  # > int(0.2 * 16) = 3 cells
    assert x.shape == (len(tok), 16)
    assert 10 not in tok and 11 not in tok and set(range(12, 22)) <= set(tok)
    assert (x[:, 10:16] == 0).all()  # donor 0 has no Other cells: padded slots are zero
    x, tok = mct.bag_tensor(dc, 0, rng, max_genes=4, min_frac=0.2)
    assert len(tok) == 4 and (np.diff(tok) > 0).all()


def test_freeze_below():
    m = mct.MctAge(small_encoder(layers=4))
    mct.freeze_below(m, 1)
    train = {n for n, p in m.named_parameters() if p.requires_grad}
    assert any(n.startswith("encoder.encoder.layers.3.") for n in train)
    assert not any(n.startswith("encoder.encoder.layers.2.") for n in train)
    assert "head.weight" in train and "encoder.gene_id_emb.weight" not in train


def test_finetune_fold_runs_on_cpu():
    torch.manual_seed(0)
    dc = make_cells()
    base = small_encoder(n_genes=40, n_input=16)
    age = np.array([30.0, 60.0])
    p = mct.finetune_fold(base, dc, np.array([0, 1]), np.array([1]), age, epochs=2, n_train_layers=1,
                          max_genes=8, n_test_bags=2, accumulate=1, device="cpu")  # fmt: skip
    assert p.shape == (1,) and np.isfinite(p).all()


def test_embed_donors_and_gene_matrix():
    dc = make_cells()
    means, per_gene, n_tok = mct.embed_donors(small_encoder(n_genes=40, n_input=16), dc, n_bags=2, max_genes=50,
                                              device="cpu")  # fmt: skip
    assert means.shape == (2, 8) and np.isfinite(means).all() and (n_tok > 0).all()
    G, toks = mct.gene_matrix(per_gene, 2, min_share=1.0)
    assert G.shape == (2, 8 * len(toks)) and np.isfinite(G).all()


def test_pool_cells_caps_per_type():
    import polars as pl

    idx = mct.Index({}, {}, ORDER, {})
    cells = pl.DataFrame({"row": np.arange(40), "donor": ["a"] * 30 + ["b"] * 10,
                          "ct": ["B cells"] * 25 + ["x"] * 5 + ["B cells"] * 10})  # fmt: skip
    out = mct.pool_cells(cells, idx, "", 2, np.random.default_rng(0))  # B cells: 4 slots × 2 = 8 per donor
    n = out.group_by("donor", "slot").len().sort("donor", "slot")
    assert n.rows() == [("a", "B cells", 8), ("a", mct.OTHER, 5), ("b", "B cells", 8)]
    assert (np.diff(out["row"].to_numpy()) > 0).all()
