"""RM37 step (c): pool files, masking and a tiny from-scratch pretraining on synthetic cells."""

from __future__ import annotations

import json

import numpy as np
import polars as pl
import scipy.sparse as sp
import torch

from rna_age.train import mct, mct_scratch, runs

ORDER = {"CD4-positive T cells": [0, 6], "B cells": [6, 10], mct.OTHER: [10, 16]}
V = 30


def write_index(d):
    d.mkdir(parents=True, exist_ok=True)
    g = {"[PAD]": 0, "[MASK]": 1} | {f"ENSG{i:011d}": i for i in range(2, V)}
    (d / "ensemble2idx.json").write_text(json.dumps(g))
    (d / "gene_symbol2idx.json").write_text(json.dumps({}))
    (d / "cell_type_input_order.json").write_text(json.dumps(ORDER))
    (d / "cell_type_map.json").write_text(json.dumps({}))


def write_pool(root, n_samples=6, seed=0):
    """A pool in dataset_pool's layout: cells sorted by sample, token ids as columns."""
    rng = np.random.default_rng(seed)
    d = root / "synthds0"
    d.mkdir(parents=True)
    sizes = rng.integers(8, 20, n_samples)
    X = sp.random(int(sizes.sum()), V, density=0.5, random_state=seed, format="csr", dtype=np.float32)
    X[:, :2] = 0
    X = X.tocsr()
    X.eliminate_zeros()
    np.save(d / "data.npy", X.data.astype(np.float16))
    np.save(d / "indices.npy", X.indices.astype(np.int32))
    np.save(d / "indptr.npy", X.indptr.astype(np.int64))
    pl.DataFrame({"slot": rng.integers(0, 3, X.shape[0]).astype(np.int16)}).write_parquet(d / "cells.parquet")
    ends = np.cumsum(sizes)
    pl.DataFrame({"sample": [f"s{i}:blood" for i in range(n_samples)], "start": ends - sizes, "end": ends,
                  "age": np.linspace(20, 80, n_samples), "disease": ["normal"] * n_samples,
                  "tissue": ["blood"] * n_samples, "dataset_id": ["synthds0"] * n_samples}).write_parquet(
        d / "samples.parquet")  # fmt: skip


def test_mask_tokens_proportions():
    tok = torch.arange(2, 20002)[None]
    ids, emask, sel = mct_scratch.mask_tokens(tok, 30000, torch.Generator().manual_seed(0))
    assert abs(float(sel.float().mean()) - 0.15) < 0.01
    assert abs(float((ids == mct_scratch.MASK_ID).float().sum() / sel.sum()) - 0.7) < 0.03
    assert abs(float(emask.float().sum() / sel.sum()) - 0.1) < 0.02
    assert not (emask & ~sel).any() and ((ids != tok) <= sel).all()


def test_pool_bag(tmp_path):
    write_index(tmp_path / "index")
    write_pool(tmp_path / "pool")
    pool = mct_scratch.Pool(tmp_path / "pool", mct.Index.load(tmp_path / "index"), min_cells=1)
    assert len(pool) == 6
    x, tok = pool.bag(2, np.random.default_rng(0), max_genes=50)
    assert x.shape == (len(tok), 16) and (tok >= 2).all() and x.dtype == np.float32


def test_pretrain_tiny_and_reload(tmp_path, monkeypatch):
    monkeypatch.setattr(runs, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(runs, "INTERIM", tmp_path / "interim")
    write_index(tmp_path / "index")
    write_pool(tmp_path / "pool", n_samples=10)
    st = mct_scratch.pretrain(steps=3, accumulate=2, warmup=1, max_genes=20, emb=8, n_layers=2, n_heads=2, hidden=16,
                              val_share=0.2, n_val_bags=2, eval_every=2, device="cpu", pool_root=tmp_path / "pool",
                              index_dir=tmp_path / "index", min_cells=1)  # fmt: skip
    assert np.isfinite(st["best_val_loss"])
    enc, cfg = mct.load_encoder(st["best_checkpoint"])
    assert len(enc.encoder.layers) == 2 and cfg["n_heads"] == 2 and enc.gene_id_emb.num_embeddings == V


def test_pool_drops_small_samples_and_tiny_bag_has_tokens(tmp_path):
    write_index(tmp_path / "index")
    write_pool(tmp_path / "pool")
    idx = mct.Index.load(tmp_path / "index")
    full = mct_scratch.Pool(tmp_path / "pool", idx, min_cells=1)
    sizes = (full.samples["end"] - full.samples["start"]).to_numpy()
    cut = int(np.median(sizes))
    part = mct_scratch.Pool(tmp_path / "pool", idx, min_cells=cut)
    assert len(part) == int((sizes >= cut).sum()) and part.dropped == len(full) - len(part)
    x, tok = full.bag(int(np.argmin(sizes)), np.random.default_rng(0), max_genes=50)  # ≤ 20 cells, MCT's rule keeps none
    assert len(tok) > 0
