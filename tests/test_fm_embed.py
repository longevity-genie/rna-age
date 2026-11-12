"""RM22: Geneformer-style tokenisation, the per-donor cell sample, and the paired comparison (synthetic data)."""

import io
import pickle

import numpy as np
import polars as pl
import pytest
import scipy.sparse as sp

from rna_age.train.fm_embed import _SafeUnpickler, compare, gene_map, paired_bootstrap, sample_cells, tokenize

VOCAB = {
    "token": {"<pad>": 0, "<cls>": 2, "<eos>": 3, "ENSG1": 10, "ENSG2": 11, "ENSG3": 12},
    "median": {"ENSG1": 1.0, "ENSG2": 4.0, "ENSG3": 0.5},
    "mapping": {"ENSG1": "ENSG1", "ENSG2": "ENSG2", "ENSG3": "ENSG3", "ENSG9": "ENSG2"},  # ENSG9 collapses into 2
    "name": {"A": "ENSG1"},
}


def test_tokenize_ranks_median_scaled_values_and_collapses_duplicates():
    G, tokens, med = gene_map(["ENSG1.5", "ENSG2", "ENSG9", "ENSG3", "ENSGX"], VOCAB)
    assert G.shape == (5, 3) and G[4].nnz == 0  # unknown gene dropped
    X = sp.csr_matrix(np.array([[3.0, 4.0, 4.0, 2.0, 9.0], [0.0, 0.0, 0.0, 3.0, 0.0]]))
    t = tokenize(X, G, tokens, med, cls=2, eos=3)
    # cell 0 (scaled by its total): ENSG1 3/1 = 3, ENSG2 (4+4)/4 = 2 (duplicates summed), ENSG3 2/0.5 = 4
    assert t[0].tolist() == [2, 12, 10, 11, 3]
    assert t[1].tolist() == [2, 12, 3]  # zeros dropped
    X2 = sp.csr_matrix(np.array([[1.0, 8.0, 0.0, 3.0, 0.0]]))  # ENSG1 1, ENSG2 2, ENSG3 6
    assert tokenize(X2, G, tokens, med, 2, 3)[0].tolist() == [2, 12, 11, 10, 3]
    assert len(tokenize(X2, G, tokens, med, 2, 3, max_len=3)[0]) == 3  # truncated, framed by cls/eos


def test_symbols_map_through_the_name_dictionary():
    G, tokens, _ = gene_map(["A", "B"], VOCAB)
    assert tokens.tolist() == [10] and G[0, 0] == 1 and G[1].nnz == 0


def test_sample_caps_per_donor_and_stratifies():
    rows = np.arange(1000)
    cells = pl.DataFrame({"row": rows, "donor": np.where(rows < 900, "d1", "d2"),
                          "cls": np.where(rows % 3 == 0, "CD4 T", "B")})  # fmt: skip
    s = sample_cells(cells, cap=90, cd4_cap=50, seed=0)
    d1 = s.filter(pl.col("donor") == "d1")
    assert d1["in_all"].sum() == 90 and d1.filter(pl.col("in_all") & (pl.col("cls") == "CD4 T")).height == 30
    assert d1["in_cd4"].sum() == 50 and (d1.filter(pl.col("in_cd4"))["cls"] == "CD4 T").all()
    d2 = s.filter(pl.col("donor") == "d2")
    assert d2["in_all"].sum() == 90 and d2["in_cd4"].sum() == 34  # fewer CD4 than the cap: all of them
    assert s["row"].n_unique() == s.height
    assert sample_cells(cells, 90, 50, seed=0).equals(s)  # fixed seed


def test_unpickler_refuses_code():
    bad = pickle.dumps(print)
    with pytest.raises(pickle.UnpicklingError):
        _SafeUnpickler(io.BytesIO(bad)).load()
    assert _SafeUnpickler(io.BytesIO(pickle.dumps({"a": np.float64(1.5)}))).load() == {"a": 1.5}


def test_paired_bootstrap_and_compare(tmp_path):
    rng = np.random.default_rng(0)
    n = 200
    age = rng.uniform(20, 80, n)
    donors = [f"d{i:03d}" for i in range(n)]
    good, bad = age + rng.normal(0, 2, n), age + rng.normal(0, 8, n)
    diff, lo, hi = paired_bootstrap(np.abs(good - age), np.abs(bad - age), np.arange(n) // 2)
    assert hi < 0 and lo < diff < hi
    oof = pl.DataFrame({"cohort": "c", "representation": "natural", "model": "geneformer-cls-ridge",
                        "donor": donors, "group": np.arange(n), "age": age, "pred": good})  # fmt: skip
    sc = pl.concat([
        pl.DataFrame({"representation": "natural", "model": m, "donor": donors, "age": age, "pred": p})
        for m, p in (("ridge", bad), ("enet", bad + 1), ("gbm", bad + 2), ("tabicl-2000g", good))
    ])  # fmt: skip
    sc.write_parquet(tmp_path / "sc_tabular_c.parquet")
    out = compare(oof, sc_dir=tmp_path, n_boot=200)
    assert set(out["ref"]) == {"pseudobulk-tabicl-2000g", "pseudobulk-ridge"}
    row = out.filter(pl.col("ref") == "pseudobulk-ridge").row(0, named=True)
    assert row["beats_ref"] and row["diff"] < 0
    assert out.filter(pl.col("ref") == "pseudobulk-tabicl-2000g")["diff"][0] == pytest.approx(0)
