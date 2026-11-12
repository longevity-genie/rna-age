"""RM21: unseen-knockdown CV on synthetic embeddings: a target tied to the embedding is recovered by ridge and
beats the mean and the shuffled-embedding ridge; a target of pure noise beats neither."""

import numpy as np
import polars as pl

from rna_age.train.unseen_kd import evaluate, folds, parse_uniprot, pool_windows


def _data(signal: float, n: int = 400, d: int = 40, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    y = signal * X[:, :3].sum(1) + rng.normal(size=n)
    genes = [f"G{i}" for i in range(n)]
    emb = pl.DataFrame({"gene": genes, "emb": pl.Series(list(X), dtype=pl.Array(pl.Float32, d))})
    tg = pl.DataFrame({"target": "clock", "culture_condition": "Rest", "perturbed_gene_name": genes + ["NONCODING"],
                       "y": list(y) + [0.0]})  # fmt: skip
    return emb, tg


def test_signal_beats_the_bar_and_noise_does_not():
    emb, tg = _data(1.0)
    d = {r["model"]: r for r in evaluate(emb, tg, ("ridge", "pca_ridge"), n_boot=200, n_pcs=40).to_dicts()}
    assert set(d) == {"mean", "ridge_shuffled_emb", "ridge", "ridge_label_shuffle", "pca_ridge",
                      "pca_ridge_label_shuffle"}  # fmt: skip
    assert d["ridge"]["n_dropped_no_protein"] == 1 and d["ridge"]["n_genes"] == 400
    assert d["ridge"]["spearman"] > 0.7 and d["ridge"]["beats_bar"] and d["pca_ridge"]["beats_bar"]
    assert abs(d["ridge_shuffled_emb"]["spearman"]) < 0.2 and abs(d["ridge_label_shuffle"]["spearman"]) < 0.2
    assert not d["mean"]["beats_bar"] and not d["ridge_label_shuffle"]["beats_bar"]
    assert d["ridge"]["spearman_lo"] < d["ridge"]["spearman"] < d["ridge"]["spearman_hi"]
    emb, tg = _data(0.0, seed=1)
    d = {r["model"]: r for r in evaluate(emb, tg, ("ridge",), n_boot=200).to_dicts()}
    assert not d["ridge"]["beats_bar"]


def test_folds_balanced_and_windows_cover():
    f = folds(103, 5, 0)
    assert sorted(np.bincount(f).tolist()) == [20, 20, 21, 21, 21]
    w = pool_windows(2500, 1022)
    assert w[0][0] == 0 and w[-1][1] == 2500 and all(b - a <= 1022 for a, b in w) and len(w) == 3
    assert pool_windows(10, 1022) == [(0, 10)]


def test_uniprot_primary_wins_synonym_only_when_unique():
    raw = (
        b"Entry\tGene Names (primary)\tGene Names (synonym)\tSequence\n"
        b"P1\tAAA\tOLD1 SHARED\tMKV\n"
        b"P2\tAAA\t\tMKVLLL\n"  # second entry with the same primary: the longer wins
        b"P3\tBBB\tSHARED AAA\tMQQ\n"  # AAA as synonym never overrides a primary
    )
    t = {r["gene"]: r for r in parse_uniprot(raw).to_dicts()}
    assert t["AAA"]["accession"] == "P2" and t["AAA"]["via"] == "primary"
    assert t["OLD1"]["accession"] == "P1" and t["OLD1"]["via"] == "synonym"
    assert "SHARED" not in t
