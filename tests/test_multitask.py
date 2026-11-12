"""RM20: the multitask network on synthetic data (units, splits, losses, metrics, arms)."""

import h5py
import numpy as np
import polars as pl
import torch

from rna_age.train.multitask import run, shuffle_coef, split
from rna_age.train.multitask_data import Rm20Data, build_units, read_logcpm_rows
from rna_age.train.multitask_eval import deg_recall, delta_metrics
from rna_age.train.multitask_model import Config, rank_loss


def _synthetic(seed: int = 0, n_genes: int = 30, n_targets: int = 40, n_donors: int = 120) -> tuple[Rm20Data, pl.DataFrame]:
    rng = np.random.default_rng(seed)
    w_age = rng.normal(0, 1, n_genes)
    age = rng.uniform(20, 80, n_donors)
    X = (np.outer((age - 50) / 17, w_age) + rng.normal(0, 0.5, (n_donors, n_genes))).astype(np.float32)
    contexts = [f"d{d}|{c}" for d in range(2) for c in ("Rest", "Stim8hr")]
    controls = rng.normal(0, 0.3, (len(contexts), n_genes)).astype(np.float32)
    feats, rows, deltas = {}, [], []
    shared = rng.normal(0, 0.2, n_genes)  # the systematic perturbed-vs-control shift
    for t in range(n_targets):
        f = rng.normal(0, 1, 2 * n_genes).astype(np.float32)
        feats[f"G{t}"] = f
        eff = f[:n_genes] * 0.5  # Δ is a function of the gene's features: learnable for unseen genes
        for i, c in enumerate(contexts):
            d, cond = c.split("|")
            rows.append({"gene": f"G{t}", "gene_id": f"E{t}", "context": c, "ctx": i, "donor": d, "condition": cond,
                         "n_profiles": 1})  # fmt: skip
            deltas.append(shared + eff + rng.normal(0, 0.05, n_genes))
    units = pl.DataFrame(rows)
    delta = np.array(deltas, dtype=np.float32)
    data = Rm20Data([f"E{i}" for i in range(n_genes)], X, age, np.arange(n_donors), contexts, controls, units, delta, feats)
    coef = pl.DataFrame({"regulator": [f"G{t}" for t in range(n_targets)] * 2, "celltype": ["Rest"] * n_targets
                         + ["Stim8hr"] * n_targets, "coef_mean": rng.normal(0, 1, 2 * n_targets)})  # fmt: skip
    return data, coef


def test_build_units_controls_and_delta():
    obs = pl.DataFrame({"donor_id": ["a"] * 4, "culture_condition": ["Rest"] * 4,
                        "guide_type": ["non-targeting", "non-targeting", "targeting", "targeting"],
                        "perturbed_gene_name": [None, None, "X", "X"], "perturbed_gene_id": [None, None, "E1", "E1"]})  # fmt: skip
    Y = np.array([[0, 2], [2, 2], [5, 0], [7, 2]], dtype=np.float32)
    contexts, controls, units, delta = build_units(obs, Y)
    assert contexts == ["a|Rest"] and controls.tolist() == [[1, 2]]
    assert units["gene"].to_list() == ["X"] and delta.tolist() == [[5, -1]]  # mean(5,7)-1, mean(0,2)-2


def test_read_logcpm_rows_csr_scattered(tmp_path):
    from scipy.sparse import csr_matrix

    X = np.array([[1, 0, 3], [0, 5, 0], [7, 0, 9], [2, 2, 0]], dtype=np.float32)
    p = tmp_path / "s.h5ad"
    with h5py.File(p, "w") as h:
        m = csr_matrix(X)
        g = h.create_group("X")
        g["data"], g["indices"], g["indptr"] = m.data, m.indices, m.indptr
        g.attrs["shape"] = X.shape
    shared = np.array([0, 2])  # file columns in the library size
    out = read_logcpm_rows(p, np.array([0, 2, 3]), shared, np.array([1]), chunk=2)
    expect = np.log1p(np.array([3 / 4, 9 / 16, 0]) * 1e6)
    assert np.allclose(out[:, 0], expect, rtol=1e-5)


def test_metrics_and_rank_loss():
    t = np.array([[3.0, 0, 0, -2], [0, 1, 0, 0]])
    assert deg_recall(t, t, k=2).tolist() == [1.0, 1.0]
    m = delta_metrics(t, t, np.zeros_like(t), k=2)
    assert abs(m["pearson_delta"] - 1) < 1e-9 and m["deg_recall50"] == 1.0
    s, c, cond = torch.tensor([0.0, 1.0]), torch.tensor([1.0, 2.0]), torch.tensor([0, 0])
    assert rank_loss(s, c, cond, 0.1) < rank_loss(-s, c, cond, 0.1)
    assert rank_loss(s, c, torch.tensor([0, 1]), 0.1) == 0  # no pair within one condition


def test_split_never_shares_genes_and_holds_a_context():
    data, _ = _synthetic()
    sp = split(data, 0.25, None, seed=0)
    g = data.units["gene"].to_numpy()
    assert not set(g[sp["train"]]) & set(g[sp["eval"]])
    assert not (data.units["context"].to_numpy()[sp["train"]] == sp["hold"]).any()


def test_shuffle_keeps_values_per_condition():
    c = pl.DataFrame({"condition": ["A"] * 5 + ["B"] * 5, "gene": list("abcdeabcde"), "coef_mean": np.arange(10.0)})
    s = shuffle_coef(c, 0)
    for cond in ("A", "B"):
        assert sorted(s.filter(pl.col("condition") == cond)["coef_mean"]) == sorted(
            c.filter(pl.col("condition") == cond)["coef_mean"])  # fmt: skip


def test_arms_run_and_delta_head_beats_the_mean_on_unseen_genes():
    data, coef = _synthetic()
    df = run(data, coef, Config(steps=300, batch=64, hidden=64, emb=16), w_rank=0.5, folds=3, device="cpu")
    assert set(df["arm"]) >= {"age", "age_delta", "age_delta_rank", "placebo", "mean_delta_context"}
    v = {(r["arm"], r["metric"], r["scope"]): r["value"] for r in df.iter_rows(named=True)}
    assert v[("age_delta", "pearson_delta_pert", "seen_ctx")] > 0.25  # learnt from gene features (mean: 0)
    assert v[("age_delta", "deg_recall50", "seen_ctx")] >= v[("mean_delta_context", "deg_recall50", "seen_ctx")]
    assert v[("age", "age_mae", "cv")] < 15
