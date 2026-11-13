"""RM28 arm ``universal``: which rows may enter a training batch (synthetic data)."""

import numpy as np
import polars as pl
import pytest
import torch

from rna_age.train import sc_universal as su
from rna_age.train.sc_pretrain import EVAL_COHORTS, ScMAE


def _bulk(n_per_study: int = 12) -> pl.DataFrame:
    rows = []
    for s, split in enumerate(["train", "train", "val", "test"]):
        for k in range(n_per_study):
            rows.append({"sample_id": f"s{s}_{k}", "study_id": f"S{s}", "split": split,
                         "age_type": "exact" if k != 0 else "binned", "age_years": 30.0 + k,
                         "group": "case" if k % 3 == 1 else "control", "category": "metabolic" if k % 3 == 1 else None,
                         "is_pseudobulk": False, "source": "archs4"})  # fmt: skip
    return pl.DataFrame(rows)


def _pool() -> tuple[pl.DataFrame, np.ndarray]:
    pool = pl.DataFrame({
        "dataset_id": ["p1"] * 6 + ["p2"] * 4,
        "donor_id": ["a", "a", "b", "b", "c", "c", "d", "d", "e", "e"],
        "age": [40.0, 40.0, 50.0, 50.0, 60.0, 60.0, 70.0, 70.0, 55.0, 55.0],
        "age_exact": [True] * 8 + [False] * 2,
    })  # fmt: skip
    is_val = np.array([False, False, False, False, True, True, False, False, False, False])
    return pool, is_val


def test_bulk_indices_never_train_on_val_or_test():
    t = _bulk()
    idx = su.bulk_indices(t)
    split = t["split"].to_numpy()
    for k in ("age_train", "head"):
        assert len(idx[k]) and (split[idx[k]] == "train").all()
    assert len(idx["pairs"]) and (split[idx["pairs"].ravel()] == "train").all()
    assert (t["group"].to_numpy()[idx["age_train"]] == "control").all()  # age only on controls
    assert (t["age_type"].to_numpy()[idx["age_train"]] == "exact").all()
    assert (split[idx["age_val"]] == "val").all()
    p = idx["pairs"]
    assert (t["group"].to_numpy()[p[:, 0]] == "case").all() and (t["group"].to_numpy()[p[:, 1]] == "control").all()
    assert (t["study_id"].to_numpy()[p[:, 0]] == t["study_id"].to_numpy()[p[:, 1]]).all()  # within study


def test_pool_donors_keep_validation_donors_out_and_refuse_eval_cohorts():
    pool, is_val = _pool()
    d = su.pool_donors(pool, is_val)
    assert d["train"][0] == ["p1:a", "p1:b", "p2:d"]  # c is a validation donor, e has no exact age
    assert d["val"][0] == ["p1:c"]
    assert [r.tolist() for r in d["train"][2]] == [[0, 1], [2, 3], [6, 7]]
    bad = pool.with_columns(dataset_id=pl.lit(EVAL_COHORTS[0]))
    with pytest.raises(ValueError):
        su.pool_donors(bad, is_val)


def test_universal_batches_use_training_rows_only():
    t = _bulk()
    pool, is_val = _pool()
    G = 5
    X = torch.rand(len(pool), G).half()
    bx = np.random.default_rng(0).random((t.height, G)).astype(np.float16)
    task = su.UniversalTask(X, pool, is_val, t, bx, emb=4, device="cpu", batch=16, donor_batch=4, cell_donors=2,
                            cells_per_donor=1)  # fmt: skip
    split = t["split"].to_numpy()
    train_donor_rows = set(np.concatenate(task.donors["train"][2]).tolist())
    for _ in range(50):
        b = task.sample()
        assert (split[b["bulk_age"]] == "train").all() and (split[b["head"]] == "train").all()
        assert (split[b["pairs"].ravel()] == "train").all()
        assert b["pool_pb"].max() < len(task.donors["train"][0])
    assert not train_donor_rows & set(np.flatnonzero(is_val).tolist())
    model = ScMAE(G, hidden=8, emb=4, depth=1, dropout=0.0)
    terms = task.loss(model)
    assert set(terms) == {"age", "rank", "health"} and all(torch.isfinite(v) for v in terms.values())
    task.total(terms).backward()
    assert task.heads.age.weight.grad is not None and model.inp.weight.grad is not None
    ev = task.evaluate(model)
    assert set(ev) == {"bulk_val_mae", "pool_val_pb_mae"}


def test_donor_pseudobulk_is_log_of_mean_cp10k():
    X = torch.log1p(torch.tensor([[1.0, 3.0], [3.0, 5.0], [10.0, 0.0]]))
    pb = su.donor_pseudobulk(X, [np.array([0, 1]), np.array([2])])
    assert torch.allclose(pb, torch.log1p(torch.tensor([[2.0, 4.0], [10.0, 0.0]])), atol=1e-6)
