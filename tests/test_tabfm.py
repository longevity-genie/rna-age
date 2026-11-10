import numpy as np
import polars as pl

from rna_age.train.tabfm import context_rows


def test_context_caps_each_study_and_total():
    allx = pl.DataFrame({"study_id": ["A"] * 50 + ["B"] * 5 + ["C"] * 30})
    is_train = np.ones(allx.height, bool)
    is_train[80:] = False  # last five of C are not train
    rows = context_rows(allx, is_train, cap_per_study=10, n_context=1000, seed=0)
    studies = allx["study_id"].to_numpy()[rows]
    assert (studies == "A").sum() == 10 and (studies == "B").sum() == 5 and (studies == "C").sum() == 10
    assert is_train[rows].all()
    assert len(context_rows(allx, is_train, cap_per_study=10, n_context=12, seed=0)) == 12
    assert np.array_equal(rows, context_rows(allx, is_train, cap_per_study=10, n_context=1000, seed=0))
