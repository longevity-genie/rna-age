"""RM25: the nearest-study prompt."""

import numpy as np

from rna_age.train.tabfm import nearest_study_context


def test_nearest_study_context_orders_studies_by_profile_and_caps_them():
    rng = np.random.default_rng(0)
    centers = {"A": 0.0, "B": 5.0, "C": 10.0, "Q": 9.0}
    study, Z = [], []
    for s, c in centers.items():
        n = 40 if s != "Q" else 10
        study += [s] * n
        Z.append(rng.normal(c, 0.1, (n, 3)))
    study, Z = np.array(study), np.vstack(Z)
    pool = np.flatnonzero(study != "Q")
    query = np.flatnonzero(study == "Q")
    ctx = nearest_study_context(Z, study, pool, query, n_context=50, cap_per_study=30, seed=0)
    got = study[ctx]
    assert (got == "C").sum() == 30 and (got == "B").sum() == 20 and (got == "A").sum() == 0
    assert not np.isin(ctx, query).any()
