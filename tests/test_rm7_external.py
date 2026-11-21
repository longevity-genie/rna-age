import numpy as np

from rna_age.train.rm7_external import study_bootstrap


def test_study_bootstrap_point_values():
    rng = np.random.default_rng(0)
    age = rng.uniform(20, 80, 200)
    pred = age + rng.normal(0, 5, 200)
    study = np.repeat(np.arange(20), 10)
    m = study_bootstrap(np.abs(pred - age), age, pred, study, n_boot=200)
    assert abs(m["mae"] - np.abs(pred - age).mean()) < 1e-12
    assert m["mae_lo"] <= m["mae"] <= m["mae_hi"] and m["r_lo"] <= m["r"] <= m["r_hi"]
