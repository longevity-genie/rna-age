import numpy as np

from rna_age.train.sys_gap_meta import dersimonian_laird


def test_dl_equal_studies_is_fixed_effect():
    m = dersimonian_laird(np.array([0.5, 0.5]), np.array([0.2, 0.2]))
    assert abs(m["mu"] - 0.5) < 1e-9 and m["tau2"] == 0.0
    assert abs(m["se"] - 0.2 / np.sqrt(2)) < 1e-9


def test_dl_heterogeneity_widens():
    m = dersimonian_laird(np.array([0.0, 2.0]), np.array([0.1, 0.1]))
    assert m["tau2"] > 0 and m["se"] > 0.1 / np.sqrt(2) and m["i2"] > 0.9
