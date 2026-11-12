"""RM18 step 1: the same-donor comparison of two clocks."""

import numpy as np
import polars as pl

from rna_age.train.clock_compare import _gap, compare_cohort


def test_compare_cohort_reads_accuracy_offset_and_gap_agreement():
    rng = np.random.default_rng(0)
    age = rng.uniform(20, 80, 400)
    shared = rng.normal(0, 4, 400)  # who is old for their age, seen by both clocks
    sc = age + shared + rng.normal(0, 1, 400)
    bulk = age + 10 + shared + rng.normal(0, 1, 400)  # same ranking, a level offset of 10 y
    r = compare_cohort(pl.DataFrame({"age": age, "sc": sc, "bulk": bulk}), ["bulk"])[0]
    assert abs(r["bias_bulk"] - 10) < 0.5 and abs(r["bias_sc"]) < 0.5
    assert r["mae_diff_lo"] > 5  # the offset makes the bulk clock's MAE clearly worse
    assert abs(r["r_sc"] - r["r_bulk"]) < 0.02  # but not its correlation
    assert r["gap_rho"] > 0.8


def test_gap_is_free_of_age():
    age = np.linspace(20, 80, 100)
    g = _gap(0.5 * age + 30 + np.sin(age), age)
    assert abs(np.corrcoef(g, age)[0, 1]) < 1e-6
