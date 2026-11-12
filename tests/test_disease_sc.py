"""RM18 step 2: the patients-vs-controls gap test."""

import numpy as np

from rna_age.train.disease_sc import gap_test


def test_gap_test_finds_an_older_patient_group_despite_a_clock_offset():
    rng = np.random.default_rng(0)
    age = rng.uniform(20, 80, 200)
    patient = np.arange(200) < 80
    pred = 0.6 * age + 25 + 4.0 * patient + rng.normal(0, 3, 200)  # compressed, offset clock; patients +4 y
    r = gap_test(pred, age, patient, n_perm=500)
    assert 3 < r["gap_diff"] < 5 and r["p"] < 0.01
    null = gap_test(0.6 * age + 25 + rng.normal(0, 3, 200), age, patient, n_perm=500)
    assert null["p"] > 0.05


def test_gap_test_drops_donors_without_a_prediction():
    age = np.linspace(20, 80, 60)
    pred = age.copy()
    pred[:5] = np.nan
    r = gap_test(pred, age, np.arange(60) % 2 == 0, n_perm=50)
    assert r["n_patients"] + r["n_controls"] == 55
