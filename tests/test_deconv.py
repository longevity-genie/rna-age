import numpy as np

from rna_age.train.deconv import fractions, markers


def test_fractions_recover_a_known_mixture():
    rng = np.random.default_rng(0)
    ref = rng.gamma(1.0, 50.0, size=(4, 300))
    w = np.array([[0.5, 0.2, 0.2, 0.1], [0.1, 0.1, 0.1, 0.7]])
    bulk = w @ (ref / ref.sum(axis=1, keepdims=True)) * 1e5
    assert np.allclose(fractions(bulk, ref), w, atol=1e-6)


def test_markers_pick_class_specific_genes():
    cpm = np.full((2, 6), 100.0)
    cpm[0, 1] = cpm[1, 4] = 5000.0
    m = markers(cpm, n_markers=1)
    assert set(m.tolist()) == {1, 4}
