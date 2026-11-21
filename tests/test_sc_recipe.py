import numpy as np

from rna_age.train import sc_recipe as sr


def _data(n=120, g=60, seed=0):
    rng = np.random.default_rng(seed)
    age = rng.uniform(20, 80, n)
    X = rng.normal(size=(n, g))
    X[:, 3] += (age - 50) / 10
    X[:, 7] -= (age - 50) / 12
    X[:, 10] *= 5  # variable, not age-related
    return X, age


def test_age_genes_finds_age_columns_not_variable_ones():
    X, age = _data()
    top = set(sr.age_genes(X, age, 2).tolist())
    assert top == {3, 7}


def test_boot_units_shapes_and_mean():
    x = np.arange(12, dtype=np.float32).reshape(6, 2)
    donor = np.array(["a", "a", "a", "b", "b", "b"])
    U, own = sr.boot_units(x, donor, np.array(["a", "b"]), 1, 0, np.random.default_rng(0))
    assert U.shape == (2, 2) and own.tolist() == [0, 1]
    assert np.allclose(U[0], x[:3].mean(0))
    U, own = sr.boot_units(x, donor, np.array(["a", "b"]), 5, 2, np.random.default_rng(0))
    assert U.shape == (10, 2) and (own == 0).sum() == 5


def test_agesel_class_features_and_stack():
    X, age = _data()
    groups = np.arange(len(age))
    ok = np.ones(len(age), bool)
    ok[:10] = False
    blocks = [("A", X, ok), ("B", X[:, ::-1].copy(), np.ones(len(age), bool))]
    tr, te = np.arange(90), np.arange(90, 120)
    a, b = sr.agesel_class_features(blocks, tr, te, age, 4)
    assert a.shape == (90, 4) and b.shape == (30, 4)
    assert np.all(a[:10, :2] == 0)  # missing class → fold mean
    p = sr.stack_predict(X, blocks, tr, te, age, groups, "ridge", 20, 0)
    assert np.corrcoef(p, age[te])[0, 1] > 0.7
