import numpy as np

from rna_age.train.sys_gap import (
    assign_folds,
    bar_passes,
    coarse_type,
    contrast,
    disattenuate,
    fit_fold,
    gaps,
    one_factor,
    oof_clock,
    spearman_brown,
    thin_halves,
)


def test_coarse_type_brain_and_blood():
    assert coarse_type("oligodendrocyte precursor cell") == "OPC"
    assert coarse_type("oligodendrocyte") == "oligodendrocyte"
    assert coarse_type("microglial cell") == "microglia"
    assert coarse_type("perivascular macrophage") == "other"
    assert coarse_type("VIP GABAergic cortical interneuron") == "inhibitory neuron"
    assert coarse_type("L2/3 intratelencephalic projecting glutamatergic neuron") == "excitatory neuron"
    assert coarse_type("Myeloid cells") == "monocyte"
    assert coarse_type("CD8-positive, alpha-beta T cell") == "CD8 T"


def test_bias_line_removes_age_trend():
    rng = np.random.default_rng(0)
    age = rng.uniform(20, 90, 400)
    pred = 30 + 0.5 * age + rng.normal(0, 4, 400)  # regression to the mean: slope < 1
    ctrl = np.ones(400, dtype=bool)
    g = gaps(pred, age, ctrl)
    assert abs(np.polyfit(age, pred - age, 1)[0]) > 0.4  # raw gap trends with age
    assert abs(np.polyfit(age, g["gap_z"], 1)[0]) < 1e-8
    assert abs(g["gap_z"].std() - 1) < 0.01
    assert abs(g["slope"] - 0.5) < 0.05
    assert np.allclose(g["accel"], g["gap"] / g["slope"])


def _synthetic_counts(n: int, rng: np.random.Generator, age: np.ndarray, genes: int = 60) -> np.ndarray:
    base = rng.uniform(5, 50, genes)
    eff = np.zeros(genes)
    eff[:10] = 0.02
    mu = base * np.exp(np.outer(age - 50, eff))
    return rng.poisson(mu * 20)


def test_cases_scored_by_one_fold_model_and_never_train():
    rng = np.random.default_rng(1)
    n = 150
    age = rng.uniform(20, 80, n)
    ctrl = np.arange(n) < 100
    person = np.arange(n)
    C = _synthetic_counts(n, rng, age)
    X = np.log1p(C / C.sum(1, keepdims=True) * 1e6)
    fold = assign_folds(person, ctrl, 5, np.random.default_rng(2))
    assert (fold >= 0).all() and set(fold[~ctrl]) <= set(range(5))
    (pred,) = oof_clock([X], age, ctrl, fold, n_genes=30)
    # a case's prediction is its own fold model's, trained on the other folds' controls
    for i in np.flatnonzero(~ctrl)[:5]:
        k = fold[i]
        f = fit_fold(X[ctrl & (fold != k)], age[ctrl & (fold != k)], 30)
        assert np.isclose(pred[i], f(X[i : i + 1])[0])
    X2 = X.copy()
    X2[~ctrl] = rng.normal(size=X2[~ctrl].shape) * 100  # cases changed: controls' predictions unchanged
    (pred2,) = oof_clock([X2], age, ctrl, fold, n_genes=30)
    assert np.allclose(pred[ctrl], pred2[ctrl])


def test_folds_keep_a_person_on_one_side():
    rng = np.random.default_rng(3)
    person = np.repeat(np.arange(40), 2)
    ctrl = person < 30
    fold = assign_folds(person, ctrl, 5, rng)
    for p in np.unique(person):
        assert len(set(fold[person == p])) == 1


def test_thinning_halves_sum_to_original():
    rng = np.random.default_rng(4)
    C = rng.poisson(30, size=(20, 50))
    a, b = thin_halves(C, rng)
    assert (a + b == C).all() and (a >= 0).all() and (b >= 0).all()
    assert abs(a.sum() / C.sum() - 0.5) < 0.01


def test_disattenuation_and_spearman_brown():
    rng = np.random.default_rng(5)
    n = 20000
    t1 = rng.normal(size=n)
    t2 = 0.6 * t1 + 0.8 * rng.normal(size=n)  # true r = 0.6
    rel1, rel2 = 0.5, 0.8
    o1 = t1 + rng.normal(0, np.sqrt(1 / rel1 - 1), n)
    o2 = t2 + rng.normal(0, np.sqrt(1 / rel2 - 1), n)
    r_obs = np.corrcoef(o1, o2)[0, 1]
    assert r_obs < 0.5
    assert abs(disattenuate(r_obs, rel1, rel2) - 0.6) < 0.03
    # two half-length measures with reliability r each: the whole has 2r/(1+r)
    assert abs(spearman_brown(1 / 3) - 0.5) < 1e-12
    assert np.isnan(disattenuate(0.3, -0.1, 0.8))


def test_one_factor_recovers_planted_shared_factor():
    rng = np.random.default_rng(6)
    n, lam = 3000, np.array([0.8, 0.7, 0.6, 0.5, 0.7, 0.6])
    u = rng.normal(size=n)
    Z = u[:, None] * lam + rng.normal(size=(n, len(lam))) * np.sqrt(1 - lam**2)
    Z[rng.random(Z.shape) < 0.05] = np.nan  # some donors lack some types
    f = one_factor(Z, np.ones(n, dtype=bool))
    ok = ~np.isnan(f["u"])
    assert abs(np.corrcoef(f["u"][ok], u[ok])[0, 1]) > 0.85
    assert np.all(np.abs(f["loading"] ** 2 - lam**2) < 0.15)  # PC1 leans high, within tolerance
    S = f["std"]
    assert np.allclose(np.nan_to_num(f["shared"] + f["specific"]), np.nan_to_num(S))


def test_contrast_recovers_planted_acceleration():
    rng = np.random.default_rng(7)
    n = 600
    case = rng.random(n) < 0.4
    ctrl = ~case
    age = np.where(case, rng.uniform(60, 90, n), rng.uniform(30, 90, n))  # cases older, as in AD
    sex = np.where(rng.random(n) < 0.5, "male", "female")
    batch = np.where(rng.random(n) < 0.5, "b1", "b2")
    accel = {"microglia": 6.0, "astrocyte": 0.0, "oligodendrocyte": 0.0, "OPC": 0.0}
    slope = {"microglia": 0.5, "astrocyte": 0.7, "oligodendrocyte": 0.4, "OPC": 0.3}
    preds = {}
    for t in accel:
        bio = age + accel[t] * case + rng.normal(0, 3, n)
        p = 20 + slope[t] * bio + 1.5 * (batch == "b2") + rng.normal(0, 1, n)
        p[rng.random(n) < 0.05] = np.nan
        preds[t] = p
    pairs = [(g, r) for g in ("microglia", "astrocyte") for r in ("oligodendrocyte", "OPC")]
    rows = contrast(preds, age, sex, batch, case, ctrl, np.arange(n), pairs, n_boot=200)
    by = {r["type"]: r for r in rows}
    assert by["microglia"]["lo"] < 6.0 < by["microglia"]["hi"]
    assert by["microglia"]["lo"] > 2
    assert abs(by["astrocyte"]["delta_years"]) < 1.5 and abs(by["OPC"]["delta_years"]) < 1.5  # planted nulls
    assert by["microglia - oligodendrocyte"]["lo"] > 0 and by["microglia - OPC"]["lo"] > 0
    assert bar_passes(rows)
    assert not bar_passes([r for r in rows if not r["type"].startswith("microglia")])
