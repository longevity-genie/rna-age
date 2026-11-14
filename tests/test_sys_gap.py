import numpy as np
import polars as pl

from rna_age.train.sys_gap import (
    CaseDef,
    Cohort,
    arms,
    assign_folds,
    bar_passes,
    coarse_type,
    combine_de,
    contrast,
    de_rows,
    disattenuate,
    first_visits,
    fit_fold,
    gaps,
    gene_halves,
    one_factor,
    oof_clock,
    predictions_pass,
    retest_reliability,
    score_predictions,
    spearman_brown,
    thin_halves,
    top_de_genes,
    verdict,
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


def _de_cohort(rng: np.random.Generator, n: int, planted: np.ndarray, genes: int = 300, name: str = "x") -> Cohort:
    age = rng.uniform(60, 95, n)
    disease = np.where(rng.random(n) < 0.6, "dementia || Alzheimer disease", "normal")
    sex = np.where(rng.random(n) < 0.5, "male", "female")
    base = rng.uniform(20, 200, genes)
    eff = np.zeros(genes)
    eff[planted] = np.where(np.arange(len(planted)) % 2 == 0, 1.0, -1.0)  # same direction in every cohort
    age_eff = np.zeros(genes)
    age_eff[-20:] = 0.02  # age genes: must not rank as DE, age is a covariate
    case = disease != "normal"
    mu = base * np.exp(np.outer(case, eff) + np.outer(age - 75, age_eff))
    C = rng.poisson(mu * rng.uniform(0.5, 2, n)[:, None])
    co = Cohort(name=name, donors=[f"d{i}" for i in range(n)], person=np.arange(n), age=age,
                disease=list(disease), sex=list(sex), batch=None, batch_name=None,
                genes=[f"G{j}" for j in range(genes)], age_censored=age > 90)  # fmt: skip
    co.types["astrocyte"] = (np.arange(n), C, np.full(n, 100.0))
    return co


def test_de_out_removes_the_planted_genes():
    rng = np.random.default_rng(8)
    planted = rng.choice(np.arange(200), 15, replace=False)
    a = de_rows(_de_cohort(rng, 80, planted, name="a"), r"alzheimer disease", "a")
    b = de_rows(_de_cohort(rng, 40, planted, name="b"), r"alzheimer disease", "b")
    de = pl.DataFrame(a + b)
    de = pl.concat([de, combine_de(de)])
    assert set(de["cohort"].unique().to_list()) == {"a", "b", "combined"}
    top = top_de_genes(de, "combined", "astrocyte", 15)
    assert set(top) == {f"G{j}" for j in planted}
    # the candidate mask built from them leaves none of the planted genes and keeps the rest
    genes = [f"G{j}" for j in range(300)]
    mask = ~np.isin(genes, top_de_genes(de, "combined", "astrocyte", 50))
    assert not mask[planted].any() and mask.sum() == 250
    # too few donors in an arm: no DE for that type
    assert de_rows(_de_cohort(rng, 10, planted), r"alzheimer disease", "c", min_arm=8) == []


def test_gene_halves_are_disjoint_cover_and_differ():
    splits = gene_halves(1001, 5, seed=0)
    assert len(splits) == 5
    for a, b in splits:
        assert len(np.intersect1d(a, b)) == 0
        assert np.array_equal(np.sort(np.concatenate([a, b])), np.arange(1001))
        assert abs(len(a) - len(b)) <= 1
    assert not np.array_equal(splits[0][0], splits[1][0])
    assert all(np.array_equal(x[0], y[0]) for x, y in zip(splits, gene_halves(1001, 5, seed=0), strict=True))


def test_condition_from_a_metadata_column():
    cmv = CaseDef("CMV", column="subject.cmv", positive=("Positive",), negative=("Negative",),
                  label_rx=r"cytomegalovirus infection")  # fmt: skip
    disease = ["cytomegalovirus infection", "normal", "normal", "cytomegalovirus infection || rheumatoid arthritis",
               "rheumatoid arthritis", "normal", "normal"]  # fmt: skip
    values = ["Positive", "Negative", "Equivocal", "Positive", "Negative", None, "Negative"]
    case, ctrl = arms(cmv, disease, values)
    assert case.tolist() == [True, False, False, False, False, False, False]  # RA donor leaves the contrast
    assert ctrl.tolist() == [False, True, False, False, False, False, True]  # equivocal / missing: neither arm
    # disease-label definitions are unchanged; no definition: controls are "normal", no cases
    ad = CaseDef("AD", r"alzheimer disease")
    case, ctrl = arms(ad, ["dementia || Alzheimer disease", "normal", "Lewy body dementia"])
    assert case.tolist() == [True, False, False] and ctrl.tolist() == [False, True, False]
    case, ctrl = arms(None, ["normal", "COVID-19"])
    assert not case.any() and ctrl.tolist() == [True, False]


def test_one_visit_per_subject_and_repeat_visits_share_the_fold():
    person = np.array([0, 0, 1, 2, 2, 2, 3])
    age = np.array([61.0, 60.0, 45.0, 70.0, 72.0, 70.0, 50.0])
    first = first_visits(person, age)
    assert first.tolist() == [False, True, True, True, False, False, True]  # lowest age; ties to the first row
    # repeat visits (not controls) of a control's person take that person's fold, not a random one
    rng = np.random.default_rng(0)
    person = np.repeat(np.arange(60), 2)
    age = np.tile([50.0, 53.0], 60) + np.repeat(np.arange(60), 2)
    first = first_visits(person, age)
    ctrl = first & (person < 45)
    fold = assign_folds(person, ctrl, 5, rng)
    for p in np.unique(person):
        assert len(set(fold[person == p])) == 1


def test_test_retest_recovers_a_stable_gap():
    rng = np.random.default_rng(9)
    n = 400
    person = np.repeat(np.arange(n), 2)
    age = np.repeat(rng.uniform(40, 80, n), 2) + np.tile([0.0, 1.5], n)
    trait = np.repeat(rng.normal(size=n), 2)  # a person's stable gap; noise var 1/3 -> retest r = 0.75
    gap = trait + rng.normal(0, np.sqrt(1 / 3), 2 * n)
    ctrl = np.repeat(rng.random(n) < 0.7, 2) & first_visits(person, age)
    gap[5] = np.nan  # a missing second visit drops that person
    r, m = retest_reliability(gap, person, age, ctrl)
    assert abs(r - 0.75) < 0.08
    assert m == ctrl.sum() - (1 if ctrl[4] else 0)
    r1, m1 = retest_reliability(gap[::2], person[::2], age[::2], ctrl[::2])  # one visit each: nothing to correlate
    assert m1 == 0 and np.isnan(r1)


def test_prediction_verdicts():
    assert verdict("accelerates", 0.5, 6.0, 1.0, 5.5) == "hold"
    assert verdict("accelerates", -6.0, -0.5, -5.5, -1.0) == "fail"
    assert verdict("accelerates", -1.9, 1.9, -1.5, 1.5) == "fail"  # equivalent to no acceleration
    assert verdict("accelerates", -1.0, 6.0, -0.5, 5.0) == "inconclusive"
    assert verdict("no_acceleration", -1.5, 2.2, -1.2, 1.8) == "hold"
    assert verdict("no_acceleration", 2.5, 8.0, 3.0, 7.0) == "fail"
    assert verdict("no_acceleration", -1.0, 5.0, -0.5, 4.0) == "inconclusive"
    assert verdict("no_acceleration", np.nan, 1.0, 0.0, 0.5) == "inconclusive"
    cd = CaseDef("CMV", predictions=(("CD8 T", "accelerates"), ("B", "no_acceleration"), ("NK", "exploratory")))
    rows = [{"type": "CD8 T", "delta_years": 3.0, "lo": 0.5, "hi": 6.0, "lo90": 1.0, "hi90": 5.5},
            {"type": "B", "delta_years": 0.1, "lo": -1.5, "hi": 1.6, "lo90": -1.2, "hi90": 1.3}]  # fmt: skip
    p = score_predictions(cd, rows)
    assert [x["verdict"] for x in p] == ["hold", "hold", "inconclusive"]  # NK absent
    assert predictions_pass(cd, rows)  # exploratory NK does not count
