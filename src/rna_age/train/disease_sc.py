"""RM18 step 2: do the clocks see disease? Patients vs controls of the same single-cell cohort, per clock.

Every clock is trained outside the cohort it scores, so patients and controls are treated alike: the single-cell
clocks are ridge on one cell class (or the whole sample) of a large healthy blood cohort (default the 981 donors of
``3faad104``), the bulk clocks are RM18's ``BULK_RUNS`` on the cohort's whole-sample pseudobulk. A donor's gap is its
prediction minus the line the controls' predictions draw against age (fitted on the controls), so a level offset of a
clock on a new cohort cancels. The statistic: mean gap of patients minus mean gap of controls; its null: the disease
labels shuffled across donors (the control line refitted each time), two-sided.
"""

from __future__ import annotations

import logging

import numpy as np
import polars as pl

from rna_age.paths import OUTPUT
from rna_age.train.clock_compare import BULK_RUNS, bulk_on_pseudobulk
from rna_age.train.composition import _cpm, donor_profiles

log = logging.getLogger(__name__)

TRAIN = "3faad104-2ab8-4434-816d-474d8d2641db"
COHORTS = ("218acb0f-9f2f-4f76-b90b-15a4b7c7f629", "9dbab10c-118d-496b-966a-67f1763a6b7d")  # lupus, COVID-19
CLASSES = ("natural", "CD4 T", "CD8 T", "NK", "B", "monocyte")


def _profile(dp: dict, cls: str, min_cells: int) -> tuple[np.ndarray, np.ndarray]:
    """(counts donors × genes, usable mask) of one class, or of the whole sample (``natural``)."""
    if cls == "natural":
        return dp["P"].sum(1), np.ones(len(dp["donors"]), dtype=bool)
    if cls not in dp["cix"]:
        return np.zeros((len(dp["donors"]), len(dp["genes"]))), np.zeros(len(dp["donors"]), dtype=bool)
    k = dp["cix"][cls]
    return dp["P"][:, k], dp["N"][:, k] >= min_cells


def clock_predictions(train: dict, target: dict, cls: str, n_genes: int = 2000, min_cells: int = 20) -> np.ndarray:
    """Ridge on ``train``'s ``cls`` profiles, applied to ``target``'s; NaN where a target donor lacks the class.
    Library size over the genes both cohorts measure; the ``n_genes`` most variable in training."""
    from sklearn.linear_model import RidgeCV

    pos = {g: i for i, g in enumerate(target["genes"])}
    shared = [i for i, g in enumerate(train["genes"]) if g in pos]
    tcols = np.array([pos[train["genes"][i]] for i in shared])
    Ptr, ok = _profile(train, cls, min_cells)
    X = np.log1p(_cpm(Ptr[ok][:, shared]))
    keep = np.argsort(-X.var(0))[:n_genes]
    mu, sd = X[:, keep].mean(0), X[:, keep].std(0) + 1e-6
    model = RidgeCV(alphas=np.logspace(-1, 5, 13)).fit((X[:, keep] - mu) / sd, train["age"][ok])
    Pt, okt = _profile(target, cls, min_cells)
    out = np.full(len(target["donors"]), np.nan)
    if okt.any():
        xt = np.log1p(_cpm(Pt[okt][:, tcols]))[:, keep]
        out[okt] = model.predict((xt - mu) / sd)
    return out


def _diff(gap_pred: np.ndarray, age: np.ndarray, patient: np.ndarray) -> float:
    b = np.polyfit(age[~patient], gap_pred[~patient], 1)
    gap = gap_pred - np.polyval(b, age)
    return float(gap[patient].mean() - gap[~patient].mean())


def gap_test(pred: np.ndarray, age: np.ndarray, patient: np.ndarray, n_perm: int = 1000, seed: int = 0) -> dict:
    """Patients' mean age-adjusted gap minus controls', and the label-shuffle two-sided p."""
    ok = ~np.isnan(pred)
    pred, age, patient = pred[ok], age[ok], patient[ok]
    if min(patient.sum(), (~patient).sum()) < 5:  # the class is absent or too rare in this cohort
        return {"n_patients": int(patient.sum()), "n_controls": int((~patient).sum()), "gap_diff": None, "p": None}
    d = _diff(pred, age, patient)
    rng = np.random.default_rng(seed)
    null = np.array([_diff(pred, age, rng.permutation(patient)) for _ in range(n_perm)])
    p = (1 + (np.abs(null) >= abs(d)).sum()) / (1 + n_perm)
    return {"n_patients": int(patient.sum()), "n_controls": int((~patient).sum()), "gap_diff": d, "p": float(p)}


def run(cohorts: tuple[str, ...] = COHORTS, train_id: str = TRAIN, n_perm: int = 1000) -> pl.DataFrame:
    train = donor_profiles(train_id, "broad")
    bulk = {name: bulk_on_pseudobulk(r) for name, r in BULK_RUNS.items()}
    rows = []
    for cid in cohorts:
        tg = donor_profiles(cid, "broad")
        dis = tg["g"].group_by("donor").agg(pl.col("disease").first(), pl.col("tissue").first()).sort("donor")
        blood = (dis["tissue"] == "blood").to_numpy()
        patient = (dis["disease"] != "normal").to_numpy()
        disease = dis.filter(pl.col("disease") != "normal")["disease"].mode()[0]
        age = tg["age"]
        clocks = {f"sc:{c}": clock_predictions(train, tg, c) for c in CLASSES}
        for name, b in bulk.items():
            m = dict(
                zip(
                    *b.filter(pl.col("ds8") == cid[:8]).select("donor", "pred").to_dict(as_series=False).values(),
                    strict=True,
                )
            )
            clocks[f"bulk:{name}"] = np.array([m.get(d, np.nan) for d in tg["donors"]], dtype=float)
        for clock, pred in clocks.items():
            pr = np.where(blood, pred, np.nan)
            r = gap_test(pr, age, patient, n_perm)
            rows.append({"cohort": cid[:8], "disease": disease, "clock": clock, **r})
            log.info("%s %s %s: %s", cid[:8], disease, clock, r)
    out = pl.DataFrame(rows)
    out.write_parquet(OUTPUT / "summary" / "clock_compare_disease.parquet")
    return out
