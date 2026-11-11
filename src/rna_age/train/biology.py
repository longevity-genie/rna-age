"""Does the disease age gap mean biology? (RM10, plan line 2, desk #19.)

Each test is written down in :data:`REGISTRY` **before it is run**: a held-out study, a covariate read from its GEO
characteristics, the covariate's order, and the expected direction of its association with the age gap. The
registry is committed alone first, so a test's expectation predates its result (``git log`` shows it).

The gap is the within-study residual of the model's prediction regressed on chronological age (so a covariate that
tracks age, such as Braak stage, cannot pass through age alone). Per test: Spearman's rho between the gap and the
ordinal covariate over the study's samples with a mapped value, and a permutation p (covariate shuffled within the
study) in the registered direction; direction 0 (a negative control) is tested two-sided and passes when it shows
nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from rna_age.paths import INPUT, OUTPUT
from rna_age.sources.archs4 import split_items
from rna_age.train.features import sample_table


@dataclass(frozen=True)
class GapTest:
    name: str
    study: str
    key: str  # characteristics key, lower-case
    order: dict[str, float]  # value (lower-case, stripped) -> ordinal; values not listed are left out
    direction: int  # +1: a higher covariate means an older gap; -1: younger; 0: no association expected
    why: str


REGISTRY: list[GapTest] = [
    # --- severity within a disease (expected: worse disease, older gap) ---
    GapTest("covid_severity", "GSE169687", "disease severity",
         {"healthy": 0, "mild": 1, "moderate": 2, "severe": 3, "critical": 4}, +1,
         "COVID-19 convalescents, blood: worse acute disease, older"),
    GapTest("viral_severity", "GSE99992", "severity", {"non-severe": 0, "severe": 1}, +1,
         "acute viral infection, blood: severe older than non-severe"),
    GapTest("lung_severity", "GSE213001", "severity", {"moderate": 1, "severe": 2, "advanced": 3}, +1,
         "chronic lung disease, whole lung: more advanced, older"),
    GapTest("liver_fibrosis_a", "GSE162694", "fibrosis stage",
         {"normal liver histology": 0, "0": 0, "1": 1, "2": 2, "3": 3, "4": 4}, +1,
         "NAFLD liver: higher fibrosis stage, older"),
    GapTest("liver_fibrosis_b", "GSE174478", "fibrosis stage", {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4}, +1,
         "NAFLD liver, a second cohort: higher fibrosis stage, older"),
    GapTest("liver_nas", "GSE174478", "nafld activity score", {str(i): i for i in range(9)}, +1,
         "NAFLD activity score: more active disease, older"),
    GapTest("braak_ad", "GSE95587", "braak score", {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6}, +1,
         "Alzheimer fusiform gyrus: higher Braak stage, older (age regressed out)"),
    GapTest("lewy_stage", "GSE216281", "braak lewy_body_stage", {str(i): i for i in range(7)}, +1,
         "Parkinson frontal cortex: higher Lewy body stage, older"),
    GapTest("mmse", "GSE243743", "mmse", {str(i): i for i in range(31)}, -1,
         "neutrophils: lower MMSE (worse cognition), older"),
    GapTest("ra_activity", "GSE117769", "disease activity", {"remission": 0, "low": 1, "moderate": 2, "high": 3}, +1,
         "rheumatoid arthritis, blood: higher disease activity, older"),
    GapTest("heart_failure_nyha", "GSE53080", "nyha",
         {"ii-iii": 2.5, "iii": 3, "iii-iv": 3.5, "iv": 4, "4": 4}, +1,
         "failing myocardium: higher NYHA class, older"),
    GapTest("lvad_unloading", "GSE53080", "lvad status",
         {"no lvad": 0, "3 months lvad support": 1, "6 months lvad support": 2, "explantation": 3}, -1,
         "failing myocardium: longer mechanical unloading (reverse remodelling), younger"),
    # --- negative control ---
    GapTest("acute_exercise_control", "GSE202295", "timepoint", {"basal": 0, "post": 1, "recovery": 2}, 0,
         "skeletal muscle before, just after and 3 h after one 30-min cycling bout: no change in biological age"),
]  # fmt: skip


@dataclass(frozen=True)
class PairedTest:
    """Before/after the same intervention. ``delta`` = prediction after − before, within a person when ``subject``
    names them (a characteristics key, or ``title:<regex>`` with one group), else between the two groups' means.
    Age needs no adjustment within a person over weeks or months."""

    name: str
    study: str
    time_key: str
    before: tuple[str, ...]
    after: tuple[str, ...]
    direction: int  # -1: younger after; +1: older after; 0: no change expected
    why: str
    subject: str | None = None
    where: tuple[tuple[str, str], ...] = ()  # (key, value) filters, lower-case


PAIRED: list[PairedTest] = [
    PairedTest("bariatric_adipose", "GSE65540", "time", ("t0",), ("t3",), -1,
               "subcutaneous adipose 3 months after gastric bypass or banding: younger", subject=r"title:sat(\d+)"),
    PairedTest("weight_loss_saat", "GSE274137", "time", ("pre weight-loss",), ("post weight-loss",), -1,
               "subcutaneous abdominal adipose after a weight-loss programme: younger", subject=r"title:lip\.(\d+)",
               where=(("tissue type", "saat"),)),
    PairedTest("weight_loss_sfat", "GSE274137", "time", ("pre weight-loss",), ("post weight-loss",), -1,
               "subcutaneous femoral adipose after a weight-loss programme: younger", subject=r"title:lip\.(\d+)",
               where=(("tissue type", "sfat"),)),
    PairedTest("calorie_restriction_adipose", "GSE192587", "timepoint", ("before",), ("after",), -1,
               "adipose tissue after calorie restriction: younger", subject="participant",
               where=(("treatment", "calorie restriction"),)),
    PairedTest("protein_restriction_adipose", "GSE192587", "timepoint", ("before",), ("after",), -1,
               "adipose tissue after protein restriction: younger", subject="participant",
               where=(("treatment", "protein restriction"),)),
    PairedTest("bariatric_monocytes", "GSE66306", "time point", ("t0; before bariatric surgery",),
               ("t3; 3 months after bariatric surgery",), -1,
               "blood monocytes 3 months after bariatric surgery: younger (less inflammation)", subject="subject id"),
    PairedTest("rygb_muscle", "GSE137631", "group", ("rygb pre surgery", "rygb + exercise training pre surgery"),
               ("rygb post surgery", "rygb + exercise training post surgery"), -1,
               "vastus lateralis after gastric bypass (with or without training): younger; groups, no subject ids"),
    PairedTest("metformin_muscle", "GSE107894", "drug", ("placebo",), ("metformin",), -1,
               "vastus lateralis on metformin vs placebo, crossover within person: younger", subject="subject id",
               where=(("tissue", "vastus lateralis muscle"),)),
    PairedTest("metformin_adipose", "GSE107894", "drug", ("placebo",), ("metformin",), -1,
               "subcutaneous adipose on metformin vs placebo, crossover within person: younger", subject="subject id",
               where=(("tissue", "abdominal subcutaneous adipose"),)),
]  # fmt: skip


def covariates(test: GapTest) -> pl.DataFrame:
    """``sample_id``, ``x`` for the test's study: its characteristics value mapped through ``order``."""
    meta = pl.read_parquet(
        INPUT / "archs4" / "metadata_all.parquet", columns=["geo_accession", "characteristics_ch1", "series_id"]
    ).filter(pl.col("series_id").str.contains(rf"\b{test.study}\b"))
    rows = []
    for gsm, chars in meta.select("geo_accession", "characteristics_ch1").iter_rows():
        d = {k.strip().lower(): v.strip().lower() for k, v in split_items(chars or "")}
        v = d.get(test.key)
        if v is not None and v in test.order:
            rows.append({"sample_id": gsm, "x": float(test.order[v])})
    return pl.DataFrame(rows, schema={"sample_id": pl.String, "x": pl.Float64})


def _rank(a: np.ndarray) -> np.ndarray:
    return pl.Series(a).rank("average").to_numpy()


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = _rank(x), _rank(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def gap_residual(pred: np.ndarray, age: np.ndarray) -> np.ndarray:
    """Prediction regressed on age within the study; the residual is the gap."""
    if np.std(age) == 0:
        return pred - pred.mean()
    b = np.polyfit(age, pred, 1)
    return pred - np.polyval(b, age)


def run_test(test: GapTest, pred: pl.DataFrame, n_perm: int = 2000, seed: int = 0) -> dict:
    """One registered test on one run's predictions (``sample_id``, ``pred``, ``age_years``)."""
    d = covariates(test).join(pred, on="sample_id", how="inner").drop_nulls(["pred", "age_years"])
    out = {"test": test.name, "study": test.study, "direction": test.direction, "n": d.height,
           "levels": d["x"].n_unique() if d.height else 0}  # fmt: skip
    if d.height < 10 or out["levels"] < 2:
        return out | {"rho": float("nan"), "p": float("nan"), "verdict": "too few"}
    gap = gap_residual(d["pred"].to_numpy(), d["age_years"].to_numpy())
    x = d["x"].to_numpy()
    rho = spearman(x, gap)
    rng = np.random.default_rng(seed)
    null = np.array([spearman(rng.permutation(x), gap) for _ in range(n_perm)])
    if test.direction == 0:
        p = float((1 + (np.abs(null) >= abs(rho)).sum()) / (1 + n_perm))
        verdict = "as expected (nothing)" if p >= 0.05 else "unexpected change"
    else:
        s = test.direction
        p = float((1 + (s * null >= s * rho).sum()) / (1 + n_perm))
        verdict = "as expected" if p < 0.05 else ("opposite" if s * rho < 0 else "not significant")
    return out | {"rho": rho, "p": p, "verdict": verdict}


def run_all(run_ids: list[str], n_perm: int = 2000) -> pl.DataFrame:
    eligible = sample_table().select("sample_id", "age_years")
    rows = []
    for run_id in run_ids:
        p = pl.read_parquet(OUTPUT / "runs" / run_id / "predictions.parquet", columns=["sample_id", "pred"])
        p = p.join(eligible, on="sample_id", how="inner")  # samples eligible in the current table only
        for t in REGISTRY:
            rows.append({"run_id": run_id, **run_test(t, p, n_perm)})
    out = pl.DataFrame(rows)
    out.write_parquet(OUTPUT / "summary" / "biology_tests.parquet")
    return out


def paired_frame(test: PairedTest) -> pl.DataFrame:
    """``sample_id``, ``subject`` (or null), ``after`` (bool) for the test's samples, from their characteristics."""
    import re

    meta = pl.read_parquet(
        INPUT / "archs4" / "metadata_all.parquet",
        columns=["geo_accession", "characteristics_ch1", "series_id", "title"],
    ).filter(pl.col("series_id").str.contains(rf"\b{test.study}\b"))
    rows = []
    for gsm, chars, title in meta.select("geo_accession", "characteristics_ch1", "title").iter_rows():
        d = {k.strip().lower(): v.strip().lower() for k, v in split_items(chars or "")}
        if any(d.get(k) != v for k, v in test.where):
            continue
        v = d.get(test.time_key)
        if v not in test.before and v not in test.after:
            continue
        subj = None
        if test.subject and test.subject.startswith("title:"):
            m = re.search(test.subject[6:], (title or "").lower())
            subj = m.group(1) if m else None
        elif test.subject:
            subj = d.get(test.subject)
        rows.append({"sample_id": gsm, "subject": subj, "after": v in test.after})
    return pl.DataFrame(rows, schema={"sample_id": pl.String, "subject": pl.String, "after": pl.Boolean})


def paired_stat(d: pl.DataFrame, direction: int, n_perm: int = 5000, seed: int = 0) -> dict:
    """``d``: ``subject`` (may be null), ``after``, ``pred``. Within subjects: the mean of (after − before) per
    subject, sign-flip permutation; without: difference of group means, label permutation. p in ``direction``
    (two-sided for 0)."""
    rng = np.random.default_rng(seed)
    paired = d["subject"].null_count() == 0 and d.height > 0
    if paired:
        per = d.group_by("subject", "after").agg(pl.col("pred").mean()).pivot(on="after", index="subject",
                                                                               values="pred")  # fmt: skip
        if "true" not in per.columns or "false" not in per.columns:
            return {"design": "paired", "n": 0, "delta": float("nan"), "p": float("nan")}
        diff = (per["true"] - per["false"]).drop_nulls().to_numpy()
        obs = float(diff.mean()) if len(diff) else float("nan")
        null = np.array([(diff * rng.choice([-1, 1], len(diff))).mean() for _ in range(n_perm)])
        n = len(diff)
    else:
        a, x = d["after"].to_numpy(), d["pred"].to_numpy()
        obs = float(x[a].mean() - x[~a].mean()) if a.any() and (~a).any() else float("nan")
        null = np.array([(lambda p: x[p].mean() - x[~p].mean())(rng.permutation(a)) for _ in range(n_perm)])
        n = d.height
    if not np.isfinite(obs) or n < 3:
        return {"design": "paired" if paired else "groups", "n": n, "delta": obs, "p": float("nan")}
    if direction == 0:
        p = float((1 + (np.abs(null) >= abs(obs)).sum()) / (1 + n_perm))
    else:
        p = float((1 + (direction * null >= direction * obs).sum()) / (1 + n_perm))
    return {"design": "paired" if paired else "groups", "n": n, "delta": obs, "p": p}


def run_paired_all(run_ids: list[str], name: str = "model", n_perm: int = 5000) -> pl.DataFrame:
    """The registered paired tests on the mean external prediction of ``run_ids``."""
    parts = [pl.read_parquet(OUTPUT / "runs" / r / "predictions_external.parquet") for r in run_ids]
    pred = pl.concat(parts).group_by("sample_id").agg(pl.col("pred").mean())
    rows = []
    for t in PAIRED:
        d = paired_frame(t).join(pred, on="sample_id", how="inner")
        s = paired_stat(d, t.direction, n_perm)
        verdict = ("too few" if not np.isfinite(s["p"]) else "as expected" if s["p"] < 0.05 and t.direction
                   else "opposite" if t.direction and s["delta"] * t.direction < 0 else "not significant")  # fmt: skip
        rows.append({"model": name, "test": t.name, "study": t.study, "direction": t.direction, **s,
                     "verdict": verdict})  # fmt: skip
    out = pl.DataFrame(rows)
    out.write_parquet(OUTPUT / "summary" / f"biology_paired_{name}.parquet")
    return out
