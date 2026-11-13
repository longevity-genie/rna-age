"""RM33 stage 0: per-person age gaps by coarse cell type (system), shared vs type-specific, and AD in the brain.

Per cohort and coarse cell type (blood: ``composition.broad_class``; brain: ``BRAIN``), on the type's donor pseudobulk:

1. **Clock.** Ridge on log-CPM, the training fold's ``n_genes`` most variable genes standardised on the training fold,
   trained on control donors only (disease ``normal``; in healthy cohorts every donor). Controls follow the D7 folds
   (5-fold, grouped by person); cases (non-normal donors) are assigned at random, by person, to the same 5 folds and
   scored by that fold's model only, exactly like the controls.
2. **Gap.** A *linear* bias line of out-of-fold prediction on age, fitted on the controls; ``gap = pred - line(age)``,
   ``gap_z = gap / sd`` (the controls' residual SD); ``slope`` is the line's slope, ``sensitivity = slope / sd``
   (z per year), ``accel_years = gap / slope``. A type whose slope is not positive has no years scale.
3. **Reliability.** Each donor's counts are split by binomial thinning (p = 0.5, seeded) into two halves; both halves
   are scored by the donor's own fold model (no refit), each half gets its own control bias line, and the controls'
   half-gap correlation is Spearman-Brown corrected. Reliability < 0.5 means 'unmeasured', not null. Thinning splits
   molecules, not cells, so it misses cell-sampling noise: an upper bound on test-retest reliability.
4. **Decomposition** (controls, measured types only). Pairwise correlation of the types' z-gaps, disattenuated by
   ``r / sqrt(rel_a rel_b)``; a one-factor model, the first principal component of that (raw) correlation matrix,
   gives loadings ``l``; each donor's shared score ``u`` is the least-squares fit of ``l`` to its available z-gaps,
   ``shared = l * u`` and ``specific = z - shared``. ``shared_share = l**2`` (of the observed gap variance); PC1
   includes some unique variance, so with few types it leans high.
5. **Contrast** (cases vs controls within a cohort). Per type: ``accel_years ~ case + age + age² + sex [+ batch]``
   by OLS on the donors inside the arms' common age range; a donor bootstrap (persons resampled, the control bias
   line refitted in each draw) gives the CI; paired differences between types use the same draws.

Pre-registered go/no-go (ROADMAP RM33, stage 0): in PsychAD (37a17b78), microglia or astrocytes show a larger AD
acceleration than both oligodendrocytes and OPCs, with the bootstrap CI of each difference above 0. RADC (5e57cd50)
is replication only, SLE (218acb0f) descriptive.
Outputs (``data/output/summary/``): ``sys_gap_donor``, ``sys_gap_type``, ``sys_gap_contrast``, ``sys_gap_corr``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import numpy as np
import polars as pl
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold

from rna_age.paths import INPUT, OUTPUT
from rna_age.sources.bgee import stage_age
from rna_age.train.composition import _cpm, broad_class

log = logging.getLogger(__name__)

SUMMARY = OUTPUT / "summary"
COHORTS = (
    "3faad104-2ab8-4434-816d-474d8d2641db",
    "c838aec3-03ef-4398-b882-0e3912abfff0",
    "218acb0f-9f2f-4f76-b90b-15a4b7c7f629",
    "terekhova2023",
    "37a17b78-4864-4a42-b67b-31c00962795a",
    "5c97eeeb-7e52-44b3-b010-b832b1f5424c",
    "5e57cd50-8e42-42d6-940d-5c1660d06864",
)
# Coarse brain classes, first match wins (OPC before oligodendrocyte). Perivascular macrophages, T, NK, B: "other".
BRAIN = [
    ("OPC", r"oligodendrocyte precursor"),
    ("oligodendrocyte", r"oligodendrocyte"),
    ("astrocyte", r"astrocyte"),
    ("microglia", r"microglial"),
    ("excitatory neuron", r"glutamatergic"),
    ("inhibitory neuron", r"gabaergic|interneuron"),
    ("vascular", r"endothelial|pericyte|smooth muscle|leptomeningeal"),
]
# Case definitions per cohort: (condition name, regex on the disease label). Controls are ``normal``; donors with any
# other label (other dementias, schizophrenia, ...) are left out of the contrast. A donor whose labels include
# Alzheimer disease among others counts as AD.
CONDITIONS = {
    "37a17b78-4864-4a42-b67b-31c00962795a": ("Alzheimer disease", r"alzheimer disease"),
    "5e57cd50-8e42-42d6-940d-5c1660d06864": ("Alzheimer disease", r"alzheimer disease"),
    "218acb0f-9f2f-4f76-b90b-15a4b7c7f629": ("systemic lupus erythematosus", r"systemic lupus erythematosus"),
}
BAR_COHORT = "37a17b78-4864-4a42-b67b-31c00962795a"
GLIA, REFS = ("microglia", "astrocyte"), ("oligodendrocyte", "OPC")
BATCH_COLUMNS = ("Processing_Cohort",)  # donor-level batch read from donor_obs/<id>.mixed.parquet (D9)
ALPHAS = np.logspace(-1, 5, 13)


def coarse_type(cell_type: str | None) -> str:
    s = (cell_type or "").lower()
    for name, rx in BRAIN:
        if re.search(rx, s):
            return name
    if re.search(r"myeloid", s):  # Terekhova's "Myeloid cells" (mostly monocytes)
        return "monocyte"
    return broad_class(cell_type)


# ---------------------------------------------------------------- data


@dataclass
class Cohort:
    name: str
    donors: list[str]
    person: np.ndarray  # CV group per donor (several visits of one person share it)
    age: np.ndarray
    disease: list[str]
    sex: list[str]
    batch: list[str] | None
    batch_name: str | None
    types: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = field(default_factory=dict)  # idx, counts, n_cells


def _cohort_dir(dataset_id: str):
    d = INPUT / "census_pseudobulk" / "by_dataset_celltype" / dataset_id
    return d if d.exists() else INPUT / dataset_id / "by_dataset_celltype" / dataset_id


def donor_batch(dataset_id: str, donor_ids: list[str]) -> tuple[list[str] | None, str | None]:
    """Each donor's majority (by cells) value of a known batch column in ``donor_obs/<id>.mixed.parquet``."""
    p = INPUT / "census_pseudobulk" / "donor_obs" / f"{dataset_id}.mixed.parquet"
    if not p.exists():
        return None, None
    m = pl.read_parquet(p)
    for col in BATCH_COLUMNS:
        b = m.filter(pl.col("column") == col)
        if b.height:
            top = b.sort("n_cells", descending=True).unique("donor_id", keep="first", maintain_order=True)
            lookup = dict(zip(top["donor_id"].to_list(), top["value"].to_list(), strict=True))
            return [str(lookup.get(d, "NA")) for d in donor_ids], col
    return None, None


def load_cohort(dataset_id: str, min_cells: int = 20) -> Cohort:
    """Donor metadata and, per coarse type, the summed counts of donors with at least ``min_cells`` cells."""
    d = _cohort_dir(dataset_id)
    g = pl.read_parquet(d / "groups.parquet")
    if "sex" not in g.columns and "Sex" in g.columns:
        g = g.with_columns(sex=pl.col("Sex").cast(pl.Utf8))
    ages = [stage_age(s) for s in g["development_stage"].to_list()]
    g = g.with_columns(
        age=pl.Series([a[1] if a else None for a in ages], dtype=pl.Float64),
        ctype=pl.Series([coarse_type(c) for c in g["cell_type"].to_list()]),
        donor=pl.col("donor_id").cast(pl.Utf8) + ":" + pl.col("tissue"),
    ).filter(pl.col("age").is_not_null())
    if "cv_group" not in g.columns:
        g = g.with_columns(cv_group=pl.col("donor"))
    meta = (
        g.group_by("donor")
        .agg(
            pl.col("donor_id").first(),
            pl.col("age").first(),
            pl.col("disease").first(),
            pl.col("sex").first().cast(pl.Utf8),
            pl.col("cv_group").first().cast(pl.Utf8),
        )  # fmt: skip
        .sort("donor")
    )
    donors = meta["donor"].to_list()
    pid = {p: i for i, p in enumerate(sorted(set(meta["cv_group"].to_list())))}
    batch, batch_name = donor_batch(dataset_id, [str(x) for x in meta["donor_id"].to_list()])
    co = Cohort(
        name=dataset_id,
        donors=donors,
        person=np.array([pid[p] for p in meta["cv_group"].to_list()]),
        age=meta["age"].to_numpy(),
        disease=[str(x) for x in meta["disease"].to_list()],
        sex=[str(x) for x in meta["sex"].to_list()],
        batch=batch,
        batch_name=batch_name,
    )
    dix = {x: i for i, x in enumerate(donors)}
    for t in sorted(set(g["ctype"].to_list()) - {"other"}):
        gt = g.filter(pl.col("ctype") == t)
        ncell = gt.group_by("donor").agg(pl.col("n_cells").sum())
        keep = set(ncell.filter(pl.col("n_cells") >= min_cells)["donor"].to_list())
        gt = gt.filter(pl.col("donor").is_in(keep))
        if gt.height == 0:
            continue
        M = pl.read_parquet(d / "counts.parquet", columns=gt["sample_id"].to_list()).to_numpy()  # genes × groups
        tdon = sorted(keep)
        tix = {x: i for i, x in enumerate(tdon)}
        rows = np.array([tix[x] for x in gt["donor"].to_list()])
        order = np.argsort(rows, kind="stable")
        starts = np.flatnonzero(np.r_[True, np.diff(rows[order]) != 0])
        C = np.add.reduceat(M.T[order], starts, axis=0, dtype=np.int64)  # donors (sorted) × genes
        del M
        nc = np.zeros(len(tdon))
        np.add.at(nc, rows, gt["n_cells"].to_numpy().astype(float))
        co.types[t] = (np.array([dix[x] for x in tdon]), C, nc)
    return co


# ---------------------------------------------------------------- clock


def log_cpm(C: np.ndarray) -> np.ndarray:
    return np.log1p(_cpm(np.asarray(C, dtype=np.float64)))


def thin_halves(C: np.ndarray, rng: np.random.Generator, p: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Binomial thinning: each count split into two halves that sum to it."""
    a = rng.binomial(np.asarray(C, dtype=np.int64), p)
    return a, C - a


def assign_folds(person: np.ndarray, is_ctrl: np.ndarray, n_folds: int, rng: np.random.Generator) -> np.ndarray:
    """D7 folds (GroupKFold by person) for the controls; cases' persons drawn uniformly at random into a fold."""
    fold = np.full(len(person), -1)
    ci = np.flatnonzero(is_ctrl)
    for k, (_, te) in enumerate(GroupKFold(n_splits=n_folds).split(ci, groups=person[ci])):
        fold[ci[te]] = k
    case_persons = np.unique(person[~is_ctrl])
    draw = dict(zip(case_persons, rng.integers(0, n_folds, len(case_persons)), strict=True))
    for i in np.flatnonzero(~is_ctrl):
        fold[i] = draw[person[i]]
    return fold


def fit_fold(Xtr: np.ndarray, y: np.ndarray, n_genes: int):
    """Ridge on the training rows' most variable columns, standardised on them; returns a predict function."""
    keep = np.argsort(-Xtr.var(0), kind="stable")[: min(n_genes, Xtr.shape[1])]
    mu, sd = Xtr[:, keep].mean(0), Xtr[:, keep].std(0) + 1e-6
    m = RidgeCV(alphas=ALPHAS).fit((Xtr[:, keep] - mu) / sd, y)
    return lambda X: m.predict((X[:, keep] - mu) / sd)


def oof_clock(
    Xs: list[np.ndarray], age: np.ndarray, is_ctrl: np.ndarray, fold: np.ndarray, n_genes: int
) -> list[np.ndarray]:
    """Fold k's model (trained on the controls outside fold k, from ``Xs[0]``) scores every row of fold k in every
    matrix of ``Xs`` (full depth, then the thinned halves). Cases never train."""
    out = [np.full(len(age), np.nan) for _ in Xs]
    for k in np.unique(fold):
        tr = is_ctrl & (fold != k)
        te = fold == k
        f = fit_fold(Xs[0][tr], age[tr], n_genes)
        for o, X in zip(out, Xs, strict=True):
            o[te] = f(X[te])
    return out


# ---------------------------------------------------------------- gap


def bias_line(pred: np.ndarray, age: np.ndarray, ctrl: np.ndarray) -> tuple[float, float, float]:
    """Linear line of prediction on age on the controls: (intercept, slope, residual SD)."""
    b, a = np.polyfit(age[ctrl], pred[ctrl], 1)
    resid = pred[ctrl] - (a + b * age[ctrl])
    return float(a), float(b), float(resid.std(ddof=2))


def gaps(pred: np.ndarray, age: np.ndarray, ctrl: np.ndarray) -> dict[str, np.ndarray | float]:
    a, b, sd = bias_line(pred, age, ctrl)
    gap = pred - (a + b * age)
    accel = gap / b if b > 0 else np.full_like(gap, np.nan)
    return {"gap": gap, "gap_z": gap / sd, "accel": accel, "intercept": a, "slope": b, "sd": sd}


def spearman_brown(r: float) -> float:
    return 2 * r / (1 + r) if r > -1 else float("nan")


def disattenuate(r: float, rel_a: float, rel_b: float) -> float:
    return r / np.sqrt(rel_a * rel_b) if rel_a > 0 and rel_b > 0 else float("nan")


def pairwise_corr(Z: np.ndarray, min_n: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Pearson r between columns over rows where both are present, and the n behind each."""
    k = Z.shape[1]
    R, N = np.full((k, k), np.nan), np.zeros((k, k), dtype=int)
    for i in range(k):
        for j in range(k):
            ok = ~np.isnan(Z[:, i]) & ~np.isnan(Z[:, j])
            N[i, j] = ok.sum()
            if ok.sum() >= min_n:
                R[i, j] = np.corrcoef(Z[ok, i], Z[ok, j])[0, 1]
    return R, N


def one_factor(Z: np.ndarray, fit: np.ndarray) -> dict[str, np.ndarray]:
    """First principal component of the columns' correlation matrix on the ``fit`` rows; loadings ``l``, each row's
    score from its available columns (standardised on the fit rows), shared ``l*u`` and specific remainder."""
    mu, sd = np.nanmean(Z[fit], 0), np.nanstd(Z[fit], 0)
    S = (Z - mu) / sd
    R, _ = pairwise_corr(S[fit])
    R = np.nan_to_num(R, nan=0.0)
    np.fill_diagonal(R, 1.0)
    w, V = np.linalg.eigh(R)
    v = V[:, -1] * (1 if V[:, -1].sum() >= 0 else -1)
    load = np.sqrt(max(w[-1], 0)) * v
    avail = ~np.isnan(S)
    num = np.where(avail, S * load, 0).sum(1)
    den = np.where(avail, load**2, 0).sum(1)
    u = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
    shared = u[:, None] * load[None, :]
    shared[~avail] = np.nan
    return {"loading": load, "eigval": w[-1], "u": u, "shared": shared, "specific": S - shared, "std": S}


# ---------------------------------------------------------------- contrast


def design(age: np.ndarray, sex: np.ndarray, batch: np.ndarray | None, case: np.ndarray) -> np.ndarray:
    """[1, case, age, age², male, batch dummies (first level dropped)]; age centred for conditioning."""
    a = age - age.mean()
    cols = [np.ones(len(age)), case.astype(float), a, a**2, (sex == "male").astype(float)]
    if batch is not None:
        for lev in np.unique(batch)[1:]:
            cols.append((batch == lev).astype(float))
    return np.column_stack(cols)


def case_effect(y: np.ndarray, X: np.ndarray) -> float:
    return float(np.linalg.lstsq(X, y, rcond=None)[0][1])


def _deltas(
    preds: dict[str, np.ndarray], idx: np.ndarray, age, sex, batch, case, ctrl, inside
) -> dict[str, tuple[float, float]]:
    """Per type, on the rows ``idx`` (a person resample): (delta years, delta z) with the control line refitted."""
    out = {}
    for t, p in preds.items():
        r = idx[~np.isnan(p[idx])]
        c = r[ctrl[r]]
        if len(c) < 10 or case[r].sum() < 3:
            out[t] = (np.nan, np.nan)
            continue
        b, a = np.polyfit(age[c], p[c], 1)
        sd = (p[c] - a - b * age[c]).std(ddof=2)
        rr = r[inside[r]]
        gap = p[rr] - a - b * age[rr]
        X = design(age[rr], sex[rr], None if batch is None else batch[rr], case[rr])
        dz = case_effect(gap / sd, X)
        out[t] = (case_effect(gap / b, X) if b > 0 else np.nan, dz)
    return out


def contrast(
    preds: dict[str, np.ndarray],
    age: np.ndarray,
    sex: np.ndarray,
    batch: np.ndarray | None,
    case: np.ndarray,
    ctrl: np.ndarray,
    person: np.ndarray,
    pairs: list[tuple[str, str]],
    n_boot: int = 2000,
    seed: int = 0,
) -> list[dict]:
    """Case minus control acceleration per type (and per pair of types: first minus second) with a person-bootstrap
    95 % CI. Rows outside the arms' common age range are kept for the control line but not in the regression."""
    use = case | ctrl
    lo_age, hi_age = max(age[case].min(), age[ctrl].min()), min(age[case].max(), age[ctrl].max())
    inside = use & (age >= lo_age) & (age <= hi_age)
    idx = np.flatnonzero(use)
    point = _deltas(preds, idx, age, sex, batch, case, ctrl, inside)
    persons = np.unique(person[idx])
    rows_of = {p: idx[person[idx] == p] for p in persons}
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(persons, len(persons))
        boots.append(_deltas(preds, np.concatenate([rows_of[p] for p in pick]), age, sex, batch, case, ctrl, inside))
    out = []

    def ci(v):
        v = np.asarray(v, dtype=float)
        v = v[np.isfinite(v)]
        return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if len(v) > 20 else (np.nan, np.nan)

    for t, p in preds.items():
        ok = inside & ~np.isnan(p)
        lo, hi = ci([b[t][0] for b in boots])
        lz, hz = ci([b[t][1] for b in boots])
        out.append({"kind": "type", "type": t, "delta_years": point[t][0], "lo": lo, "hi": hi,
                    "delta_z": point[t][1], "lo_z": lz, "hi_z": hz,
                    "n_case": int((ok & case).sum()), "n_control": int((ok & ctrl).sum())})  # fmt: skip
    for a, b in pairs:
        if a not in preds or b not in preds:
            continue
        lo, hi = ci([x[a][0] - x[b][0] for x in boots])
        lz, hz = ci([x[a][1] - x[b][1] for x in boots])
        ok = inside & ~np.isnan(preds[a]) & ~np.isnan(preds[b])
        out.append({"kind": "difference", "type": f"{a} - {b}", "delta_years": point[a][0] - point[b][0],
                    "lo": lo, "hi": hi, "delta_z": point[a][1] - point[b][1], "lo_z": lz, "hi_z": hz,
                    "n_case": int((ok & case).sum()), "n_control": int((ok & ctrl).sum())})  # fmt: skip
    for r in out:
        r["age_range"] = f"{lo_age:g}-{hi_age:g}"
    return out


def bar_passes(rows: list[dict]) -> bool:
    """Pre-registered: some glial type's difference to every reference type has its CI above 0 (years)."""
    diff = {r["type"]: r for r in rows if r["kind"] == "difference"}
    return any(all((d := diff.get(f"{g} - {r}")) is not None and d["lo"] > 0 for r in REFS) for g in GLIA)


# ---------------------------------------------------------------- per cohort


def run_cohort(
    dataset_id: str, n_genes: int = 2000, min_cells: int = 20, min_controls: int = 20, seed: int = 0
) -> dict[str, list[dict]]:
    co = load_cohort(dataset_id, min_cells)
    ctrl_all = np.array([d == "normal" for d in co.disease])
    tag = dataset_id[:8] if len(dataset_id) == 36 else dataset_id
    log.info("%s: %d donors (%d controls), types %s", tag, len(co.donors), ctrl_all.sum(), list(co.types))
    rng = np.random.default_rng(seed)
    nd = len(co.donors)
    res: dict[str, dict] = {}
    for t, (idx, C, nc) in co.types.items():
        ctrl = ctrl_all[idx]
        if ctrl.sum() < min_controls:
            log.info("%s %s: %d controls < %d; skipped", tag, t, ctrl.sum(), min_controls)
            continue
        age, person = co.age[idx], co.person[idx]
        fold = assign_folds(person, ctrl, 5, rng)
        A, B = thin_halves(C, rng)
        pred, pa, pb = oof_clock([log_cpm(C), log_cpm(A), log_cpm(B)], age, ctrl, fold, n_genes)
        g, ga, gb = gaps(pred, age, ctrl), gaps(pa, age, ctrl), gaps(pb, age, ctrl)
        r_half = float(np.corrcoef(ga["gap"][ctrl], gb["gap"][ctrl])[0, 1])
        rel = spearman_brown(r_half)
        status = "ok" if rel >= 0.5 and g["slope"] > 0 else ("unmeasured: slope <= 0" if g["slope"] <= 0
                                                              else "unmeasured: reliability < 0.5")  # fmt: skip
        full = {k: np.full(nd, np.nan) for k in ("pred", "gap_z", "accel", "n_cells")}
        full["pred"][idx], full["gap_z"][idx], full["accel"][idx], full["n_cells"][idx] = (
            pred,
            g["gap_z"],
            g["accel"],
            nc,
        )
        fold_full = np.full(nd, -1)
        fold_full[idx] = fold
        res[t] = {**full, "fold": fold_full, "slope": g["slope"], "intercept": g["intercept"], "sd": g["sd"],
                  "r_half": r_half, "rel": rel, "status": status, "n": len(idx), "n_ctrl": int(ctrl.sum()),
                  "r_age": float(np.corrcoef(pred[ctrl], age[ctrl])[0, 1]),
                  "mae": float(np.abs(pred[ctrl] - age[ctrl]).mean())}  # fmt: skip
        log.info("%s %-18s n=%d ctrl=%d slope %.2f sd %.1f r %.2f rel %.2f %s", tag, t, len(idx), ctrl.sum(),
                 g["slope"], g["sd"], res[t]["r_age"], rel, status)  # fmt: skip
    types = list(res)
    measured = [t for t in types if res[t]["status"] == "ok"]
    Z = np.column_stack([res[t]["gap_z"] for t in types]) if types else np.zeros((nd, 0))
    shared = np.full_like(Z, np.nan)
    specific = np.full_like(Z, np.nan)
    load = dict.fromkeys(types, np.nan)
    if len(measured) >= 2:
        mi = [types.index(t) for t in measured]
        f = one_factor(Z[:, mi], ctrl_all)
        shared[:, mi], specific[:, mi] = f["shared"], f["specific"]
        load.update(dict(zip(measured, f["loading"], strict=True)))
    R, N = pairwise_corr(np.where(ctrl_all[:, None], Z, np.nan))
    # tables
    donor_rows, type_rows, corr_rows = [], [], []
    for j, t in enumerate(types):
        r = res[t]
        type_rows.append({"cohort": tag, "type": t, "n_donors": r["n"], "n_controls": r["n_ctrl"],
                          "slope": r["slope"], "intercept": r["intercept"], "resid_sd": r["sd"],
                          "sensitivity": r["slope"] / r["sd"], "r_age": r["r_age"], "mae_controls": r["mae"],
                          "r_halves": r["r_half"], "reliability": r["rel"], "shared_share": load[t] ** 2,
                          "shared_share_reliable": load[t] ** 2 / r["rel"] if r["rel"] > 0 else np.nan,
                          "status": r["status"]})  # fmt: skip
        for i in np.flatnonzero(~np.isnan(r["pred"])):
            donor_rows.append({"cohort": tag, "donor": co.donors[i], "person": int(co.person[i]), "type": t,
                               "age": float(co.age[i]), "disease": co.disease[i], "sex": co.sex[i],
                               "is_control": bool(ctrl_all[i]), "fold": int(r["fold"][i]),
                               "n_cells": float(r["n_cells"][i]), "pred": float(r["pred"][i]),
                               "gap_z": float(r["gap_z"][i]), "accel_years": float(r["accel"][i]),
                               "shared": float(shared[i, j]), "specific": float(specific[i, j])})  # fmt: skip
        for k in range(j + 1, len(types)):
            u = types[k]
            corr_rows.append({"cohort": tag, "type_a": t, "type_b": u, "n": int(N[j, k]), "r_raw": float(R[j, k]),
                              "r_disattenuated": disattenuate(R[j, k], r["rel"], res[u]["rel"]),
                              "both_measured": t in measured and u in measured})  # fmt: skip
    contrast_rows = []
    if dataset_id in CONDITIONS:
        cond, rx = CONDITIONS[dataset_id]
        case = np.array([bool(re.search(rx, d.lower())) for d in co.disease])
        n_other = int((~case & ~ctrl_all).sum())
        preds = {t: res[t]["pred"] for t in types}
        pairs = [(g, r) for g in GLIA for r in REFS]
        batch = None if co.batch is None else np.array(co.batch)
        cov = "age+age2+sex" + (f"+batch:{co.batch_name}" if co.batch else " (no batch-like column with >1 level)")
        rows = contrast(preds, co.age, np.array(co.sex), batch, case, ctrl_all, co.person, pairs, seed=seed)
        for r in rows:
            st = res[r["type"]]["status"] if r["type"] in res else ""
            contrast_rows.append({"cohort": tag, "condition": cond, **r, "status": st, "covariates": cov,
                                  "n_other_excluded": n_other, "beyond_ci": bool(r["lo"] > 0 or r["hi"] < 0)})  # fmt: skip
        if dataset_id == BAR_COHORT:
            log.info("%s go/no-go bar: %s", tag, "PASS" if bar_passes(rows) else "FAIL")
    return {"donor": donor_rows, "type": type_rows, "contrast": contrast_rows, "corr": corr_rows}


def run(
    cohorts: tuple[str, ...] = COHORTS, n_genes: int = 2000, min_cells: int = 20, min_controls: int = 20, seed: int = 0
) -> dict[str, pl.DataFrame]:
    SUMMARY.mkdir(parents=True, exist_ok=True)
    acc: dict[str, list[dict]] = {k: [] for k in ("donor", "type", "contrast", "corr")}
    out: dict[str, pl.DataFrame] = {}
    for c in cohorts:
        part = run_cohort(c, n_genes, min_cells, min_controls, seed)
        for k, v in part.items():
            acc[k] += v
        for k, v in acc.items():  # after every cohort: a crash keeps the rest
            if v:
                out[k] = pl.DataFrame(v, infer_schema_length=None)
                out[k].write_parquet(SUMMARY / f"sys_gap_{k}.parquet", compression="zstd")
    return out
