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

6. **Guards** (``guards=True``, the bar cohort only; same folds, case assignment and bootstrap draws as the plain run):
   *leave-DE-genes-out*: per type, the condition's top ``N`` genes by |t| in an independent cohort (AD vs control on
   donor pseudobulk log-CPM, ``gene ~ case + age + sex [+ censored-age flag]``; cohorts combined by a sqrt(n)-weighted
   Stouffer sum) are removed from the clock's candidate genes and the clocks retrained (``de_out_200``,
   ``de_out_500``); a type no independent cohort can give falls back to DE on a held-out half of the bar cohort's
   persons, who then leave that type's clock and contrast. *Gene halves*: the genes split at random into two disjoint
   halves (5 seeded splits), a clock per half per type (``gene_half_<split><a|b>``).

7. **Case definitions** (``CONDITIONS``, one ``CaseDef`` per cohort): cases by a regex on the disease label with
   controls ``normal`` (AD, SLE), or by a donor-metadata column (``donor_obs/<id>.parquet``, e.g. ``subject.cmv``
   ``Positive`` vs ``Negative``), both arms healthy by the disease label once the condition's own term is ignored.
   Without a case definition the controls are the ``normal`` donors. Clocks always train on the controls only.
8. **Visits.** Where a cohort names a subject column (``PERSON_COLUMNS``), the subject is the person; a subject with
   several samples enters clock and contrast with its first visit (lowest age), later visits are scored by the same
   fold model and give the type's test-retest reliability (Pearson r of visit-1 and visit-2 gaps, controls only).

Pre-registered go/no-go (ROADMAP RM33, stage 0): in PsychAD (37a17b78), microglia or astrocytes show a larger AD
acceleration than both oligodendrocytes and OPCs, with the bootstrap CI of each difference above 0. RADC (5e57cd50)
is replication only, SLE (218acb0f) descriptive.
CMV (ROADMAP RM33, CMV pre-registration): Allen (ff4235bb) and ALTRA (789ad837, replication); predictions per type or
pair scored hold / fail / inconclusive in ``sys_gap_predictions`` (per guard in Allen: gene halves only).
Outputs (``data/output/summary/``): ``sys_gap_donor``, ``sys_gap_type``, ``sys_gap_contrast``, ``sys_gap_corr`` (a run on
some cohorts replaces only those cohorts' rows); with guards ``sys_gap_guards`` (contrast rows with ``guard``) and
``sys_gap_de`` (the DE table used: cohort, type, gene, t); with predictions ``sys_gap_predictions``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field, replace

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
GLIA, REFS = ("microglia", "astrocyte"), ("oligodendrocyte", "OPC")
EQUIV_MARGIN = 2.0  # years: equivalence band of a "does not accelerate" prediction (90 % CI inside +-margin)


@dataclass(frozen=True)
class CaseDef:
    """A cohort's condition. With ``column`` unset: cases match ``disease_rx`` on the disease label, controls are
    ``normal``. With ``column``: cases have a value in ``positive``, controls in ``negative``, and both must be healthy
    by the disease label once terms matching ``label_rx`` (the condition's own term) are ignored. ``predictions``:
    (type or "a - b", kind) with kind ``accelerates``, ``no_acceleration`` or ``exploratory`` (scored, not counted)."""

    name: str
    disease_rx: str | None = None
    column: str | None = None
    positive: tuple[str, ...] = ()
    negative: tuple[str, ...] = ()
    label_rx: str | None = None
    pairs: tuple[tuple[str, str], ...] = tuple((g, r) for g in GLIA for r in REFS)
    predictions: tuple[tuple[str, str], ...] = ()


ALLEN = "ff4235bb-1cdf-4dcf-a13a-dc4f32244400"
ALTRA = "789ad837-9bf5-436c-9b60-9f1153546e4c"
CMV_COHORTS = (ALLEN, ALTRA)
CMV = CaseDef(
    "CMV seropositive", column="subject.cmv", positive=("Positive",), negative=("Negative",),
    label_rx=r"cytomegalovirus infection",
    pairs=(("CD8 T", "monocyte"), ("CD4 T", "monocyte"), ("CD8 T", "B"), ("CD4 T", "B"), ("NK", "monocyte")),
    predictions=(("CD8 T", "accelerates"), ("CD4 T", "accelerates"), ("CD8 T - monocyte", "accelerates"),
                 ("monocyte", "no_acceleration"), ("B", "no_acceleration"), ("NK", "exploratory")),
)  # fmt: skip
# Case definitions per cohort. Disease-label conditions: controls are ``normal``; donors with any other label (other
# dementias, schizophrenia, ...) are left out of the contrast. A donor whose labels include Alzheimer disease among
# others counts as AD.
CONDITIONS = {
    "37a17b78-4864-4a42-b67b-31c00962795a": CaseDef("Alzheimer disease", r"alzheimer disease"),
    "5e57cd50-8e42-42d6-940d-5c1660d06864": CaseDef("Alzheimer disease", r"alzheimer disease"),
    "218acb0f-9f2f-4f76-b90b-15a4b7c7f629": CaseDef("systemic lupus erythematosus", r"systemic lupus erythematosus"),
    ALLEN: CMV,
    ALTRA: CMV,
}
# Subject column in donor_obs/<id>.parquet: the person when a cohort has several samples (visits) per subject.
PERSON_COLUMNS = {ALLEN: "subject.subjectGuid", ALTRA: "subject.subjectGuid"}
BAR_COHORT = "37a17b78-4864-4a42-b67b-31c00962795a"
GUARD_COHORTS = (BAR_COHORT, ALLEN)  # --guards runs here; leave-DE-genes-out only where DE_COHORTS names a source
# Independent cohorts for the leave-DE-genes-out guard (same condition, never the bar cohort itself).
DE_COHORTS = {
    BAR_COHORT: ("5e57cd50-8e42-42d6-940d-5c1660d06864", "cff99df2-4904-44f7-9173-ff837f95606e"),
}
DE_TOPS = (200, 500)
N_GENE_SPLITS = 5
BATCH_COLUMNS = ("Processing_Cohort", "batch_id")  # donor-level batch read from donor_obs/<id>.mixed.parquet (D9)
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
    genes: list[str] = field(default_factory=list)
    age_censored: np.ndarray | None = None  # True where the age is a censoring bound (e.g. "80 and over")
    donor_ids: list[str] = field(default_factory=list)  # the dataset's donor_id behind each unit
    obs: dict[str, list[str | None]] = field(default_factory=dict)  # donor_obs columns per unit (as strings)


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


def donor_obs(dataset_id: str, donor_ids: list[str], columns: list[str]) -> dict[str, list[str | None]]:
    """Per donor, the values of ``columns`` in ``donor_obs/<id>.parquet`` (strings; None where absent)."""
    p = INPUT / "census_pseudobulk" / "donor_obs" / f"{dataset_id}.parquet"
    if not columns or not p.exists():
        return {}
    o = pl.read_parquet(p)
    out = {}
    for col in columns:
        if col not in o.columns:
            continue
        lookup = dict(zip(o["donor_id"].cast(pl.Utf8).to_list(), o[col].cast(pl.Utf8).to_list(), strict=True))
        out[col] = [lookup.get(d) for d in donor_ids]
    return out


def load_cohort(dataset_id: str, min_cells: int = 20, by_donor: bool = False, keep_censored: bool = False) -> Cohort:
    """Donor metadata and, per coarse type, the summed counts of donors with at least ``min_cells`` cells.
    ``by_donor`` sums a donor's tissues (default: one unit per donor × tissue); ``keep_censored`` keeps donors whose
    age is only a bound (age = the bound, flagged in ``age_censored``); used for DE, never for clocks."""
    d = _cohort_dir(dataset_id)
    g = pl.read_parquet(d / "groups.parquet")
    if "sex" not in g.columns and "Sex" in g.columns:
        g = g.with_columns(sex=pl.col("Sex").cast(pl.Utf8))
    ages = [stage_age(s) for s in g["development_stage"].to_list()]
    if keep_censored:
        age_v = [(a[1] if a[1] is not None else a[2]) if a else None for a in ages]
    else:
        age_v = [a[1] if a else None for a in ages]
    g = g.with_columns(
        age=pl.Series(age_v, dtype=pl.Float64),
        censored=pl.Series([bool(a) and a[0] != "exact" for a in ages], dtype=pl.Boolean),
        ctype=pl.Series([coarse_type(c) for c in g["cell_type"].to_list()]),
        donor=pl.col("donor_id").cast(pl.Utf8)
        if by_donor
        else pl.col("donor_id").cast(pl.Utf8) + ":" + pl.col("tissue"),
    ).filter(pl.col("age").is_not_null())
    if dataset_id in PERSON_COLUMNS:  # the subject is the person (several samples per subject share it)
        ids = g["donor_id"].cast(pl.Utf8).unique().to_list()
        subj = donor_obs(dataset_id, ids, [PERSON_COLUMNS[dataset_id]]).get(PERSON_COLUMNS[dataset_id])
        if subj is not None:
            lookup = {d: s if s is not None else d for d, s in zip(ids, subj, strict=True)}
            g = g.with_columns(cv_group=pl.col("donor_id").cast(pl.Utf8).replace_strict(lookup))
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
            pl.col("censored").any(),
        )  # fmt: skip
        .sort("donor")
    )
    donors = meta["donor"].to_list()
    pid = {p: i for i, p in enumerate(sorted(set(meta["cv_group"].to_list())))}
    ids = [str(x) for x in meta["donor_id"].to_list()]
    batch, batch_name = donor_batch(dataset_id, ids)
    cd = CONDITIONS.get(dataset_id)
    co = Cohort(
        name=dataset_id,
        donors=donors,
        person=np.array([pid[p] for p in meta["cv_group"].to_list()]),
        age=meta["age"].to_numpy(),
        disease=[str(x) for x in meta["disease"].to_list()],
        sex=[str(x) for x in meta["sex"].to_list()],
        batch=batch,
        batch_name=batch_name,
        genes=pl.read_parquet(d / "counts.parquet", columns=["gene_id"])["gene_id"].to_list(),
        age_censored=meta["censored"].to_numpy(),
        donor_ids=ids,
        obs=donor_obs(dataset_id, ids, [cd.column] if cd is not None and cd.column else []),
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


def arms(
    cd: CaseDef | None, disease: list[str], values: list[str | None] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """(case, control) masks under a case definition (none: controls ``normal``, no cases)."""
    dis = [d.lower() for d in disease]
    if cd is None:
        return np.zeros(len(dis), dtype=bool), np.array([d == "normal" for d in dis])
    if cd.column is None:
        return np.array([bool(re.search(cd.disease_rx, d)) for d in dis]), np.array([d == "normal" for d in dis])
    if values is None:
        raise ValueError(f"{cd.name}: column {cd.column} not in donor_obs")

    def healthy(d: str) -> bool:
        return all(p == "normal" or (cd.label_rx and re.search(cd.label_rx, p)) for p in map(str.strip, d.split("||")))

    ok = np.array([healthy(d) for d in dis])
    case = ok & np.array([v in cd.positive for v in values])
    ctrl = ok & np.array([v in cd.negative for v in values])
    return case, ctrl


def first_visits(person: np.ndarray, age: np.ndarray) -> np.ndarray:
    """True on each person's first visit: the lowest age, ties to the first row."""
    out = np.zeros(len(person), dtype=bool)
    order = np.lexsort((np.arange(len(person)), age, person))
    _, first = np.unique(person[order], return_index=True)
    out[order[first]] = True
    return out


def retest_reliability(
    gap: np.ndarray, person: np.ndarray, age: np.ndarray, ctrl: np.ndarray, min_n: int = 5
) -> tuple[float, int]:
    """Pearson r between the gaps of visit 1 and visit 2 (the two lowest ages) of persons whose first visit is a
    control and who have a second visit with a gap; (r, n persons). NaN below ``min_n`` persons."""
    a, b = [], []
    first = first_visits(person, age)
    for i in np.flatnonzero(first & ctrl & ~np.isnan(gap)):
        rest = np.flatnonzero((person == person[i]) & ~first & ~np.isnan(gap))
        if len(rest):
            j = rest[np.lexsort((rest, age[rest]))[0]]
            a.append(gap[i])
            b.append(gap[j])
    n = len(a)
    return (float(np.corrcoef(a, b)[0, 1]) if n >= min_n else float("nan")), n


# ---------------------------------------------------------------- clock


def log_cpm(C: np.ndarray) -> np.ndarray:
    return np.log1p(_cpm(np.asarray(C, dtype=np.float64)))


def thin_halves(C: np.ndarray, rng: np.random.Generator, p: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Binomial thinning: each count split into two halves that sum to it."""
    a = rng.binomial(np.asarray(C, dtype=np.int64), p)
    return a, C - a


def assign_folds(person: np.ndarray, is_ctrl: np.ndarray, n_folds: int, rng: np.random.Generator) -> np.ndarray:
    """D7 folds (GroupKFold by person) for the controls; other rows of a control's person (repeat visits) take its
    fold; the remaining persons (cases, others) drawn uniformly at random into a fold."""
    fold = np.full(len(person), -1)
    ci = np.flatnonzero(is_ctrl)
    for k, (_, te) in enumerate(GroupKFold(n_splits=n_folds).split(ci, groups=person[ci])):
        fold[ci[te]] = k
    ctrl_fold = dict(zip(person[ci], fold[ci], strict=True))
    case_persons = np.array([p for p in np.unique(person[~is_ctrl]) if p not in ctrl_fold])
    draw = {**dict(zip(case_persons, rng.integers(0, n_folds, len(case_persons)), strict=True)), **ctrl_fold}
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

    def ci(v, q=2.5):
        v = np.asarray(v, dtype=float)
        v = v[np.isfinite(v)]
        return (float(np.percentile(v, q)), float(np.percentile(v, 100 - q))) if len(v) > 20 else (np.nan, np.nan)

    for t, p in preds.items():
        ok = inside & ~np.isnan(p)
        lo, hi = ci([b[t][0] for b in boots])
        lo90, hi90 = ci([b[t][0] for b in boots], 5)
        lz, hz = ci([b[t][1] for b in boots])
        out.append({"kind": "type", "type": t, "delta_years": point[t][0], "lo": lo, "hi": hi,
                    "delta_z": point[t][1], "lo_z": lz, "hi_z": hz, "lo90": lo90, "hi90": hi90,
                    "n_case": int((ok & case).sum()), "n_control": int((ok & ctrl).sum())})  # fmt: skip
    for a, b in pairs:
        if a not in preds or b not in preds:
            continue
        lo, hi = ci([x[a][0] - x[b][0] for x in boots])
        lo90, hi90 = ci([x[a][0] - x[b][0] for x in boots], 5)
        lz, hz = ci([x[a][1] - x[b][1] for x in boots])
        ok = inside & ~np.isnan(preds[a]) & ~np.isnan(preds[b])
        out.append({"kind": "difference", "type": f"{a} - {b}", "delta_years": point[a][0] - point[b][0],
                    "lo": lo, "hi": hi, "delta_z": point[a][1] - point[b][1], "lo_z": lz, "hi_z": hz,
                    "lo90": lo90, "hi90": hi90,
                    "n_case": int((ok & case).sum()), "n_control": int((ok & ctrl).sum())})  # fmt: skip
    for r in out:
        r["age_range"] = f"{lo_age:g}-{hi_age:g}"
    return out


def bar_passes(rows: list[dict]) -> bool:
    """Pre-registered: some glial type's difference to every reference type has its CI above 0 (years)."""
    diff = {r["type"]: r for r in rows if r["kind"] == "difference"}
    return any(all((d := diff.get(f"{g} - {r}")) is not None and d["lo"] > 0 for r in REFS) for g in GLIA)


def verdict(kind: str, lo: float, hi: float, lo90: float, hi90: float, margin: float = EQUIV_MARGIN) -> str:
    """Pre-registered scoring (ROADMAP RM33, CMV). ``accelerates`` / ``exploratory``: hold if the 95 % CI is above 0,
    else fail if it is below 0 or the 90 % CI lies inside +-margin. ``no_acceleration``: hold if the 90 % CI lies
    inside +-margin, fail if the 95 % CI lies wholly beyond it. Otherwise (or without a CI) inconclusive."""
    if not all(np.isfinite([lo, hi, lo90, hi90])):
        return "inconclusive"
    equiv = lo90 > -margin and hi90 < margin
    if kind in ("accelerates", "exploratory"):
        return "hold" if lo > 0 else "fail" if hi < 0 or equiv else "inconclusive"
    if kind == "no_acceleration":
        return "hold" if equiv else "fail" if lo > margin or hi < -margin else "inconclusive"
    raise ValueError(kind)


def score_predictions(cd: CaseDef, rows: list[dict]) -> list[dict]:
    """One row per pre-registered prediction from contrast rows (a missing type or pair is inconclusive)."""
    by = {r["type"]: r for r in rows}
    out = []
    for target, kind in cd.predictions:
        r = by.get(target, {})
        v = [float(r.get(k, np.nan)) for k in ("delta_years", "lo", "hi", "lo90", "hi90")]
        out.append({"condition": cd.name, "type": target, "prediction": kind, "counted": kind != "exploratory",
                    "delta_years": v[0], "lo": v[1], "hi": v[2], "lo90": v[3], "hi90": v[4],
                    "n_case": int(r.get("n_case", 0)), "n_control": int(r.get("n_control", 0)),
                    "verdict": verdict(kind, *v[1:])})  # fmt: skip
    return out


def predictions_pass(cd: CaseDef, rows: list[dict]) -> bool:
    """Every counted prediction holds."""
    return all(p["verdict"] == "hold" for p in score_predictions(cd, rows) if p["counted"])


# ---------------------------------------------------------------- guards


def de_t(X: np.ndarray, case: np.ndarray, covars: list[np.ndarray]) -> np.ndarray:
    """Per column of ``X`` (donors × genes), the OLS t of ``case`` in ``x ~ 1 + case + covars``."""
    D = np.column_stack([np.ones(len(case)), case.astype(float), *covars])
    keep = [0, 1] + [j for j in range(2, D.shape[1]) if D[:, j].std() > 0]  # drop constant covariates
    D = D[:, keep]
    beta, *_ = np.linalg.lstsq(D, X, rcond=None)
    resid = X - D @ beta
    dof = len(case) - D.shape[1]
    s2 = (resid**2).sum(0) / dof
    v = np.linalg.pinv(D.T @ D)[1, 1]
    se = np.sqrt(s2 * v)
    return np.where(se > 0, beta[1] / np.where(se > 0, se, 1), np.nan)


def de_rows(
    co: Cohort, rx: str, tag: str, min_arm: int = 8, min_frac: float = 0.25, keep: dict | None = None
) -> list[dict]:
    """AD (``rx``) vs control DE per type on donor pseudobulk log-CPM, ``gene ~ case + age + sex [+ censored]``; genes
    with CPM >= 1 in at least ``min_frac`` of the type's donors. ``keep`` (type -> donor mask) restricts the donors."""
    case_all = np.array([bool(re.search(rx, d.lower())) for d in co.disease])
    ctrl_all = np.array([d == "normal" for d in co.disease])
    rows = []
    for t, (idx, C, _) in co.types.items():
        use = (case_all[idx] | ctrl_all[idx]) & (keep[t] if keep and t in keep else True)
        case, ctrl = case_all[idx][use], ctrl_all[idx][use]
        if case.sum() < min_arm or ctrl.sum() < min_arm:
            log.info("DE %s %s: %d cases, %d controls < %d; no DE", tag, t, case.sum(), ctrl.sum(), min_arm)
            continue
        cpm = _cpm(np.asarray(C[use], dtype=np.float64))
        ok = (cpm >= 1).mean(0) >= min_frac
        X = np.log1p(cpm[:, ok])
        age = co.age[idx][use]
        cov = [age - age.mean(), (np.array(co.sex)[idx][use] == "male").astype(float)]
        if co.age_censored is not None:
            cov.append(co.age_censored[idx][use].astype(float))
        tv = de_t(X, case, cov)
        n_case, n_ctrl = int(case.sum()), int(ctrl.sum())
        rows += [{"cohort": tag, "type": t, "gene": g, "t": float(v), "n_case": n_case, "n_control": n_ctrl}
                 for g, v in zip(np.asarray(co.genes)[ok], tv, strict=True) if np.isfinite(v)]  # fmt: skip
    return rows


def combine_de(de: pl.DataFrame) -> pl.DataFrame:
    """Per type and gene, a sqrt(n)-weighted Stouffer sum of the cohorts' t (cohort ``combined``)."""
    w = (pl.col("n_case") + pl.col("n_control")).sqrt()
    return (
        de.group_by("type", "gene")
        .agg(
            t=(w * pl.col("t")).sum() / (w**2).sum().sqrt(),
            n_case=pl.col("n_case").sum(),
            n_control=pl.col("n_control").sum(),
        )  # fmt: skip
        .with_columns(cohort=pl.lit("combined"))
        .select(de.columns)
    )


def top_de_genes(de: pl.DataFrame, cohort: str, cell_type: str, n: int) -> list[str]:
    """The ``n`` genes with the largest |t| for one cohort (or ``combined``) and type; ties broken by gene id."""
    d = de.filter((pl.col("cohort") == cohort) & (pl.col("type") == cell_type))
    return d.with_columns(a=pl.col("t").abs()).sort(["a", "gene"], descending=[True, False]).head(n)["gene"].to_list()


def gene_halves(n_genes: int, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """``n_splits`` seeded random splits of ``range(n_genes)`` into two disjoint halves that cover it."""
    out = []
    for k in range(n_splits):
        perm = np.random.default_rng([seed, 7919, k]).permutation(n_genes)
        out.append((np.sort(perm[: n_genes // 2]), np.sort(perm[n_genes // 2 :])))
    return out


def independent_de(dataset_id: str, rx: str, min_cells: int) -> pl.DataFrame | None:
    """DE rows from the condition's independent cohorts, per cohort and combined."""
    rows: list[dict] = []
    for c in DE_COHORTS.get(dataset_id, ()):
        if not _cohort_dir(c).exists():
            log.info("DE cohort %s not on disk; skipped", c[:8])
            continue
        rows += de_rows(load_cohort(c, min_cells, by_donor=True, keep_censored=True), rx, c[:8])
    if not rows:
        return None
    de = pl.DataFrame(rows)
    return pl.concat([de, combine_de(de)])


def run_guards(
    co: Cohort, res: dict, logx: dict[str, np.ndarray], case: np.ndarray, ctrl_all: np.ndarray,
    batch: np.ndarray | None, pairs: list[tuple[str, str]], n_genes: int, min_cells: int, seed: int,
) -> tuple[list[dict], pl.DataFrame | None]:  # fmt: skip
    """Leave-DE-genes-out and gene-half clocks on the plain run's folds; each guard's contrast with the same seed."""
    tag = co.name[:8] if len(co.name) == 36 else co.name
    nd, types = len(co.donors), list(res)
    cd = CONDITIONS[co.name]
    with_de = co.name in DE_COHORTS
    de = independent_de(co.name, cd.disease_rx, min_cells) if with_de else None
    have = set() if de is None else set(de.filter(pl.col("cohort") == "combined")["type"].to_list())
    source = {t: "combined" for t in types if t in have}
    no_drop = {t: np.zeros(len(co.types[t][0]), dtype=bool) for t in types}
    drop = dict(no_drop)
    missing = [t for t in types if t not in have]
    rx = cd.disease_rx
    if with_de and missing:  # fallback: DE on a held-out half of this cohort's persons, who then leave that type
        persons = np.unique(co.person)
        held = set(np.random.default_rng([seed, 4049]).choice(persons, len(persons) // 2, replace=False).tolist())
        held_mask = np.array([p in held for p in co.person])
        keep = {t: held_mask[co.types[t][0]] for t in missing}
        sub = replace(co, types={t: co.types[t] for t in missing})
        fb = de_rows(sub, rx, f"{tag}-heldout", keep=keep)
        for t in missing:
            if any(r["type"] == t for r in fb):
                source[t] = f"{tag}-heldout"
                drop[t] = keep[t]
        log.info("%s: DE fallback (held-out half) for %s", tag, [t for t in missing if t in source])
        if fb:
            de = pl.concat([de, pl.DataFrame(fb)]) if de is not None else pl.DataFrame(fb)
    gene_ix = {g: i for i, g in enumerate(co.genes)}
    specs: list[tuple[str, dict[str, np.ndarray], dict[str, np.ndarray]]] = []
    for n in DE_TOPS if with_de else ():
        masks = {}
        for t in types:
            m = np.ones(len(co.genes), dtype=bool)
            if t in source:
                m[[gene_ix[g] for g in top_de_genes(de, source[t], t, n) if g in gene_ix]] = False
            masks[t] = m
        specs.append((f"de_out_{n}", masks, drop))
    for k, halves in enumerate(gene_halves(len(co.genes), N_GENE_SPLITS, seed)):
        for h, cols in zip("ab", halves, strict=True):
            m = np.zeros(len(co.genes), dtype=bool)
            m[cols] = True
            specs.append((f"gene_half_{k}{h}", dict.fromkeys(types, m), no_drop))
    out = []
    for name, masks, drops in specs:
        preds, info = {}, {}
        for t in types:
            idx = co.types[t][0]
            age, ctrl, dr = co.age[idx], ctrl_all[idx], drops[t]
            if t in source or not name.startswith("de_out"):
                (pred,) = oof_clock([logx[t][:, masks[t]]], age, ctrl & ~dr, res[t]["fold"][idx], n_genes)
                pred[dr] = np.nan
            else:  # no DE for this type at all: no guarded clock
                pred = np.full(len(idx), np.nan)
            full = np.full(nd, np.nan)
            full[idx] = pred
            preds[t] = full
            c = ctrl & ~np.isnan(pred)
            ok = c.sum() >= 10
            info[t] = {"r_age": float(np.corrcoef(pred[c], age[c])[0, 1]) if ok else np.nan,
                       "slope": float(np.polyfit(age[c], pred[c], 1)[0]) if ok else np.nan,
                       "n_candidate_genes": int(masks[t].sum()), "n_dropped_donors": int(dr.sum()),
                       "de_source": source.get(t, "") if name.startswith("de_out") else ""}  # fmt: skip
        rows = contrast(preds, co.age, np.array(co.sex), batch, case, ctrl_all, co.person, pairs, seed=seed)
        bar = predictions_pass(cd, rows) if cd.predictions else bar_passes(rows)
        log.info("%s guard %s: bar %s", tag, name, "PASS" if bar else "FAIL")
        for r in rows:
            i = info.get(r["type"], {})
            out.append({"guard": name, **r, "r_age": i.get("r_age", np.nan), "slope": i.get("slope", np.nan),
                        "n_candidate_genes": i.get("n_candidate_genes", -1),
                        "n_dropped_donors": i.get("n_dropped_donors", 0), "de_source": i.get("de_source", ""),
                        "bar_pass": bar})  # fmt: skip
    if de is not None:
        de = de.with_columns(for_cohort=pl.lit(tag))
    return out, de


# ---------------------------------------------------------------- per cohort


def run_cohort(
    dataset_id: str, n_genes: int = 2000, min_cells: int = 20, min_controls: int = 20, seed: int = 0,
    guards: bool = False,
) -> dict[str, list[dict]]:  # fmt: skip
    co = load_cohort(dataset_id, min_cells)
    cd = CONDITIONS.get(dataset_id)
    case_all, ctrl_all = arms(cd, co.disease, co.obs.get(cd.column) if cd is not None and cd.column else None)
    visits = dataset_id in PERSON_COLUMNS  # elsewhere a person's units are tissues or pooled samples: all kept
    # repeat visits leave clock and contrast, kept for test-retest
    primary = first_visits(co.person, co.age) if visits else np.ones(len(co.donors), dtype=bool)
    case_all, ctrl_all = case_all & primary, ctrl_all & primary
    tag = dataset_id[:8] if len(dataset_id) == 36 else dataset_id
    log.info("%s: %d donors (%d persons, %d repeat visits; %d controls, %d cases), types %s", tag, len(co.donors),
             len(np.unique(co.person)), (~primary).sum(), ctrl_all.sum(), case_all.sum(), list(co.types))  # fmt: skip
    rng = np.random.default_rng(seed)
    nd = len(co.donors)
    res: dict[str, dict] = {}
    with_guards = guards and dataset_id in GUARD_COHORTS and dataset_id in CONDITIONS
    logx: dict[str, np.ndarray] = {}
    for t, (idx, C, nc) in co.types.items():
        ctrl = ctrl_all[idx]
        if ctrl.sum() < min_controls:
            log.info("%s %s: %d controls < %d; skipped", tag, t, ctrl.sum(), min_controls)
            continue
        age, person = co.age[idx], co.person[idx]
        fold = assign_folds(person, ctrl, 5, rng)
        A, B = thin_halves(C, rng)
        X = log_cpm(C)
        if with_guards:
            logx[t] = X
        pred, pa, pb = oof_clock([X, log_cpm(A), log_cpm(B)], age, ctrl, fold, n_genes)
        g, ga, gb = gaps(pred, age, ctrl), gaps(pa, age, ctrl), gaps(pb, age, ctrl)
        r_half = float(np.corrcoef(ga["gap"][ctrl], gb["gap"][ctrl])[0, 1])
        rel = spearman_brown(r_half)
        r_retest, n_retest = retest_reliability(g["gap"], person, age, ctrl) if visits else (np.nan, 0)
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
                  "r_half": r_half, "rel": rel, "r_retest": r_retest, "n_retest": n_retest, "status": status,
                  "n": len(idx), "n_ctrl": int(ctrl.sum()),
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
                          "r_halves": r["r_half"], "reliability": r["rel"], "test_retest_r": r["r_retest"],
                          "n_test_retest": r["n_retest"], "shared_share": load[t] ** 2,
                          "shared_share_reliable": load[t] ** 2 / r["rel"] if r["rel"] > 0 else np.nan,
                          "status": r["status"]})  # fmt: skip
        for i in np.flatnonzero(~np.isnan(r["pred"])):
            donor_rows.append({"cohort": tag, "donor": co.donors[i], "person": int(co.person[i]), "type": t,
                               "age": float(co.age[i]), "disease": co.disease[i], "sex": co.sex[i],
                               "is_control": bool(ctrl_all[i]), "is_case": bool(case_all[i]),
                               "first_visit": bool(primary[i]), "fold": int(r["fold"][i]),
                               "n_cells": float(r["n_cells"][i]), "pred": float(r["pred"][i]),
                               "gap_z": float(r["gap_z"][i]), "accel_years": float(r["accel"][i]),
                               "shared": float(shared[i, j]), "specific": float(specific[i, j])})  # fmt: skip
        for k in range(j + 1, len(types)):
            u = types[k]
            corr_rows.append({"cohort": tag, "type_a": t, "type_b": u, "n": int(N[j, k]), "r_raw": float(R[j, k]),
                              "r_disattenuated": disattenuate(R[j, k], r["rel"], res[u]["rel"]),
                              "both_measured": t in measured and u in measured})  # fmt: skip
    contrast_rows: list[dict] = []
    guard_rows: list[dict] = []
    de_rows_used: list[dict] = []
    pred_rows: list[dict] = []
    if cd is not None:
        cond, case = cd.name, case_all
        n_other = int((~case & ~ctrl_all).sum())
        preds = {t: res[t]["pred"] for t in types}
        pairs = list(cd.pairs)
        batch = None if co.batch is None else np.array(co.batch)
        cov = "age+age2+sex" + (f"+batch:{co.batch_name}" if co.batch and len(set(co.batch)) > 1
                                 else " (no batch-like column with >1 level)")
        rows = contrast(preds, co.age, np.array(co.sex), batch, case, ctrl_all, co.person, pairs, seed=seed)
        for r in rows:
            st = res[r["type"]]["status"] if r["type"] in res else ""
            contrast_rows.append({"cohort": tag, "condition": cond, **r, "status": st, "covariates": cov,
                                  "n_other_excluded": n_other, "beyond_ci": bool(r["lo"] > 0 or r["hi"] < 0)})  # fmt: skip
        bar = predictions_pass(cd, rows) if cd.predictions else bar_passes(rows)
        if dataset_id == BAR_COHORT:
            log.info("%s go/no-go bar: %s", tag, "PASS" if bar else "FAIL")
        status_of = {t: res[t]["status"] for t in types}

        def scored(guard: str, rr: list[dict]) -> list[dict]:
            return [{"cohort": tag, "guard": guard, **p, "status": " / ".join(status_of.get(x.strip(), "absent")
                     for x in p["type"].split(" - "))} for p in score_predictions(cd, rr)]  # fmt: skip

        pred_rows += scored("none", rows)
        if with_guards:
            plain = [{"guard": "none", **r, "r_age": res[r["type"]]["r_age"] if r["type"] in res else np.nan,
                      "slope": res[r["type"]]["slope"] if r["type"] in res else np.nan,
                      "n_candidate_genes": len(co.genes) if r["type"] in res else -1, "n_dropped_donors": 0,
                      "de_source": "", "bar_pass": bar} for r in rows]  # fmt: skip
            more, de = run_guards(co, res, logx, case, ctrl_all, batch, pairs, n_genes, min_cells, seed)
            extra = {"cohort": tag, "condition": cond, "covariates": cov, "n_other_excluded": n_other}
            for r in plain + more:
                st = res[r["type"]]["status"] if r["type"] in res else ""
                guard_rows.append({**extra, **r, "status": st, "beyond_ci": bool(r["lo"] > 0 or r["hi"] < 0)})
            for gname in dict.fromkeys(r["guard"] for r in more):
                pred_rows += scored(gname, [r for r in more if r["guard"] == gname])
            if de is not None:
                de_rows_used = de.to_dicts()
    return {"donor": donor_rows, "type": type_rows, "contrast": contrast_rows, "corr": corr_rows,
            "guards": guard_rows, "de": de_rows_used,
            "predictions": pred_rows if cd and cd.predictions else []}  # fmt: skip


def _merge_write(k: str, new: pl.DataFrame) -> pl.DataFrame:
    """Replace the rows of the cohorts in ``new`` in ``sys_gap_<k>.parquet``; other cohorts' rows stay."""
    path = SUMMARY / f"sys_gap_{k}.parquet"
    key = "for_cohort" if k == "de" else "cohort"
    if path.exists():
        old = pl.read_parquet(path)
        if key in old.columns:
            new = pl.concat([old.filter(~pl.col(key).is_in(new[key].unique().to_list())), new], how="diagonal_relaxed")
    new.write_parquet(path, compression="zstd")
    return new


def run(
    cohorts: tuple[str, ...] = COHORTS, n_genes: int = 2000, min_cells: int = 20, min_controls: int = 20, seed: int = 0,
    guards: bool = False,
) -> dict[str, pl.DataFrame]:  # fmt: skip
    SUMMARY.mkdir(parents=True, exist_ok=True)
    acc: dict[str, list[dict]] = {k: [] for k in ("donor", "type", "contrast", "corr", "guards", "de", "predictions")}
    out: dict[str, pl.DataFrame] = {}
    for c in cohorts:
        part = run_cohort(c, n_genes, min_cells, min_controls, seed, guards=guards)
        for k, v in part.items():
            acc[k] += v
        for k, v in acc.items():  # after every cohort: a crash keeps the rest
            if v:
                out[k] = _merge_write(k, pl.DataFrame(v, infer_schema_length=None))
    return out
