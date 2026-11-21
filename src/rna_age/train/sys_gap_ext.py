"""RM33 external clocks: per-type ridge clocks trained on other cohorts' healthy donors of the same tissue.

Pre-registered in ROADMAP RM33 ("External clocks", 2026-11-20). For a target cohort too small for clocks on its own
controls:

1. **Clock.** Per coarse type, ridge on the pooled healthy (``normal``) adult donors with exact ages of the tissue's
   other cohorts (``POOLS``; the target itself is never in its pool), ≥ ``min_cells`` cells of the type. Genes: the
   intersection of the pool cohorts' and the target's genes; log-CPM per unit over it; the ``n_genes`` most variable
   genes picked and standardised on the training donors (``sys_gap.fit_fold``). One clock scores every target unit,
   cases and controls alike: no folds, nothing of the target trains.
2. **Gap and contrast** as stage 0 (``sys_gap.gaps``, ``sys_gap.contrast``): the bias line and the years scale on the
   target's own controls, ``accel ~ case + age + age² + sex [+ batch]``, person bootstrap with the line refitted.
3. **Measurable** only with ≥ ``min_controls`` target controls in the type, a positive slope and split-half
   reliability ≥ 0.5 (binomial thinning of the target's counts, both halves scored by the same clock). A prediction
   touching an unmeasured type is ``inconclusive`` whatever its CI (``verdict_ci`` keeps the CI's own reading).
4. **Gene halves** (targets with counted predictions): 5 seeded splits of the common genes into two halves, each half
   its own external clock, contrast with the same seed.

Rows go to ``sys_gap_type``, ``sys_gap_contrast``, ``sys_gap_guards`` and ``sys_gap_predictions`` with ``clock`` =
``external:<pool>``; a run replaces only its (cohort, level, clock) rows.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace

import numpy as np
import polars as pl

from rna_age.paths import INPUT
from rna_age.train import sys_gap as sg
from rna_age.train.sys_gap import (
    ALLEN,
    ALTRA,
    CMV,
    GLIA,
    N_GENE_SPLITS,
    PD,
    PD_BRAIN,
    PERSON_COLUMNS,
    REFS,
    SUMMARY,
    CaseDef,
    Cohort,
    _merge_write,
    arms,
    contrast,
    first_visits,
    fit_fold,
    gaps,
    gene_halves,
    load_cohort,
    log_cpm,
    gate_z,
    score_predictions,
    spearman_brown,
    thin_halves,
)

log = logging.getLogger(__name__)

PSYCHAD = "37a17b78-4864-4a42-b67b-31c00962795a"
HBCC = "5c97eeeb-7e52-44b3-b010-b832b1f5424c"
ONEK1K = "3faad104-2ab8-4434-816d-474d8d2641db"
AIDA = "c838aec3-03ef-4398-b882-0e3912abfff0"
TEREKHOVA = "terekhova2023"
RADC = "5e57cd50-8e42-42d6-940d-5c1660d06864"
AD_RESILIENCE = "cff99df2-4904-44f7-9173-ff837f95606e"
COVID_PBMC = "9dbab10c-118d-496b-966a-67f1763a6b7d"
POOLS = {"brain": (PSYCHAD, HBCC), "blood": (ONEK1K, AIDA, TEREKHOVA)}
MIN_TRAIN_AGE = 18.0
# Cohorts whose counts name genes by symbol: mapped to Ensembl through a symbol table (symbols naming one gene only).
SYMBOL_COHORTS = (TEREKHOVA,)
SYMBOL_TABLE = INPUT / "archs4" / "genes.parquet"

_BRAIN_TYPES = (
    "astrocyte",
    "microglia",
    "oligodendrocyte",
    "OPC",
    "excitatory neuron",
    "inhibitory neuron",
    "vascular",
)
PD_EXT = replace(
    PD,
    predictions=(("microglia", "accelerates"), ("microglia - OPC", "accelerates"),
                 ("oligodendrocyte - OPC", "exploratory"), ("microglia - oligodendrocyte", "exploratory"),
                 *((t, "exploratory") for t in _BRAIN_TYPES if t != "microglia")),
)  # fmt: skip
AD_EXT = CaseDef(
    "Alzheimer disease", r"alzheimer disease",
    pairs=tuple((g, r) for g in GLIA for r in REFS),
    predictions=(("astrocyte", "accelerates"), ("astrocyte - OPC", "accelerates"),
                 ("astrocyte - oligodendrocyte", "exploratory"), ("microglia - OPC", "exploratory"),
                 ("microglia - oligodendrocyte", "exploratory"),
                 *((t, "exploratory") for t in _BRAIN_TYPES if t != "astrocyte")),
)  # fmt: skip
CMV_EXT = replace(
    CMV,
    predictions=(("CD8 T", "accelerates"), ("CD8 T - monocyte", "accelerates"), ("B", "no_acceleration"),
                 ("CD4 T", "exploratory"), ("monocyte", "exploratory"), ("NK", "exploratory")),
)  # fmt: skip
COVID = CaseDef(
    "COVID-19", r"covid-19",
    pairs=(("CD8 T", "monocyte"), ("CD8 T", "B"), ("NK", "monocyte")),
    predictions=tuple((t, "exploratory") for t in ("B", "CD8 T", "NK", "monocyte", "other T", "DC")),
)  # fmt: skip


@dataclass(frozen=True)
class Target:
    """A target cohort scored by its tissue's external clock. ``counted=False``: every prediction is exploratory
    (kept with its kind, ``counted`` false). ``tissues``: only those units; ``by_donor``: a donor's tissues summed."""

    dataset_id: str
    pool: str
    cd: CaseDef
    counted: bool = True
    by_donor: bool = False
    tissues: tuple[str, ...] | None = None
    guards: bool = True
    tag: str | None = None  # the summary's cohort key, when one cohort is scored two ways


TARGETS = {
    "pd": Target(PD_BRAIN, "brain", PD_EXT, by_donor=True),
    "radc": Target(RADC, "brain", AD_EXT),
    "cff99df2": Target(AD_RESILIENCE, "brain", AD_EXT, tissues=("Brodmann (1909) area 9",)),
    # RM33 pre-registered rerun (2026-11-21, desk #79 default): every region a donor has, summed per type
    "cff99df2_pooled": Target(AD_RESILIENCE, "brain", AD_EXT, by_donor=True, tag="cff99df2-pooled"),
    "altra": Target(ALTRA, "blood", CMV_EXT),
    "covid": Target(COVID_PBMC, "blood", COVID, counted=False, tissues=("blood",), guards=False),
    "allen": Target(ALLEN, "blood", CMV, counted=False, guards=False),
}


def _tag(dataset_id: str) -> str:
    return dataset_id[:8] if len(dataset_id) == 36 else dataset_id


def pool_cohorts(pool: str, target: str) -> tuple[str, ...]:
    """The pool's cohorts without the target (never trains on its own donors)."""
    return tuple(c for c in POOLS[pool] if c != target)


def to_ensembl(genes: list[str], table: pl.DataFrame) -> list[str]:
    """Symbols to Ensembl ids by ``table`` (gene_id, symbol); a symbol naming several genes, an unknown symbol, or a
    second symbol landing on an id already taken keeps its own name (then it matches no Ensembl id)."""
    u = table.drop_nulls("symbol").group_by("symbol").agg(pl.col("gene_id").first(), pl.len().alias("k"))
    sym2ens = dict(zip(*u.filter(pl.col("k") == 1).select("symbol", "gene_id"), strict=True))
    out, taken = [], set()
    for g in genes:
        e = sym2ens.get(g)
        if e is None or e in taken:
            out.append(g)
        else:
            taken.add(e)
            out.append(e)
    return out


def load_pool(dataset_id: str, min_cells: int = 20) -> Cohort:
    """A pool cohort's healthy adults with exact ages (genes as Ensembl ids)."""
    co = load_cohort(dataset_id, min_cells, where=(pl.col("disease") == "normal") & (pl.col("age") >= MIN_TRAIN_AGE))
    if dataset_id in SYMBOL_COHORTS:
        co.genes = to_ensembl(co.genes, pl.read_parquet(SYMBOL_TABLE, columns=["gene_id", "symbol"]))
        log.info("%s: %d of %d symbols mapped to Ensembl", _tag(dataset_id), sum(g.startswith("ENSG") for g in co.genes),
                 len(co.genes))  # fmt: skip
    return co


def common_genes(genes: list[list[str]]) -> list[str]:
    """Genes present in every list, in the first list's order."""
    rest = [set(g) for g in genes[1:]]
    return [g for g in genes[0] if all(g in s for s in rest)]


def training_set(pool: list[Cohort], cell_type: str, genes: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Pool units with the type: (log-CPM over ``genes``, age, cohort of each row)."""
    Xs, ys, src = [], [], []
    for co in pool:
        if cell_type not in co.types:
            continue
        idx, C, _ = co.types[cell_type]
        gi = {g: i for i, g in enumerate(co.genes)}
        Xs.append(log_cpm(C[:, [gi[g] for g in genes]]))
        ys.append(co.age[idx])
        src += [co.name] * len(idx)
    if not Xs:
        return np.zeros((0, len(genes))), np.zeros(0), []
    return np.vstack(Xs), np.concatenate(ys), src


def external_scores(
    Xtr: np.ndarray, ytr: np.ndarray, Xs: list[np.ndarray], n_genes: int, cols: np.ndarray | None = None
) -> list[np.ndarray]:
    """One clock on the training rows (columns ``cols`` if given) scores every matrix of ``Xs``."""
    if cols is not None:
        Xtr, Xs = Xtr[:, cols], [X[:, cols] for X in Xs]
    f = fit_fold(Xtr, ytr, n_genes)
    return [f(X) for X in Xs]


def _measure(pred: np.ndarray, pa: np.ndarray, pb: np.ndarray, age: np.ndarray, ctrl: np.ndarray,
             min_controls: int) -> dict:  # fmt: skip
    """Bias line, reliability and status on the target's controls."""
    if ctrl.sum() < min_controls:
        return {"slope": np.nan, "intercept": np.nan, "sd": np.nan, "r_half": np.nan, "rel": np.nan,
                "r_age": np.nan, "mae": np.nan, "status": f"unmeasured: < {min_controls} controls"}  # fmt: skip
    g, ga, gb = gaps(pred, age, ctrl), gaps(pa, age, ctrl), gaps(pb, age, ctrl)
    r_half = float(np.corrcoef(ga["gap"][ctrl], gb["gap"][ctrl])[0, 1])
    rel = spearman_brown(r_half)
    status = "ok" if rel >= 0.5 and g["slope"] > 0 else ("unmeasured: slope <= 0" if g["slope"] <= 0
                                                          else "unmeasured: reliability < 0.5")  # fmt: skip
    return {"slope": g["slope"], "intercept": g["intercept"], "sd": g["sd"], "r_half": r_half, "rel": rel,
            "r_age": float(np.corrcoef(pred[ctrl], age[ctrl])[0, 1]),
            "mae": float(np.abs(pred[ctrl] - age[ctrl]).mean()), "status": status}  # fmt: skip


def gated(pred_rows: list[dict], status_of: dict[str, str], counted: bool) -> list[dict]:
    """Predictions with the measurability rule: a type (or either type of a pair) not ``ok`` makes the verdict
    inconclusive; ``verdict_ci`` keeps the CI's reading. ``counted=False`` marks every row exploratory."""
    out = []
    for p in pred_rows:
        parts = [x.strip() for x in p["type"].split(" - ")]
        st = [status_of.get(x, "absent") for x in parts]
        ok = all(s == "ok" for s in st)
        out.append({**p, "counted": p["counted"] and counted, "status": " / ".join(st), "verdict_ci": p["verdict"],
                    "verdict": p["verdict"] if ok else "inconclusive"})  # fmt: skip
    return gate_z(out)


def run_target(
    t: Target, pool: list[Cohort], n_genes: int = 2000, min_cells: int = 20, min_controls: int = 10, seed: int = 0,
    guards: bool = True, n_boot: int = 2000,
) -> dict[str, list[dict]]:  # fmt: skip
    """Score one target with its pool's external clocks (``pool`` must not contain the target)."""
    if any(co.name == t.dataset_id for co in pool):
        raise ValueError(f"{t.dataset_id}: the target is in its own pool")
    where = pl.col("tissue").is_in(list(t.tissues)) if t.tissues else None
    co = load_cohort(t.dataset_id, min_cells, by_donor=t.by_donor, where=where)
    cd = t.cd
    case_all, ctrl_all = arms(cd, co.disease, co.obs.get(cd.column) if cd.column else None)
    visits = t.dataset_id in PERSON_COLUMNS  # as stage 0: repeat visits leave clock line and contrast
    primary = first_visits(co.person, co.age) if visits else np.ones(len(co.donors), dtype=bool)
    case_all, ctrl_all = case_all & primary, ctrl_all & primary
    tag, clock = t.tag or _tag(t.dataset_id), f"external:{t.pool}" + sg.clock_suffix()
    genes = common_genes([co.genes] + [p.genes for p in pool])
    if len(genes) < n_genes:
        raise ValueError(f"{t.dataset_id}: {len(genes)} genes common to target and pool (< {n_genes}); ids differ?")
    tgi = {g: i for i, g in enumerate(co.genes)}
    tcols = np.array([tgi[g] for g in genes])
    log.info("%s (%s): %d units (%d controls, %d cases), %d common genes; pool %s", tag, clock, len(co.donors),
             ctrl_all.sum(), case_all.sum(), len(genes), [_tag(p.name) for p in pool])  # fmt: skip
    rng = np.random.default_rng(seed)
    nd = len(co.donors)
    res: dict[str, dict] = {}
    train: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    target_x: dict[str, np.ndarray] = {}
    for ct, (idx, C, _) in co.types.items():
        Xtr, ytr, src = training_set(pool, ct, genes)
        if len(ytr) < 20:
            log.info("%s %s: %d training donors; no external clock", tag, ct, len(ytr))
            continue
        Ct = C[:, tcols]
        A, B = thin_halves(Ct, rng)
        X = log_cpm(Ct)
        pred, pa, pb = external_scores(Xtr, ytr, [X, log_cpm(A), log_cpm(B)], n_genes)
        ctrl, age = ctrl_all[idx], co.age[idx]
        m = _measure(pred, pa, pb, age, ctrl, min_controls)
        full = np.full(nd, np.nan)
        full[idx] = pred
        res[ct] = {**m, "pred": full, "n": len(idx), "n_ctrl": int(ctrl.sum()), "n_train": len(ytr),
                   "pool": "+".join(f"{_tag(c)}:{src.count(c)}" for c in dict.fromkeys(src))}  # fmt: skip
        train[ct] = (Xtr, ytr)
        target_x[ct] = X
        log.info("%s %-18s train=%d n=%d ctrl=%d slope %.2f r %.2f rel %.2f %s", tag, ct, len(ytr), len(idx),
                 ctrl.sum(), m["slope"], m["r_age"], m["rel"], m["status"])  # fmt: skip
    type_rows = [{"cohort": tag, "type": ct, "n_donors": r["n"], "n_controls": r["n_ctrl"], "slope": r["slope"],
                  "intercept": r["intercept"], "resid_sd": r["sd"],
                  "sensitivity": r["slope"] / r["sd"] if r["sd"] and np.isfinite(r["sd"]) else np.nan,
                  "r_age": r["r_age"], "mae_controls": r["mae"], "r_halves": r["r_half"], "reliability": r["rel"],
                  "status": r["status"], "n_train": r["n_train"], "pool": r["pool"], "n_genes_common": len(genes)}
                 for ct, r in res.items()]  # fmt: skip
    status_of = {ct: r["status"] for ct, r in res.items()}
    batch = None if co.batch is None else np.array(co.batch)
    cov = "age+age2+sex" + (f"+batch:{co.batch_name}" if co.batch and len(set(co.batch)) > 1
                             else " (no batch-like column with >1 level)")  # fmt: skip
    sex = np.array(co.sex)
    n_other = int((~case_all & ~ctrl_all).sum())
    extra = {"cohort": tag, "condition": cd.name, "covariates": cov, "n_other_excluded": n_other}

    def contrast_of(preds: dict[str, np.ndarray]) -> list[dict]:
        return contrast(preds, co.age, sex, batch, case_all, ctrl_all, co.person, list(cd.pairs), n_boot=n_boot,
                        seed=seed)  # fmt: skip

    rows = contrast_of({ct: r["pred"] for ct, r in res.items()})
    contrast_rows = [{**extra, **r, "status": status_of.get(r["type"], ""),
                      "beyond_ci": bool(r["lo"] > 0 or r["hi"] < 0)} for r in rows]  # fmt: skip

    def scored(guard: str, rr: list[dict], st: dict[str, str]) -> list[dict]:
        return [{"cohort": tag, "guard": guard, **p} for p in gated(score_predictions(cd, rr), st, t.counted)]

    pred_rows = scored("none", rows, status_of)
    guard_rows: list[dict] = []
    if guards and t.guards and res:
        for k, halves in enumerate(gene_halves(len(genes), N_GENE_SPLITS, seed)):
            for h, cols in zip("ab", halves, strict=True):
                name, preds, st, info = f"gene_half_{k}{h}", {}, {}, {}
                for ct in res:
                    idx = co.types[ct][0]
                    (p,) = external_scores(*train[ct], [target_x[ct]], n_genes, cols)
                    full = np.full(nd, np.nan)
                    full[idx] = p
                    preds[ct] = full
                    c = ctrl_all[idx]
                    ok = c.sum() >= min_controls
                    slope = float(np.polyfit(co.age[idx][c], p[c], 1)[0]) if ok else np.nan
                    info[ct] = {"r_age": float(np.corrcoef(p[c], co.age[idx][c])[0, 1]) if ok else np.nan,
                                "slope": slope}  # fmt: skip
                    # halves: the slope rule only (no thinning per half); reliability is the plain run's
                    st[ct] = "ok" if ok and slope > 0 and status_of[ct] == "ok" else "unmeasured"
                rr = contrast_of(preds)
                pred_rows += scored(name, rr, st)
                guard_rows += [{**extra, "guard": name, **r, **info.get(r["type"], {}),
                                "n_candidate_genes": len(cols), "status": st.get(r["type"], ""),
                                "beyond_ci": bool(r["lo"] > 0 or r["hi"] < 0)} for r in rr]  # fmt: skip
    out = {"type": type_rows, "contrast": contrast_rows, "guards": guard_rows, "predictions": pred_rows}
    return {k: [{**r, "level": "coarse", "clock": clock} for r in v] for k, v in out.items()}


def run(
    targets: tuple[str, ...] = tuple(TARGETS), n_genes: int = 2000, min_cells: int = 20, min_controls: int = 10,
    seed: int = 0, guards: bool = True, gene_selection: str = "hvg",
) -> dict[str, pl.DataFrame]:  # fmt: skip
    if gene_selection not in ("hvg", "age"):
        raise ValueError(f"gene_selection must be hvg or age, got {gene_selection!r}")
    sg.GENE_SELECTION = gene_selection
    SUMMARY.mkdir(parents=True, exist_ok=True)
    cache: dict[str, Cohort] = {}
    acc: dict[str, list[dict]] = {k: [] for k in ("type", "contrast", "guards", "predictions")}
    out: dict[str, pl.DataFrame] = {}
    for name in targets:
        t = TARGETS[name]
        pool = []
        for c in [c for c in cache if c not in POOLS[t.pool]]:  # one tissue's pool in memory at a time
            del cache[c]
        for c in pool_cohorts(t.pool, t.dataset_id):
            if c not in cache:
                cache[c] = load_pool(c, min_cells)
                log.info("pool %s: %d healthy adults, types %s", _tag(c), len(cache[c].donors), list(cache[c].types))
            pool.append(cache[c])
        part = run_target(t, pool, n_genes, min_cells, min_controls, seed, guards)
        for k, v in part.items():
            acc[k] += v
        for k, v in acc.items():  # after every target: a crash keeps the rest
            if v:
                out[k] = _merge_write(k, pl.DataFrame(v, infer_schema_length=None))
    return out
