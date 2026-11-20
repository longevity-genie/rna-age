"""RM33 bulk positive controls: whole tissue as one system, scored by an existing bulk clock (never retrained).

Pre-registered in ROADMAP RM33 ("Four more controls", 2026-11-20). Per control (series × age subset × clock):

1. **Predictions** of an existing run: its ``predictions.parquet`` (samples of our table outside train, e.g. the
   sarcopenia series in val), else ``data/interim/positive_controls/predictions/<gse>.parquet`` written by
   :func:`predict_series` (series kept out of every table, e.g. GSE190125's NCBI counts; GPU).
2. **Gap**: prediction minus the bias line of prediction on age fitted on the series' own controls, in predicted
   years (``sys_gap.contrast`` with ``per_slope=False``); case − control by OLS ``gap ~ case + age + age² + sex
   [+ batch]`` on the donors inside the arms' common age range; donor bootstrap with the line refitted per draw.
3. **Contamination**: a series with any labelled sample in the clock's train split (``samples_all.split``) is
   scored as exploratory, never counted.

Rows go to the sys_gap tables with ``level`` ``bulk`` and ``cohort`` = the series (cohort-scoped replacement).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import polars as pl

from rna_age.paths import INPUT, INTERIM, OUTPUT
from rna_age.train.sys_gap import CaseDef, _merge_write, contrast, score_predictions

log = logging.getLogger(__name__)

LABELS = INPUT / "positive_controls" / "bulk_labels.parquet"
PRED_DIR = INTERIM / "positive_controls" / "predictions"
HEADLINE = "20261111-0747-ens-mlp3t2-plus-atlasft2"  # MLP members kept no weights: existing predictions only
ATLAS_PAIR = "20261110-1329-ens-atlasft-u4u6"
ATLAS_MEMBERS = {ATLAS_PAIR: ("20261110-0555-atlas-ft-u4-all", "20261110-0533-atlas-ft-u6-all")}
LEVEL = "bulk"


@dataclass(frozen=True)
class BulkControl:
    gse: str
    condition: str
    system: str
    clock: str
    kind: str  # accelerates (counted) or exploratory
    subset: str = "all"
    min_age: float | None = None  # inclusive
    max_age: float | None = None  # exclusive
    batch: bool = False


CONTROLS = (
    BulkControl("GSE190125", "Down syndrome", "blood", ATLAS_PAIR, "accelerates", "adults", min_age=18, batch=True),
    BulkControl("GSE190125", "Down syndrome", "blood", ATLAS_PAIR, "exploratory", "children", max_age=18, batch=True),
    BulkControl("GSE111016", "sarcopenia", "muscle", HEADLINE, "accelerates"),
    BulkControl("GSE167186", "sarcopenia", "muscle", HEADLINE, "accelerates"),
    BulkControl("GSE111016", "sarcopenia", "muscle", ATLAS_PAIR, "exploratory"),
    BulkControl("GSE167186", "sarcopenia", "muscle", ATLAS_PAIR, "exploratory"),
)


def labels(gse: str, lab: pl.DataFrame | None = None) -> pl.DataFrame:
    """A series' usable labelled samples: arm case / normal, no exclusion, an age."""
    lab = pl.read_parquet(LABELS) if lab is None else lab
    return lab.filter(
        (pl.col("gse") == gse) & pl.col("exclude_reason").is_null() & pl.col("arm").is_in(["case", "normal"])
        & pl.col("age_years").is_not_null()
    )  # fmt: skip


def run_predictions(run_id: str, sample_ids: list[str], gse: str) -> pl.DataFrame:
    """``sample_id``, ``pred`` of ``run_id`` for these samples: the run's own predictions, else the series file of
    :func:`predict_series` (an ensemble as the mean of its members there)."""
    out = pl.DataFrame(schema={"sample_id": pl.Utf8, "pred": pl.Float64})
    rp = OUTPUT / "runs" / run_id / "predictions.parquet"
    if rp.exists():
        out = pl.read_parquet(rp, columns=["sample_id", "pred"]).filter(pl.col("sample_id").is_in(sample_ids))
    sp = PRED_DIR / f"{gse}.parquet"
    if out.height == 0 and sp.exists():
        members = ATLAS_MEMBERS.get(run_id, (run_id,))
        s = pl.read_parquet(sp).filter(pl.col("run").is_in(members) & pl.col("sample_id").is_in(sample_ids))
        if s["run"].n_unique() == len(members):
            out = s.group_by("sample_id").agg(pl.col("pred").mean(), pl.len()).filter(pl.col("len") == len(members))
            out = out.select("sample_id", "pred")
    return out


def train_overlap(sample_ids: list[str]) -> int:
    """How many of these samples sit in the train split of the age table."""
    s = pl.read_parquet(INTERIM / "samples_all.parquet", columns=["sample_id", "split"])
    return s.filter(pl.col("sample_id").is_in(sample_ids) & (pl.col("split") == "train")).height


def score_control(
    spec: BulkControl, tab: pl.DataFrame, n_train: int = 0, n_boot: int = 2000, seed: int = 0
) -> dict[str, list[dict]]:
    """Type, contrast and prediction rows for one control from ``tab`` (labels joined with ``pred``)."""
    t = tab
    if spec.min_age is not None:
        t = t.filter(pl.col("age_years") >= spec.min_age)
    if spec.max_age is not None:
        t = t.filter(pl.col("age_years") < spec.max_age)
    t = t.sort("gsm")
    age = t["age_years"].to_numpy().astype(float)
    case = (t["arm"] == "case").to_numpy()
    ctrl = (t["arm"] == "normal").to_numpy()
    sex = np.array([str(s).lower() for s in t["sex"].to_list()])
    batch = np.array([str(b) for b in t["batch"].to_list()]) if spec.batch else None
    donors = t["donor"].fill_null(t["gsm"]).to_list()
    pid = {d: i for i, d in enumerate(sorted(set(donors)))}
    person = np.array([pid[d] for d in donors])
    pred = t["pred"].to_numpy().astype(float)
    contaminated = n_train > 0
    kind = "exploratory" if contaminated else spec.kind
    extra = {"cohort": spec.gse, "subset": spec.subset, "clock": spec.clock, "condition": spec.condition,
             "n_train_overlap": n_train, "contaminated": contaminated, "level": LEVEL}  # fmt: skip
    ok_line = ctrl.sum() >= 10
    if ok_line:
        b, a = np.polyfit(age[ctrl], pred[ctrl], 1)
        sd = float((pred[ctrl] - a - b * age[ctrl]).std(ddof=2))
        r = float(np.corrcoef(age[ctrl], pred[ctrl])[0, 1])
        mae = float(np.abs(pred[ctrl] - age[ctrl]).mean())
    else:
        a = b = sd = r = mae = np.nan
    status = "ok" if ok_line and b > 0 else ("unmeasured: < 10 controls" if not ok_line else "slope <= 0")
    type_rows = [{**extra, "type": spec.system, "n_donors": len(pid), "n_controls": int(ctrl.sum()),
                  "n_cases": int(case.sum()), "slope": float(b), "intercept": float(a), "resid_sd": sd,
                  "r_age": r, "mae_controls": mae, "status": status}]  # fmt: skip
    rows: list[dict] = []
    if case.sum() and ctrl.sum():
        rows = contrast({spec.system: pred}, age, sex, batch, case, ctrl, person, [], n_boot=n_boot, seed=seed,
                        per_slope=False)  # fmt: skip
    cov = "age+age2+sex" + ("+batch" if spec.batch else "")
    contrast_rows = [{**extra, **r, "status": status, "covariates": cov, "beyond_ci": bool(r["lo"] > 0 or r["hi"] < 0)}
                     for r in rows]  # fmt: skip
    cd = CaseDef(spec.condition, predictions=((spec.system, kind),))
    pred_rows = [{**extra, "guard": "none", **p, "status": status} for p in score_predictions(cd, rows)]
    return {"type": type_rows, "contrast": contrast_rows, "predictions": pred_rows}


def run(controls: tuple[BulkControl, ...] = CONTROLS, n_boot: int = 2000, seed: int = 0) -> dict[str, pl.DataFrame]:
    """Score every control whose clock has predictions; a series without any is skipped (logged), never guessed."""
    lab = pl.read_parquet(LABELS)
    acc: dict[str, list[dict]] = {"type": [], "contrast": [], "predictions": []}
    for spec in controls:
        lb = labels(spec.gse, lab)
        ids = lb["gsm"].to_list()
        pr = run_predictions(spec.clock, ids, spec.gse)
        if pr.height == 0:
            log.warning("%s (%s): no predictions of %s; skipped", spec.gse, spec.subset, spec.clock)
            continue
        tab = lb.join(pr, left_on="gsm", right_on="sample_id", how="inner")
        part = score_control(spec, tab, train_overlap(ids), n_boot=n_boot, seed=seed)
        log.info("%s %s %s: %d of %d labelled samples scored", spec.gse, spec.subset, spec.clock, tab.height,
                 lb.height)  # fmt: skip
        for k, v in part.items():
            acc[k] += v
    out = {}
    for k, v in acc.items():
        if v:
            out[k] = _merge_write(k, pl.DataFrame(v, infer_schema_length=None))
    return out


# ---------------------------------------------------------------- scoring series kept out of the table (GPU)


def series_counts(gse: str, sample_ids: list[str]) -> tuple[list[str], list[str], np.ndarray]:
    """NCBI counts of a series mapped to Ensembl as ``ncbi_geo.parse`` does: (samples, genes, samples × genes)."""
    from rna_age.paths import download_dir
    from rna_age.sources import ncbi_geo

    ncbi_geo.ensure_annotation()
    ann = ncbi_geo._annotation()
    gene_ids = ann["gene_id"].unique().sort().to_list()
    counts = ncbi_geo.read_counts(download_dir(ncbi_geo.NAME) / "counts" / ncbi_geo.counts_name(gse))
    keep = [c for c in counts.columns if c in set(sample_ids)]
    mapped = ncbi_geo.map_counts(counts.select("geneid", *keep), ann, gene_ids)
    return keep, gene_ids, mapped.select(keep).to_numpy().T.astype(np.float32)


def atlas_inputs(ids: list[str], genes: list[str], counts: np.ndarray) -> tuple[list[str], np.ndarray]:
    """Panel inputs exactly as the embedding job writes them (``prepare_atlas_input`` on counts, panel genes a
    sample lacks = 0, stored as fp16): (kept sample ids, samples × 4,096)."""
    import pandas as pd
    from bulkformer import prepare_atlas_input

    from rna_age.train.embed import PANEL, _vocab

    vocab, norm = _vocab()
    frame = pd.DataFrame(counts, index=ids, columns=genes)
    frame = frame.T.groupby(level=0).sum().T
    empty = frame.loc[:, frame.columns.isin(norm.ensembl_gene)].sum(axis=1) <= 0
    if empty.any():
        log.warning("skipping %d samples with zero matched library", int(empty.sum()))
        frame = frame.loc[~empty]
    x, _ = prepare_atlas_input(frame, vocab, norm, input_scale="counts")
    panel = pl.read_parquet(PANEL)["index"].to_numpy()
    sub = x[:, panel]
    return list(frame.index), np.where(sub < 0, 0.0, sub).astype(np.float16).astype(np.float32)


def predict_series(gse: str, clock: str = ATLAS_PAIR) -> pl.DataFrame:
    """Score a series kept out of the table with each saved member of ``clock`` (GPU); writes
    ``data/interim/positive_controls/predictions/<gse>.parquet`` (``sample_id``, ``run``, ``pred``)."""
    from rna_age.train.finetune import predict_checkpoint

    ids, genes, counts = series_counts(gse, labels(gse)["gsm"].to_list())
    ids, X = atlas_inputs(ids, genes, counts)
    parts = []
    for run_id in ATLAS_MEMBERS.get(clock, (clock,)):
        pred = predict_checkpoint(INTERIM / "checkpoints" / run_id / "best.pt", X)
        parts.append(pl.DataFrame({"sample_id": ids, "run": run_id, "pred": pred.astype(np.float64)}))
        log.info("%s: %s scored %d samples", gse, run_id, len(ids))
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    out = pl.concat(parts)
    out.write_parquet(PRED_DIR / f"{gse}.parquet", compression="zstd")
    return out
