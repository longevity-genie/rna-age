"""Do the disease age gaps survive adjustment for blood cell composition?

1. Reference: broad blood classes (``composition.BROAD``) from the single-cell pseudobulk of large healthy-ish PBMC
   cohorts (OneK1K, AIDA) plus Tabula Sapiens blood for granulocytes: per class, summed counts → CPM.
2. Markers: per class the ``n_markers`` genes with the largest log-fold change over the other classes (CPM ≥ 10).
3. Deconvolution: bulk blood samples of the held-out studies (val + test, exact ages); on the marker genes, reference
   and sample are both scaled to sum 1 and the fractions are non-negative least squares, renormalised to 1.
4. Gap: pooled over studies with ≥ 5 cases and ≥ 5 controls, ``pred − age ~ study + age + case`` (raw) vs
   ``… + fractions`` (adjusted; the largest class dropped as reference), OLS; 95 % CI by resampling studies.
   If the case coefficient survives the adjustment, the clock's disease signal is not a cell-mix artefact.

Output: ``data/output/summary/deconv_gap_<run>.parquet`` and ``data/interim/deconv_fractions.parquet``.
"""

from __future__ import annotations

import logging

import numpy as np
import polars as pl
from scipy.optimize import nnls

from rna_age.paths import INPUT, INTERIM, OUTPUT
from rna_age.train.composition import broad_class

log = logging.getLogger(__name__)

REFERENCE = ("3faad104", "c838aec3", "983d5ec9")  # OneK1K, AIDA, Tabula Sapiens blood (granulocytes)
BLOOD = r"(?i)blood|pbmc|leuk|peripheral|wbs|buffy"
NOT_BLOOD = r"(?i)nasal|cord|marrow|tumou?r|biopsy"
CLASSES = ("CD4 T", "CD8 T", "other T", "NK", "B", "monocyte", "DC", "platelet", "granulocyte")


def reference_profiles(prefixes=REFERENCE) -> tuple[list[str], np.ndarray, list[str]]:
    """Genes (shared by all reference cohorts) and a classes × genes CPM matrix pooled over the cohorts."""
    root = INPUT / "census_pseudobulk" / "by_dataset_celltype"
    tables = []
    for p in prefixes:
        d = next(root.glob(f"{p}*"), None)
        if d is None:
            log.warning("reference %s has no cell-type pseudobulk yet; skipped", p)
            continue
        g = pl.read_parquet(d / "groups.parquet")
        g = g.with_columns(cls=pl.Series([broad_class(c) for c in g["cell_type"].to_list()]))
        counts = pl.read_parquet(d / "counts.parquet")
        sums = {}
        for c in CLASSES:  # a class absent from a cohort contributes zeros
            cols = g.filter(pl.col("cls") == c)["sample_id"].to_list()
            sums[c] = counts.select(cols).to_numpy().sum(axis=1) if cols else np.zeros(counts.height)
        tables.append(pl.DataFrame({"gene_id": counts["gene_id"], **{f"{p}|{c}": v for c, v in sums.items()}}))
    t = tables[0]
    for other in tables[1:]:
        t = t.join(other, on="gene_id", how="inner")
    used = [p for p in prefixes if f"{p}|{CLASSES[0]}" in t.columns]
    acc = np.stack([sum(t[f"{p}|{c}"].to_numpy() for p in used) for c in CLASSES]).astype(np.float64)
    keep = acc.sum(axis=1) > 0
    acc = acc[keep]
    cpm = acc / acc.sum(axis=1, keepdims=True) * 1e6
    return t["gene_id"].to_list(), cpm, [c for c, k in zip(CLASSES, keep, strict=True) if k]


def markers(cpm: np.ndarray, n_markers: int = 50, min_cpm: float = 10.0) -> np.ndarray:
    lg = np.log2(cpm + 1)
    out = set()
    for k in range(len(cpm)):
        others = np.delete(lg, k, axis=0).max(axis=0)
        fc = np.where(cpm[k] >= min_cpm, lg[k] - others, -np.inf)
        out.update(np.argsort(fc)[::-1][:n_markers].tolist())
    return np.array(sorted(out))


def fractions(bulk: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Rows of ``bulk`` (samples × markers) as non-negative mixtures of ``ref`` (classes × markers)."""
    A = (ref / ref.sum(axis=1, keepdims=True)).T
    out = np.zeros((len(bulk), len(ref)))
    for i, b in enumerate(bulk):
        s = b.sum()
        if s > 0:
            w, _ = nnls(A, b / s)
            out[i] = w / w.sum() if w.sum() > 0 else w
    return out


def _ols_case(d: pl.DataFrame, frac_cols: list[str]) -> float:
    studies = d["study_id"].unique().sort().to_list()
    X = [np.ones(d.height), d["case"].to_numpy().astype(float), d["age_years"].to_numpy()]
    sid = d["study_id"].to_numpy()
    X += [(sid == s).astype(float) for s in studies[1:]]
    X += [d[c].to_numpy() for c in frac_cols]
    X = np.column_stack(X)
    beta, *_ = np.linalg.lstsq(X, d["gap"].to_numpy(), rcond=None)
    return float(beta[1])


def gap_adjusted(run_id: str, aging_only: bool = False, n_boot: int = 1000, seed: int = 0) -> pl.DataFrame:
    from rna_age.train.features import load_matrix, shared_genes
    from rna_age.train.health import AGING_LINKED, health_table

    h = health_table().filter(
        pl.col("split").is_in(["val", "test"]) & ~pl.col("is_pseudobulk") & (pl.col("age_type") == "exact")
        & pl.col("tissue").str.contains(BLOOD) & ~pl.col("tissue").str.contains(NOT_BLOOD).fill_null(False)
        & pl.col("group").is_in(["case", "control"])
    )  # fmt: skip
    if aging_only:
        h = h.filter((pl.col("group") == "control") | pl.col("category").is_in(list(AGING_LINKED)))
    ok = h.group_by("study_id").agg(
        ((pl.col("group") == "case").sum() >= 5) & ((pl.col("group") == "control").sum() >= 5)
    )  # fmt: skip
    h = h.filter(pl.col("study_id").is_in(ok.filter(pl.col("group"))["study_id"].implode()))
    genes, cpm, classes = reference_profiles()
    shared = set(shared_genes())
    mk = [i for i in markers(cpm) if genes[i] in shared]
    counts = load_matrix(h.select("sample_id", "source"), [genes[i] for i in mk])
    F = fractions(counts, cpm[:, mk])
    fcols = [f"f_{c}" for c in classes]
    fr = h.select("sample_id", "study_id").with_columns([pl.Series(c, F[:, k]) for k, c in enumerate(fcols)])
    (INTERIM / "deconv_fractions.parquet").parent.mkdir(parents=True, exist_ok=True)
    fr.write_parquet(INTERIM / "deconv_fractions.parquet")
    pred = pl.read_parquet(OUTPUT / "runs" / run_id / "predictions.parquet").select("sample_id", "pred")
    d = (
        h.join(fr.drop("study_id"), on="sample_id")
        .join(pred, on="sample_id")
        .with_columns(gap=pl.col("pred") - pl.col("age_years"), case=(pl.col("group") == "case").cast(pl.Int8))
    )
    adj_cols = [c for c in fcols if c != fcols[int(np.argmax(F.mean(axis=0)))]]  # drop the largest class
    raw, adj = _ols_case(d, []), _ols_case(d, adj_cols)
    rng = np.random.default_rng(seed)
    studies = d["study_id"].unique().to_list()
    bs = []
    for _ in range(n_boot):
        pick = rng.choice(studies, len(studies), replace=True)
        bd = pl.concat([d.filter(pl.col("study_id") == s).with_columns(study_id=pl.lit(f"{s}#{j}"))
                        for j, s in enumerate(pick)])  # fmt: skip
        bs.append((_ols_case(bd, []), _ols_case(bd, adj_cols)))
    bs = np.array(bs)
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    case_frac = d.group_by("case").agg([pl.col(c).mean() for c in fcols]).sort("case")
    out = pl.DataFrame({
        "run_id": [run_id] * 2, "cases": ["aging-linked" if aging_only else "all"] * 2,
        "model": ["raw", "composition-adjusted"], "case_gap": [raw, adj],
        "ci_low": [lo[0], lo[1]], "ci_high": [hi[0], hi[1]],
        "studies": [len(studies)] * 2, "samples": [d.height] * 2, "n_markers": [len(mk)] * 2,
    })  # fmt: skip
    log.info("composition shift case − control: %s", {c: round(float(case_frac[c][1] - case_frac[c][0]), 3)
                                                    for c in fcols})  # fmt: skip
    path = OUTPUT / "summary" / f"deconv_gap_{run_id}{'_aging' if aging_only else ''}.parquet"
    out.write_parquet(path)
    return out
