"""Robustness of an age model by tissue and platform (RM11, plan line 3).

Held-out samples (test, bulk, exact ages) of one run, grouped by a tissue class read from the free-text tissue field
(:func:`tissue_class`) and by sequencing platform; per group the MAE, Pearson r, mean error, samples and studies.
A group is flagged ``unsupported`` when its MAE exceeds ``flag_ratio`` x the run's overall test MAE, counted only for
groups with at least ``min_studies`` studies and ``min_samples`` samples (fewer is reported, never flagged).
"""

from __future__ import annotations

import re

import numpy as np
import polars as pl

from rna_age.paths import INTERIM, OUTPUT

# First match wins: sorted cells before blood, heart before muscle, airway before lung.
TISSUE_CLASSES: list[tuple[str, str]] = [
    (
        "sorted_immune_cells",
        r"\bcd\d|t[- ]?cells?\b|\btreg|monocyte|neutrophil|b[- ]?cells?\b|\bbcell|macrophage|"
        r"lymphocyte|\biels?\b|nk cells?",
    ),
    ("plasma_cfrna", r"plasma|cell-free|cfrna|serum"),
    ("pbmc", r"pbmc|mononuclear"),
    ("blood", r"blood|\bwbs?\b|\bwbc\b|leukocyte"),
    ("bone_marrow", r"bone marrow|\bbm-msc"),
    (
        "brain_cns",
        r"brain|cortex|gyrus|putamen|caudate|cerebell|hippocamp|accumbens|spinal|white matter|\bba ?\d|"
        r"dlpfc|amygdala|insula|subiculum|midbrain|forebrain|hindbrain|hypothalamus|substantia|basal ?ganglia|"
        r"temporal|frontal|pituitary|\bwm\b|\bnac\b",
    ),
    ("meningioma", r"meningioma"),
    ("airway", r"nasal|nasopharyn|airway|bronch|\bbal\b|lavage|epithelial cells and secretions"),
    ("lung", r"lung"),
    ("liver", r"liver|hepatocyte"),
    ("heart", r"heart|cardiac|atri(al|a)\b|ventric"),
    ("vascular", r"artery|aorta|vein|valve|\bavf\b"),
    ("skeletal_muscle", r"muscle|vastus"),
    ("fibroblast", r"fibroblast"),
    ("skin", r"skin|cutaneous"),
    ("adipose", r"adipose"),
    ("gut", r"esophag|stomach|colon|ileum|ileal|duoden|rectum|intestin|gastro"),
    ("pancreas", r"pancrea|islet"),
    ("thyroid", r"thyroid"),
    ("kidney", r"kidney|renal"),
    ("breast", r"breast|mammary"),
    ("reproductive", r"testis|prostate|ovary|uterus|vagina|endometri|cervix"),
    ("oral", r"saliva|oral|salivary|periodont|tooth|gingiv"),
    ("bone_joint", r"bone|cartilage|menisc|synovi|bursa"),
]
_COMPILED = [(name, re.compile(rx)) for name, rx in TISSUE_CLASSES]


def tissue_class(text: str | None) -> str:
    """A coarse tissue class for a free-text tissue field (lower-cased match; ``other`` when nothing fits)."""
    if not text:
        return "unknown"
    t = text.lower()
    for name, rx in _COMPILED:
        if rx.search(t):
            return name
    return "other"


def _stats(t: pl.DataFrame) -> dict:
    e = (t["pred"] - t["age_years"]).to_numpy()
    a, p = t["age_years"].to_numpy(), t["pred"].to_numpy()
    r = float(np.corrcoef(a, p)[0, 1]) if len(t) > 2 and a.std() > 0 and p.std() > 0 else float("nan")
    return {"mae": float(np.abs(e).mean()), "mean_error": float(e.mean()), "r": r, "samples": t.height,
            "studies": t["study_id"].n_unique()}  # fmt: skip


def by_group(
    run_id: str, flag_ratio: float = 1.5, min_studies: int = 3, min_samples: int = 50, split: str = "test"
) -> pl.DataFrame:
    """Per tissue class and per platform: the run's error on held-out bulk samples with exact ages."""
    p = pl.read_parquet(OUTPUT / "runs" / run_id / "predictions.parquet")
    p = p.filter((pl.col("split") == split) & (pl.col("age_type") == "exact") & ~pl.col("is_pseudobulk"))
    meta = pl.read_parquet(INTERIM / "samples_all.parquet", columns=["sample_id", "platform", "split"])
    # samples eligible in the current table only (a re-merge may have excluded some since the run)
    p = p.join(meta.filter(pl.col("split").is_not_null()).drop("split"), on="sample_id", how="inner").with_columns(
        tissue_class=pl.col("tissue").map_elements(tissue_class, return_dtype=pl.String),
        platform=pl.col("platform").fill_null("unknown"),
    )
    overall = _stats(p)
    rows = [{"run_id": run_id, "grouping": "all", "group": "all", **overall}]
    for grouping in ("tissue_class", "platform", "source"):
        for (g,), t in p.group_by(grouping):
            rows.append({"run_id": run_id, "grouping": grouping, "group": str(g), **_stats(t)})
    out = pl.DataFrame(rows).with_columns(
        ratio=pl.col("mae") / overall["mae"],
        unsupported=(pl.col("mae") > flag_ratio * overall["mae"])
        & (pl.col("studies") >= min_studies)
        & (pl.col("samples") >= min_samples),
    )
    out = out.sort("grouping", "mae", descending=[False, True])
    out.write_parquet(OUTPUT / "summary" / f"robustness_{run_id}.parquet")
    return out
