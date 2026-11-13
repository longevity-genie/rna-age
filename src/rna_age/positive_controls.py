"""RM34 step 1: an exhaustive metadata scan for positive controls of system-specific ageing.

A positive control is a human study where a disease or condition affects a known tissue or system and healthy
controls of the same tissue were sampled in the same study. Each (dataset or study × condition × tissue) is scored
for usability; nothing large is downloaded.

Sources:
- ``cellxgene``: every human Discover dataset with primary cells, a non-cancer disease and ``normal`` donors (plus
  the normal-only datasets the RM33 seed list names, for conditions held in author columns such as CMV status).
  The per-cell obs is read over HTTP range requests from the h5ad (Census LTS bucket when the dataset is in the
  pinned release, else the Discover asset), aggregated to donor × standard columns × cell type counts and
  donor × author column × value counts, and cached under ``data/download/positive_controls/``.
- ``bulk``: ARCHS4 series in ``data/interim/samples_all.parquet`` with disease and healthy samples of one tissue;
  series with curated GEO labels (``data/input/positive_controls/bulk_labels.parquet``, built by
  :mod:`rna_age.positive_controls_bulk`) are scored from those labels instead, with their batch field.
- ``scbasecount``: Arc's reprocessed SRA single-cell samples (local inventory), studies with diseased and healthy
  donors of one tissue; counts are not read at this step (``types: needs counts``).

Score: arms and donors per arm; share of donors with a precise age (exact, or a bin of at most 5 years); age
medians; donors left after 1:1 age matching within a 5-year caliper (:func:`match_pairs`); sex per arm; cells per
donor and per donor × coarse cell class; per batch-like column, whether the condition is nested in batch (D9:
Cramér's V ≥ 0.8, or no level holding both arms). Verdict (:func:`verdict`): usable (≥ 10 matched pairs, precise
ages, not batch-confounded), caveat (one criterion borderline), or not usable.
Output: ``data/output/summary/positive_controls_usability.parquet``.
"""

from __future__ import annotations

import io
import json
import logging
import re
import time
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

from rna_age.download import Remote
from rna_age.paths import DOWNLOAD, INPUT, INTERIM, OUTPUT
from rna_age.sources.bgee import stage_age
from rna_age.sources.census_pseudobulk import BUCKET_URL, CANCER, H5AD_PREFIX
from rna_age.train.composition import broad_class
from rna_age.train.robustness import tissue_class

log = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

NAME = "positive_controls"
OUT = OUTPUT / "summary" / "positive_controls_usability.parquet"
CACHE = DOWNLOAD / NAME
DISCOVER_JSON = DOWNLOAD / "census_pseudobulk" / "discover_datasets.json"
CENSUS_DATASETS = INTERIM / "inventory" / "census_datasets_2025-11-08.parquet"
CENSUS_OBS = INTERIM / "inventory" / "census_obs_2025-11-08.parquet"
SCBASECOUNT = INTERIM / "inventory" / "scbasecount_samples.parquet"
SCBASECOUNT_META = DOWNLOAD / "scbasecount" / "hs_sample_metadata.parquet"
BULK_LABELS = INPUT / NAME / "bulk_labels.parquet"

CALIPER = 5.0  # years, 1:1 age matching
MIN_PAIRS = 10  # matched donors per arm for "usable"
BORDER_PAIRS = 6  # 6-9 matched pairs: borderline
PRECISE_BIN = 5.0  # an age bin this wide or narrower counts as precise
V_CONFOUNDED = 0.8
V_BORDER = 0.5
MAX_OBS_BYTES = 200 << 20
CHEMISTRY = ("assay", "suspension_type", "platform", "tech_10x", "cell_prep")  # batch-like, but not a processing batch

STANDARD = ["donor_id", "disease", "development_stage", "sex", "tissue", "tissue_type", "assay", "suspension_type",
            "cell_type", "is_primary_data"]  # fmt: skip
PRENATAL = re.compile(r"(?i)carnegie|post-fertilization|embryo|fetal|foetal|prenatal|gestation|week post|newborn")
CONTROL_LABEL = re.compile(
    r"(?i)^\s*(normal|healthy|control|ctrl|none|no|non[- ]?\w+|negative|unaffected|match(ed)? control|hc|nc|"
    r"healthy control|not applicable|n/?a|unsure|not specified|unknown|other|null|)\s*$|\bcontrol|\bhealthy|^non-|^none|"
    r"^not\b|\bnot an?\b"
)
CELL_TYPE_LIKE = re.compile(r"(?i)cell_?type|annot|cluster|leiden|louvain|subtype|lineage|class|state|compartment|"
                            r"subclass|level|^l\d|azimuth|celltypist|label|ontology|_id$|^_|barcode|umap|pca|doublet")  # fmt: skip
BATCH_LIKE = re.compile(
    r"(?i)batch|librar|lib_|seq|run|lane|chip|pool|plate|site|cent(er|re)|channel|capture|gem|well|kit|chemistr|"
    r"protocol|process|date|replicate|flow_?cell|10x|sort|hash|multiplex|cohort|study|institut|\blab|experiment|"
    r"project|source|origin|dataset|collection|sample|hospital|bank|version|platform|instrument|assay|suspension"
)
CONDITION_COLUMN = re.compile(r"(?i)cmv|smok|sero|hiv|ebv|infect|diagnos|condition|status|group|treatment|vaccin")
CONDITION_WORDS = re.compile(
    r"(?i)control|healthy|normal|patient|\bcase|ctrl|disease|\bhc\b|\bnc\b|lesion|affected|infected|positive|"
    r"negative|sick|\bad\b|\bpd\b|covid|diagnos|\bms\b|\bsle\b|donor_type"
)
MISSING_LEVEL = re.compile(r"(?i)not evaluated|not assessed|unknown|missing|^n/?a$|not available|^nan$|^none$|^$|"
                           r"not reported|undetermined|indeterminate")
REFERENCE_LEVEL = re.compile(r"(?i)^(neg|negative|never|no|non|none|control|healthy|normal|false|0|uninfected|absent|"
                             r"seronegative|cmv-|cmv neg\w*|never smoker|non-?smoker)$|negative|never|non-?smok")  # fmt: skip

# disease text → tissue classes (robustness.tissue_class names) where the disease sits; blood elsewhere is indirect
IMMUNE = {"blood", "pbmc", "bone_marrow", "sorted_immune_cells", "lymphoid"}
DISEASE_SITES: list[tuple[str, set[str]]] = [
    (r"alzheimer|parkinson|dementia|lewy|amyotrophic|\bals\b|frontotemporal|huntington|multiple sclerosis|epilep|"
     r"schizophren|bipolar|autism|depress|psych|supranuclear|pick|neurodegener|stroke|cerebral|tauopath|"
     r"motor neuron|spinal|brain|neuro|nervous|tremor|ataxia|dystonia|opioid|alcohol|cocaine|addict|substance",
     {"brain_cns"}),
    (r"pulmonary|lung|copd|emphysema|asthma|interstitial|cystic fibrosis|bronch|respiratory|pneumon|smok|"
     r"sarcoidosis|tuberculosis", {"lung", "airway"}),
    (r"covid|sars-cov|influenza|respiratory syncytial", {"lung", "airway", *IMMUNE}),
    (r"kidney|renal|nephr|glomerul", {"kidney"}),
    (r"liver|hepat|cirrho|steato|nafld|nash|masld|mash|cholang|biliary", {"liver"}),
    (r"diabet|pancrea|islet", {"pancreas"}),
    (r"heart|cardi|myocard|coronary|atrial|ventric", {"heart"}),
    (r"atheroscl|aneurysm|vascular|arter|aort", {"vascular", "heart"}),
    (r"crohn|colitis|inflammatory bowel|celiac|gastr|intestin|colon|esophag|barrett", {"gut"}),
    (r"psoria|dermat|eczema|skin|acne|hidradenitis|vitiligo|scleroderma|systemic sclerosis|alopecia|keloid|lichen|"
     r"wound|ulcer", {"skin", "fibroblast"}),
    (r"arthritis|osteoarth|joint|cartilage|spondyl|gout|osteopor|bone", {"bone_joint"}),
    (r"muscle|myopath|myositis|dystroph|sarcopen", {"skeletal_muscle"}),
    (r"lupus|immunodef|hiv|cytomegalo|\bcmv|sepsis|clonal hemat|anemi|thrombocyt|neutropen|myelodysplas|vasculitis|"
     r"kawasaki|malaria|dengue|infection|immun|hemophagocyt|graft|sickle|thalass|aplastic|marrow", IMMUNE),
    (r"obes|adipos|lipodys", {"adipose"}),
    (r"endometri|ovar|infertil|prostat|testic|preeclampsia|uter|cervic", {"reproductive"}),
    (r"thyroid|graves|hashimoto", {"thyroid"}),
    (r"periodont|gingiv|oral|sjogren|sjögren|salivary", {"oral"}),
    (r"retin|macular|glaucoma|cataract|uveitis|keratoconus|eye", {"eye"}),
    (r"down syndrome|trisomy", IMMUNE | {"brain_cns"}),
    (r"progeria|hgps|werner", {"fibroblast", "skin", "vascular"}),
]
_SITES = [(re.compile(rx, re.I), s) for rx, s in DISEASE_SITES]

# coarse classes beyond composition.BROAD (immune) for solid tissues; first match wins
EXTRA_CLASSES: list[tuple[str, str]] = [
    ("microglia", r"microglia"),
    ("macrophage", r"macrophage|kupffer|histiocyte|osteoclast"),
    ("mast", r"mast cell"),
    ("neuron", r"neuron|interneuron|purkinje|granule cell|pyramidal|\bl\d|excitatory|inhibitory|neuroblast"),
    ("astrocyte", r"astrocyte|bergmann|radial glia|glial"),
    ("oligodendrocyte", r"oligodendrocyte|\bopc\b|schwann"),
    ("endocrine", r"\b(alpha|beta|delta|gamma|epsilon)\b.*cell|endocrine|islet|pancreatic [abd] cell|"
     r"type [abd] pancreatic|pp cell"),
    ("endothelial", r"endotheli|lymphatic"),
    ("stromal", r"fibroblast|stromal|pericyte|smooth muscle|mesenchymal|stellate|mesangial|myofibroblast|"
     r"chondrocyte|osteoblast|tenocyte"),
    ("muscle", r"cardiomyocyte|cardiac muscle|myocyte|muscle fiber|skeletal muscle|satellite cell|myoblast"),
    ("adipocyte", r"adipocyte|preadipocyte"),
    ("erythroid", r"erythro|reticulocyte"),
    ("epithelial", r"epitheli|keratinocyte|hepatocyte|cholangiocyte|pneumocyte|alveolar|club|ciliated|basal cell|"
     r"goblet|enterocyte|tubule|podocyte|acinar|duct|secretory|melanocyte|urothel|ionocyte|tuft|paneth|"
     r"thyrocyte|luminal|myoepithel|loop of henle|collecting|principal cell|intercalated|parietal|chief|"
     r"sebocyte|trophoblast|granulosa|theca|sertoli|leydig|spermat"),
]
_EXTRA = [(n, re.compile(rx, re.I)) for n, rx in EXTRA_CLASSES]


# ------------------------------------------------------------------ pure helpers (tested)


def is_cancer(label: str | None) -> bool:
    return bool(label) and re.search(CANCER, label) is not None


def is_prenatal(stage: str | None) -> bool:
    return bool(stage) and PRENATAL.search(stage) is not None


def coarse_class(cell_type: str | None) -> str:
    """A coarse cell class: composition.broad_class for immune cells, else a solid-tissue class, else ``other``."""
    b = broad_class(cell_type)
    if b != "other":
        return b
    s = cell_type or ""
    for name, rx in _EXTRA:
        if rx.search(s):
            return name
    return "other"


def age_from_range(lo: float | None, hi: float | None) -> tuple[str | None, float | None, float | None]:
    """(age_type, precise age, any-age midpoint) from an age range: exact if lo == hi, ``narrow`` if the bin is at
    most :data:`PRECISE_BIN` years wide, ``binned`` when wider (only the midpoint), ``censored`` without an upper
    bound (the lower bound as the any-age value)."""
    if lo is None or (isinstance(lo, float) and np.isnan(lo)):
        return None, None, None
    if hi is None or (isinstance(hi, float) and np.isnan(hi)):
        return "censored", None, float(lo)
    if hi == lo:
        return "exact", float(lo), float(lo)
    mid = (float(lo) + float(hi)) / 2
    if hi - lo <= PRECISE_BIN:
        return "narrow", mid, mid
    return "binned", None, mid


def age_from_stage(stage: str | None) -> tuple[str | None, float | None, float | None]:
    """The same triple from a CELLxGENE/HsapDv development stage (prenatal stages give no age)."""
    if is_prenatal(stage):
        return None, None, None
    kind, years, lo, hi = stage_age(stage)
    if kind == "exact":
        return "exact", years, years
    if kind == "binned":
        return age_from_range(lo, hi + 1 if hi is not None else None)  # "sixth decade" = [50, 60)
    if kind == "censored":
        return "censored", None, lo
    return None, None, None


def match_pairs(a: Iterable[float], b: Iterable[float], caliper: float = CALIPER) -> int:
    """Maximum number of 1:1 pairs (one from each arm) whose ages differ by at most ``caliper`` (sorted two-pointer
    sweep, optimal for matching on one dimension)."""
    xs = sorted(x for x in a if x is not None and not np.isnan(x))
    ys = sorted(y for y in b if y is not None and not np.isnan(y))
    i = j = n = 0
    while i < len(xs) and j < len(ys):
        if abs(xs[i] - ys[j]) <= caliper:
            n, i, j = n + 1, i + 1, j + 1
        elif xs[i] < ys[j]:
            i += 1
        else:
            j += 1
    return n


def common_support(a: Iterable[float], b: Iterable[float], caliper: float = CALIPER) -> tuple[int, int]:
    """Donors of each arm with at least one donor of the other arm within ``caliper`` years (1:k matching bound)."""
    xs = np.array([x for x in a if x is not None and not np.isnan(x)], dtype=float)
    ys = np.array([y for y in b if y is not None and not np.isnan(y)], dtype=float)
    if not len(xs) or not len(ys):
        return 0, 0
    d = np.abs(xs[:, None] - ys[None, :]) <= caliper
    return int(d.any(1).sum()), int(d.any(0).sum())


def cramers_v(x: list, y: list, corrected: bool = True) -> float:
    """Cramér's V of two categorical vectors (0 when either has a single level). ``corrected``: Bergsma's (2013)
    small-sample bias correction, so a batch column with many small levels does not look nested by chance."""
    xs, xi = np.unique(np.asarray(x, dtype=str), return_inverse=True)
    ys, yi = np.unique(np.asarray(y, dtype=str), return_inverse=True)
    if len(xs) < 2 or len(ys) < 2:
        return 0.0
    t = np.zeros((len(xs), len(ys)))
    np.add.at(t, (xi, yi), 1)
    n = t.sum()
    e = t.sum(1, keepdims=True) * t.sum(0, keepdims=True) / n
    phi2 = ((t - e) ** 2 / e).sum() / n
    r, k = t.shape
    if not corrected or n <= 1:
        return float(np.sqrt(phi2 / (min(r, k) - 1)))
    phi2 = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rc, kc = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    d = min(rc - 1, kc - 1)
    return float(np.sqrt(phi2 / d)) if d > 0 else 0.0


def batch_check(arm: list[str], batch: list[str | None]) -> dict:
    """Is the condition nested in this batch column? ``confounded`` when the bias-corrected V ≥ 0.8 or no level
    holds both arms, ``borderline`` when V ≥ 0.5, else ``ok`` (one level: everything in one batch, ``ok``). A column
    filled in one arm only describes the condition (e.g. days since hospitalisation): ``none``."""
    pairs = [(a, b) for a, b in zip(arm, batch, strict=True) if b is not None]
    if not pairs or len({a for a, _ in pairs}) < len(set(arm)):
        return {"status": "none", "v": None, "levels": 0, "shared": 0, "mixed_share": None}
    a, b = zip(*pairs, strict=True)
    levels = sorted(set(b))
    mixed = {lv for lv in levels if len({x for x, y in pairs if y == lv}) > 1}
    mixed_share = round(sum(1 for _, y in pairs if y in mixed) / len(pairs), 3)
    if len(levels) < 2:
        return {"status": "ok", "v": 0.0, "levels": 1, "shared": len(mixed), "mixed_share": mixed_share}
    v = cramers_v(list(a), list(b))
    if v >= V_CONFOUNDED or not mixed:
        status = "confounded"
    elif v >= V_BORDER:
        status = "borderline"
    else:
        status = "ok"
    return {"status": status, "v": round(v, 3), "levels": len(levels), "shared": len(mixed), "mixed_share": mixed_share}


def is_condition_synonym(values: Iterable[str], condition: str) -> bool:
    """A 'batch' column whose levels read like the condition itself (control/patient/the disease's words)."""
    words = [w for w in re.findall(r"[a-z]{4,}", condition.lower()) if w not in {"disease", "syndrome", "disorder"}]
    for v in values:
        s = str(v).lower()
        if CONDITION_WORDS.search(s) or any(w in s for w in words):
            return True
    return False


TISSUE_EXTRA = [("brain_cns", r"brodmann|cerebral|lobe|thalam|pons|medulla|olfactory bulb|entorhinal|cingulate"), ("oral", r"saliva|labial|parotid|submandibular|tongue|tonsil"), ("eye", r"eye|retina|cornea|macula|lens"),
                ("lymphoid", r"lymph node|spleen|thymus")]


def system_class(tissue: str | None) -> str:
    """robustness.tissue_class, with a few extra classes for tissues it calls ``other``."""
    c = tissue_class(tissue)
    if c == "other":
        for name, rx in TISSUE_EXTRA:
            if re.search(rx, (tissue or "").lower()):
                return name
    return c


def target_relation(condition: str, tissue_cls: str) -> tuple[str, str]:
    """('direct' | 'indirect' | 'unknown', the disease's site classes) for a condition sampled in a tissue class."""
    sites: set[str] = set()
    for rx, s in _SITES:
        if rx.search(condition):
            sites |= s
    if not sites:
        return ("indirect" if tissue_cls in IMMUNE else "unknown"), ""
    return ("direct" if tissue_cls in sites else "indirect"), ",".join(sorted(sites))


def verdict(n_case: int, n_control: int, pairs: int, pairs_any: int, batch_status: str, needs_batch: bool
            ) -> tuple[str, str]:  # fmt: skip
    """usable / caveat / not usable with the reason (criteria: donors after matching, precise ages, batch)."""
    fails, borders = [], []
    if pairs >= MIN_PAIRS:
        pass
    elif pairs_any >= MIN_PAIRS:
        borders.append(f"ages: {pairs} matched pairs with precise ages, {pairs_any} counting wide bins")
    elif max(pairs, pairs_any) >= BORDER_PAIRS:
        borders.append(f"donors: {max(pairs, pairs_any)} matched pairs (< {MIN_PAIRS})")
        if pairs < BORDER_PAIRS:
            borders.append("ages: wide bins")
    elif min(n_case, n_control) < MIN_PAIRS:
        fails.append(f"too few donors ({n_case} vs {n_control})")
    elif pairs_any == 0 and pairs == 0:
        fails.append("no ages to match on")
    else:
        fails.append(f"only {max(pairs, pairs_any)} pairs after age matching")
    if batch_status == "confounded":
        fails.append("condition nested in batch")
    elif batch_status == "borderline":
        borders.append("batch: partly nested (Cramér's V 0.5-0.8)")
    elif batch_status in ("none", "unknown") and needs_batch:
        borders.append("batch: no batch column, within-batch test impossible (D9)")
    if fails:
        return "not usable", "; ".join(fails + borders)
    if len(borders) >= 2:
        return "not usable", "several borderline: " + "; ".join(borders)
    if borders:
        return "caveat", borders[0]
    return "usable", ""


def _median(xs: list[float | None]) -> float | None:
    v = [x for x in xs if x is not None and not np.isnan(x)]
    return float(np.median(v)) if v else None


def score(donors: pl.DataFrame, batch_cols: list[str], needs_batch: bool, condition: str) -> dict:
    """Score one arm pair. ``donors``: one row per donor with ``arm`` ('case'/'control'), ``age`` (precise or
    null), ``age_any`` (any midpoint or null), ``sex``, ``n_cells`` (null in bulk) and the batch columns."""
    case, ctrl = donors.filter(pl.col("arm") == "case"), donors.filter(pl.col("arm") == "control")
    pairs = match_pairs(case["age"].to_list(), ctrl["age"].to_list())
    pairs_any = match_pairs(case["age_any"].to_list(), ctrl["age_any"].to_list())
    sup_case, sup_ctrl = common_support(case["age"].to_list(), ctrl["age"].to_list())
    checks, skipped = [], []
    for c in batch_cols:
        vals = [v for v in donors[c].drop_nulls().unique().to_list()]
        if len(vals) >= max(3, 0.8 * donors.height):
            skipped.append(f"{c}: one level per donor")  # an id, or each donor its own batch
            continue
        if c not in CHEMISTRY and is_condition_synonym(vals, condition):
            skipped.append(f"{c}: reads like the condition")
            continue
        r = batch_check(donors["arm"].to_list(), donors[c].to_list())
        if r["status"] != "none":
            checks.append((c, r))
    rank = {"confounded": 3, "borderline": 2, "ok": 1}
    if checks:
        worst_col, worst = max(checks, key=lambda t: (rank[t[1]["status"]], t[1]["v"] or 0))
        batch_status = worst["status"]
    else:
        worst_col, worst, batch_status = None, None, "none"
    if needs_batch and batch_status == "ok" and all(c in CHEMISTRY or r["levels"] < 2 for c, r in checks):
        batch_status = "none"  # chemistry balanced, but no processing batch with two levels to test within (D9)

    def sex(df: pl.DataFrame, s: str) -> int:
        return int((df["sex"] == s).sum())

    def share(df: pl.DataFrame) -> float | None:
        return round(float(df["age"].is_not_null().mean()), 3) if df.height else None

    v, reason = verdict(case.height, ctrl.height, pairs, pairs_any, batch_status, needs_batch)
    return {
        "n_case": case.height, "n_control": ctrl.height,
        "share_precise_age_case": share(case), "share_precise_age_control": share(ctrl),
        "age_median_case": _median(case["age_any"].to_list()), "age_median_control": _median(ctrl["age_any"].to_list()),
        "n_pairs_matched": pairs, "n_pairs_matched_any_age": pairs_any,
        "n_case_support": sup_case, "n_control_support": sup_ctrl,
        "female_case": sex(case, "female"), "male_case": sex(case, "male"),
        "female_control": sex(ctrl, "female"), "male_control": sex(ctrl, "male"),
        "cells_per_donor_case": _median(case["n_cells"].cast(pl.Float64).to_list()) if "n_cells" in case.columns else None,
        "cells_per_donor_control": _median(ctrl["n_cells"].cast(pl.Float64).to_list()) if "n_cells" in ctrl.columns else None,
        "batch_columns": ",".join(c for c, _ in checks),
        "batch_status": batch_status, "batch_column": worst_col,
        "batch_cramers_v": worst["v"] if worst else None,
        "batch_detail": "; ".join([*(f"{c}: {r['status']} V={r['v']} levels={r['levels']} shared={r['shared']} in_mixed={r['mixed_share']}"
                                     for c, r in checks), *skipped]),
        "verdict": v, "reason": reason,
    }  # fmt: skip


def coarse_summary(cells: pl.DataFrame) -> tuple[str, list[dict]]:
    """Per coarse class, median cells per donor in each arm and donors with ≥ 50 cells. ``cells``: donor_id, arm,
    coarse, n_cells (one row per donor × class)."""
    if not cells.height:
        return "", []
    donors = cells.select("donor_id", "arm").unique()
    full = donors.join(cells.select("coarse").unique(), how="cross").join(
        cells.group_by("donor_id", "arm", "coarse").agg(pl.col("n_cells").sum()), on=["donor_id", "arm", "coarse"],
        how="left").with_columns(pl.col("n_cells").fill_null(0))  # fmt: skip
    t = full.group_by("coarse").agg(
        pl.col("n_cells").filter(pl.col("arm") == "case").median().alias("median_case"),
        pl.col("n_cells").filter(pl.col("arm") == "control").median().alias("median_control"),
        (pl.col("n_cells").filter(pl.col("arm") == "case") >= 50).sum().cast(pl.Int64).alias("donors_ge50_case"),
        (pl.col("n_cells").filter(pl.col("arm") == "control") >= 50).sum().cast(pl.Int64).alias("donors_ge50_control"),
        pl.col("n_cells").sum().alias("total"),
    ).sort("total", descending=True)
    text = "; ".join(f"{r['coarse']} {r['median_case']:.0f}/{r['median_control']:.0f}"
                     for r in t.head(10).iter_rows(named=True))  # fmt: skip
    return text, t.drop("total").to_dicts()


# ------------------------------------------------------------------ remote h5ad obs


class RangeFile(io.RawIOBase):
    """A read-only file over HTTP range requests with a block cache and a byte budget (h5py reads it like a file)."""

    def __init__(self, url: str, block: int = 2 << 20, limit: int = MAX_OBS_BYTES) -> None:
        import httpx

        self.url, self.block, self.limit = url, block, limit
        self.client = httpx.Client(timeout=180, follow_redirects=True)
        r = self.client.head(url).raise_for_status()
        self.size = int(r.headers["content-length"])
        self.pos, self.fetched = 0, 0
        self.cache: dict[int, bytes] = {}

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, off: int, whence: int = 0) -> int:
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def _get(self, i: int) -> bytes:
        if i not in self.cache:
            lo = i * self.block
            hi = min(self.size, lo + self.block) - 1
            for attempt in range(5):
                try:
                    r = self.client.get(self.url, headers={"Range": f"bytes={lo}-{hi}"})
                    r.raise_for_status()
                    break
                except Exception:
                    if attempt == 4:
                        raise
                    time.sleep(5 * (attempt + 1))
            self.cache[i] = r.content
            self.fetched += len(r.content)
            if self.fetched > self.limit:
                raise ObsTooLarge(f"obs read exceeded {self.limit >> 20} MB")
        return self.cache[i]

    def readinto(self, b) -> int:  # noqa: ANN001
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        out = bytearray()
        p = self.pos
        while len(out) < n:
            i = p // self.block
            blk = self._get(i)
            o = p - i * self.block
            take = blk[o : o + n - len(out)]
            out += take
            p += len(take)
        b[:n] = out
        self.pos += n
        return n

    def close(self) -> None:
        self.client.close()
        super().close()


class ObsTooLarge(RuntimeError):
    pass


def _codes(obs, name: str) -> tuple[np.ndarray, list[str]] | None:  # noqa: ANN001
    """(integer codes, category strings) of one obs column; None for numeric columns."""
    import h5py

    item = obs[name]
    if isinstance(item, h5py.Group):
        if "codes" not in item:
            return None
        cats = [c.decode() if isinstance(c, bytes) else str(c) for c in item["categories"][()]]
        return item["codes"][()].astype(np.int64), cats
    if "__categories" in obs and name in obs["__categories"]:  # anndata < 0.8
        cats = [c.decode() if isinstance(c, bytes) else str(c) for c in obs["__categories"][name][()]]
        return item[()].astype(np.int64), cats
    if item.dtype.kind == "b":
        return item[()].astype(np.int64), ["False", "True"]
    if item.dtype.kind in ("S", "O", "U"):
        arr = item[()]
        cats, inv = np.unique(arr, return_inverse=True)
        return inv.astype(np.int64), [c.decode() if isinstance(c, bytes) else str(c) for c in cats]
    return None


def _decode(codes: np.ndarray, cats: list[str]) -> pl.Series:
    lut = np.array([*cats, None], dtype=object)
    return pl.Series(lut[np.where(codes < 0, len(cats), codes)].tolist(), dtype=pl.String)


def read_obs(url: str, max_bytes: int = MAX_OBS_BYTES, local: Path | None = None
             ) -> tuple[pl.DataFrame, pl.DataFrame, dict]:  # fmt: skip
    """Per-cell obs of an h5ad (remote over range requests, or ``local`` when the file was fetched before),
    aggregated: (groups: standard columns + cell_type → n_cells, extra: donor_id × batch- or condition-like author
    column × value → n_cells, info)."""
    import h5py

    f = open(local, "rb") if local is not None else RangeFile(url, limit=max_bytes)  # noqa: SIM115
    try:
        with h5py.File(f, "r") as h:
            obs = h["obs"]
            index = obs.attrs.get("_index", "_index")
            names = [k for k in obs.keys() if k not in (index, "__categories")]
            std: dict[str, tuple[np.ndarray, list[str]]] = {}
            for c in STANDARD:
                if c in names and (r := _codes(obs, c)) is not None:
                    std[c] = r
            extras: dict[str, tuple[np.ndarray, list[str]]] = {}
            for c in names:
                if c in STANDARD or c.endswith("ontology_term_id") or CELL_TYPE_LIKE.search(c):
                    continue
                if not (BATCH_LIKE.search(c) or CONDITION_COLUMN.search(c)):
                    continue
                item = obs[c]
                n_cats = None
                if isinstance(item, h5py.Group) and "categories" in item:
                    n_cats = item["categories"].shape[0]
                elif not isinstance(item, h5py.Group) and item.dtype.kind == "b":
                    n_cats = 2
                if n_cats is None or n_cats > 200:
                    continue
                if (r := _codes(obs, c)) is not None:
                    extras[c] = r
            fetched = getattr(f, "fetched", 0)
    finally:
        f.close()
    code_df = pl.DataFrame({c: pl.Series(v[0]) for c, v in std.items()})
    groups = code_df.group_by(code_df.columns).len("n_cells")
    groups = groups.with_columns(
        [_decode(groups[c].to_numpy(), std[c][1]).alias(c) for c in std]
    ).with_columns(pl.col("n_cells").cast(pl.Int64))
    parts = []
    donor_codes = std["donor_id"][0]
    for c, (codes, cats) in extras.items():
        g = pl.DataFrame({"d": donor_codes, "v": codes}).group_by("d", "v").len("n_cells")
        parts.append(pl.DataFrame({
            "donor_id": _decode(g["d"].to_numpy(), std["donor_id"][1]), "column": c,
            "value": _decode(g["v"].to_numpy(), cats), "n_cells": g["n_cells"].cast(pl.Int64),
        }))  # fmt: skip
    extra = pl.concat(parts) if parts else pl.DataFrame(
        schema={"donor_id": pl.String, "column": pl.String, "value": pl.String, "n_cells": pl.Int64})
    info = {"url": str(local) if local is not None else url, "bytes_fetched": fetched, "n_cells": int(len(donor_codes)), "extra_columns": sorted(extras)}
    return groups, extra, info


def cached_obs(dataset_id: str, url: str, max_bytes: int = MAX_OBS_BYTES, fetch: bool = True
               ) -> tuple[pl.DataFrame, pl.DataFrame] | str:  # fmt: skip
    """The cached aggregated obs of a dataset (read and cached once); a string is the reason it is missing."""
    CACHE.mkdir(parents=True, exist_ok=True)
    gp, ep, mp = (CACHE / f"{dataset_id}.{k}" for k in ("groups.parquet", "extra.parquet", "meta.json"))
    if gp.exists() and ep.exists():
        return pl.read_parquet(gp), pl.read_parquet(ep)
    if mp.exists():
        meta = json.loads(mp.read_text())
        # a size skip stands only for a limit at least as large as this run's (older markers carry none: retried)
        if meta.get("skipped") and meta.get("extras") == "filtered" and meta.get("max_bytes", 0) >= max_bytes:
            return meta["skipped"]
    local = DOWNLOAD / "census_pseudobulk" / "h5ads" / f"{dataset_id}.h5ad"
    local = local if local.exists() else None
    if not fetch and local is None:
        return "not cached (--no-fetch)"
    t0 = time.time()
    try:
        groups, extra, info = read_obs(url, max_bytes, local)
    except ObsTooLarge:
        mp.write_text(json.dumps({"url": url, "fetched_at": _now(), "skipped": "obs too large, needs job",
                                  "extras": "filtered", "max_bytes": max_bytes}))  # fmt: skip
        return "obs too large, needs job"
    except Exception as e:  # network or format: retried on the next run (not written as skipped)
        log.warning("%s: obs read failed: %s", dataset_id[:8], e)
        return f"obs read failed: {type(e).__name__}"
    groups.write_parquet(gp, compression="zstd")
    extra.write_parquet(ep, compression="zstd")
    mp.write_text(json.dumps({**info, "fetched_at": _now(), "seconds": round(time.time() - t0, 1)}, indent=1))
    log.info("%s: obs %d cells, %.1f MB read in %.0fs", dataset_id[:8], info["n_cells"], info["bytes_fetched"] / 1e6,
             time.time() - t0)  # fmt: skip
    return groups, extra


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _cached_worker(args: tuple[str, str, int]) -> tuple[str, str | None]:
    dataset_id, url, max_bytes = args
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    r = cached_obs(dataset_id, url, max_bytes)
    return dataset_id, r if isinstance(r, str) else None


# ------------------------------------------------------------------ CELLxGENE


def _labels(x: dict, key: str) -> list[str]:
    v = x.get(key)
    return [e["label"] for e in v] if isinstance(v, list) and v and isinstance(v[0], dict) else []


def cellxgene_candidates(discover: list[dict], extra_ids: set[str] = frozenset()) -> list[dict]:
    """Human datasets with primary cells, a non-cancer disease and ``normal`` donors (plus ``extra_ids``)."""
    out = []
    for x in discover:
        if "Homo sapiens" not in _labels(x, "organism") or True not in (x.get("is_primary_data") or []):
            continue
        dis = _labels(x, "disease")
        has = "normal" in dis and any(d != "normal" and not is_cancer(d) for d in dis)
        if has or x["dataset_id"] in extra_ids:
            out.append(x)
    return out


def _h5ad_url(x: dict, census_ids: set[str]) -> str:
    if x["dataset_id"] in census_ids:
        return f"{BUCKET_URL}/{H5AD_PREFIX}{x['dataset_id']}.h5ad"
    return next(a["url"] for a in x["assets"] if a["filetype"] == "H5AD")


def h5ad_remote(dataset_id: str, discover: list[dict], listing: dict[str, tuple[int, str]]) -> Remote:
    """The full h5ad of one dataset: the pinned Census release when it holds it (size and etag checked), else the
    current Discover asset (size from the listing)."""
    if dataset_id in listing:
        size, etag = listing[dataset_id]
        return Remote(url=f"{BUCKET_URL}/{H5AD_PREFIX}{dataset_id}.h5ad", name=f"{dataset_id}.h5ad", size=size,
                      checksum=f"s3etag:{etag}")  # fmt: skip
    x = next((d for d in discover if d["dataset_id"] == dataset_id), None)
    if x is None:
        raise KeyError(f"{dataset_id}: neither in the Census release nor in the Discover listing")
    a = next(a for a in x["assets"] if a["filetype"] == "H5AD")
    return Remote(url=a["url"], name=f"{dataset_id}.h5ad", size=a.get("filesize"))


def fetch_h5ads(dataset_ids: list[str], connections: int = 4) -> list[Path]:
    """Download whole h5ads of positive-control datasets next to the Census ones (resumable, verified)."""
    from rna_age.download import fetch as _fetch
    from rna_age.sources.census_pseudobulk import _listing

    discover = json.loads(DISCOVER_JSON.read_text())
    listing = _listing()
    dest = DOWNLOAD / "census_pseudobulk" / "h5ads"
    return [_fetch(h5ad_remote(i, discover, listing), dest, connections=connections) for i in dataset_ids]


_EMPTY_EXTRA = pl.DataFrame(schema={"donor_id": pl.String, "column": pl.String, "value": pl.String, "n_cells": pl.Int64})


def inventory_groups(dataset_id: str) -> pl.DataFrame:
    """The Census obs inventory of one dataset in the ``read_obs`` layout (no sex, cell type or author columns;
    tissue is the Census ``tissue_general``): the fallback when the h5ad obs is too large to read here."""
    g = pl.read_parquet(CENSUS_OBS).filter(pl.col("dataset_id") == dataset_id)
    return g.select(
        "donor_id", "disease", "development_stage", pl.lit(None, pl.String).alias("sex"),
        pl.col("tissue_general").alias("tissue"), "tissue_type", "assay", "suspension_type",
        pl.lit(None, pl.String).alias("cell_type"), pl.col("is_primary_data").cast(pl.String).str.to_titlecase(),
        pl.col("n_cells").cast(pl.Int64),
    )  # fmt: skip


def _donor_table(groups: pl.DataFrame) -> pl.DataFrame:
    """Primary, non-culture, postnatal cells → one row per donor × tissue × disease with its age, sex and cells."""
    g = groups
    if "is_primary_data" in g.columns:
        g = g.filter(pl.col("is_primary_data") == "True")
    if "tissue_type" in g.columns:
        g = g.filter(~pl.col("tissue_type").is_in(["cell culture", "organoid"]))
    for c in ("development_stage", "sex", "assay", "suspension_type", "cell_type", "tissue"):
        if c not in g.columns:
            g = g.with_columns(pl.lit(None, pl.String).alias(c))
    g = g.filter(~pl.col("development_stage").map_elements(is_prenatal, return_dtype=pl.Boolean))
    return g


def _arm_frames_sc(g: pl.DataFrame, tissue: str, cond: str | None) -> pl.DataFrame:
    """Donors of one tissue with arm: case = the disease ``cond``, control = normal (donors in both → case)."""
    t = g.filter(pl.col("tissue") == tissue)
    per = t.group_by("donor_id").agg(
        pl.col("disease").str.split(" || ").explode().unique().alias("diseases"),
        pl.col("development_stage").sort_by("n_cells", descending=True).first(),
        pl.col("sex").sort_by("n_cells", descending=True).first(),
        pl.col("assay").sort_by("n_cells", descending=True).first(),
        pl.col("suspension_type").sort_by("n_cells", descending=True).first(),
        pl.col("n_cells").sum(),
    )
    return per.with_columns(
        arm=pl.when(pl.col("diseases").list.contains(cond)).then(pl.lit("case"))
        .when(pl.col("diseases").list.eval(pl.element() == "normal").list.all()).then(pl.lit("control"))
    ).filter(pl.col("arm").is_not_null())  # fmt: skip


def _with_ages(donors: pl.DataFrame) -> pl.DataFrame:
    ages = [age_from_stage(s) for s in donors["development_stage"].to_list()]
    return donors.with_columns(
        age_type=pl.Series([a[0] for a in ages], dtype=pl.String),
        age=pl.Series([a[1] for a in ages], dtype=pl.Float64),
        age_any=pl.Series([a[2] for a in ages], dtype=pl.Float64),
    )


def _batch_values(extra: pl.DataFrame) -> pl.DataFrame:
    """Per donor, the majority value of every batch-like author column (wide)."""
    e = extra.filter(pl.col("column").str.contains(BATCH_LIKE.pattern))
    if not e.height:
        return pl.DataFrame(schema={"donor_id": pl.String})
    top = e.sort("n_cells", descending=True).group_by("donor_id", "column").agg(pl.col("value").first())
    return top.pivot(on="column", index="donor_id", values="value")


def _condition_columns(extra: pl.DataFrame, normal_donors: list[str]) -> list[tuple[str, str, list[str]]]:
    """Author condition columns (CMV, smoking, ...) constant within donor among normal donors: (column, reference
    level, other levels with ≥ 3 donors)."""
    out = []
    e = extra.filter(pl.col("column").str.contains(CONDITION_COLUMN.pattern) & pl.col("donor_id").is_in(normal_donors))
    for col in e["column"].unique().sort().to_list():
        c = e.filter(pl.col("column") == col)
        if c.group_by("donor_id").agg(pl.col("value").n_unique())["value"].max() > 1:
            continue
        counts = c.group_by("value").agg(pl.col("donor_id").n_unique().alias("n")).filter(pl.col("value").is_not_null())
        counts = counts.filter(~pl.col("value").str.contains(MISSING_LEVEL.pattern))
        levels = counts.filter(pl.col("n") >= 3)["value"].to_list()
        if not 2 <= len(counts) <= 6 or len(levels) < 2:
            continue
        ref = [v for v in levels if REFERENCE_LEVEL.search(v.strip())]
        if len(ref) != 1:
            continue
        out.append((col, ref[0], [v for v in levels if v != ref[0]]))
    return out


def scan_cellxgene(seed_ids: set[str], workers: int = 4, fetch: bool = True, max_bytes: int = MAX_OBS_BYTES
                   ) -> list[dict]:  # fmt: skip
    discover = json.loads(DISCOVER_JSON.read_text())
    census_ids = set(pl.read_parquet(CENSUS_DATASETS, columns=["dataset_id"])["dataset_id"].to_list())
    cands = cellxgene_candidates(discover, {d for d in seed_ids if len(d) == 36})
    inventory_ids = set(pl.read_parquet(CENSUS_OBS, columns=["dataset_id"])["dataset_id"].unique().to_list())
    log.info("CELLxGENE: %d candidate datasets", len(cands))
    jobs = [(x["dataset_id"], _h5ad_url(x, census_ids), max_bytes) for x in cands]
    status: dict[str, str | None] = {}
    if fetch:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        def done(did: str) -> bool:
            meta = CACHE / f"{did}.meta.json"
            skipped = meta.exists() and json.loads(meta.read_text()).get("extras") == "filtered"
            return (CACHE / f"{did}.groups.parquet").exists() or skipped

        todo = [j for j in jobs if not done(j[0])]
        log.info("CELLxGENE: %d obs to read over HTTP (%d cached)", len(todo), len(jobs) - len(todo))
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
            for did, err in ex.map(_cached_worker, todo):
                status[did] = err
    rows: list[dict] = []
    for x, (did, url, _) in zip(cands, jobs, strict=True):
        base = {"source": "cellxgene", "dataset_id": did, "collection_id": x["collection_id"],
                "title": f"{x['collection_name']} / {x['title']}", "url": url,
                "assay": ",".join(_labels(x, "assay")), "suspension_type": ",".join(x.get("suspension_type") or [])}  # fmt: skip
        r = cached_obs(did, url, max_bytes, fetch=False)
        if isinstance(r, str) and did in inventory_ids:
            ontology = {e["label"]: e["ontology_term_id"] for e in x.get("disease") or [] if isinstance(e, dict)}
            for row in _score_cellxgene(base, inventory_groups(did), _EMPTY_EXTRA, ontology):
                rows.append({**row, "scan_status": f"census inventory only ({r})", "coarse_cells": "types: needs job",
                             "coarse_cells_detail": []})  # fmt: skip
            continue
        if isinstance(r, str):
            for d in _labels(x, "disease"):
                if d != "normal" and not is_cancer(d):
                    oid = next((e["ontology_term_id"] for e in x["disease"] if e["label"] == d), None)
                    rows.append({**base, "condition": d, "condition_column": "disease", "condition_ontology": oid, "tissue": ",".join(_labels(x, "tissue")),
                                 "verdict": "not usable", "reason": r, "scan_status": r})  # fmt: skip
            continue
        ontology = {e["label"]: e["ontology_term_id"] for e in x.get("disease") or [] if isinstance(e, dict)}
        rows.extend(_score_cellxgene(base, *r, ontology))
    return rows


def _score_cellxgene(base: dict, groups: pl.DataFrame, extra: pl.DataFrame, ontology: dict[str, str]) -> list[dict]:
    g = _donor_table(groups)
    batch = _batch_values(extra)
    batch_cols = ["assay", "suspension_type", *[c for c in batch.columns if c != "donor_id"]]
    rows = []
    labels = {x for d in g["disease"].unique().drop_nulls().to_list() for x in d.split(" || ")}  # multi-valued
    diseases = sorted(d for d in labels if d != "normal" and not is_cancer(d))
    # a donor with a cancer label anywhere is no control and no case
    cancer_donors = g.filter(pl.col("disease").map_elements(is_cancer, return_dtype=pl.Boolean))["donor_id"].unique()
    g = g.filter(~pl.col("donor_id").is_in(cancer_donors.to_list()))
    for tissue in sorted(g["tissue"].unique().drop_nulls().to_list()):
        tcls = system_class(tissue)
        arms: list[tuple[str, str, str, pl.DataFrame]] = []
        for d in diseases:
            donors = _arm_frames_sc(g, tissue, d)
            if donors.filter(pl.col("arm") == "case").height and donors.filter(pl.col("arm") == "control").height:
                arms.append((d, "disease", "normal", donors))
        normal = _arm_frames_sc(g, tissue, "__none__").filter(pl.col("arm") == "control")
        for col, ref, others in _condition_columns(extra, normal["donor_id"].to_list()):
            vals = extra.filter((pl.col("column") == col) & pl.col("donor_id").is_in(normal["donor_id"].to_list()))
            vals = vals.group_by("donor_id").agg(pl.col("value").first())
            for lv in others:
                dn = normal.join(vals, on="donor_id").with_columns(
                    arm=pl.when(pl.col("value") == lv).then(pl.lit("case"))
                    .when(pl.col("value") == ref).then(pl.lit("control"))
                ).filter(pl.col("arm").is_not_null()).drop("value")  # fmt: skip
                if dn.filter(pl.col("arm") == "case").height:
                    arms.append((f"{col}={lv}", col, f"{col}={ref}", dn))
        for cond, ccol, ctrl_label, donors in arms:
            donors = _with_ages(donors).join(batch, on="donor_id", how="left")
            cond_text = cond if ccol == "disease" else ccol + " " + cond
            s = score(donors, [c for c in batch_cols if c in donors.columns], needs_batch=True, condition=cond_text)
            rel, sites = target_relation(cond_text, tcls)
            cells = (g.filter(pl.col("tissue") == tissue)
                     .join(donors.select("donor_id", "arm"), on="donor_id")
                     .with_columns(coarse=pl.col("cell_type").fill_null("").map_elements(coarse_class, return_dtype=pl.String)))  # fmt: skip
            text, per = coarse_summary(cells.select("donor_id", "arm", "coarse", "n_cells"))
            assays = ",".join(sorted(donors["assay"].drop_nulls().unique().to_list()))
            susp = ",".join(sorted(donors["suspension_type"].drop_nulls().unique().to_list()))
            age_types = donors["age_type"].fill_null("missing").value_counts().sort("count", descending=True)
            rows.append({**base, "condition": cond, "condition_column": ccol, "arm_case": cond,
                         "condition_ontology": ontology.get(cond) if ccol == "disease" else None,
                         "arm_control": ctrl_label, "tissue": tissue, "tissue_class": tcls,
                         "target_relation": rel, "disease_sites": sites,
                         "target_system": tcls + (" (cell types: " + ",".join(sorted({p['coarse'] for p in per})) + ")" if per else ""),
                         "assay": assays, "suspension_type": susp,
                         "age_types": ", ".join(f"{r[0]} {r[1]}" for r in age_types.iter_rows()),
                         "coarse_cells": text, "coarse_cells_detail": per, "scan_status": "scored", **s})  # fmt: skip
    return rows


# ------------------------------------------------------------------ bulk (ARCHS4 in samples_all)

BULK_DROP = ["single_cell", "tumor", "cell_line", "stem_or_organoid", "cancer_patient", "prenatal"]


def bulk_donors(samples: pl.DataFrame) -> pl.DataFrame:
    """Bulk samples → one row per study × tissue × donor with arm label (case: the disease label; control:
    is_healthy) and ages; tumour, cell line, organoid, prenatal and single-cell libraries dropped."""
    s = samples.filter(
        ~pl.col("is_pseudobulk").fill_null(False) & ~pl.col("is_tumor").fill_null(False)
        & ~pl.col("is_cell_line").fill_null(False) & ~pl.col("exclude_reason").fill_null("").is_in(BULK_DROP)
        & pl.col("is_healthy").is_not_null()
    ).with_columns(
        tissue_key=pl.col("tissue").str.to_lowercase().str.strip_chars().fill_null("unspecified"),
        label=pl.when(pl.col("is_healthy")).then(pl.lit("healthy")).otherwise(pl.col("disease").str.strip_chars()),
        donor=pl.coalesce("donor_id", "sample_id"),
    )  # fmt: skip
    s = s.filter(pl.col("is_healthy") | (pl.col("label").is_not_null()
                 & ~pl.col("label").str.contains(CONTROL_LABEL.pattern) & ~pl.col("label").str.contains(CANCER)))  # fmt: skip
    lo = pl.when(pl.col("age_type") == "exact").then(pl.col("age_years")).otherwise(pl.col("age_min"))
    hi = pl.when(pl.col("age_type") == "exact").then(pl.col("age_years")).when(pl.col("age_type") == "censored") \
        .then(pl.lit(None, pl.Float64)).otherwise(pl.col("age_max"))  # fmt: skip
    return s.with_columns(lo=lo, hi=hi)


def scan_bulk(samples: pl.DataFrame, batch_cols: tuple[str, ...] = ("platform",)) -> list[dict]:
    s = bulk_donors(samples)
    extra_cols = [c for c in batch_cols if c != "platform"]
    studies = s.group_by("study_id", "tissue_key").agg(
        pl.col("is_healthy").any().alias("h"), (~pl.col("is_healthy")).any().alias("d")
    ).filter("h", "d")
    log.info("bulk: %d study × tissue groups with disease and healthy samples", studies.height)
    rows = []
    for study, tkey in studies.select("study_id", "tissue_key").sort("study_id", "tissue_key").iter_rows():
        t = s.filter((pl.col("study_id") == study) & (pl.col("tissue_key") == tkey))
        per = t.group_by("donor").agg(
            pl.col("label").unique().alias("labels"), pl.col("lo").first(), pl.col("hi").first(),
            pl.col("sex").first(), pl.col("platform").first(), pl.col("tissue").first(), pl.len().alias("n_samples"),
            pl.col("source").first(), pl.col("cohort").first(), *[pl.col(c).first() for c in extra_cols],
        )
        tissue = t["tissue"].drop_nulls().first() if t["tissue"].drop_nulls().len() else tkey
        tcls = system_class(tissue)
        for lab in sorted({x for xs in per["labels"].to_list() for x in xs if x != "healthy"}):
            donors = per.with_columns(
                arm=pl.when(pl.col("labels").list.contains(lab)).then(pl.lit("case"))
                .when(pl.col("labels").list.eval(pl.element() == "healthy").list.all()).then(pl.lit("control"))
            ).filter(pl.col("arm").is_not_null())  # fmt: skip
            ages = [age_from_range(a, b) for a, b in zip(donors["lo"].to_list(), donors["hi"].to_list(), strict=True)]
            donors = donors.with_columns(
                age_type=pl.Series([a[0] for a in ages], dtype=pl.String),
                age=pl.Series([a[1] for a in ages], dtype=pl.Float64),
                age_any=pl.Series([a[2] for a in ages], dtype=pl.Float64),
                n_cells=pl.lit(None, pl.Int64),
            )
            if not donors.filter(pl.col("arm") == "control").height or not donors.filter(pl.col("arm") == "case").height:
                continue
            sc = score(donors, list(batch_cols), needs_batch=False, condition=lab)
            rel, sites = target_relation(lab, tcls)
            note = "vague label: arm meaningful only within the study" if is_generic(lab) else ""
            rows.append({"source": "bulk", "dataset_id": study, "collection_id": None, "condition_ontology": None,
                         "title": t["cohort"].drop_nulls().first() if t["cohort"].drop_nulls().len() else None,
                         "url": f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={study}" if study.startswith("GSE") else None,
                         "condition": lab, "condition_column": "disease", "arm_case": lab, "arm_control": "healthy",
                         "tissue": tissue, "tissue_class": tcls, "target_relation": rel, "disease_sites": sites,
                         "target_system": tcls, "assay": ",".join(sorted(donors["platform"].drop_nulls().unique().to_list())),
                         "suspension_type": "bulk",
                         "age_types": ", ".join(f"{r[0]} {r[1]}" for r in donors["age_type"].fill_null("missing")
                                                .value_counts().sort("count", descending=True).iter_rows()),
                         "coarse_cells": "bulk", "coarse_cells_detail": [], "scan_status": "scored",
                         "notes": note, **sc})  # fmt: skip
    return rows


def labels_as_samples(labels: pl.DataFrame) -> pl.DataFrame:
    """Curated GEO labels (``bulk_labels.parquet``) → the sample columns :func:`bulk_donors` reads. Samples left
    out by the rules are dropped, except microarray ones (scored, then marked not usable for want of counts)."""
    keep = labels.filter(pl.col("arm").is_not_null()
                         & (pl.col("exclude_reason").is_null() | (pl.col("exclude_reason") == "microarray")))  # fmt: skip
    age = pl.col("age_years")
    return keep.select(
        sample_id="gsm", study_id="gse", donor_id=pl.col("gse") + ":" + pl.col("donor"),
        is_healthy=pl.col("arm") == "normal", disease=pl.when(pl.col("arm") == "case").then(pl.col("condition")),
        tissue="tissue", age_type=pl.when(age.is_not_null()).then(pl.col("age_type")),
        age_years=age, age_min=age, age_max=age, sex="sex", platform="assay", batch="batch",
        source=pl.lit("bulk_labels"), cohort=pl.lit("curated GEO labels; counts: ") + pl.col("counts_source"),
        is_pseudobulk=pl.lit(False), is_tumor=pl.lit(False), is_cell_line=pl.lit(False),
        exclude_reason=pl.lit(None, pl.String),
    )  # fmt: skip


def scan_bulk_labels(labels: pl.DataFrame) -> list[dict]:
    """Bulk rows for the series with curated labels; no RNA-seq counts (microarray, or none at NCBI) → not usable."""
    counts = dict(labels.group_by("gse").agg(pl.col("counts_source").first()).iter_rows())
    rows = scan_bulk(labels_as_samples(labels), batch_cols=("platform", "batch"))
    for r in rows:
        src = counts.get(r["dataset_id"], "")
        notes = [f"curated GEO labels; counts: {src}"]
        if "cultured" in (r["tissue"] or ""):
            notes.append("cultured cells, not tissue")
        if src.startswith("none"):
            r["verdict"] = "not usable"
            r["reason"] = "; ".join(x for x in (src.removeprefix("none: "), r["reason"]) if x)
        r["notes"] = "; ".join(x for x in (r["notes"], *notes) if x)
    return rows


# ------------------------------------------------------------------ scBaseCount


def scbasecount_donors(inv: pl.DataFrame) -> pl.DataFrame:
    """scBaseCount SRX samples → usable rows (no tumour, culture, prenatal or cell line) with labels."""
    cl = pl.col("cell_line").fill_null("none").str.to_lowercase()
    s = inv.filter(
        ~pl.col("tumor").fill_null(False) & ~pl.col("culture").fill_null(False) & ~pl.col("prenatal").fill_null(False)
        & cl.is_in(["none", "unsure", "not_applicable", "na", "n/a", ""])
    ).with_columns(
        tissue_key=pl.col("tissue").str.to_lowercase().str.strip_chars().fill_null("unspecified"),
        label=pl.when(pl.col("healthy")).then(pl.lit("healthy")).otherwise(pl.col("disease").str.strip_chars()),
    )  # fmt: skip
    return s.filter(pl.col("healthy") | (~pl.col("not_diseased").fill_null(False) & pl.col("label").is_not_null()
                    & ~pl.col("label").str.contains(CONTROL_LABEL.pattern) & ~pl.col("label").str.contains(CANCER)))  # fmt: skip


def scan_scbasecount(inv: pl.DataFrame, meta: pl.DataFrame | None = None) -> list[dict]:
    s = scbasecount_donors(inv)
    if meta is not None:
        s = s.join(meta.select("srx_accession", "single_disease_confidence", "disease_ontology_term_id"),
                   on="srx_accession", how="left")
    else:
        s = s.with_columns(single_disease_confidence=pl.lit(None, pl.String),
                           disease_ontology_term_id=pl.lit(None, pl.String))
    groups = s.group_by("study", "tissue_key").agg(
        pl.col("healthy").any().alias("h"), (~pl.col("healthy")).any().alias("d")).filter("h", "d")
    log.info("scBaseCount: %d study × tissue groups with disease and healthy donors", groups.height)
    rows = []
    for study, tkey in groups.select("study", "tissue_key").sort("study", "tissue_key").iter_rows():
        t = s.filter((pl.col("study") == study) & (pl.col("tissue_key") == tkey))
        per = t.group_by("donor_key").agg(
            pl.col("label").unique().alias("labels"), pl.col("age_lo").first(), pl.col("age_hi").first(),
            pl.col("obs_count").sum().alias("n_cells"), pl.col("srx_accession").n_unique().alias("n_srx"),
            pl.col("tech_10x").first(), pl.col("cell_prep").first(), pl.col("in_census").any(),
            pl.col("single_disease_confidence").first(), pl.col("disease_ontology_term_id").first().alias("oid"),
        ).rename({"donor_key": "donor_id"})
        tissue = t["tissue"].drop_nulls().first() if t["tissue"].drop_nulls().len() else tkey
        tcls = system_class(tissue)
        for lab in sorted({x for xs in per["labels"].to_list() for x in xs if x != "healthy"}):
            donors = per.with_columns(
                arm=pl.when(pl.col("labels").list.contains(lab)).then(pl.lit("case"))
                .when(pl.col("labels").list.eval(pl.element() == "healthy").list.all()).then(pl.lit("control")),
                sex=pl.lit(None, pl.String),
            ).filter(pl.col("arm").is_not_null())  # fmt: skip
            if not donors.filter(pl.col("arm") == "control").height or not donors.filter(pl.col("arm") == "case").height:
                continue
            ages = [age_from_range(a, b) for a, b in zip(donors["age_lo"].to_list(), donors["age_hi"].to_list(), strict=True)]
            donors = donors.with_columns(
                age_type=pl.Series([a[0] for a in ages], dtype=pl.String),
                age=pl.Series([a[1] for a in ages], dtype=pl.Float64),
                age_any=pl.Series([a[2] for a in ages], dtype=pl.Float64),
            )
            sc = score(donors, ["tech_10x", "cell_prep"], needs_batch=True, condition=lab)
            one_srx = bool((donors["n_srx"] == 1).all())
            if sc["batch_status"] not in ("confounded", "borderline"):
                # chemistry and prep are no processing batch: SRX/run structure is all there is
                sc["batch_status"] = "unknown"
                sc["verdict"], sc["reason"] = verdict(sc["n_case"], sc["n_control"], sc["n_pairs_matched"],
                                                      sc["n_pairs_matched_any_age"], "unknown", True)  # fmt: skip
            notes = []
            if one_srx:
                notes.append("one SRX per donor: batch not separable from donor")
            if donors["in_census"].any():
                notes.append("study also in CELLxGENE Census (double count)")
            conf = donors.filter(pl.col("arm") == "case")["single_disease_confidence"].drop_nulls()
            if conf.len():
                notes.append("disease confidence " + ",".join(sorted(conf.unique().to_list())))
            rel, sites = target_relation(lab, tcls)
            oids = donors.filter(pl.col("arm") == "case")["oid"].drop_nulls()
            oids = oids.filter(oids.str.contains(r"^[A-Z]+:\d+$"))
            rows.append({"source": "scbasecount", "dataset_id": study, "collection_id": None, "title": None,
                         "condition_ontology": oids.mode().sort().first() if oids.len() else None,
                         "url": f"https://www.ncbi.nlm.nih.gov/sra/?term={study}",
                         "condition": lab, "condition_column": "disease", "arm_case": lab, "arm_control": "healthy",
                         "tissue": tissue, "tissue_class": tcls, "target_relation": rel, "disease_sites": sites,
                         "target_system": tcls, "assay": ",".join(sorted(donors["tech_10x"].drop_nulls().unique().to_list())),
                         "suspension_type": ",".join(sorted(donors["cell_prep"].drop_nulls().unique().to_list())),
                         "age_types": ", ".join(f"{r[0]} {r[1]}" for r in donors["age_type"].fill_null("missing")
                                                .value_counts().sort("count", descending=True).iter_rows()),
                         "coarse_cells": "types: needs counts", "coarse_cells_detail": [], "scan_status": "scored",
                         "in_census": bool(donors["in_census"].any()), "notes": "; ".join(notes), **sc})  # fmt: skip
    return rows


# ------------------------------------------------------------------ seed list, duplicates, assembly

SEED_SYNONYMS = {
    "cmv": "cmv|cytomegalo", "smoking": "smok", "lupus": "lupus", "alzheimer": "alzheimer|dementia",
    "parkinson": "parkinson|lewy", "pulmonary fibrosis": "fibrosis|interstitial", "ild": "interstitial|fibrosis",
    "ipf": "fibrosis", "copd": "obstructive|copd", "kidney": "kidney|renal", "ckd": "kidney|renal",
    "diabetes": "diabet", "covid": "covid", "arthritis": "arthritis", "sclerosis": "sclerosis", "als": "amyotrophic|als",
    "crohn": "crohn|colitis|bowel", "chip": "clonal", "immunodeficiency": "immunodef", "sjogren": "sj.gren",
    "heart failure": "cardiomyopathy|heart|myocard", "atherosclerosis": "atheroscl", "masld": "fatty|steato|nash|masld",
    "down syndrome": "down|trisomy", "hiv": "hiv|human immunodeficiency", "supranuclear": "supranuclear|pick",
    "diabetic": "diabet", "sepsis": "sepsis", "progeria": "progeria", "sarcopenia": "sarcopen", "osteoarthritis": "osteoarth",
}


def seed_ids(seed: pl.DataFrame) -> list[list[str]]:
    """Accession tokens (CELLxGENE ids or 8-hex prefixes, GEO series) of each seed row."""
    out = []
    for acc in seed["accession"].fill_null("").to_list():
        toks = re.findall(r"[0-9a-f]{8}(?:-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})?|GSE\d+|SRP\d+", acc)
        out.append(toks)
    return out


def condition_matches(seed_condition: str, condition: str) -> bool:
    sc, c = seed_condition.lower(), condition.lower()
    for key, rx in SEED_SYNONYMS.items():
        if re.search(rf"\b{re.escape(key)}", sc) and re.search(rx, c):
            return True
    words = [w for w in re.findall(r"[a-z]{5,}", sc) if w not in {"disease", "active", "acute", "chronic", "multi"}]
    return any(w in c for w in words)


def join_seed(out: pl.DataFrame, seed: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Attach seed annotations to matching rows; return (out, seed rows the scan missed with a reason column)."""
    ids = seed_ids(seed)
    ann: dict[int, dict] = {}
    found = []
    did = out["dataset_id"].to_list()
    cond = out["condition"].to_list()
    for k, row in enumerate(seed.iter_rows(named=True)):
        hits = [i for i, d in enumerate(did) if any(d == t or (len(t) == 8 and d.startswith(t)) for t in ids[k])]
        good = [i for i in hits if condition_matches(row["condition"], cond[i])]
        found.append((hits, good))
        for i in good or hits:
            if i not in ann or (ann[i]["seed_match"] == "dataset only" and good):
                ann[i] = {"seed_rank": row["rank"], "seed_condition": row["condition"],
                          "seed_system_expected_older": row["system_expected_older"], "seed_evidence": row["evidence"],
                          "seed_match": "condition" if good else "dataset only"}  # fmt: skip
    cols = ["seed_rank", "seed_condition", "seed_system_expected_older", "seed_evidence", "seed_match"]
    extra = pl.DataFrame([ann.get(i, dict.fromkeys(cols)) for i in range(out.height)],
                         schema={"seed_rank": pl.Int64, **{c: pl.String for c in cols[1:]}})  # fmt: skip
    out = pl.concat([out, extra], how="horizontal")
    missed = [k for k, (hits, good) in enumerate(found) if not good]
    return out, seed[missed].with_columns(
        hit_rows=pl.Series([len(found[k][0]) for k in missed], dtype=pl.Int64))


def miss_reason(row: dict, discover_ids: dict[str, dict], cand_ids: set[str], bulk_ids: set[str],
                bulk_any: dict[str, str], sbc_ids: set[str], hit_rows: int) -> str:  # fmt: skip
    toks = seed_ids(pl.DataFrame({"accession": [row["accession"]]}))[0]
    if hit_rows:
        return "dataset scanned but its condition is not a disease label or author column with a reference level"
    if not toks:
        return f"accession not in a scanned source ({row['source']})"
    reasons = []
    for t in toks:
        if t.startswith("GSE") or t.startswith("SRP"):
            if t in bulk_any and t not in bulk_ids:
                reasons.append(f"{t}: in ARCHS4 table, but no disease + healthy samples of one tissue in our annotation"
                               f" ({bulk_any[t]})")
            elif t in sbc_ids:
                reasons.append(f"{t}: in scBaseCount, no healthy + diseased donors of one tissue")
            else:
                reasons.append(f"{t}: series not in the ARCHS4 table (or single-cell only)")
            continue
        x = next((v for k, v in discover_ids.items() if k == t or k.startswith(t)), None)
        if x is None:
            reasons.append(f"{t}: not in the Discover listing")
        elif x["dataset_id"] in cand_ids:
            reasons.append(f"{t}: scanned, no arm pair (no normal donors of the same tissue, or prenatal/culture)")
        elif True not in (x.get("is_primary_data") or []):
            reasons.append(f"{t}: only non-primary cells (a re-published subset)")
        elif "normal" not in _labels(x, "disease"):
            reasons.append(f"{t}: no normal donors in the dataset")
        else:
            reasons.append(f"{t}: no non-cancer disease label besides normal in the Discover listing")
    return "; ".join(reasons)


def mark_duplicates(out: pl.DataFrame) -> pl.DataFrame:
    """Within a CELLxGENE collection, a row whose (condition, tissue) also appears in a dataset of the same
    collection with more case and control donors is ``covered_by`` that dataset (re-published subset)."""
    cx = out
    key = ["collection_id", "condition", "tissue"]
    best = (cx.filter(pl.col("collection_id").is_not_null() & pl.col("n_case").is_not_null())
            .with_columns(size=pl.col("n_case") + pl.col("n_control"))
            .sort("size", "n_pairs_matched", descending=True).group_by(key).agg(pl.col("dataset_id").first().alias("best"),
                                                                                 pl.len().alias("k")))  # fmt: skip
    out = out.join(best, on=key, how="left")
    return out.with_columns(
        covered_by=pl.when((pl.col("k") > 1) & (pl.col("best") != pl.col("dataset_id"))).then("best")
    ).drop("best", "k")


def run(seed_path: Path | None = None, workers: int = 4, fetch: bool = True, max_mb: int = 200) -> pl.DataFrame:
    seed = pl.read_parquet(seed_path) if seed_path and seed_path.exists() else None
    sids = {t for ts in seed_ids(seed) for t in ts} if seed is not None else set()
    discover = json.loads(DISCOVER_JSON.read_text())
    full_seed = {x["dataset_id"] for x in discover for t in sids if len(t) >= 8 and x["dataset_id"].startswith(t)}
    rows = scan_cellxgene(full_seed, workers=workers, fetch=fetch, max_bytes=max_mb << 20)
    samples = pl.read_parquet(INTERIM / "samples_all.parquet")
    labels = pl.read_parquet(BULK_LABELS) if BULK_LABELS.exists() else None
    a4 = samples.filter(pl.col("source") == "archs4")
    if labels is not None:  # curated labels override our ARCHS4 annotation for their series
        a4 = a4.filter(~pl.col("study_id").is_in(labels["gse"].unique().to_list()))
    rows += scan_bulk(a4)
    if labels is not None:
        rows += scan_bulk_labels(labels)
    inv = pl.read_parquet(SCBASECOUNT)
    meta = pl.read_parquet(SCBASECOUNT_META) if SCBASECOUNT_META.exists() else None
    rows += scan_scbasecount(inv, meta)
    out = pl.DataFrame(rows, infer_schema_length=None)
    for c in ("in_census", "notes", "covered_by"):
        if c not in out.columns:
            out = out.with_columns(pl.lit(None).alias(c))
    out = mark_duplicates(out)
    if "condition_ontology" not in out.columns:
        out = out.with_columns(condition_ontology=pl.lit(None, pl.String))
    out = condition_keys(out, ontology_lookup(discover, meta))
    if seed is not None:
        out, missed = join_seed(out, seed)
        disc = {x["dataset_id"]: x for x in discover}
        cand = {r["dataset_id"] for r in rows if r["source"] == "cellxgene"}
        bulk_ids = set(out.filter(pl.col("source") == "bulk")["dataset_id"].to_list())
        bulk_any = {r[0]: f"{r[1]} samples, healthy {r[2]}, disease {r[3]}, unlabelled {r[4]}, excluded {r[5]}"
                    for r in samples.group_by("study_id").agg(
                        pl.len(), pl.col("is_healthy").sum().alias("h"), (~pl.col("is_healthy")).sum().alias("d"),
                        pl.col("is_healthy").is_null().sum().alias("u"),
                        pl.col("exclude_reason").drop_nulls().unique().sort().str.join("/").alias("x"),
                    ).iter_rows()}  # fmt: skip
        sbc = set(inv["study"].unique().to_list())
        reasons = [miss_reason(r, disc, cand, bulk_ids, bulk_any, sbc, r["hit_rows"]) for r in missed.iter_rows(named=True)]
        missed = missed.with_columns(miss_reason=pl.Series(reasons, dtype=pl.String))
        missed.write_parquet(OUT.with_name("positive_controls_seed_missed.parquet"), compression="zstd")
    order = pl.col("verdict").replace_strict({"usable": 0, "caveat": 1, "not usable": 2}, default=3)
    out = out.with_columns(rank_key=order).sort("rank_key", "n_pairs_matched", descending=[False, True],
                                                nulls_last=True).drop("rank_key")  # fmt: skip
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(OUT, compression="zstd")
    conds = conditions_table(out, seed)
    conds.write_parquet(CONDITIONS_OUT, compression="zstd")
    log.info("positive controls: %d rows → %s; %d conditions → %s", out.height, OUT, conds.height, CONDITIONS_OUT)
    return out


# ------------------------------------------------------------------ conditions across sources

CONDITIONS_OUT = OUTPUT / "summary" / "positive_controls_conditions.parquet"


ABBREVIATIONS = {
    "sle": "systemic lupus erythematosus", "uc": "ulcerative colitis", "cd": "crohn disease", "ibd": "inflammatory bowel disease",
    "ms": "multiple sclerosis", "ra": "rheumatoid arthritis", "t2d": "type 2 diabetes mellitus",
    "t1d": "type 1 diabetes mellitus", "ad": "alzheimer disease", "pd": "parkinson disease",
    "mdd": "major depressive disorder", "copd": "chronic obstructive pulmonary disease",
    "ipf": "idiopathic pulmonary fibrosis", "tb": "tuberculosis", "ptb": "pulmonary tuberculosis",
    "active tb": "tuberculosis", "active_tb": "tuberculosis", "ltbi": "latent tuberculosis", "psor": "psoriasis",
    "pso": "psoriasis", "psc": "primary sclerosing cholangitis", "pbc": "primary biliary cholangitis",
    "als": "amyotrophic lateral sclerosis", "hd": "huntington disease", "scz": "schizophrenia", "sz": "schizophrenia",
    "bd": "bipolar disorder", "asd": "autism spectrum disorder", "ckd": "chronic kidney disease",
    "aki": "acute kidney injury", "nafld": "non-alcoholic fatty liver disease", "nash": "non-alcoholic steatohepatitis",
    "dcm": "dilated cardiomyopathy", "hcm": "hypertrophic cardiomyopathy", "oa": "osteoarthritis", "as": "ankylosing spondylitis",
    "pah": "pulmonary arterial hypertension", "cf": "cystic fibrosis", "hiv": "hiv infection",
    "covid": "covid-19", "covid19": "covid-19", "sars-cov-2": "covid-19", "ssc": "systemic sclerosis",
    "pss": "sjogren syndrome", "aps": "antiphospholipid syndrome", "mci": "mild cognitive impairment",
}
GENERIC_LABEL = re.compile(
    r"^(case|cases|affected|patient|patients|disease|diseased|yes|first|second|mild|moderate|severe|critical|exposed|"
    r"asymptomatic|symptomatic|pre|post|mid|acute|chronic|active|inactive|infected|lesion|lesional|positive|treated|"
    r"untreated|baseline|responder|non-responder|r|nr|progressor|recovered|convalescent|sick|ill|\w{1,2})$"
)


def clean_label(label: str | None) -> str:
    """Lower case, '(from text)', possessives and a trailing parenthesised abbreviation removed, spaces collapsed,
    known abbreviations expanded ('SLE' → 'systemic lupus erythematosus')."""
    s = (label or "").lower().replace("(from text)", "").replace("'s ", " ").replace("’s ", " ")
    s = re.sub(r"\s*\([^)]{1,12}\)\s*$", "", s)
    s = re.sub(r"\s+", " ", s).strip(" .,;:-_'\"")
    return ABBREVIATIONS.get(s, s)


def is_generic(label: str | None) -> bool:
    """A label meaningful only inside its own study ('case', 'Mid', 'affected', 'Infected-Vax')."""
    s = clean_label(label)
    return bool(GENERIC_LABEL.match(s)) or bool(re.search(r"\b(case|affected|patients?|group|arm|cohort)\b", s)) \
        or bool(re.fullmatch(r"[a-z]+[-_ ](vax|unvax|pre|post)", s))  # fmt: skip


def ontology_lookup(discover: list[dict], sbc_meta: pl.DataFrame | None) -> dict[str, str]:
    """Cleaned disease label → ontology id (MONDO etc.) from CELLxGENE Discover and scBaseCount's curation."""
    out: dict[str, str] = {}
    for x in discover:
        for e in x.get("disease") or []:
            if isinstance(e, dict) and e.get("ontology_term_id") and e["label"] != "normal":
                out.setdefault(clean_label(e["label"]), e["ontology_term_id"])
    if sbc_meta is not None:
        m = sbc_meta.filter(pl.col("disease_ontology_term_id").str.contains(r"^[A-Z]+:\d+$"))
        for lab, oid in m.select("disease", "disease_ontology_term_id").unique().iter_rows():
            out.setdefault(clean_label(lab), oid)
    return out


def condition_keys(out: pl.DataFrame, lookup: dict[str, str]) -> pl.DataFrame:
    """``condition_id``: the ontology id where the label (or a row's own curated id) has one, else the cleaned
    label; author-column conditions keep 'column=value'."""
    ids = []
    for cond, col, own, did in zip(out["condition"].to_list(), out["condition_column"].to_list(),
                                   out["condition_ontology"].to_list(), out["dataset_id"].to_list(), strict=True):  # fmt: skip
        if col != "disease":
            ids.append(clean_label(cond))
        elif is_generic(cond):
            ids.append(f"{did}:{clean_label(cond)}")  # 'case', 'Mid', 'affected': only meaningful within its study
        else:
            ids.append(own or lookup.get(clean_label(cond)) or clean_label(cond))
    return out.with_columns(condition_id=pl.Series(ids, dtype=pl.String))


def conditions_table(out: pl.DataFrame, seed: pl.DataFrame | None) -> pl.DataFrame:
    """One row per normalised condition across sources: studies with same-tissue healthy controls, donors (each
    study counted once, at its largest tissue), tissues, usable studies and donors, and whether the seed list had it.
    Re-published CELLxGENE subsets (``covered_by``) and scBaseCount studies also in the Census are left out."""
    o = out.filter(pl.col("covered_by").is_null() & ~pl.col("in_census").fill_null(False))
    scored = o.filter(pl.col("scan_status") == "scored")
    per_study = scored.group_by("condition_id", "source", "dataset_id").agg(
        pl.col("condition").first(), pl.col("n_case").max(), pl.col("n_control").max(),
        pl.col("n_pairs_matched").filter(pl.col("verdict") == "usable").max().alias("usable_pairs"),
        (pl.col("verdict") == "usable").any().alias("usable"), (pl.col("verdict") == "caveat").any().alias("caveat"),
        pl.col("tissue_class").unique().alias("tc"),
    )
    t = per_study.group_by("condition_id").agg(
        pl.col("condition").mode().first().alias("condition_name"),
        pl.col("condition").unique().sort().str.join(" | ").alias("labels"),
        pl.len().alias("n_studies"),
        pl.col("source").unique().sort().str.join(",").alias("sources"),
        pl.col("n_case").sum().alias("case_donors"), pl.col("n_control").sum().alias("control_donors"),
        pl.col("tc").list.explode(keep_nulls=False, empty_as_null=False).unique().sort().str.join(",").alias("tissues"),
        pl.col("usable").sum().cast(pl.Int64).alias("n_usable_studies"),
        pl.col("caveat").sum().cast(pl.Int64).alias("n_caveat_studies"),
        pl.col("usable_pairs").sum().fill_null(0).alias("usable_matched_pairs"),
    )
    unscored = o.filter(pl.col("scan_status") != "scored").group_by("condition_id").agg(
        pl.col("dataset_id").n_unique().alias("n_studies_unscored"), pl.col("condition").first().alias("c2"))
    t = t.join(unscored, on="condition_id", how="full", coalesce=True).with_columns(
        condition_name=pl.coalesce("condition_name", "c2"), n_studies=pl.col("n_studies").fill_null(0),
        n_studies_unscored=pl.col("n_studies_unscored").fill_null(0)).drop("c2")  # fmt: skip
    seeds = seed["condition"].to_list() if seed is not None else []
    has = [any(condition_matches(sc, f"{n} {lab or ''} {clean_label(n)}") for sc in seeds)
           for n, lab in zip(t["condition_name"].to_list(), t["labels"].to_list(), strict=True)]  # fmt: skip
    return t.with_columns(seed=pl.Series(has, dtype=pl.Boolean)).sort(
        "usable_matched_pairs", "n_usable_studies", "case_donors", descending=True, nulls_last=True)
