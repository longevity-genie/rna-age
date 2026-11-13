"""RM34 step 2: curated case/control labels for the bulk GEO series among RM33's positive controls.

The positive-controls scan reads bulk series from our ARCHS4 annotation, which has no usable arms for some seeds
(sarcopenia status, OA grade, karyotype sit in GEO characteristics our generic annotation does not read) and does
not hold the series missing from ARCHS4 at all. Here each series gets a rule (:data:`RULES`) that turns GEO's
sample characteristics into one label row per GSM: arm (``case`` / ``normal``), condition, exact age, sex, tissue,
a batch-like field and why a sample is left out. Rules work on the fetched characteristics, never on hand-typed
samples.

Raw text: GEO's per-series sample text (``ncbi_geo.samples_text_url``) under
``data/download/positive_controls/geo/`` with ``.meta.json`` sidecars. Output:
``data/input/positive_controls/bulk_labels.parquet``. Counts: ARCHS4 where the series is there, else
NCBI-generated GEO counts (``ncbi_geo``, targets with reason ``positive_control``); a microarray series has none.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from rna_age.ages import find_age
from rna_age.download import Remote, meta_path
from rna_age.paths import DOWNLOAD, INPUT, INTERIM
from rna_age.sources.ncbi_geo import NAME as NCBI
from rna_age.sources.ncbi_geo import _polite_fetch, counts_name, counts_url, samples_text_url

log = logging.getLogger(__name__)

GEO_DIR = DOWNLOAD / "positive_controls" / "geo"
OUT = INPUT / "positive_controls" / "bulk_labels.parquet"
SCHEMA = {
    "gse": pl.String, "gsm": pl.String, "donor": pl.String, "arm": pl.String, "condition": pl.String,
    "age_years": pl.Float64, "age_type": pl.String, "sex": pl.String, "tissue": pl.String, "batch": pl.String,
    "assay": pl.String, "counts_source": pl.String, "exclude_reason": pl.String,
}  # fmt: skip


@dataclass
class GeoSample:
    gsm: str
    title: str = ""
    source: str = ""
    chars: dict[str, str] = field(default_factory=dict)  # characteristic key (lower case) → value
    sample_type: str = ""  # SRA (sequencing) or RNA (array)
    platform: str = ""
    instrument: str = ""
    library_strategy: str = ""
    description: list[str] = field(default_factory=list)

    def get(self, key: str) -> str | None:
        return self.chars.get(key.lower())


def read_samples(text: str) -> list[GeoSample]:
    """GEO ``targ=gsm&form=text`` page → one :class:`GeoSample` per GSM."""
    out: list[GeoSample] = []
    cur: GeoSample | None = None
    for line in text.splitlines():
        if line.startswith("^SAMPLE"):
            cur = GeoSample(gsm=line.split("=", 1)[1].strip())
            out.append(cur)
            continue
        if cur is None or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if key == "!Sample_characteristics_ch1":
            k, colon, v = value.partition(":")
            if colon:
                cur.chars[k.strip().lower()] = v.strip()
        elif key == "!Sample_title":
            cur.title = value
        elif key == "!Sample_source_name_ch1":
            cur.source = value
        elif key == "!Sample_type":
            cur.sample_type = value
        elif key == "!Sample_platform_id":
            cur.platform = value
        elif key == "!Sample_instrument_model":
            cur.instrument = value
        elif key == "!Sample_library_strategy":
            cur.library_strategy = value
        elif key == "!Sample_description":
            cur.description.append(value)
    return out


# ------------------------------------------------------------------ helpers shared by the rules


def age_of(s: GeoSample, *keys: str) -> tuple[float | None, str | None]:
    """(years, age_type) from the first of ``keys`` that parses (rna_age.ages)."""
    for k in keys:
        v = s.get(k)
        if v is None:
            continue
        a = find_age([(k, v)])
        if a is not None:
            return a.age_years if a.age_type == "exact" else None, a.age_type
    return None, None


def sex_of(value: str | None) -> str | None:
    v = re.sub(r"^\d+\s*-\s*", "", (value or "").strip()).lower()  # "1-Male"
    return {"m": "male", "male": "male", "f": "female", "female": "female"}.get(v)


def assay_of(s: GeoSample) -> str:
    if s.sample_type.upper() == "RNA" or not s.library_strategy:
        return "microarray"
    return s.library_strategy


def _label(arm: str | None, condition: str | None, tissue: str, batch: str | None = None, donor: str | None = None,
           exclude: str | None = None) -> dict:  # fmt: skip
    return {"arm": arm, "condition": condition, "tissue": tissue, "batch": batch, "donor": donor,
            "exclude_reason": exclude}  # fmt: skip


# ------------------------------------------------------------------ one rule per series

HGPS = "Hutchinson-Gilford progeria (HGPS)"
SARCOPENIA = "sarcopenia"
OA = "osteoarthritis"
DOWN = "Down syndrome (trisomy 21)"
MASLD = "MASLD (NAFLD)"
ILD = "interstitial lung disease"
COPD = "chronic obstructive pulmonary disease"
# PRF: Progeria Research Foundation, abbreviated so the level does not read like the condition (scan skips those)
REPOSITORY = {"AG": "Coriell", "GM": "Coriell", "HGADFN": "PRF", "PRF": "PRF"}


def rule_gse113957(s: GeoSample) -> dict:
    """Fleischer 2018: cultured dermal fibroblasts; ``disease: HGPS|Normal``; batch = cell repository
    (Coriell vs PRF, the Progeria Research Foundation; from the ``cell id`` prefix)."""
    d = (s.get("disease") or "").strip().lower()
    cid = s.get("cell id") or ""
    prefix = re.match(r"[A-Za-z]*", cid).group().upper()
    batch = REPOSITORY.get(prefix, prefix or None)
    tissue = "dermal fibroblast (cultured)"
    if d == "hgps":
        return _label("case", HGPS, tissue, batch, cid or None)
    if d == "normal":
        return _label("normal", "normal", tissue, batch, cid or None)
    return _label(None, None, tissue, batch, cid or None, f"disease {d!r}")


def rule_gse111016(s: GeoSample) -> dict:
    """Singapore sarcopenia study: ``sarcopenia status: yes|no``, vastus lateralis; no batch field (instrument)."""
    st = (s.get("sarcopenia status") or "").strip().lower()
    tissue = s.get("tissue") or s.source or "skeletal muscle"
    arm = {"yes": "case", "no": "normal"}.get(st)
    return _label(arm, SARCOPENIA if arm == "case" else "normal" if arm else None, tissue, s.instrument or None,
                  exclude=None if arm else f"sarcopenia status {st!r}")  # fmt: skip


def rule_gse167186(s: GeoSample) -> dict:
    """Buck muscle study: ``group: Sarcopenia | Old Healthy | Young Healthy`` (bulk); ``Old``/``Young`` are the
    single-nucleus libraries, ``UNCLASSIFIED`` bulk samples without a group."""
    g = (s.get("group") or "").strip().lower()
    tissue = s.get("tissue") or s.source or "skeletal muscle"
    if any("single-nuclei" in d.lower() or "single nuclei" in d.lower() for d in s.description):
        return _label(None, None, tissue, s.instrument or None, exclude="single_nucleus")
    if g == "sarcopenia":
        return _label("case", SARCOPENIA, tissue, s.instrument or None)
    if g in ("old healthy", "young healthy"):
        return _label("normal", "normal", tissue, s.instrument or None)
    return _label(None, None, tissue, s.instrument or None, exclude=f"group {g!r}")


def rule_gse114007(s: GeoSample) -> dict:
    """Fisch 2018 knee cartilage: ``oa grade: 4`` (OA) or ``1`` (normal); two sequencing sets (HiSeq 2000 and
    NextSeq 500), each with both arms: batch = instrument."""
    g = (s.get("oa grade") or "").strip()
    tissue = s.source or "knee articular cartilage"
    arm = {"4": "case", "1": "normal"}.get(g)
    return _label(arm, OA if arm == "case" else "normal" if arm else None, tissue, s.instrument or None,
                  exclude=None if arm else f"oa grade {g!r}")  # fmt: skip


def rule_gse190125(s: GeoSample) -> dict:
    """Human Trisome Project whole blood: ``karyotype: T21|Control``; batch = ``sample source`` (A-D);
    donor = HTP participant id in the title."""
    k = (s.get("karyotype") or "").strip().lower()
    m = re.search(r"HTP\d+", s.title)
    tissue = s.get("tissue") or s.source or "whole blood"
    batch = s.get("sample source")
    donor = m.group() if m else None
    if k == "t21":
        return _label("case", DOWN, tissue, batch, donor)
    if k == "control":
        return _label("normal", "normal", tissue, batch, donor)
    return _label(None, None, tissue, batch, donor, f"karyotype {k!r}")


def rule_gse135251(s: GeoSample) -> dict:
    """Govaere 2020 liver biopsies: ``disease: NAFLD|Control`` (NAFL to NASH F4); no ages in GEO."""
    d = (s.get("disease") or "").strip().lower()
    tissue = "liver"
    if d == "nafld":
        return _label("case", MASLD, tissue, s.instrument or None)
    if d == "control":
        return _label("normal", "normal", tissue, s.instrument or None)
    return _label(None, None, tissue, s.instrument or None, exclude=f"disease {d!r}")


def rule_gse47460(s: GeoSample) -> dict:
    """LGRC lung: ``disease state: Interstitial lung disease | Chronic Obstructive Lung Disease | Control``;
    Agilent microarrays (batch = GPL); donor = LT id in the title."""
    d = (s.get("disease state") or "").strip().lower()
    m = re.match(r"LT\d+", s.title)
    donor = m.group() if m else None
    tissue = "lung"
    if d.startswith("interstitial"):
        return _label("case", ILD, tissue, s.platform or None, donor)
    if d.startswith("chronic obstructive"):
        return _label("case", COPD, tissue, s.platform or None, donor)
    if d == "control":
        return _label("normal", "normal", tissue, s.platform or None, donor)
    return _label(None, None, tissue, s.platform or None, donor, f"disease state {d!r}")


RULES: dict[str, Callable[[GeoSample], dict]] = {
    "GSE113957": rule_gse113957,
    "GSE111016": rule_gse111016,
    "GSE167186": rule_gse167186,
    "GSE114007": rule_gse114007,
    "GSE190125": rule_gse190125,
    "GSE135251": rule_gse135251,
    "GSE47460": rule_gse47460,
}
NOT_IN_ARCHS4 = ["GSE190125", "GSE135251", "GSE47460"]


def label_series(gse: str, samples: list[GeoSample], counts_source: str) -> pl.DataFrame:
    rule = RULES[gse]
    rows = []
    for s in samples:
        lab = rule(s)
        age, age_type = age_of(s, "age", "age (yr)", "age (years)")
        assay = assay_of(s)
        exclude = lab["exclude_reason"]
        if exclude is None and assay == "microarray":
            exclude = "microarray"
        rows.append({
            "gse": gse, "gsm": s.gsm, "donor": lab["donor"] or s.gsm, "arm": lab["arm"], "condition": lab["condition"],
            "age_years": age, "age_type": age_type, "sex": sex_of(s.get("sex")), "tissue": lab["tissue"],
            "batch": lab["batch"], "assay": assay, "counts_source": counts_source, "exclude_reason": exclude,
        })  # fmt: skip
    return pl.DataFrame(rows, schema=SCHEMA)


# ------------------------------------------------------------------ fetching and building


def text_path(gse: str) -> Path:
    return GEO_DIR / f"{gse}_samples.txt"


def fetch_texts(gses: list[str] | None = None) -> list[Path]:
    """GEO sample text of each series (one request at a time, NCBI's pause between)."""
    GEO_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for gse in gses or list(RULES):
        p = _polite_fetch(Remote(url=samples_text_url(gse), name=text_path(gse).name), GEO_DIR)
        if p is not None:
            out.append(p)
    return out


def counts_source(gse: str, samples: list[GeoSample], archs4: set[str], ncbi: set[str]) -> str:
    if gse in archs4:
        return "archs4"
    if samples and all(assay_of(s) == "microarray" for s in samples):
        return "none: microarray only (no RNA-seq counts)"
    if gse in ncbi:
        return "ncbi_geo"
    return "none: no NCBI-generated counts"


def build(fetch: bool = True) -> pl.DataFrame:
    """Label every series in :data:`RULES` → ``bulk_labels.parquet``."""
    if fetch:
        fetch_texts([g for g in RULES if not meta_path(text_path(g)).exists()])
    samples_all = pl.read_parquet(INTERIM / "samples_all.parquet", columns=["source", "study_id"])
    archs4 = set(samples_all.filter(pl.col("source") == "archs4")["study_id"].unique().to_list())
    ncbi_dir = DOWNLOAD / NCBI / "counts"
    frames = []
    for gse in RULES:
        path = text_path(gse)
        if not path.exists():
            log.warning("%s: no GEO text at %s", gse, path)
            continue
        samples = read_samples(path.read_text(errors="replace"))
        ncbi = {gse} if (ncbi_dir / counts_name(gse)).exists() else set()
        frames.append(label_series(gse, samples, counts_source(gse, samples, archs4, ncbi)))
    out = pl.concat(frames)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(OUT, compression="zstd")
    log.info("bulk labels: %d samples of %d series → %s", out.height, out["gse"].n_unique(), OUT)
    return out


def add_ncbi_targets(gses: list[str], reason: str = "positive_control") -> pl.DataFrame:
    """Add ``gses`` to ``ncbi_geo_targets.parquet`` (existing rows kept, a series already listed keeps its reason)."""
    from rna_age.sources.ncbi_geo import TARGETS

    old = pl.read_parquet(TARGETS) if TARGETS.exists() else pl.DataFrame(schema={"gse": pl.String, "reason": pl.String})
    new = pl.DataFrame({"gse": gses, "reason": [reason] * len(gses)}).filter(~pl.col("gse").is_in(old["gse"].to_list()))
    out = pl.concat([old, new])
    TARGETS.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(TARGETS)
    return out


def fetch_ncbi_counts(gses: list[str] | None = None) -> dict[str, str]:
    """NCBI-generated counts for the positive-control series missing from ARCHS4 (microarray series skipped),
    one request at a time. Returns gse → status."""
    gses = gses or NOT_IN_ARCHS4
    status: dict[str, str] = {}
    want = []
    for gse in gses:
        path = text_path(gse)
        samples = read_samples(path.read_text(errors="replace")) if path.exists() else []
        if samples and all(assay_of(s) == "microarray" for s in samples):
            status[gse] = "skipped: microarray"
            continue
        want.append(gse)
    add_ncbi_targets(want)
    d = DOWNLOAD / NCBI / "counts"
    d.mkdir(parents=True, exist_ok=True)
    for gse in want:
        try:
            p = _polite_fetch(Remote(url=counts_url(gse), name=counts_name(gse)), d)
        except Exception as e:  # noqa: BLE001 - recorded, the next series still runs
            status[gse] = f"failed: {type(e).__name__}: {e}"[:300]
            continue
        status[gse] = f"fetched {p.name}" if p is not None else "none: NCBI answered 404"
    for g, st in status.items():
        log.info("NCBI counts %s: %s", g, st)
    return status
