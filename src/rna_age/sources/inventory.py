"""Inventory of age-labelled human transcriptomes beyond what we use, from metadata only (no expression).

``rna-age inventory`` writes ``data/output/summary/age_inventory.parquet``: one row per dataset (Census) or
study (scBaseCount, recount3), with counts at each quality step. Three sources:

- **CELLxGENE Census** (pinned LTS ``2025-11-08``, all human datasets): the obs table read through TileDB-SOMA
  with only metadata columns, aggregated to dataset × donor × stage × disease × tissue × assay cell counts
  (``data/interim/inventory/census_obs_<version>.parquet``). Ages from ``development_stage``.
- **scBaseCount** (Arc Institute, release ``2026-01-12``, ``GeneFull_Ex50pAS``): the per-SRX sample table from the
  public GCS bucket, plus each SRX's BioSample attributes from ENA (the sample table carries no age).
- **recount3** human SRA projects: the per-project ``sra.sra.*.MD.gz`` (SRA sample attributes) and
  ``sra.recount_pred.*.MD.gz`` (predicted bulk / single cell) from the public S3 bucket; matched to ARCHS4 by
  SRX and GSM.

Counting rules (each a column, so the steps can be read separately):
- ``usable`` = exact age or a bin of ≤ 10 years (a decade), age ≥ 18. Censored (``90+``) and wide bins are counted
  apart.
- ``clean`` = usable, healthy (Census: ``normal``; SRA sources: healthy/control wording **or no disease stated**,
  since most SRA samples state no health at all; ``n_usable_healthy`` counts explicit healthy only), not tumour, not cell line /
  culture / organoid / stem-cell derived, not prenatal, and (single cell) ≥ :data:`MIN_CELLS` primary cells.
- ``new`` = clean and not already in our data (Census datasets we pseudobulked, ARCHS4 for SRA bulk, Census for
  scBaseCount via its CZI collection link).

Units: Census and scBaseCount count donors (Census: dataset collection × ``donor_id``; scBaseCount: study × donor
attribute, else BioSample), recount3 counts samples (SRA experiments).
"""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import polars as pl

from rna_age.ages import find_age, looks_prenatal, parse_age_value
from rna_age.annotate import CELL_LINE_NAMES, HEALTHY, PRENATAL, SINGLE_CELL, STEM, TUMOR, TUMOR_NORMAL
from rna_age.download import Remote, gcs_md5_hex
from rna_age.download import fetch as _fetch
from rna_age.paths import INTERIM, OUTPUT, download_dir, input_dir
from rna_age.sources import census_pseudobulk as cp
from rna_age.sources.bgee import stage_age

log = logging.getLogger(__name__)

MIN_CELLS = 200
ADULT = 18.0
OUT = OUTPUT / "summary" / "age_inventory.parquet"
INTERIM_DIR = INTERIM / "inventory"

CENSUS_LICENCE = "CC BY 4.0 (CELLxGENE Discover datasets)"
SBC_LICENCE = "not stated in Arc's scBaseCount docs (checked 2026-11-12); SRA-derived; ask Arc before redistribution"
R3_LICENCE = "recount3: CC BY 4.0 (processed data); underlying SRA data public"

# --------------------------------------------------------------------------------------------- age helpers


def classify(age_type: str | None, lo: float | None, hi: float | None, years: float | None) -> str | None:
    """exact | decade | wide_bin | censored | None (no age). A bin of ≤ 10 years counts as a decade."""
    if age_type is None:
        return None
    if age_type == "exact":
        return "exact"
    if age_type == "censored":
        return "censored"
    if lo is not None and hi is not None and hi - lo <= 10:
        return "decade"
    return "wide_bin"


def stage_years(stage: str | None) -> tuple[str | None, float | None, float | None]:
    """(age class, lower bound, upper bound) of a CELLxGENE ``development_stage`` label.

    Uses the Census source's HsapDv rules (``bgee.stage_age``) and then our free-text parser for what those miss
    (``60-79 year-old stage``, ``15-19 year-old``); ``90 year-old and over stage`` is censored. Vague stages
    (``adult stage``, ``prime adult stage``) and prenatal ones give no age.
    """
    if not stage or looks_prenatal(stage) or "post-fertilization" in stage or "LMP month" in stage:
        return None, None, None
    t, years, lo, hi = stage_age(stage)
    if t is None:
        s = re.sub(r"\s*stage$", "", stage.strip(), flags=re.I)
        if m := re.match(r"^(\d+) year-old and over$", s):
            return "censored", float(m.group(1)), None
        a = parse_age_value(s.replace("year-old", "years"), "age")
        if a is None:
            return None, None, None
        t, years, lo, hi = a.age_type, a.age_years, a.age_min, a.age_max
    if t == "exact":
        lo = hi = years
    return classify(t, lo, hi, years), lo, hi


def _adult_usable(cls: pl.Expr, lo: pl.Expr) -> pl.Expr:
    return (cls.is_in(["exact", "decade"]) & (lo >= ADULT)).fill_null(False)


# --------------------------------------------------------------------------------------------- Census


def census_obs(force: bool = False) -> Path:
    """Scan the Census LTS human obs (metadata columns only) into a dataset × donor × … cell-count table."""
    import cellxgene_census  # heavy import, only here

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    out = INTERIM_DIR / f"census_obs_{cp.CENSUS_VERSION}.parquet"
    if out.exists() and not force:
        return out
    # TileDB's S3 client ignores http(s)_proxy; pass the proxy from the environment explicitly
    import os

    cfg = {"vfs.s3.region": "us-west-2", "vfs.s3.no_sign_request": "true"}
    proxy = os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY")
    if proxy and (m := re.match(r"^(\w+)://([^:/]+):(\d+)", proxy)):
        cfg |= {"vfs.s3.proxy_scheme": m.group(1), "vfs.s3.proxy_host": m.group(2), "vfs.s3.proxy_port": m.group(3)}
    keys = ["dataset_id", "donor_id", "development_stage", "is_primary_data", "disease", "tissue_general",
            "tissue_type", "assay", "suspension_type"]  # fmt: skip
    parts: list[pl.DataFrame] = []
    t0 = time.time()
    with cellxgene_census.open_soma(census_version=cp.CENSUS_VERSION, tiledb_config=cfg) as census:
        datasets = pl.from_pandas(census["census_info"]["datasets"].read().concat().to_pandas())
        datasets.write_parquet(INTERIM_DIR / f"census_datasets_{cp.CENSUS_VERSION}.parquet", compression="zstd")
        n = 0
        for batch in census["census_data"]["homo_sapiens"].obs.read(column_names=keys):
            df = pl.from_arrow(batch).with_columns(pl.col(pl.Categorical).cast(pl.String))
            n += df.height
            parts.append(df.group_by(keys).len("n_cells"))
            if len(parts) >= 20:
                parts = [pl.concat(parts).group_by(keys).agg(pl.col("n_cells").sum())]
                log.info("Census obs: %d cells (%.0fs)", n, time.time() - t0)
    pl.concat(parts).group_by(keys).agg(pl.col("n_cells").sum()).write_parquet(out, compression="zstd")
    log.info("Census obs: %d cells scanned in %.0fs → %s", n, time.time() - t0, out)
    return out


def census_donors(obs: pl.DataFrame, datasets: pl.DataFrame) -> pl.DataFrame:
    """One row per dataset × donor with age class, flags and primary cell counts."""
    stages = obs["development_stage"].unique().to_list()
    parsed = [stage_years(s) for s in stages]
    st = pl.DataFrame(
        {"development_stage": stages, "age_class": [p[0] for p in parsed], "age_lo": [p[1] for p in parsed],
         "age_hi": [p[2] for p in parsed]},
        schema={"development_stage": pl.String, "age_class": pl.String, "age_lo": pl.Float64, "age_hi": pl.Float64},
    )  # fmt: skip
    o = obs.join(st, on="development_stage", how="left").with_columns(
        prenatal=pl.col("development_stage").str.contains(r"(?i)post-fertilization|carnegie|embryonic|LMP month|prenatal|blastula|organogenesis|gastrula"),
        culture=pl.col("tissue_type").is_in(["cell culture", "organoid"]),
        tumor=pl.col("disease").str.contains(cp.CANCER),
        normal=pl.col("disease") == "normal",
    )  # fmt: skip
    prim = pl.col("n_cells").filter(pl.col("is_primary_data"))
    d = o.group_by("dataset_id", "donor_id").agg(
        n_cells=pl.col("n_cells").sum(),
        n_primary=prim.sum(),
        n_stages=pl.col("development_stage").n_unique(),
        # the stage carrying most cells (donors sampled at one age; a few datasets list several)
        development_stage=pl.col("development_stage").sort_by("n_cells").last(),
        age_class=pl.col("age_class").sort_by("n_cells").last(),
        age_lo=pl.col("age_lo").sort_by("n_cells").last(),
        age_hi=pl.col("age_hi").sort_by("n_cells").last(),
        healthy=pl.col("normal").all(),
        tumor=pl.col("tumor").any(),
        culture=pl.col("culture").any(),
        prenatal=pl.col("prenatal").any(),
        tissue=pl.col("tissue_general").unique().sort().str.join(","),
        assay=pl.col("assay").unique().sort().str.join(","),
        disease=pl.col("disease").unique().sort().str.join(","),
    )
    d = d.join(datasets.select("dataset_id", "collection_id"), on="dataset_id", how="left")
    usable = _adult_usable(pl.col("age_class"), pl.col("age_lo"))
    ok = usable & ~pl.col("tumor") & ~pl.col("culture") & ~pl.col("prenatal") & (pl.col("n_primary") >= MIN_CELLS)
    return d.with_columns(usable=usable, ok=ok, clean=ok & pl.col("healthy"))


def census_ours() -> tuple[set[str], set[str]]:
    """(datasets we pseudobulked, datasets selected in census_pseudobulk but not yet pseudobulked)."""
    root = input_dir(cp.NAME) / "by_dataset"
    have = {p.parent.name for p in root.glob("*/groups.parquet")}
    selected = set(cp.DATASETS) | set(cp.TIER2)
    return have, selected - have


def census_inventory() -> pl.DataFrame:
    obs = pl.read_parquet(census_obs())
    datasets = pl.read_parquet(INTERIM_DIR / f"census_datasets_{cp.CENSUS_VERSION}.parquet")
    donors = census_donors(obs, datasets)
    have, selected = census_ours()
    # a donor of a dataset we have, seen again in another dataset of the same collection, is not new
    have_keys = donors.filter(pl.col("dataset_id").is_in(list(have))).select("collection_id", "donor_id").unique()
    donors = donors.join(have_keys.with_columns(_have=pl.lit(True)), on=["collection_id", "donor_id"], how="left")
    ours = pl.col("_have").is_not_null() | pl.col("dataset_id").is_in(list(have))
    donors = donors.with_columns(new=pl.col("clean") & ~ours, new_any=pl.col("ok") & ~ours)
    listing = cp._listing()
    size = pl.DataFrame({"dataset_id": list(listing), "size_gb": [v[0] / 1e9 for v in listing.values()]})
    status = (
        pl.when(pl.col("dataset_id").is_in(list(have))).then(pl.lit("have"))
        .when(pl.col("dataset_id").is_in(list(selected))).then(pl.lit("selected_not_fetched"))
        .otherwise(pl.lit("new"))
    )  # fmt: skip
    agg = donors.group_by("dataset_id").agg(_counts("n_cells", "n_primary"), _mode("tissue"), _mode("assay"))
    agg = agg.unnest("c")
    return (
        agg.join(datasets.select("dataset_id", "collection_id", pl.col("dataset_title").alias("title"),
                                 pl.col("collection_name").alias("study")), on="dataset_id", how="left")
        .join(size, on="dataset_id", how="left")
        .with_columns(source=pl.lit("census"), unit_id=pl.col("dataset_id"), count_unit=pl.lit("donor"),
                      modality=pl.lit("single_cell"), status=status, licence=pl.lit(CENSUS_LICENCE),
                      access=pl.lit(f"s3 cell-census/{cp.CENSUS_VERSION}/h5ads/<dataset_id>.h5ad (public)"))
    )  # fmt: skip


SPATIAL = r"(?i)visium|slide-seq|merfish|xenium|cosmx|stereo-seq"


def discover_after_lts() -> pl.DataFrame:
    """Human non-spatial datasets on CELLxGENE Discover that are not in the pinned Census release (published or
    revised later). Discover lists donors and stages per dataset but not which donor has which stage, so only an
    upper bound is given: ``n_usable_max`` = all donors when some stage is an adult exact/decade age."""
    discover = json.loads((download_dir(cp.NAME) / "discover_datasets.json").read_text())
    census_ids = set(pl.read_parquet(INTERIM_DIR / f"census_datasets_{cp.CENSUS_VERSION}.parquet")["dataset_id"])
    rows = []
    for x in discover:
        if x["dataset_id"] in census_ids or not any(o["label"] == "Homo sapiens" for o in x["organism"]):
            continue
        assays = [a["label"] for a in x["assay"]]
        if all(re.search(SPATIAL, a) for a in assays):
            continue
        stages = [stage_years(s["label"]) for s in x["development_stage"]]
        adult = any(c in ("exact", "decade") and lo is not None and lo >= ADULT for c, lo, _ in stages)
        diseases = [d["label"] for d in x["disease"]]
        n_donors = len(x["donor_id"] or [])
        rows.append({
            "unit_id": x["dataset_id"], "title": x["title"], "study": x["collection_name"],
            "n_units": n_donors, "n_cells": int(x["primary_cell_count"] or 0),
            "n_usable_max": n_donors if adult else 0,
            "tissue": ",".join(sorted({t["label"] for t in x["tissue"]}))[:200], "assay": ",".join(assays)[:200],
            "size_gb": sum(a["filesize"] for a in x["assets"] if a["filetype"] == "H5AD") / 1e9,
            "status": "after_lts_" + ("tumor" if any(re.search(cp.CANCER, d) for d in diseases)
                                       else "normal" if diseases == ["normal"] else "mixed"),
        })  # fmt: skip
    # a collection often publishes the same donors as several subsets (cell-type splits, "all cells"): count the
    # donors once per collection, on its largest dataset
    first = pl.col("n_units").rank("ordinal", descending=True).over("study") == 1
    return pl.DataFrame(rows).with_columns(
        n_usable_max=pl.when(first).then(pl.col("n_usable_max")).otherwise(0),
        source=pl.lit("cellxgene_discover"), count_unit=pl.lit("donor"), modality=pl.lit("single_cell"),
        licence=pl.lit(CENSUS_LICENCE),
        access=pl.lit("Discover dataset h5ad (datasets.cellxgene.cziscience.com, public); not in Census LTS"),
    )  # fmt: skip


def _mode(col: str) -> pl.Expr:
    return pl.col(col).drop_nulls().mode().sort().first().alias(col)


def _counts(cells: str, primary: str | None = None) -> pl.Expr:
    """Per-group counts at each quality step, packed in a struct ``c``."""
    p = pl.col(primary or cells)
    u, cl = pl.col("usable"), pl.col("clean")
    adult_any = pl.col("age_class").is_not_null() & (pl.col("age_lo") >= ADULT)
    return pl.struct(
        n_units=pl.len(),
        n_cells=pl.col(cells).sum(),
        n_age_any=pl.col("age_class").is_not_null().sum(),
        n_adult_exact=(adult_any & (pl.col("age_class") == "exact")).sum(),
        n_adult_decade=(adult_any & (pl.col("age_class") == "decade")).sum(),
        n_adult_censored=(adult_any & (pl.col("age_class") == "censored")).sum(),
        n_adult_wide_bin=(adult_any & (pl.col("age_class") == "wide_bin")).sum(),
        n_usable=u.sum(),
        n_usable_healthy=(u & pl.col("healthy")).sum(),
        n_ok_any_health=pl.col("ok").sum(),
        n_clean=cl.sum(),
        n_new=pl.col("new").sum(),
        n_new_any_health=pl.col("new_any").sum(),
        cells_usable=p.filter(u).sum(),
        cells_clean=p.filter(cl).sum(),
        cells_new=p.filter(pl.col("new")).sum(),
        cells_new_any_health=p.filter(pl.col("new_any")).sum(),
    ).alias("c")


# --------------------------------------------------------------------------------------------- SRA helpers

ENA_SEARCH = "https://www.ebi.ac.uk/ena/portal/api/search"
ENA_XML = "https://www.ebi.ac.uk/ena/browser/api/xml/"
_ATTR = re.compile(r"<SAMPLE_ATTRIBUTE>\s*<TAG>(.*?)</TAG>\s*<VALUE>(.*?)</VALUE>", re.S)
_SAMPLE = re.compile(r'<SAMPLE accession="([^"]+)".*?</SAMPLE>', re.S)
_TITLE = re.compile(r"<TITLE>(.*?)</TITLE>", re.S)


def _get(c: httpx.Client, url: str, data: dict | None = None, tries: int = 6) -> bytes:
    for i in range(tries):
        try:
            r = c.post(url, data=data) if data is not None else c.get(url)
            if r.status_code in (429, 500, 502, 503, 504):
                raise httpx.HTTPStatusError(str(r.status_code), request=r.request, response=r)
            return r.raise_for_status().content
        except (httpx.HTTPError, httpx.TransportError) as e:
            wait = 10 * 2**i
            log.warning("%s: %s; retry in %ds", url[:80], e, wait)
            time.sleep(wait)
    raise RuntimeError(f"giving up on {url}")


def ena_samples(srx: list[str], dest: Path, batch: int = 200, pause: float = 0.5, workers: int = 3) -> None:
    """SRX → BioSample mapping (ENA portal TSV) and the BioSamples' attributes (ENA browser XML), in batches.

    Raw responses are kept (``ena/experiments_NNNNN.tsv``, ``ena/samples_NNNNN.xml.gz``); finished batches are
    skipped on rerun. Experiments: one connection; samples: a few batches in flight (EBI asks for moderate rates).
    """
    dest.mkdir(parents=True, exist_ok=True)
    srx = sorted({x for x in srx if re.match(r"^[SED]RX\d+$", x)})  # scBaseCount also has non-INSDC NRX ids
    with httpx.Client(timeout=300, follow_redirects=True) as c:
        for i in range(0, len(srx), batch):
            tsv = dest / f"experiments_{i // batch:05d}.tsv"
            if tsv.exists():
                continue
            body = _get(c, ENA_SEARCH, data={
                "result": "read_experiment", "includeAccessions": ",".join(srx[i : i + batch]), "format": "tsv",
                "fields": "experiment_accession,sample_accession,secondary_sample_accession,study_accession,"
                          "secondary_study_accession"})  # fmt: skip
            tsv.with_suffix(".tmp").write_bytes(body)
            tsv.with_suffix(".tmp").rename(tsv)
            time.sleep(pause)
        mapping = ena_experiments(dest)
        samples = sorted(set(mapping["sample_accession"].drop_nulls().to_list()))

    def one(i: int) -> None:
        xml = dest / f"samples_{i // batch:05d}.xml.gz"
        if xml.exists():
            return
        failed: list[str] = []
        with httpx.Client(timeout=300, follow_redirects=True) as c:
            body = _ena_xml(c, samples[i : i + batch], failed)
        if failed:
            (dest / f"samples_{i // batch:05d}.failed.json").write_text(json.dumps(failed))
        tmp = xml.with_suffix(".tmp")
        tmp.write_bytes(gzip.compress(body))
        tmp.rename(xml)
        if (i // batch) % 10 == 0:
            log.info("ENA samples: batch %d of %d", i // batch, len(samples) // batch + 1)
        time.sleep(pause)

    # a batch of 200 SAMN records takes ~30-60 s on ENA's side; a few in flight keep the rate moderate
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(one, range(0, len(samples), batch)))


def _ena_xml(c: httpx.Client, accs: list[str], failed: list[str]) -> bytes:
    """ENA sample XML for a batch; a batch the server keeps failing (HTTP 500 on some records) is split in halves,
    and single records that still fail are listed in ``failed``. The parts are concatenated as fetched."""
    try:
        return _get(c, ENA_XML + ",".join(accs), tries=2)
    except RuntimeError:
        if len(accs) == 1:
            log.warning("ENA: %s not served", accs[0])
            failed.append(accs[0])
            return b""
        mid = len(accs) // 2
        return _ena_xml(c, accs[:mid], failed) + b"\n" + _ena_xml(c, accs[mid:], failed)


def ena_experiments(dest: Path) -> pl.DataFrame:
    parts = [pl.read_csv(p, separator="\t", infer_schema=False) for p in sorted(dest.glob("experiments_*.tsv"))
             if p.stat().st_size > 0]  # fmt: skip
    return pl.concat(parts).drop("run_accession", strict=False).unique() if parts else pl.DataFrame()


def ena_attributes(dest: Path) -> pl.DataFrame:
    """sample_accession, title, fields (list of key/value) from the kept ENA sample XML."""
    rows = []
    for p in sorted(dest.glob("samples_*.xml.gz")):
        text = gzip.decompress(p.read_bytes()).decode("utf-8", "replace")
        for m in _SAMPLE.finditer(text):
            block = m.group(0)
            t = _TITLE.search(block)
            rows.append({
                "sample_accession": m.group(1),
                "title": t.group(1).strip() if t else "",
                "fields": [{"key": _unxml(k).strip(), "value": _unxml(v).strip()} for k, v in _ATTR.findall(block)],
            })  # fmt: skip
    return pl.DataFrame(
        rows,
        schema={"sample_accession": pl.String, "title": pl.String,
                "fields": pl.List(pl.Struct({"key": pl.String, "value": pl.String}))},
    )  # fmt: skip


def _unxml(s: str) -> str:
    return (
        s.replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"').replace("&apos;", "'").replace("&amp;", "&")
    )


def sra_flags(df: pl.DataFrame, fields: str = "fields", extra_text: list[str] | None = None) -> pl.DataFrame:
    """Age (via ``ages.find_age``) and quality flags from SRA/BioSample key-value attributes.

    Adds age_class, age_lo, age_hi, age_raw, donor (a donor-like attribute), healthy, tumor, culture, prenatal.
    """
    ages = [find_age([(f["key"], f["value"]) for f in fs]) if fs else None for fs in df[fields].to_list()]
    cls, lo, hi = [], [], []
    for a in ages:
        if a is None:
            cls.append(None), lo.append(None), hi.append(None)
            continue
        low = a.age_years if a.age_type == "exact" else a.age_min
        high = a.age_years if a.age_type == "exact" else a.age_max
        cls.append(classify(a.age_type, low, high, a.age_years)), lo.append(low), hi.append(high)
    kv = (
        pl.col(fields)
        .list.eval(pl.element().struct.field("key") + pl.lit(": ") + pl.element().struct.field("value"))
        .list.join(" | ")
    )
    text = pl.concat_str(
        [kv, *[pl.col(c).fill_null("") for c in (extra_text or [])]], separator=" | "
    ).str.to_lowercase()
    from rna_age.annotate import CELL_LINE_KEYS, DISEASE_KEYS, DONOR_KEYS, NA_LIKE, field

    cell_line = field(fields, CELL_LINE_KEYS).str.to_lowercase().str.strip_chars()

    disease = field(fields, DISEASE_KEYS).str.to_lowercase()
    return (
        df.with_columns(
            age_class=pl.Series(cls, dtype=pl.String),
            age_lo=pl.Series(lo, dtype=pl.Float64),
            age_hi=pl.Series(hi, dtype=pl.Float64),
            age_raw=pl.Series([a.raw if a else None for a in ages], dtype=pl.String),
            donor=field(fields, DONOR_KEYS),
            _text=text,
            _disease=disease,
            _cell_line=(cell_line.is_not_null() & ~cell_line.str.contains(NA_LIKE)).fill_null(False),
        )
        .with_columns(
            tumor=pl.col("_text").str.contains(TUMOR) & ~pl.col("_text").str.contains(TUMOR_NORMAL),
            culture=pl.col("_text").str.contains(CELL_LINE_NAMES)
            | pl.col("_text").str.contains(STEM)
            | pl.col("_cell_line"),
            prenatal=pl.col("_text").str.contains(PRENATAL),
            healthy=pl.col("_disease").str.contains(HEALTHY).fill_null(False)
            | pl.col("_text").str.contains(r"healthy (?:donor|control|volunteer|subject|individual)"),
        )
        .with_columns(
            not_diseased=(pl.col("healthy") | pl.col("_disease").is_null()) & ~pl.col("tumor"),
        )
        .drop("_text", "_disease", "_cell_line")
    )


# --------------------------------------------------------------------------------------------- scBaseCount

SBC_BUCKET = "arc-institute-virtual-cell-atlas"
SBC_RELEASE = "2026-01-12"
SBC_SAMPLE = f"scbasecount/{SBC_RELEASE}/metadata/GeneFull_Ex50pAS/Homo_sapiens/sample_metadata.parquet"
SBC_NA = r"^(?:|none|none reported|not applicable|not_applicable|n/?a|na|null|not specified|unsure|no|primary|primary cells?|unknown|other)$"


def scbasecount_fetch() -> None:
    d = download_dir("scbasecount")
    meta = httpx.get(f"https://storage.googleapis.com/storage/v1/b/{SBC_BUCKET}/o/" + SBC_SAMPLE.replace("/", "%2F"),
                     timeout=60).raise_for_status().json()  # fmt: skip
    _fetch(Remote(url=f"https://storage.googleapis.com/{SBC_BUCKET}/{SBC_SAMPLE}", name="hs_sample_metadata.parquet",
                  size=int(meta["size"]), checksum="md5:" + gcs_md5_hex(meta["md5Hash"])), d, connections=1)  # fmt: skip
    gcs_list(f"scbasecount/{SBC_RELEASE}/h5ad/GeneFull_Ex50pAS/Homo_sapiens/", d / "listing_h5ad")
    s = pl.read_parquet(d / "hs_sample_metadata.parquet")
    ena_samples(s["srx_accession"].to_list(), d / "ena")


def gcs_list(prefix: str, dest: Path) -> pl.DataFrame:
    """name, size of every object under a prefix of the scBaseCount bucket (raw JSON pages kept)."""
    dest.mkdir(parents=True, exist_ok=True)
    url = f"https://storage.googleapis.com/storage/v1/b/{SBC_BUCKET}/o"
    rows, token, page = [], None, 0
    with httpx.Client(timeout=120) as c:
        while True:
            path = dest / f"list_{page:04d}.json"
            if path.exists():
                body = json.loads(path.read_text())
            else:
                params = {"prefix": prefix, "maxResults": "1000", "fields": "nextPageToken,items(name,size)"}
                if token:
                    params["pageToken"] = token
                body = json.loads(_get(c, url + "?" + str(httpx.QueryParams(params))))
                path.write_text(json.dumps(body))
            rows += [{"name": x["name"], "size": int(x["size"])} for x in body.get("items", [])]
            token, page = body.get("nextPageToken"), page + 1
            if not token:
                return pl.DataFrame(rows, schema={"name": pl.String, "size": pl.Int64})


def scbasecount_inventory(archs4_ids: pl.DataFrame) -> pl.DataFrame:
    d = download_dir("scbasecount")
    s = pl.read_parquet(d / "hs_sample_metadata.parquet")
    exp = ena_experiments(d / "ena")
    attrs = ena_attributes(d / "ena")
    s = (
        s.join(exp.rename({"experiment_accession": "srx_accession"}), on="srx_accession", how="left")
        .join(attrs, on="sample_accession", how="left")
        .with_columns(pl.col("fields").fill_null([]))
    )
    s = sra_flags(s, extra_text=["title", "tissue", "cell_line", "disease", "perturbation"])
    sbc_cell_line = ~pl.col("cell_line").str.to_lowercase().str.strip_chars().str.contains(SBC_NA).fill_null(True)
    sbc_healthy = (
        pl.col("disease").str.to_lowercase().str.contains(r"^(?:none|normal|healthy|none reported|healthy control)$")
    )
    s = s.with_columns(
        culture=pl.col("culture") | sbc_cell_line,
        tumor=pl.col("tumor") | pl.col("disease").str.to_lowercase().str.contains(TUMOR).fill_null(False),
        healthy=(pl.col("healthy") | sbc_healthy.fill_null(False)),
        study=pl.coalesce("secondary_study_accession", "study_accession", pl.lit("unknown")),
        biosample=pl.coalesce("sample_accession", "srx_accession"),
        in_census=pl.col("czi_collection_id").is_not_null(),
        in_archs4=pl.col("srx_accession").is_in(archs4_ids["srx"].drop_nulls().implode()),
    )
    sizes = gcs_list(f"scbasecount/{SBC_RELEASE}/h5ad/GeneFull_Ex50pAS/Homo_sapiens/", d / "listing_h5ad")
    sizes = sizes.select(srx_accession=pl.col("name").str.extract(r"([A-Z]RX\d+)\.h5ad$"), h5ad_bytes="size")
    s = s.join(sizes, on="srx_accession", how="left")
    s = s.with_columns(
        healthy=pl.col("healthy") & ~pl.col("tumor"),
        not_diseased=(
            pl.col("healthy")
            | (
                pl.col("not_diseased")
                & pl.col("disease").str.to_lowercase().str.strip_chars().str.contains(SBC_NA).fill_null(True)
            )
        )
        & ~pl.col("tumor"),
        donor_key=pl.col("study") + ":" + pl.coalesce("donor", "biosample"),
    )
    s.select("srx_accession", "study", "biosample", "donor_key", "obs_count", "age_class", "age_lo", "age_hi",
             "age_raw", "healthy", "not_diseased", "tumor", "culture", "prenatal", "in_census", "in_archs4", "cell_prep", "tech_10x",
             "tissue", "disease", "cell_line", "czi_collection_name").write_parquet(
        INTERIM_DIR / "scbasecount_samples.parquet", compression="zstd")  # fmt: skip
    donors = s.group_by("study", "donor_key").agg(
        n_cells=pl.col("obs_count").sum(),
        n_srx=pl.len(),
        h5ad_bytes=pl.col("h5ad_bytes").sum(),
        age_class=pl.col("age_class").drop_nulls().first(),
        age_lo=pl.col("age_lo").drop_nulls().first(),
        age_hi=pl.col("age_hi").drop_nulls().first(),
        n_ages=pl.col("age_lo").drop_nulls().n_unique(),
        healthy=pl.col("healthy").all(),
        not_diseased=pl.col("not_diseased").all(),
        tumor=pl.col("tumor").any(),
        culture=pl.col("culture").any(),
        prenatal=pl.col("prenatal").any(),
        in_census=pl.col("in_census").any(),
        tissue=pl.col("tissue").drop_nulls().mode().first(),
        assay=pl.col("tech_10x").drop_nulls().mode().first(),
        title=pl.col("czi_collection_name").drop_nulls().first(),
    )
    usable = _adult_usable(pl.col("age_class"), pl.col("age_lo")) & (pl.col("n_ages") <= 1)
    ok = usable & ~pl.col("tumor") & ~pl.col("culture") & ~pl.col("prenatal") & (pl.col("n_cells") >= MIN_CELLS)
    donors = donors.with_columns(usable=usable, ok=ok, clean=ok & pl.col("not_diseased")).with_columns(
        new=pl.col("clean") & ~pl.col("in_census"), new_any=pl.col("ok") & ~pl.col("in_census")
    )
    agg = donors.group_by("study").agg(_counts("n_cells"), _mode("tissue"), _mode("assay"),
                                      pl.col("title").drop_nulls().first(), pl.col("in_census").any().alias("_c"),
                                      pl.col("n_srx").sum().alias("n_samples"),
                                      (pl.col("h5ad_bytes").sum() / 1e9).alias("size_gb")).unnest("c")  # fmt: skip
    return agg.with_columns(
        source=pl.lit("scbasecount"), unit_id=pl.col("study"), study=pl.col("study"), count_unit=pl.lit("donor"),
        modality=pl.lit("single_cell"), status=pl.when("_c").then(pl.lit("in_census")).otherwise(pl.lit("new")),
        licence=pl.lit(SBC_LICENCE),
        access=pl.lit(f"gs://{SBC_BUCKET}/scbasecount/{SBC_RELEASE}/h5ad/GeneFull_Ex50pAS/Homo_sapiens/<SRX>.h5ad (public)"),
    ).drop("_c")  # fmt: skip


# --------------------------------------------------------------------------------------------- recount3

R3_BUCKET = "https://recount-opendata.s3.amazonaws.com"
R3_PREFIX = "recount3/release/human/data_sources/sra/metadata/"
R3_KINDS = ("sra.sra.", "sra.recount_pred.")


def _s3_list(c: httpx.Client, prefix: str, dest: Path) -> list[tuple[str, str, int]]:
    """(key, etag, size) under a prefix; raw XML pages kept in ``dest``."""
    dest.mkdir(parents=True, exist_ok=True)
    out, token, page = [], None, 0
    while True:
        path = dest / f"list_{page:04d}.xml"
        if path.exists():
            xml = path.read_text()
        else:
            params = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
            if token:
                params["continuation-token"] = token
            xml = c.get(R3_BUCKET + "/", params=params).raise_for_status().text
            path.write_text(xml)
        out += [(k, e, int(s)) for k, e, s in re.findall(
            r"<Key>([^<]+)</Key>.*?<ETag>&quot;([^&]+)&quot;</ETag>.*?<Size>(\d+)</Size>", xml, re.S)]  # fmt: skip
        m = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", xml)
        if not m:
            return out
        token, page = m.group(1), page + 1


def recount3_fetch(concurrency: int = 8) -> None:
    """Per-project SRA attribute and predicted-type tables of recount3 human (small files, checked by S3 md5)."""
    d = download_dir("recount3")
    (d / "listing").mkdir(exist_ok=True)
    with httpx.Client(timeout=120) as c:
        keys = _s3_list(c, R3_PREFIX, d / "listing")
        (d / "listing_gene_sums").mkdir(exist_ok=True)
        _s3_list(c, R3_PREFIX.replace("/metadata/", "/gene_sums/"), d / "listing_gene_sums")
    want = [(k, e, s) for k, e, s in keys if k.rsplit("/", 1)[-1].startswith(R3_KINDS)]
    log.info("recount3: %d listed, %d to keep", len(keys), len(want))

    async def one(client: httpx.AsyncClient, sem: asyncio.Semaphore, key: str, etag: str, size: int) -> dict:
        dest = d / "metadata" / key.removeprefix(R3_PREFIX)
        if dest.exists() and dest.stat().st_size == size:
            return {"key": key, "etag": etag, "size": size}
        async with sem:
            for i in range(5):
                try:
                    r = await client.get(f"{R3_BUCKET}/{key}")
                    body = r.raise_for_status().content
                    if hashlib.md5(body).hexdigest() != etag:
                        raise ValueError("md5 mismatch")
                    break
                except (httpx.HTTPError, ValueError) as e:
                    log.warning("%s: %s", key, e)
                    await asyncio.sleep(5 * 2**i)
            else:
                raise RuntimeError(f"giving up on {key}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(body)
        return {"key": key, "etag": etag, "size": size}

    async def run() -> list[dict]:
        sem = asyncio.Semaphore(concurrency)
        async with httpx.AsyncClient(timeout=120) as client:
            return await asyncio.gather(*(one(client, sem, *w) for w in want))

    done = asyncio.run(run())
    (d / "metadata.meta.json").write_text(json.dumps(
        {"url": f"{R3_BUCKET}/{R3_PREFIX}", "files": len(done), "bytes": sum(x["size"] for x in done),
         "checksum": "s3 md5 (ETag) per file, verified", "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")},
        indent=2))  # fmt: skip


def _read_md(path: Path) -> pl.DataFrame:
    return pl.read_csv(path, separator="\t", infer_schema=False, quote_char=None, truncate_ragged_lines=True)


def recount3_samples() -> pl.DataFrame:
    """All projects' SRA metadata joined with the predicted type; cached in interim (reading 17k files is slow)."""
    root = download_dir("recount3") / "metadata"
    cache = INTERIM_DIR / "recount3_sra_metadata.parquet"
    stamp = download_dir("recount3") / "metadata.meta.json"
    if cache.exists() and stamp.exists() and cache.stat().st_mtime > stamp.stat().st_mtime:
        return pl.read_parquet(cache)
    cols = ["external_id", "study", "sample_acc", "experiment_acc", "sample_attributes", "sample_name",
            "sample_title", "study_title", "library_strategy", "library_source", "num_bases"]  # fmt: skip
    parts, preds = [], []
    for p in sorted(root.glob("*/*/sra.sra.*.MD.gz")):
        df = _read_md(p)
        parts.append(df.select([c for c in cols if c in df.columns]))
    for p in sorted(root.glob("*/*/sra.recount_pred.*.MD.gz")):
        preds.append(_read_md(p))
    s = pl.concat(parts, how="diagonal_relaxed")
    pred = pl.concat(preds, how="diagonal_relaxed")
    if "external_id" in pred.columns:
        type_col = next((c for c in pred.columns if c.endswith("pred.type")), None)
        if type_col:
            s = s.join(pred.select("external_id", pl.col(type_col).alias("pred_type")), on="external_id", how="left")
    s.write_parquet(cache, compression="zstd")
    return s


def recount3_inventory(archs4_ids: pl.DataFrame) -> pl.DataFrame:
    s = recount3_samples()
    # runs → experiments (one sample per SRX)
    s = s.group_by("experiment_acc").agg(pl.all().first(), pl.len().alias("n_runs"))
    s = s.with_columns(
        fields=pl.col("sample_attributes").fill_null("").str.split("|").list.eval(
            pl.struct(key=pl.element().str.split(";;").list.first(),
                      value=pl.element().str.split(";;").list.slice(1).list.join(";;"))
        ).list.eval(pl.element().filter(pl.element().struct.field("value") != "")),
    )  # fmt: skip
    s = sra_flags(s, extra_text=["sample_title", "study_title"])
    gsm = pl.col("sample_name").str.extract(r"^(GSM\d+)")
    s = s.with_columns(
        in_archs4=pl.col("experiment_acc").is_in(archs4_ids["srx"].drop_nulls().implode())
        | gsm.is_in(archs4_ids["gsm"].drop_nulls().implode()).fill_null(False),
        # recount3's predicted type is "rna-seq" / "scrna-seq" (and a few others); text catches the rest
        single_cell=pl.col("pred_type").str.contains("sc").fill_null(False)
        | pl.concat_str("sample_title", "study_title", separator=" ")
        .str.to_lowercase()
        .str.contains(SINGLE_CELL)
        .fill_null(False),
    )
    usable = _adult_usable(pl.col("age_class"), pl.col("age_lo"))
    ok = usable & ~pl.col("tumor") & ~pl.col("culture") & ~pl.col("prenatal") & ~pl.col("single_cell")
    s = s.with_columns(
        usable=usable, ok=ok, clean=ok & pl.col("not_diseased"), n_cells=pl.lit(None, pl.Int64)
    ).with_columns(new=pl.col("clean") & ~pl.col("in_archs4"), new_any=pl.col("ok") & ~pl.col("in_archs4"))
    s.select("experiment_acc", "external_id", "study", "sample_name", "age_class", "age_lo", "age_hi", "age_raw",
             "healthy", "not_diseased", "tumor", "culture", "prenatal", "single_cell", "in_archs4", "usable", "clean",
             "new").write_parquet(INTERIM_DIR / "recount3_samples.parquet", compression="zstd")  # fmt: skip
    # extra per-study counts: usable adult samples not in ARCHS4 (any health)
    with httpx.Client(timeout=120) as c:
        gs = _s3_list(c, R3_PREFIX.replace("/metadata/", "/gene_sums/"), download_dir("recount3") / "listing_gene_sums")
    size = (
        pl.DataFrame(
            [(k.rsplit("/", 2)[-2], sz) for k, _, sz in gs if k.endswith(".G026.gz")],
            schema=["study", "bytes"],
            orient="row",
        )
        .group_by("study")
        .agg(size_gb=pl.col("bytes").sum() / 1e9)
    )
    agg = (
        s.group_by("study")
        .agg(
            _counts("n_cells"),
            pl.col("study_title").first().alias("title"),
            (pl.col("usable") & ~pl.col("in_archs4")).sum().alias("n_usable_not_archs4"),
            pl.col("in_archs4").all().alias("_all"),
            pl.col("in_archs4").any().alias("_any"),
        )
        .unnest("c")
        .join(size, on="study", how="left")
    )
    no_cells = [c for c in agg.columns if c == "n_cells" or c.startswith("cells_")]
    return agg.with_columns(
        *[pl.lit(None, pl.Int64).alias(c) for c in no_cells],
        source=pl.lit("recount3"), unit_id=pl.col("study"), count_unit=pl.lit("sample"), modality=pl.lit("bulk"),
        status=pl.when("_all").then(pl.lit("in_archs4")).when("_any").then(pl.lit("partly_in_archs4"))
        .otherwise(pl.lit("new")),
        licence=pl.lit(R3_LICENCE),
        access=pl.lit(f"{R3_BUCKET}/recount3/release/human/data_sources/sra/ (gene sums per project, public)"),
    ).drop("_all", "_any")  # fmt: skip


# --------------------------------------------------------------------------------------------- all


def archs4_ids() -> pl.DataFrame:
    """GSM and SRX of every ARCHS4 human sample (our bulk source), from the parsed full metadata."""
    a = pl.scan_parquet(input_dir("archs4") / "metadata_all.parquet").select("geo_accession", "relation").collect()
    return a.select(gsm=pl.col("geo_accession"), srx=pl.col("relation").str.extract(r"term=([SED]RX\d+)"))


COLUMNS = ["source", "unit_id", "study", "title", "modality", "count_unit", "status", "n_units", "n_samples",
           "n_cells", "n_age_any", "n_adult_exact", "n_adult_decade", "n_adult_censored", "n_adult_wide_bin",
           "n_usable", "n_usable_healthy", "n_ok_any_health", "n_clean", "n_new", "n_new_any_health", "n_usable_max",
           "n_usable_not_archs4", "cells_usable", "cells_clean", "cells_new", "cells_new_any_health", "tissue", "assay", "size_gb", "licence", "access"]  # fmt: skip


def inventory(fetch: bool = True) -> Path:
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    if fetch:
        scbasecount_fetch()
        recount3_fetch()
    ids = archs4_ids()
    parts = [census_inventory(), discover_after_lts(), scbasecount_inventory(ids), recount3_inventory(ids)]
    out = pl.concat(
        [p.select([pl.col(c) if c in p.columns else pl.lit(None).alias(c) for c in COLUMNS]) for p in parts],
        how="vertical_relaxed",
    ).sort("source", "n_new", descending=[False, True])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(OUT, compression="zstd")
    log.info("inventory: %d rows → %s", out.height, OUT)
    return OUT
