"""NCBI-generated GEO RNA-seq counts (beta): all human GEO RNA-seq runs recounted by NCBI on GRCh38.p13.

Two uses:
1. **Cross-check**: the same GSMs counted by an independent pipeline (NCBI vs ARCHS4) for the largest
   age-annotated ARCHS4 series.
2. **Gap filling**: age-annotated series that have NCBI counts but are missing from ARCHS4.

NCBI throttles bursts (it answered 403 for about an hour after ~4 GB): one request at a time, a pause between
requests (``PAUSE`` s), long back-off on 403/429, and a per-run cap on new files so a run never bursts.

Targets are chosen by :func:`select_targets` from ARCHS4's parsed tables and written to
``data/interim/ncbi_geo_targets.parquet`` (columns ``gse``, ``reason``); ``fetch`` reads that list.
Counts are keyed by Entrez GeneID; ``Human.GRCh38.p13.annot.tsv.gz`` maps them to Ensembl IDs (GeneIDs without
an Ensembl ID are dropped, several GeneIDs on one Ensembl ID are summed).
"""

from __future__ import annotations

import gzip
import io
import json
import logging
import re
import time
from pathlib import Path

import httpx
import numpy as np
import polars as pl

from rna_age.ages import find_age
from rna_age.annotate import add_flags
from rna_age.counts import blocks_of, write_parts
from rna_age.download import Remote, meta_path
from rna_age.download import fetch as _fetch
from rna_age.paths import INPUT, INTERIM, download_dir, input_dir
from rna_age.schema import conform

log = logging.getLogger(__name__)

NAME = "ncbi_geo"
PAUSE = 4.0  # seconds between NCBI requests
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
SEARCH_TERM = '"rnaseq counts"[Filter] AND "Homo sapiens"[Organism] AND gse[ETYP]'
ANNOT = "Human.GRCh38.p13.annot.tsv.gz"
ANNOT_URL = f"https://www.ncbi.nlm.nih.gov/geo/download/?format=file&type=rnaseq_counts&file={ANNOT}"
TARGETS = INTERIM / "ncbi_geo_targets.parquet"
CROSSCHECK_SERIES = 30


def counts_name(gse: str) -> str:
    return f"{gse}_raw_counts_GRCh38.p13_NCBI.tsv.gz"


def counts_url(gse: str) -> str:
    return (
        f"https://www.ncbi.nlm.nih.gov/geo/download/?type=rnaseq_counts&acc={gse}&format=file&file={counts_name(gse)}"
    )


def samples_text_url(gse: str) -> str:
    """GEO's text view of every sample of a series (title, source, characteristics, relations)."""
    return f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={gse}&targ=gsm&form=text&view=brief"


def _polite_fetch(remote: Remote, dest_dir: Path, max_403: int = 3) -> Path | None:
    for attempt in range(1, max_403 + 1):
        try:
            path = _fetch(remote, dest_dir, connections=1)
            time.sleep(PAUSE)
            return path
        except httpx.HTTPStatusError as e:
            code = e.response.status_code
            if code == 404:
                log.warning("%s: 404", remote.name)
                time.sleep(PAUSE)
                return None
            if code in (403, 429, 503) and attempt < max_403:
                log.warning("%s: NCBI answered %d; backing off 20 min (attempt %d)", remote.name, code, attempt)
                time.sleep(1200)
                continue
            raise
    return None


def fetch_series_list() -> list[str]:
    """All human GSEs with NCBI-generated counts, via one esearch (saved as raw XML pages)."""
    d = download_dir(NAME) / "esearch"
    d.mkdir(parents=True, exist_ok=True)
    gses: list[str] = []
    start, step = 0, 10000
    while True:
        name = f"esearch_rnaseq_counts_human_{start:06d}.xml"
        url = f"{EUTILS}/esearch.fcgi?db=gds&tool=rna-age&retmax={step}&retstart={start}&term=" + httpx.QueryParams(
            {"t": SEARCH_TERM}
        )["t"].replace(" ", "+")
        path = d / name
        if not path.exists():
            _polite_fetch(Remote(url=url, name=name), d)
        xml = path.read_text()
        total = int(re.search(r"<Count>(\d+)</Count>", xml).group(1))
        ids = re.findall(r"<Id>(\d+)</Id>", xml)
        gses += [f"GSE{int(i) - 200000000}" for i in ids if int(i) >= 200000000]
        start += step
        if start >= total or not ids:
            break
    log.info("NCBI GEO: %d human series with NCBI-generated counts", len(set(gses)))
    return sorted(set(gses), key=lambda g: int(g[3:]))


def select_targets() -> pl.DataFrame:
    """Cross-check series (largest age-annotated ARCHS4 series) + age series missing from ARCHS4.

    For gap filling we cannot know ages before reading GEO, so every NCBI-counted series absent from ARCHS4
    becomes a ``metadata`` target (one small text request each); its counts are fetched only when the text
    shows donor ages (see :func:`fetch`).
    """
    ncbi = set(fetch_series_list())
    a4 = pl.read_parquet(INPUT / "archs4" / "samples.parquet")
    elig = a4.filter(pl.col("exclude_reason").is_null() & (pl.col("age_type") == "exact"))
    top = (
        elig.group_by("study_id")
        .len()
        .filter(pl.col("study_id").is_in(list(ncbi)))
        .sort("len", descending=True)
        .head(CROSSCHECK_SERIES)["study_id"]
        .to_list()
    )
    meta_all = pl.read_parquet(INPUT / "archs4" / "metadata_all.parquet", columns=["series_id"])
    in_archs4 = set(meta_all["series_id"].str.replace_all(r"\s+", "").str.split(",").explode().unique().to_list())
    missing = sorted(ncbi - in_archs4, key=lambda g: int(g[3:]))
    targets = pl.concat(
        [
            pl.DataFrame({"gse": top, "reason": ["crosscheck"] * len(top)}),
            pl.DataFrame({"gse": missing, "reason": ["not_in_archs4"] * len(missing)}),
        ]
    )
    TARGETS.parent.mkdir(parents=True, exist_ok=True)
    targets.write_parquet(TARGETS)
    log.info("NCBI GEO targets: %d cross-check series, %d series missing from ARCHS4", len(top), len(missing))
    return targets


def parse_samples_text(text: str, gse: str) -> pl.DataFrame:
    """Samples of a GEO ``targ=gsm&form=text`` page → one row per GSM with characteristics as pairs."""
    rows, cur = [], None
    for line in text.splitlines():
        if line.startswith("^SAMPLE"):
            if cur:
                rows.append(cur)
            cur = {"geo_accession": line.split("=", 1)[1].strip(), "fields": [], "title": "", "source": "",
                   "organism": "", "library_strategy": "", "relation": [], "series": []}  # fmt: skip
        elif cur is None or "=" not in line:
            continue
        else:
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if key == "!Sample_characteristics_ch1":
                k, colon, v = value.partition(":")
                cur["fields"].append({"key": k.strip() if colon else "", "value": v.strip() if colon else value})
            elif key == "!Sample_title":
                cur["title"] = value
            elif key == "!Sample_source_name_ch1":
                cur["source"] = value
            elif key == "!Sample_organism_ch1":
                cur["organism"] = value
            elif key == "!Sample_library_strategy":
                cur["library_strategy"] = value
            elif key == "!Sample_relation":
                cur["relation"].append(value)
            elif key == "!Sample_series_id":
                cur["series"].append(value)
    if cur:
        rows.append(cur)
    return pl.DataFrame(
        [
            {
                "geo_accession": r["geo_accession"],
                "fields": r["fields"],
                "title": r["title"],
                "source_name_ch1": r["source"],
                "organism_ch1": r["organism"],
                "library_strategy": r["library_strategy"],
                "relation": " ".join(r["relation"]),
                "series_id": ",".join(r["series"]) or gse,
            }
            for r in rows
        ],
        schema={
            "geo_accession": pl.String,
            "fields": pl.List(pl.Struct({"key": pl.String, "value": pl.String})),
            "title": pl.String,
            "source_name_ch1": pl.String,
            "organism_ch1": pl.String,
            "library_strategy": pl.String,
            "relation": pl.String,
            "series_id": pl.String,
        },
    )


def _has_age(df: pl.DataFrame) -> pl.Series:
    return pl.Series(
        [find_age([(f["key"], f["value"]) for f in fl]) is not None for fl in df["fields"].to_list()],
        dtype=pl.Boolean,
    )


def fetch(connections: int = 1, max_new: int = 400) -> None:
    """Annotation + metadata text for gap series + counts for cross-check and age-annotated gap series.

    ``max_new`` caps how many new NCBI files one run fetches (resumable: rerun to continue).
    """
    d = download_dir(NAME)
    _polite_fetch(Remote(url=ANNOT_URL, name=ANNOT), d)
    targets = pl.read_parquet(TARGETS) if TARGETS.exists() else select_targets()
    new = 0
    for gse, reason in targets.iter_rows():
        if new >= max_new:
            log.info("NCBI GEO: reached max_new=%d for this run; rerun to continue", max_new)
            break
        want_counts = reason == "crosscheck"
        if reason == "not_in_archs4":
            name = f"{gse}_samples.txt"
            if not meta_path(d / "series" / name).exists():
                new += 1
            path = _polite_fetch(Remote(url=samples_text_url(gse), name=name), d / "series")
            if path is None:
                continue
            df = parse_samples_text(path.read_text(errors="replace"), gse)
            want_counts = df.height > 0 and bool(_has_age(df).any())
        if want_counts:
            if not meta_path(d / "counts" / counts_name(gse)).exists():
                new += 1
            _polite_fetch(Remote(url=counts_url(gse), name=counts_name(gse)), d / "counts")


def _annotation() -> pl.DataFrame:
    with gzip.open(download_dir(NAME) / ANNOT, "rb") as f:
        ann = pl.read_csv(f.read(), separator="\t", infer_schema_length=0, quote_char=None)
    return ann.select(
        geneid=pl.col("GeneID").cast(pl.Int64),
        gene_id=pl.col("EnsemblGeneID").str.split(";").list.first(),
        symbol=pl.col("Symbol"),
        biotype=pl.col("GeneType"),
    ).filter(pl.col("gene_id").is_not_null() & (pl.col("gene_id") != ""))


def parse() -> None:
    out = input_dir(NAME)
    d = download_dir(NAME)
    targets = pl.read_parquet(TARGETS)
    ann = _annotation()
    genes = ann.group_by("gene_id").agg(pl.col("symbol").first(), pl.col("biotype").first()).sort("gene_id")
    gene_ids = genes["gene_id"].to_list()
    a4_path = INPUT / "archs4" / "samples.parquet"
    a4 = pl.read_parquet(a4_path) if a4_path.exists() else None

    sample_frames, columns = [], {}
    for gse, reason in targets.iter_rows():
        cpath = d / "counts" / counts_name(gse)
        if not cpath.exists():
            continue
        with gzip.open(cpath, "rb") as f:
            counts = pl.read_csv(io.BytesIO(f.read()), separator="\t")
        counts = counts.rename({counts.columns[0]: "geneid"}).with_columns(pl.col("geneid").cast(pl.Int64))
        mapped = (
            counts.join(ann.select("geneid", "gene_id"), on="geneid", how="inner")
            .drop("geneid")
            .group_by("gene_id")
            .sum()
        )
        mapped = pl.DataFrame({"gene_id": gene_ids}).join(mapped, on="gene_id", how="left").fill_null(0)
        gsms = [c for c in mapped.columns if c != "gene_id"]
        if reason == "crosscheck" and a4 is not None:
            meta = a4.filter(pl.col("sample_id").is_in(gsms))
        else:
            spath = d / "series" / f"{gse}_samples.txt"
            if not spath.exists():
                continue
            raw = parse_samples_text(spath.read_text(errors="replace"), gse)
            meta = _conform_text_samples(raw)
        meta = meta.with_columns(source=pl.lit(NAME), cohort=pl.lit(reason))
        for gsm in meta["sample_id"].to_list():
            if gsm in mapped.columns and gsm not in columns:
                columns[gsm] = mapped[gsm].to_numpy()
        sample_frames.append(meta.filter(pl.col("sample_id").is_in(list(columns))))
        log.info("NCBI GEO %s (%s): %d samples", gse, reason, meta.height)
    if not sample_frames:
        log.warning("NCBI GEO: no counts parsed yet")
        return
    samples = pl.concat(sample_frames, how="vertical_relaxed").unique("sample_id", keep="first")
    ids = samples["sample_id"].to_list()
    samples = samples.with_columns(read_count_total=pl.Series([int(columns[s].sum()) for s in ids], dtype=pl.Int64))
    write_parts(out, gene_ids, ((b, np.column_stack([columns[s] for s in b])) for b in blocks_of(ids)))
    genes.write_parquet(out / "genes.parquet")
    conform(samples).write_parquet(out / "samples.parquet")
    (out / "README.json").write_text(
        json.dumps({"source": NAME, "genome": "GRCh38.p13", "samples": samples.height}, indent=2)
    )
    log.info("NCBI GEO parsed: %d samples", samples.height)


def _conform_text_samples(raw: pl.DataFrame) -> pl.DataFrame:
    ages = [find_age([(f["key"], f["value"]) for f in fl]) for fl in raw["fields"].to_list()]
    df = raw.with_columns(
        age_type=pl.Series([a.age_type if a else None for a in ages], dtype=pl.String),
        age_years=pl.Series([a.age_years if a else None for a in ages], dtype=pl.Float64),
        age_min=pl.Series([a.age_min if a else None for a in ages], dtype=pl.Float64),
        age_max=pl.Series([a.age_max if a else None for a in ages], dtype=pl.Float64),
        age_raw=pl.Series([a.raw if a else None for a in ages], dtype=pl.String),
        text=pl.concat_str(
            [
                pl.col("title"),
                pl.col("source_name_ch1"),
                pl.col("fields")
                .list.eval(pl.element().struct.field("key") + pl.lit(": ") + pl.element().struct.field("value"))
                .list.join(" | "),
            ],
            separator=" | ",
        ).str.to_lowercase(),
        exclude_reason=pl.when(~pl.col("organism_ch1").str.contains("(?i)homo sapiens")).then(pl.lit("non_human")),
    ).filter(pl.col("age_type").is_not_null())
    df = add_flags(df)
    series = pl.col("series_id").str.split(",")
    return conform(
        df.with_columns(
            sample_id=pl.col("geo_accession"),
            study_id=series.list.first(),
            series_ids=pl.col("series_id"),
            donor_id=pl.when(pl.col("donor_id").is_not_null())
            .then(series.list.first() + ":" + pl.col("donor_id"))
            .otherwise(pl.col("geo_accession")),
            tissue=pl.coalesce("tissue", "source_name_ch1"),
            assay=pl.col("library_strategy"),
            library_type=pl.col("library_strategy"),
            srr_ids=pl.col("relation").str.extract_all(r"SR[RX]\d+").list.join(","),
        )
    )
