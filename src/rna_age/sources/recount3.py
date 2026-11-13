"""recount3 human SRA projects: the adult bulk samples the 2026-11-12 inventory found missing from our table (RM26).

Targets are the rows of ``data/interim/inventory/recount3_samples.parquet`` with ``new == True`` (exact or decade
age ≥ 18, healthy or no disease stated, not tumour / culture / prenatal / single cell, not in ARCHS4; one row per
SRA experiment). ``rna-age inventory`` must have run first.

Fetch (``rna-age fetch recount3``): for every target project, ``gene_sums/<xx>/<SRP>/sra.gene_sums.<SRP>.G026.gz``
(GENCODE v26 gene sums of all the project's runs) and ``metadata/<xx>/<SRP>/sra.recount_qc.<SRP>.MD.gz`` (per-run
QC), each checked against its S3 ETag (md5) and given a ``.meta.json`` sidecar.

Parse (``rna-age parse recount3``) → ``data/input/recount3/``:

- **Coverage → read counts.** recount3's gene sums are base-pair coverage sums, not read counts. As recount3's
  ``compute_read_counts`` does, each run's sums are divided by its average mapped read length
  (``star.average_mapped_length``; for paired-end runs STAR reports both mates together, so the result counts
  fragments) and rounded.
- **Runs → experiments.** One sample per SRX: the converted counts of all its runs are summed; the runs go to
  ``srr_ids``. ``sample_id`` is always the SRX (only ~9 % of targets carry a GSM sample name, and ARCHS4 rows are
  GSMs, so SRX keeps one namespace for this source); ``read_count_total`` is the sum of the runs' STAR mapped reads.
- **De-duplication.** A target is dropped when its SRX, any of its runs (SRR) or its GSM appears in ARCHS4 (all of
  ``metadata_all.parquet``, not only aged samples), Bgee or GTEx (GTEx's samples carry no SRA ids; recount3's
  SRA source excludes GTEx's project SRP012682, both checked and counted).
- **study_id** keeps the split rule (no study on both sides): for a project with experiments in ARCHS4, the first
  GEO series ARCHS4 gives those experiments, when that is one series; a project whose ARCHS4 experiments span
  several first series cannot be mapped and is dropped. Other projects keep the SRA project id (SRP/ERP/DRP).
- **D4 rules.** Single cell: a project whose mapped ARCHS4 series is single-cell as a whole
  (:func:`rna_age.sources.archs4.single_cell_series`, median probability > 0.4), or more than
  :data:`SC_STUDY_FRACTION` of whose runs recount3 predicts as single cell (``pred.type == "scrna-seq"``, the
  analogue of D4's median on a binary prediction), is excluded whole. Tumours, cell lines, stem cells, prenatal and
  cancer patients through :func:`rna_age.annotate.add_flags` (meningioma among the tumour names) on top of the
  inventory's flags; every other sample of a project with a tumour sample is ``cancer_patient`` (the patients'
  normal tissue, e.g. SRP114315's LUSC_N beside its LUSC_C tumours).
- **Splits** (:mod:`rna_age.merge`): recount3 studies hash by the usual study rule, a study mapped to an ARCHS4
  series takes that series' split, and recount3 samples landing in test get split ``test_new`` (unseen new
  studies, scored apart from ``test``).
- **Health** follows ARCHS4's convention (``add_flags``): ``is_healthy`` True when healthy/control wording is
  stated, False when a disease is stated, null when nothing is; ``disease`` keeps the stated value (null = nothing
  stated).
"""

from __future__ import annotations

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np
import polars as pl

from rna_age.annotate import add_flags
from rna_age.counts import CHUNK, write_parts
from rna_age.download import Remote
from rna_age.download import fetch as _fetch
from rna_age.paths import INPUT, INTERIM, download_dir, input_dir
from rna_age.schema import conform
from rna_age.sources.inventory import R3_BUCKET, R3_PREFIX, _read_md, _s3_list

log = logging.getLogger(__name__)

NAME = "recount3"
GS_PREFIX = R3_PREFIX.replace("/metadata/", "/gene_sums/")
ANNOTATION = "G026"
TARGETS = INTERIM / "inventory" / "recount3_samples.parquet"
GTEX_SRA_PROJECT = "SRP012682"
# a project is single cell as a whole when more than this fraction of its predicted runs are scrna-seq (D4 analogue)
SC_STUDY_FRACTION = 0.4
MAX_CONCURRENCY = 8


def shard(project: str) -> str:
    """recount3's two-character directory of a project: its last two characters (``SRP053794`` → ``94``)."""
    return project[-2:]


def targets() -> pl.DataFrame:
    """The inventory's new samples: one row per SRA experiment."""
    if not TARGETS.exists():
        raise FileNotFoundError(f"{TARGETS} missing; run `rna-age inventory` first")
    return pl.read_parquet(TARGETS).filter("new")


# --------------------------------------------------------------------------------------------- fetch


def _remotes(projects: list[str]) -> list[tuple[Remote, Path]]:
    d = download_dir(NAME)
    with httpx.Client(timeout=120) as c:
        # the metadata listing is cached in listing/ by the inventory; read it again (no network when complete)
        meta = {k: (e, s) for k, e, s in _s3_list(c, R3_PREFIX, d / "listing")}
        out = []
        for p in projects:
            qc_key = f"{R3_PREFIX}{shard(p)}/{p}/sra.recount_qc.{p}.MD.gz"
            if qc_key not in meta:
                raise KeyError(f"{qc_key} not in the metadata listing")
            gs_dir = f"{GS_PREFIX}{shard(p)}/{p}/"
            gs = {k: (e, s) for k, e, s in _s3_list(c, gs_dir, d / "listing_gene_sums_by_project" / p)}
            gs_key = f"{gs_dir}sra.gene_sums.{p}.{ANNOTATION}.gz"
            if gs_key not in gs:
                raise KeyError(f"{gs_key} not listed")
            for key, (etag, size) in ((qc_key, meta[qc_key]), (gs_key, gs[gs_key])):
                rel = key.removeprefix("recount3/release/human/data_sources/sra/")
                remote = Remote(url=f"{R3_BUCKET}/{key}", name=rel.rsplit("/", 1)[1], size=size,
                                checksum=f"s3etag:{etag}")  # fmt: skip
                out.append((remote, d / rel.rsplit("/", 1)[0]))
    return out


def fetch(connections: int = 8) -> None:
    """Gene sums and QC tables of the target projects (resumable: verified files with sidecars are skipped)."""
    projects = sorted(targets()["study"].unique().to_list())
    remotes = _remotes(projects)
    total = sum(r.size or 0 for r, _ in remotes)
    log.info("recount3: %d projects, %d files, %.2f GB", len(projects), len(remotes), total / 1e9)
    workers = max(1, min(connections, MAX_CONCURRENCY))
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, path in enumerate(pool.map(lambda rd: _fetch(rd[0], rd[1], connections=1), remotes), 1):
            if i % 50 == 0 or i == len(remotes):
                log.info("recount3: %d/%d files (%.0fs), last %s", i, len(remotes), time.time() - t0, path.name)


def gene_sums_path(project: str) -> Path:
    return download_dir(NAME) / "gene_sums" / shard(project) / project / f"sra.gene_sums.{project}.{ANNOTATION}.gz"


def qc_path(project: str) -> Path:
    return download_dir(NAME) / "metadata" / shard(project) / project / f"sra.recount_qc.{project}.MD.gz"


def sra_path(project: str) -> Path:
    return download_dir(NAME) / "metadata" / shard(project) / project / f"sra.sra.{project}.MD.gz"


def pred_path(project: str) -> Path:
    return download_dir(NAME) / "metadata" / shard(project) / project / f"sra.recount_pred.{project}.MD.gz"


# --------------------------------------------------------------------------------------------- pure helpers


def coverage_to_counts(coverage: np.ndarray, avg_mapped_length: np.ndarray) -> np.ndarray:
    """genes × runs base-pair coverage sums → read (fragment) counts, as recount3's ``compute_read_counts``.

    Divides each run's column by its average mapped read length and rounds.
    """
    length = np.asarray(avg_mapped_length, dtype=np.float64)
    if coverage.shape[1] != length.shape[0]:
        raise ValueError(f"{coverage.shape[1]} runs vs {length.shape[0]} read lengths")
    if not np.all(np.isfinite(length) & (length > 0)):
        raise ValueError("average mapped length must be positive for every run")
    return np.rint(coverage / length[None, :])


def sum_runs(counts: np.ndarray, runs: list[str], run_to_exp: dict[str, str]) -> tuple[list[str], np.ndarray]:
    """Sum genes × runs counts into genes × experiments; experiments in order of first appearance."""
    exps: list[str] = []
    col: dict[str, int] = {}
    for r in runs:
        e = run_to_exp[r]
        if e not in col:
            col[e] = len(exps)
            exps.append(e)
    out = np.zeros((counts.shape[0], len(exps)), dtype=np.float64)
    for j, r in enumerate(runs):
        out[:, col[run_to_exp[r]]] += counts[:, j]
    return exps, out


def age_columns(df: pl.DataFrame) -> pl.DataFrame:
    """The inventory's age class / bounds → the schema's age columns (exact; decade → binned)."""
    cls = pl.col("age_class")
    return df.with_columns(
        age_type=pl.when(cls == "exact").then(pl.lit("exact")).when(cls == "decade").then(pl.lit("binned")),
        age_years=pl.when(cls == "exact").then(pl.col("age_lo")),
        age_min=pl.when(cls.is_in(["exact", "decade"])).then(pl.col("age_lo")),
        age_max=pl.when(cls.is_in(["exact", "decade"])).then(pl.col("age_hi")),
    )


def map_studies(runs: pl.DataFrame, archs4: pl.DataFrame) -> pl.DataFrame:
    """Project → study_id through ARCHS4.

    ``runs``: every run of the target projects (``study``, ``experiment_acc``). ``archs4``: ``srx``, ``gse`` (first
    series) of every ARCHS4 sample. Returns ``study, study_id, archs4_series, mapped`` where ``mapped`` is
    ``"archs4"`` (one series), ``"ambiguous"`` (several; study_id null, to drop) or ``"none"`` (study_id = project).
    """
    hits = (
        runs.select("study", "experiment_acc")
        .unique()
        .join(archs4.select("srx", "gse").drop_nulls(), left_on="experiment_acc", right_on="srx", how="inner")
        .group_by("study")
        .agg(pl.col("gse").unique().sort().alias("archs4_series"))
    )
    out = runs.select("study").unique().join(hits, on="study", how="left")
    n = pl.col("archs4_series").list.len().fill_null(0)
    return out.with_columns(
        mapped=pl.when(n == 0).then(pl.lit("none")).when(n == 1).then(pl.lit("archs4")).otherwise(pl.lit("ambiguous")),
    ).with_columns(
        study_id=pl.when(pl.col("mapped") == "none")
        .then(pl.col("study"))
        .when(pl.col("mapped") == "archs4")
        .then(pl.col("archs4_series").list.first()),
        archs4_series=pl.col("archs4_series").list.join(","),
    )


def sra_fields(df: pl.DataFrame, col: str = "sample_attributes") -> pl.DataFrame:
    """SRA ``key;;value|key;;value`` attributes → the ``fields`` list of key/value structs ``add_flags`` reads."""
    return df.with_columns(
        fields=pl.col(col).fill_null("").str.split("|").list.eval(
            pl.struct(key=pl.element().str.split(";;").list.first(),
                      value=pl.element().str.split(";;").list.slice(1).list.join(";;"))
        ).list.eval(pl.element().filter(pl.element().struct.field("value") != "")),
    )  # fmt: skip


# --------------------------------------------------------------------------------------------- parse


def _project_tables(projects: list[str]) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """All runs' SRA metadata, recount3 predictions and QC of the projects."""
    sra, pred, qc = [], [], []
    for p in projects:
        sra.append(_read_md(sra_path(p)))
        if pred_path(p).exists():
            pred.append(_read_md(pred_path(p)).select("external_id", pl.col("pred.type").alias("pred_type")))
        qc.append(
            _read_md(qc_path(p)).select(
                "external_id",
                pl.col("star.average_mapped_length").cast(pl.Float64, strict=False).alias("avg_mapped_length"),
                pl.col("star.all_mapped_reads").cast(pl.Float64, strict=False).alias("star_mapped_reads"),
                pl.col("gene_fc_count_all.assigned").cast(pl.Float64, strict=False).alias("fc_assigned"),
            )
        )
    return (pl.concat(sra, how="diagonal_relaxed"), pl.concat(pred, how="diagonal_relaxed"),
            pl.concat(qc, how="diagonal_relaxed"))  # fmt: skip


def _other_ids() -> dict[str, set[str]]:
    """GSM / SRX / SRR ids already in our bulk sources (ARCHS4 all samples, Bgee, GTEx)."""
    a = pl.read_parquet(INPUT / "archs4" / "metadata_all.parquet", columns=["geo_accession", "relation"])
    ids = {
        "archs4_gsm": set(a["geo_accession"].drop_nulls().to_list()),
        "archs4_sra": set(a["relation"].str.extract_all(r"[SED]R[XR]\d+").explode().drop_nulls().to_list()),
    }
    for src in ("bgee", "gtex"):
        p = INPUT / src / "samples.parquet"
        if p.exists():
            s = pl.read_parquet(p, columns=["sample_id", "srr_ids"])
            acc = s["srr_ids"].str.split(",").explode().drop_nulls().to_list() + s["sample_id"].to_list()
            ids[src] = {x for x in acc if re.match(r"^[SED]R[XRS]\d+$", x or "")} | set(s["sample_id"].to_list())
    return ids


def _archs4_series() -> tuple[pl.DataFrame, set[str]]:
    from rna_age.sources.archs4 import single_cell_series

    meta = pl.read_parquet(
        INPUT / "archs4" / "metadata_all.parquet", columns=["geo_accession", "series_id", "relation",
                                                            "singlecellprobability"]
    )  # fmt: skip
    ids = meta.select(
        srx=pl.col("relation").str.extract(r"term=([SED]RX\d+)"),
        gse=pl.col("series_id").str.replace_all(r"\s+", "").str.split(",").list.first(),
    )
    return ids, single_cell_series(meta)


def build_samples() -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    """(samples table for every kept target with its runs, run table for the eligible ones, report)."""
    t = targets()
    projects = sorted(t["study"].unique().to_list())
    sra, pred, qc = _project_tables(projects)
    report: dict = {"targets": t.height, "target_projects": len(projects)}

    # de-duplication against our other bulk sources, by SRX, SRR and GSM
    other = _other_ids()
    exp_runs = sra.group_by("experiment_acc").agg(pl.col("external_id").unique().sort().alias("runs"))
    t = t.drop("external_id").join(exp_runs, on="experiment_acc", how="left")
    gsm = pl.col("sample_name").str.extract(r"^(GSM\d+)")
    dup = {}
    for name, acc in other.items():
        hit = (
            pl.col("experiment_acc").is_in(list(acc))
            | pl.col("runs").list.eval(pl.element().is_in(list(acc))).list.any().fill_null(False)
            | gsm.is_in(list(acc)).fill_null(False)
        )
        dup[name] = t.filter(hit)["experiment_acc"].to_list()
    report["dropped_duplicate"] = {k: len(v) for k, v in dup.items()}
    report["gtex_project_in_targets"] = int((t["study"] == GTEX_SRA_PROJECT).sum())
    gtex_names = t.filter(pl.col("sample_name").str.contains("GTEX-")).height
    report["gtex_named_samples_in_targets"] = gtex_names
    drop = {e for v in dup.values() for e in v}
    t = t.filter(~pl.col("experiment_acc").is_in(list(drop)) & (pl.col("study") != GTEX_SRA_PROJECT))

    # study_id through ARCHS4, and D4's whole-series single-cell rule
    a4, sc_series = _archs4_series()
    studies = map_studies(sra.filter(pl.col("study").is_in(projects)), a4)
    frac = (
        sra.select("external_id", "study")
        .join(pred, on="external_id", how="left")
        .filter(pl.col("pred_type").is_in(["rna-seq", "scrna-seq"]))
        .group_by("study")
        .agg(sc_fraction=(pl.col("pred_type") == "scrna-seq").mean())
    )
    studies = studies.join(frac, on="study", how="left").with_columns(
        sc_study=(pl.col("sc_fraction") > SC_STUDY_FRACTION).fill_null(False)
        | pl.col("study_id").is_in(list(sc_series)).fill_null(False)
    )
    studies.write_parquet(input_dir(NAME) / "studies.parquet", compression="zstd")
    t = t.join(studies.select("study", "study_id", "mapped", "archs4_series", "sc_study"), on="study", how="left")
    amb = t.filter(pl.col("mapped") == "ambiguous")
    report["dropped_ambiguous_study"] = {"samples": amb.height, "studies": amb["study"].n_unique()}
    t = t.filter(pl.col("mapped") != "ambiguous")
    report["mapped_to_archs4_series"] = {
        "samples": t.filter(pl.col("mapped") == "archs4").height,
        "studies": t.filter(pl.col("mapped") == "archs4")["study"].n_unique(),
    }

    # metadata and flags of the kept experiments (the first run's SRA record; runs share the sample)
    meta = (
        sra.filter(pl.col("experiment_acc").is_in(t["experiment_acc"].implode()))
        .sort("external_id")
        .group_by("experiment_acc")
        .agg(pl.all().first())
        .select(
            "experiment_acc",
            "sample_attributes",
            "sample_title",
            "study_title",
            "experiment_title",
            "library_strategy",
            "library_selection",
            "library_source",
            "library_layout",
            "platform_model",
            "sample_acc",
        )  # fmt: skip
    )
    df = sra_fields(t.join(meta, on="experiment_acc", how="left"))
    df = df.with_columns(
        text=pl.concat_str(
            [
                pl.col("fields")
                .list.eval(pl.element().struct.field("key") + pl.lit(": ") + pl.element().struct.field("value"))
                .list.join(" | "),
                *[pl.col(c).fill_null("") for c in ("sample_title", "study_title", "experiment_title")],
            ],
            separator=" | ",
        ).str.to_lowercase(),
        exclude_reason=pl.when(pl.col("sc_study")).then(pl.lit("single_cell")),
    )
    df = add_flags(df)
    # a project with tumour samples is a cancer study: its other samples are the patients' normal tissue
    # (SRP114315: LUSC_C tumours beside LUSC_N normals whose own text names no cancer)
    tumor_study = pl.col("is_tumor").fill_null(False).any().over("study")
    report["cancer_study_normals"] = df.filter(tumor_study & pl.col("exclude_reason").is_null()).height
    df = df.with_columns(
        exclude_reason=pl.coalesce("exclude_reason", pl.when(tumor_study).then(pl.lit("cancer_patient")))
    )
    df = age_columns(df)
    run_qc = (
        df.select("experiment_acc", pl.col("runs").alias("external_id"))
        .explode("external_id")
        .join(qc, on="external_id", how="left")
    )
    lib = run_qc.group_by("experiment_acc").agg(
        read_count_total=pl.col("star_mapped_reads").sum().cast(pl.Int64),
        _bad_qc=(pl.col("avg_mapped_length").is_null() | (pl.col("avg_mapped_length") <= 0)).any(),
    )
    df = df.join(lib, on="experiment_acc", how="left").with_columns(
        exclude_reason=pl.coalesce("exclude_reason", pl.when("_bad_qc").then(pl.lit("no_qc")))
    )
    df = df.with_columns(
        sample_id=pl.col("experiment_acc"),
        source=pl.lit(NAME),
        series_ids=pl.when(pl.col("mapped") == "archs4").then(pl.col("study_id")),
        donor_id=pl.when(pl.col("donor_id").is_not_null())
        .then(pl.col("study_id") + ":" + pl.col("donor_id"))
        .otherwise(pl.col("experiment_acc")),
        tissue=pl.coalesce("tissue", "sample_title"),
        platform=pl.col("platform_model"),
        library_type=pl.concat_str(
            ["library_strategy", "library_selection", "library_source"], separator="/", ignore_nulls=True
        ),  # fmt: skip
        assay=pl.col("library_strategy"),
        srr_ids=pl.col("runs").list.join(","),
    )
    report["samples"] = df.height
    report["studies"] = df["study_id"].n_unique()
    report["exclude_reason"] = dict(df.group_by("exclude_reason").len().sort("len", descending=True).iter_rows())
    el = df.filter(pl.col("exclude_reason").is_null() & ~pl.col("is_tumor"))
    report["eligible"] = el.height
    report["eligible_studies"] = el["study_id"].n_unique()
    report["eligible_by_age_type"] = dict(el.group_by("age_type").len().iter_rows())
    report["eligible_by_health"] = {
        "healthy_stated": int((el["is_healthy"] == True).sum()),  # noqa: E712
        "disease_stated": int((el["is_healthy"] == False).sum()),  # noqa: E712
        "nothing_stated": int(el["is_healthy"].is_null().sum()),
    }
    runs = run_qc.filter(pl.col("experiment_acc").is_in(el["experiment_acc"].implode())).join(
        df.select("experiment_acc", "study"), on="experiment_acc"
    )
    audit = df.select("sample_id", "study", "mapped", "archs4_series", "sample_name", "sample_attributes")
    audit.write_parquet(input_dir(NAME) / "audit.parquet", compression="zstd")
    return conform(df), runs, report


def read_gene_sums(path: Path, runs: list[str]) -> tuple[list[str], np.ndarray]:
    """(gene ids as in the file, genes × runs coverage) of the requested runs of one project's gene-sums file."""
    df = pl.read_csv(path, separator="\t", comment_prefix="##", columns=["gene_id", *runs],
                     schema_overrides={r: pl.Float64 for r in runs})  # fmt: skip
    return df["gene_id"].to_list(), df.select(runs).to_numpy()


def gene_rows(raw_ids: list[str]) -> tuple[np.ndarray, list[str]]:
    """Rows to keep (drop the ``_PAR_Y`` copies) and their Ensembl ids without version."""
    keep = np.array([not g.endswith("_PAR_Y") for g in raw_ids])
    ids = [re.sub(r"\.\d+$", "", g) for g, k in zip(raw_ids, keep, strict=True) if k]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate gene ids after removing versions")
    return np.flatnonzero(keep), ids


def _blocks(runs: pl.DataFrame, check: list[dict]):
    """Yield (experiment ids, genes × experiments counts) in blocks of ≤ CHUNK, project by project."""
    run_to_exp = dict(zip(runs["external_id"].to_list(), runs["experiment_acc"].to_list(), strict=True))
    length = dict(zip(runs["external_id"].to_list(), runs["avg_mapped_length"].to_list(), strict=True))
    ref: list[str] | None = None
    keep_rows = None
    ids_buf: list[str] = []
    cols_buf: list[np.ndarray] = []
    for (project,), pr in runs.sort("study", "external_id").group_by("study", maintain_order=True):
        rs = pr["external_id"].to_list()
        raw, cov = read_gene_sums(gene_sums_path(project), rs)
        if ref is None:
            ref = raw
            keep_rows, _ = gene_rows(raw)
        elif raw != ref:
            raise ValueError(f"{project}: gene rows differ from the first project's")
        counts = coverage_to_counts(cov, np.array([length[r] for r in rs]))
        for r, j in zip(rs, range(len(rs)), strict=True):
            row = pr.filter(pl.col("external_id") == r).row(0, named=True)
            check.append({"external_id": r, "study": project, "count_total": float(counts[:, j].sum()),
                          "star_mapped_reads": row["star_mapped_reads"], "fc_assigned": row["fc_assigned"],
                          "avg_mapped_length": row["avg_mapped_length"]})  # fmt: skip
        exps, mat = sum_runs(counts[keep_rows], rs, run_to_exp)
        ids_buf += exps
        cols_buf.append(mat)
        while len(ids_buf) >= CHUNK:
            m = np.concatenate(cols_buf, axis=1)
            yield ids_buf[:CHUNK], m[:, :CHUNK]
            ids_buf, cols_buf = ids_buf[CHUNK:], [m[:, CHUNK:]]
        log.info("recount3: %s, %d runs → %d samples", project, len(rs), len(exps))
    if ids_buf:
        yield ids_buf, np.concatenate(cols_buf, axis=1)


def parse() -> None:
    out = input_dir(NAME)
    samples, runs, report = build_samples()
    missing = [p for p in runs["study"].unique().to_list() if not gene_sums_path(p).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} gene-sums files missing (run `rna-age fetch recount3`): {missing[:5]}")
    check: list[dict] = []
    raw = pl.read_csv(gene_sums_path(runs["study"][0]), separator="\t", comment_prefix="##", columns=["gene_id"])
    _, gene_ids = gene_rows(raw["gene_id"].to_list())
    write_parts(out, gene_ids, _blocks(runs, check))
    genes = pl.DataFrame({"gene_id": gene_ids})
    a4 = INPUT / "archs4" / "genes.parquet"
    if a4.exists():
        sym = pl.read_parquet(a4).unique("gene_id", keep="first")
        genes = genes.join(sym, on="gene_id", how="left")
    genes.write_parquet(out / "genes.parquet", compression="zstd")
    # samples whose counts were written; the rest keep their exclude_reason and get no counts
    samples.write_parquet(out / "samples.parquet", compression="zstd")
    chk = pl.DataFrame(check).with_columns(
        ratio_mapped=pl.col("count_total") / pl.col("star_mapped_reads"),
        ratio_fc=pl.col("count_total") / pl.col("fc_assigned"),
    )
    chk.write_parquet(out / "read_length_check.parquet", compression="zstd")
    report["read_length_check"] = {
        "runs": chk.height,
        "median_counts_over_star_mapped": chk["ratio_mapped"].median(),
        "p05_p95_counts_over_star_mapped": [chk["ratio_mapped"].quantile(0.05), chk["ratio_mapped"].quantile(0.95)],
        "median_counts_over_featurecounts_assigned": chk["ratio_fc"].median(),
        "p05_p95_counts_over_featurecounts_assigned": [chk["ratio_fc"].quantile(0.05), chk["ratio_fc"].quantile(0.95)],
    }
    report["genes"] = len(gene_ids)
    (out / "README.json").write_text(json.dumps({"source": NAME, "annotation": ANNOTATION, **report}, indent=2,
                                                default=str))  # fmt: skip
    log.info("recount3: %s", json.dumps(report, default=str))
