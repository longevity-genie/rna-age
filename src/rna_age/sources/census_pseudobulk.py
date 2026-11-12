"""CZ CELLxGENE Census (LTS 2025-11-08): donor × tissue pseudobulk of age-annotated human scRNA-seq.

Why not ``cellxgene_census``/tiledbsoma: reading the SOMA arrays over the cluster's HTTP proxy stalled, so we
use the same LTS release's source h5ad files over HTTPS (``cell-census/2025-11-08/h5ads/<dataset_id>.h5ad``,
checked against their S3 ETags) and the CELLxGENE Discover API to choose datasets. The h5ads are the exact
inputs of the Census build.

Steps:
- ``fetch``: Discover dataset list (raw JSON), the LTS h5ad listing (raw XML pages), then the h5ads of
  :data:`DATASETS` (tier 1; ``RNA_AGE_CENSUS_TIER=2`` adds tier 2).
- ``pseudobulk <dataset_id>`` (one Slurm job per dataset): primary cells only (``is_primary_data``), raw
  counts (``raw/X`` if present, else ``X``) summed per donor × tissue in row chunks; writes
  ``data/input/census_pseudobulk/by_dataset/<dataset_id>/{samples,counts}.parquet``.
- ``parse``: merges every dataset into the standard layout (union of genes, absent genes = 0), keeps groups
  with ≥ :data:`MIN_CELLS` cells and ≥ :data:`MIN_UMIS` counts.

Ages come from ``development_stage`` (``"54-year-old stage"`` exact, ``"sixth decade stage"`` binned,
``"80 year-old and over stage"`` censored; vaguer stages are dropped).
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path

import h5py
import httpx
import numpy as np
import polars as pl
import scipy.sparse as sp

from rna_age.counts import blocks_of, write_parts
from rna_age.download import Remote
from rna_age.download import fetch as _fetch
from rna_age.paths import download_dir, input_dir
from rna_age.schema import conform
from rna_age.sources.bgee import stage_age

log = logging.getLogger(__name__)

NAME = "census_pseudobulk"
CENSUS_VERSION = "2025-11-08"
BUCKET_URL = "https://cellxgene-census-public-us-west-2.s3.us-west-2.amazonaws.com"
H5AD_PREFIX = f"cell-census/{CENSUS_VERSION}/h5ads/"
DISCOVER_URL = "https://api.cellxgene.cziscience.com/curation/v1/datasets"
MIN_CELLS = 200
MIN_UMIS = 500_000

# Chosen 2026-11-09 from the Discover listing: non-cancer human datasets with many donors carrying exact-age
# stages, weighed against file size (see STATUS.md). Integrated atlases are fine: only primary cells are used.
DATASETS: dict[str, str] = {
    "3faad104-2ab8-4434-816d-474d8d2641db": "OneK1K PBMC (Yazar 2022), 981 donors",
    "c838aec3-03ef-4398-b882-0e3912abfff0": "AIDA phase 1 v2 PBMC, 625 donors",
    "9f222629-9e39-47d0-b83f-e08d610c7479": "Human Lung Cell Atlas core+extended, 484 donors",
    "5c97eeeb-7e52-44b3-b010-b832b1f5424c": "PsychENCODE HBCC cohort brain snRNA, 300 donors",
    "218acb0f-9f2f-4f76-b90b-15a4b7c7f629": "SLE + control PBMC (Perez 2022), 261 donors",
    "a19d1667-a7b5-4556-9e5f-f9bfa690c0f1": "Human dermal fibroblast atlas, 221 donors",
    "ca7d95ac-53aa-4a94-8c76-300c608ce6a0": "Indonesia PBMC, 199 donors",
    "9dbab10c-118d-496b-966a-67f1763a6b7d": "COVID-19 PBMC (Ren 2021), 196 donors",
    "d9f3a82c-24bb-4277-b715-8ec92a33dd32": "circulating HSPCs (Illumina), 132 donors",
    "ef3055e1-0cad-40a3-961d-7110a323dc4f": "HLiCA liver, 110 donors",
    "e522d2cd-7927-4e59-a4ed-064009569279": "Human Immune Health Atlas, 108 donors",
    "eb499fd8-7000-419f-8854-926b9b61b11f": "CellCards CellRef 1.0 lung, 104 donors",
    "5e57cd50-8e42-42d6-940d-5c1660d06864": "PsychENCODE RADC cohort brain, 152 donors",
    "440da0a0-f321-4f40-9080-bb04031681e5": "microglia/perivascular myeloid, 137 donors",
    "f14bc322-1322-4184-8d16-409557525ea5": "interstitial lung disease, 119 donors",
    "91f31e05-56d8-46fc-b408-d90c9228a81b": "adult kidney scRNA, 112 donors",
}
TIER2: dict[str, str] = {
    "37a17b78-4864-4a42-b67b-31c00962795a": "PsychENCODE MSSM cohort brain, 1042 donors (36 GB)",
    "ff4235bb-1cdf-4dcf-a13a-dc4f32244400": "Immunobiology of Aging cohort PBMC, 234 donors (40 GB)",
    "55003f67-c494-46f1-83fb-902745646379": "Integrated human breast cell atlas, 287 donors (19 GB)",
}
# reference-only datasets: fetched and pseudobulked by cell type for deconvolution, never parsed into age samples
REFERENCES: dict[str, str] = {
    "983d5ec9-40e8-4512-9e65-a572a9c486cb": "Tabula Sapiens blood (healthy donors, includes neutrophils)",
}
CANCER = r"(?i)cancer|carcinoma|tumou?r|leuka?emia|lymphoma|melanoma|glioma|glioblastoma|myeloma|neoplas|sarcoma"


def selected() -> dict[str, str]:
    tier = int(os.environ.get("RNA_AGE_CENSUS_TIER", "1"))
    return DATASETS | (TIER2 if tier >= 2 else {})


def _listing() -> dict[str, tuple[int, str]]:
    """dataset_id → (size, etag) for every h5ad of the pinned Census release (raw XML pages kept)."""
    d = download_dir(NAME) / "listing"
    d.mkdir(parents=True, exist_ok=True)
    out: dict[str, tuple[int, str]] = {}
    token, page = None, 0
    with httpx.Client(timeout=120) as c:
        while True:
            path = d / f"h5ads_{CENSUS_VERSION}_{page:03d}.xml"
            if path.exists():
                xml = path.read_text()
            else:
                params = {"list-type": "2", "prefix": H5AD_PREFIX, "max-keys": "1000"}
                if token:
                    params["continuation-token"] = token
                xml = c.get(BUCKET_URL + "/", params=params).raise_for_status().text
                path.write_text(xml)
            for key, etag, size in re.findall(
                r"<Key>([^<]+)</Key>.*?<ETag>&quot;([^&]+)&quot;</ETag>.*?<Size>(\d+)</Size>", xml, re.S
            ):
                out[key.rsplit("/", 1)[1].removesuffix(".h5ad")] = (int(size), etag)
            m = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", xml)
            if not m:
                break
            token, page = m.group(1), page + 1
    return out


def fetch(connections: int = 8) -> None:
    d = download_dir(NAME)
    discover = d / "discover_datasets.json"
    if not discover.exists():
        discover.write_bytes(httpx.get(DISCOVER_URL, timeout=300).raise_for_status().content)
    listing = _listing()
    log.info("Census %s: %d h5ads in the release", CENSUS_VERSION, len(listing))
    for dataset_id, label in (selected() | REFERENCES).items():
        if dataset_id not in listing:
            log.warning("Census: %s (%s) not in release %s", dataset_id, label, CENSUS_VERSION)
            continue
        size, etag = listing[dataset_id]
        _fetch(
            Remote(
                url=f"{BUCKET_URL}/{H5AD_PREFIX}{dataset_id}.h5ad",
                name=f"{dataset_id}.h5ad",
                size=size,
                checksum=f"s3etag:{etag}",
            ),
            d / "h5ads",
            connections=connections,
        )


# ------------------------------------------------------------------ pseudobulk


def _obs_column(obs: h5py.Group, name: str) -> np.ndarray:
    """An obs column as an array of python strings / values (handles anndata categoricals)."""
    item = obs[name]
    if isinstance(item, h5py.Group) and "codes" in item:
        cats = item["categories"][()]
        cats = np.array([c.decode() if isinstance(c, bytes) else c for c in cats], dtype=object)
        codes = item["codes"][()]
        out = np.empty(len(codes), dtype=object)
        ok = codes >= 0
        out[ok] = cats[codes[ok]]
        return out
    arr = item[()]
    if arr.dtype.kind in ("S", "O"):
        return np.array([x.decode() if isinstance(x, bytes) else x for x in arr], dtype=object)
    return arr


def _var_ids(f: h5py.File, raw: bool) -> list[str]:
    var = f["raw/var"] if raw else f["var"]
    key = var.attrs.get("_index", "_index")
    ids = var[key][()]
    return [x.decode() if isinstance(x, bytes) else str(x) for x in ids]


def donor_columns(obs: pl.DataFrame, donor: str = "donor_id") -> pl.DataFrame:
    """The obs columns constant within each donor and varying between donors, one row per donor."""
    keep = [donor]
    for c in obs.columns:
        if c == donor:
            continue
        per = obs.group_by(donor).agg(pl.col(c).n_unique().alias("k"))
        if per["k"].max() == 1 and obs[c].n_unique() > 1:
            keep.append(c)
    return obs.select(keep).unique(donor).sort(donor)


def donor_obs(dataset_id: str) -> Path:
    """Per-donor metadata of one fetched Census dataset (obs only, no counts): every column constant within a donor."""
    src = download_dir(NAME) / "h5ads" / f"{dataset_id}.h5ad"
    with h5py.File(src, "r") as f:
        obs = f["obs"]
        index = obs.attrs.get("_index", "_index")
        names = [k for k in obs.keys() if k != index and k != "__categories"]
        cols = {}
        for k in names:
            v = _obs_column(obs, k)
            cols[k] = [None if x is None else str(x) for x in v] if v.dtype == object else v
    out = donor_columns(pl.DataFrame(cols, strict=False))
    path = input_dir(NAME) / "donor_obs" / f"{dataset_id}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(path, compression="zstd")
    log.info("%s: %d donors, columns %s", dataset_id[:8], out.height, out.columns)
    return path


def pseudobulk(dataset_id: str, chunk: int = 200_000, by_cell_type: bool = False) -> Path:
    """Sum raw counts of primary cells per donor × tissue (× cell type) for one dataset (Slurm job).

    ``by_cell_type`` writes donor × tissue × cell type groups to ``by_dataset_celltype/`` (composition analyses);
    the default donor × tissue groups in ``by_dataset/`` are the model inputs.
    """
    src = download_dir(NAME) / "h5ads" / f"{dataset_id}.h5ad"
    out = input_dir(NAME) / ("by_dataset_celltype" if by_cell_type else "by_dataset") / dataset_id
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with h5py.File(src, "r") as f:
        obs = f["obs"]
        cols = {c: _obs_column(obs, c) for c in obs.keys() if c in {
            "donor_id", "tissue", "development_stage", "is_primary_data", "assay", "suspension_type",
            "disease", "sex", "tissue_type", "self_reported_ethnicity", "cell_type"}}  # fmt: skip
        n = len(cols["donor_id"])
        primary = cols.get("is_primary_data", np.ones(n, dtype=bool)).astype(bool)
        if dataset_id in REFERENCES:  # reference-only subsets (e.g. Tabula Sapiens blood) re-publish primary cells
            primary = np.ones(n, dtype=bool)  # held elsewhere; not double counted since never parsed into samples
        raw = "raw" in f and "X" in f["raw"]
        X = f["raw/X"] if raw else f["X"]
        genes = _var_ids(f, raw)
        keys = pl.DataFrame(
            {c: pl.Series(cols[c].astype(str)) for c in ("donor_id", "tissue", "development_stage", "assay",
                                                       "suspension_type", "disease", "sex", "cell_type") if c in cols}
        ).with_columns(primary=pl.Series(primary))  # fmt: skip
        group_cols = ["donor_id", "tissue", "cell_type"] if by_cell_type else ["donor_id", "tissue"]
        groups = (
            keys.with_row_index("cell")
            .filter("primary")
            .with_columns(group=pl.concat_str(group_cols, separator="\x1f").rank("dense").cast(pl.Int64) - 1)
        )
        n_groups = int(groups["group"].max()) + 1 if groups.height else 0
        cell_group = np.full(n, -1, dtype=np.int64)
        cell_group[groups["cell"].to_numpy()] = groups["group"].to_numpy()
        sums = np.zeros((n_groups, len(genes)), dtype=np.float64)
        n_cells = np.bincount(cell_group[cell_group >= 0], minlength=n_groups)
        if isinstance(X, h5py.Group):  # CSR
            indptr = X["indptr"][()]
            for lo in range(0, n, chunk):
                hi = min(n, lo + chunk)
                g = cell_group[lo:hi]
                if not (g >= 0).any():
                    continue
                a, b = int(indptr[lo]), int(indptr[hi])
                m = sp.csr_matrix(
                    (X["data"][a:b], X["indices"][a:b], indptr[lo : hi + 1] - a), shape=(hi - lo, len(genes))
                )
                keep = np.flatnonzero(g >= 0)
                ind = sp.csr_matrix((np.ones(len(keep)), (g[keep], keep)), shape=(n_groups, hi - lo))
                sums += (ind @ m).toarray()
                log.info("%s: %d/%d cells (%.0fs)", dataset_id[:8], hi, n, time.time() - t0)
        else:  # dense
            for lo in range(0, n, chunk):
                hi = min(n, lo + chunk)
                g = cell_group[lo:hi]
                keep = np.flatnonzero(g >= 0)
                if len(keep):
                    np.add.at(sums, g[keep], X[lo:hi][keep])
    if not np.allclose(sums, np.rint(sums)):
        log.warning("%s: non-integer sums; X may not hold raw counts", dataset_id)
    meta = (
        groups.group_by("group")
        .agg(
            pl.col("donor_id").first(),
            pl.col("tissue").first(),
            (pl.col("cell_type").first() if "cell_type" in groups.columns else pl.lit(None, pl.String)).alias(
                "cell_type"
            ),
            pl.col("development_stage").first(),
            pl.col("development_stage").n_unique().alias("n_stages"),
            *[
                pl.col(c).unique().sort().str.join(",").alias(c)
                if c in groups.columns
                else pl.lit(None, pl.String).alias(c)
                for c in ("assay", "suspension_type", "disease")
            ],
            (pl.col("sex").first() if "sex" in groups.columns else pl.lit(None, pl.String)).alias("sex"),
        )
        .sort("group")
        .with_columns(
            dataset_id=pl.lit(dataset_id),
            n_cells=pl.Series(n_cells.astype(np.int64)),
            n_umis=pl.Series(np.rint(sums.sum(axis=1)).astype(np.int64)),
            sample_id=pl.lit(dataset_id[:8] + ":")
            + pl.col("donor_id")
            + ":"
            + pl.col("tissue")
            + (pl.lit(":") + pl.col("cell_type") if by_cell_type else pl.lit("")),
        )
    )
    meta.write_parquet(out / "groups.parquet")
    counts = pl.DataFrame({"gene_id": genes}).with_columns(
        [pl.Series(sid, np.rint(sums[i]).astype(np.uint32)) for i, sid in enumerate(meta["sample_id"].to_list())]
    )
    counts.write_parquet(out / "counts.parquet", compression="zstd")
    log.info("%s: %d groups, %d genes in %.0fs", dataset_id, n_groups, len(genes), time.time() - t0)
    return out


def parse() -> None:
    root = input_dir(NAME)
    discover = json.loads((download_dir(NAME) / "discover_datasets.json").read_text())
    titles = {x["dataset_id"]: (x["collection_name"], x["title"]) for x in discover}
    metas, counts = [], []
    for d in sorted((root / "by_dataset").glob("*/groups.parquet")):
        metas.append(pl.read_parquet(d))
        counts.append(d.parent / "counts.parquet")
    if not metas:
        log.warning("Census: no pseudobulk yet; run `rna-age census-pseudobulk <dataset_id>` jobs first")
        return
    meta = pl.concat(metas, how="vertical_relaxed")
    ages = [stage_age(s) if s else (None,) * 4 for s in meta["development_stage"].to_list()]
    meta = meta.with_columns(
        age_type=pl.Series([a[0] for a in ages], dtype=pl.String),
        age_years=pl.Series([a[1] for a in ages], dtype=pl.Float64),
        age_min=pl.Series([a[2] for a in ages], dtype=pl.Float64),
        age_max=pl.Series([a[3] for a in ages], dtype=pl.Float64),
    )
    enough = (pl.col("n_cells") >= MIN_CELLS) & (pl.col("n_umis") >= MIN_UMIS)
    meta = meta.filter(pl.col("age_type").is_not_null() & enough & (pl.col("n_stages") == 1))
    is_normal = pl.col("disease").str.split(",").list.eval(pl.element() == "normal").list.all()
    samples = conform(
        meta.with_columns(
            source=pl.lit(NAME),
            study_id=pl.col("dataset_id"),
            donor_id=pl.col("dataset_id").str.slice(0, 8) + ":" + pl.col("donor_id"),
            age_raw=pl.col("development_stage"),
            sex=pl.col("sex").replace_strict({"female": "female", "male": "male"}, default=None),
            is_healthy=is_normal,
            is_tumor=pl.col("disease").str.contains(CANCER),
            is_cell_line=pl.lit(False),
            platform=pl.col("assay"),
            library_type=pl.col("assay"),
            read_count_total=pl.col("n_umis"),
            is_pseudobulk=pl.lit(True),
            cohort=pl.col("dataset_id").replace_strict({k: " / ".join(v) for k, v in titles.items()}, default=None),
            exclude_reason=pl.when(pl.col("disease").str.contains(CANCER)).then(pl.lit("tumor")),
        )
    )
    keep = set(samples["sample_id"].to_list())
    gene_sets = [pl.read_parquet(p, columns=["gene_id"])["gene_id"] for p in counts]
    gene_ids = sorted(set().union(*[set(g.to_list()) for g in gene_sets]))
    index = pl.DataFrame({"gene_id": gene_ids})
    columns: dict[str, np.ndarray] = {}
    for p in counts:
        df = pl.read_parquet(p)
        want = [c for c in df.columns if c in keep]
        if not want:
            continue
        df = index.join(df.select(["gene_id", *want]), on="gene_id", how="left").fill_null(0)
        columns.update({c: df[c].to_numpy() for c in want})
    ids = [s for s in samples["sample_id"].to_list() if s in columns]
    samples = samples.filter(pl.col("sample_id").is_in(ids))
    write_parts(root, gene_ids, ((b, np.column_stack([columns[s] for s in b])) for b in blocks_of(ids)))
    index.with_columns(symbol=pl.lit(None, pl.String), biotype=pl.lit(None, pl.String)).write_parquet(
        root / "genes.parquet"
    )
    samples.write_parquet(root / "samples.parquet")
    (root / "README.json").write_text(
        json.dumps(
            {"source": NAME, "census_version": CENSUS_VERSION, "min_cells": MIN_CELLS, "min_umis": MIN_UMIS,
             "datasets": sorted(meta["dataset_id"].unique().to_list()), "samples": samples.height},
            indent=2,
        )
    )  # fmt: skip
    log.info("Census pseudobulk: %d donor × tissue samples", samples.height)
