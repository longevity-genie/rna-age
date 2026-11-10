"""Terekhova et al. 2023 (Immunity): single-cell atlas of human blood during healthy aging.

166 healthy donors aged 25–85, 317 samples (some longitudinal), ~2 M PBMC; not in CELLxGENE. Synapse
(syn49637038) and its AWS mirror refuse anonymous access, but the HCA Data Portal serves it openly: project
``896f377c-8e88-463e-82b0-b2a5409d6fe4`` in Azul catalog ``dcp60``. We fetch ``all_pbmcs.tar.gz`` (count matrix +
sample annotation) and the metadata xlsx through Azul's ``/repository/files/<uuid>`` redirect, verified against
Azul's sha256.

Parse (Slurm): unpack, pseudobulk per sample (donor × visit) like ``census_pseudobulk``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import httpx

from rna_age.download import Remote
from rna_age.download import fetch as _fetch
from rna_age.paths import download_dir

log = logging.getLogger(__name__)

NAME = "terekhova2023"
AZUL = "https://service.azul.data.humancellatlas.org"
CATALOG = "dcp60"
PROJECT = "896f377c-8e88-463e-82b0-b2a5409d6fe4"
WANTED = ("all_pbmcs.tar.gz", "BloodAge10xArtyomov_metadata_09-01-2024.xlsx")


def files() -> list[dict]:
    d = download_dir(NAME)
    params = {
        "catalog": CATALOG,
        "filters": json.dumps({"projectId": {"is": [PROJECT]}}),
        "size": "100",
    }
    with httpx.Client(timeout=120) as c:
        data = c.get(f"{AZUL}/index/files", params=params).raise_for_status().json()
    (d / "azul_files.json").write_text(json.dumps(data, indent=1))
    return [f for h in data["hits"] for f in h["files"]]


def _resolve(c: httpx.Client, f: dict) -> str:
    """Follow Azul's repository redirect(s) to the storage URL."""
    url = f"{AZUL}/repository/files/{f['uuid']}"
    params = {"catalog": CATALOG, "version": f["version"]}
    for _ in range(10):
        r = c.get(url, params=params, follow_redirects=False)
        if r.status_code in (301, 302, 303, 307, 308):
            return r.headers["location"]
        if r.status_code == 200 and r.headers.get("content-type", "").startswith("application/json"):
            body = r.json()
            if body.get("Status") == 302:
                return body["Location"]
            if body.get("Status") == 301:  # still preparing: retry after the suggested delay
                url, params = body["Location"], None
                continue
        r.raise_for_status()
    raise RuntimeError(f"Azul did not resolve {f['name']}")


def fetch(connections: int = 8) -> None:
    try:
        fetch_synapse_raw(connections)
    except (httpx.HTTPStatusError, SystemExit) as e:  # token missing/invalid: HCA copy + count recovery instead
        log.warning("Synapse raw counts not fetched (%s); using the HCA archive", type(e).__name__)
    d = download_dir(NAME)
    with httpx.Client(timeout=120) as c:
        for f in files():
            if f["name"] not in WANTED:
                continue
            url = _resolve(c, f)
            _fetch(
                Remote(url=url, name=f["name"], size=int(f["size"]), checksum=f"sha256:{f['sha256']}"),
                d,
                connections=connections,
            )


SYNAPSE = "https://repo-prod.prod.sagebase.org/repo/v1"
SYNAPSE_RAW = ("syn56693935", "raw_counts_h5ad.tar.gz", 10_500_000_000, "20fcbe2531a999c51ed03dfce5d29b7b")


def _synapse_token() -> str:
    """Personal access token from ~/.synapseConfig, read at runtime only (never logged or passed on)."""
    import configparser

    cfg = configparser.RawConfigParser()
    cfg.read(Path.home() / ".synapseConfig")
    for section in cfg.sections():
        for key in ("authtoken", "auth_token"):
            if cfg.has_option(section, key):
                return cfg.get(section, key).strip().strip("\"'")
    raise SystemExit("no Synapse auth token in ~/.synapseConfig")


def fetch_synapse_raw(connections: int = 8) -> Path:
    """Raw-count h5ads (syn56693935) from Synapse via a pre-signed S3 URL, md5-verified."""
    entity, name, _, md5 = SYNAPSE_RAW
    d = download_dir(NAME)
    headers = {"Authorization": f"Bearer {_synapse_token()}"}
    with httpx.Client(timeout=120, headers=headers) as c:
        r = c.get(f"{SYNAPSE}/entity/{entity}/file", params={"redirect": "false"})
        r.raise_for_status()
        url = r.text.strip().strip('"')
    with httpx.Client(timeout=120) as c:
        size = int(c.get(url, headers={"Range": "bytes=0-0"}).headers["content-range"].rpartition("/")[2])
    path = _fetch(Remote(url=url, name=name, size=size, checksum=f"md5:{md5}"), d, connections=connections)
    meta = path.with_name(path.name + ".meta.json")
    m = json.loads(meta.read_text())
    m["url"] = f"synapse:{entity}"  # the pre-signed URL is a temporary credential; do not keep it
    meta.write_text(json.dumps(m, indent=2) + "\n")
    return path


MIN_CELLS = 200
MIN_UMIS = 500_000


def _unpack(member: str, dest_root: Path) -> Path:
    import os
    import subprocess

    dest = dest_root / member
    if not dest.exists():
        dest_root.mkdir(parents=True, exist_ok=True)
        archive = download_dir(NAME) / "all_pbmcs.tar.gz"
        log.info("unpacking %s from %s", member, archive.name)
        subprocess.run(["tar", "xzf", str(archive), "-C", str(dest_root), member], check=True)
        os.chmod(dest, 0o644)
    return dest


def parse(chunk: int = 100_000, by_cell_type: bool = False) -> None:
    """Pseudobulk per tube (donor × visit): raw UMIs per cell summed per sample.

    ``by_cell_type``: per tube × published cell cluster (``Cluster_names``) instead, written as
    ``by_dataset_celltype/terekhova2023/{groups,counts}.parquet`` in the layout of the Census cell-type pseudobulk, for
    the composition analyses (never merged into the age samples).

    Uses ``.raw.X`` / ``layers/counts`` / integer ``X`` when present; otherwise recovers UMIs from log1p-CP10k
    as ``round(expm1(x) * nCount_RNA / 1e4)`` with the per-cell total from the metadata, after checking on the
    first chunk that the recovered values are integral.
    """
    import os
    import time

    import h5py
    import numpy as np
    import polars as pl
    import scipy.sparse as sp

    from rna_age.counts import blocks_of, write_parts
    from rna_age.paths import INTERIM, input_dir
    from rna_age.schema import conform
    from rna_age.sources.census_pseudobulk import _obs_column, _var_ids

    out = input_dir(NAME)
    scratch = Path(os.environ.get("RNA_AGE_LOCAL_TMP") or INTERIM / NAME / "unpacked")
    meta_csv = _unpack("all_pbmcs/all_pbmcs_metadata.csv", INTERIM / NAME / "unpacked")
    h5ad = _unpack("all_pbmcs/all_pbmcs_rna.h5ad", scratch)
    meta = pl.read_csv(meta_csv, infer_schema_length=10000).rename({"": "barcode"})
    t0 = time.time()
    with h5py.File(h5ad, "r") as f:
        obs_index_key = f["obs"].attrs.get("_index", "_index")
        barcodes = _obs_column(f["obs"], obs_index_key)
        if "layers" in f and "counts" in f["layers"]:
            X, genes, mode = f["layers/counts"], _var_ids(f, False), "layers/counts"
        elif "raw" in f and "X" in f["raw"]:
            X, genes, mode = f["raw/X"], _var_ids(f, True), "raw/X"
        else:
            X, genes, mode = f["X"], _var_ids(f, False), "X"
        log.info("Terekhova h5ad: %d cells × %d genes, matrix %s", len(barcodes), len(genes), mode)
        cells = pl.DataFrame({"barcode": barcodes.astype(str)}).with_row_index("cell")
        cells = cells.join(meta.select("barcode", "Tube_id", "Cluster_names", "nCount_RNA"), on="barcode", how="left")
        key = (
            pl.when(pl.col("Tube_id").is_not_null() & pl.col("Cluster_names").is_not_null()).then(
                pl.col("Tube_id") + "\x1f" + pl.col("Cluster_names")
            )
            if by_cell_type
            else pl.col("Tube_id")
        )
        cells = cells.with_columns(key=key)
        tubes = cells["key"].drop_nulls().unique().sort().to_list()
        tindex = {t: i for i, t in enumerate(tubes)}
        cell_group = np.array([tindex.get(t, -1) if t is not None else -1 for t in cells["key"].to_list()])
        ncount = cells["nCount_RNA"].fill_null(0).to_numpy().astype(np.float64)
        n = len(barcodes)
        sums = np.zeros((len(tubes), len(genes)), dtype=np.float64)
        indptr = X["indptr"][()]
        recover = None
        for lo in range(0, n, chunk):
            hi = min(n, lo + chunk)
            a, b = int(indptr[lo]), int(indptr[hi])
            data = X["data"][a:b].astype(np.float64)
            m = sp.csr_matrix((data, X["indices"][a:b], indptr[lo : hi + 1] - a), shape=(hi - lo, len(genes)))
            if recover is None:
                integral = np.allclose(data[:100000], np.rint(data[:100000]))
                recover = not integral
                if recover:
                    rows = np.repeat(np.arange(hi - lo), np.diff(m.indptr))
                    test = np.expm1(m.data) * ncount[lo:hi][rows] / 1e4
                    frac = float(np.mean(np.abs(test - np.rint(test)) < 0.05))
                    log.info("X is not integral; recovered UMIs integral for %.1f%% of entries", 100 * frac)
                    if frac < 0.95:
                        raise ValueError("cannot recover integer UMIs from X (not log1p-CP10k with nCount_RNA?)")
            if recover:
                rows = np.repeat(np.arange(hi - lo), np.diff(m.indptr))
                m.data = np.rint(np.expm1(m.data) * ncount[lo:hi][rows] / 1e4)
            g = cell_group[lo:hi]
            keep = np.flatnonzero(g >= 0)
            ind = sp.csr_matrix((np.ones(len(keep)), (g[keep], keep)), shape=(len(tubes), hi - lo))
            sums += (ind @ m).toarray()
            log.info("Terekhova: %d/%d cells (%.0fs)", hi, n, time.time() - t0)
    n_cells = np.bincount(cell_group[cell_group >= 0], minlength=len(tubes))
    if by_cell_type:
        _write_cell_type(meta, tubes, n_cells, sums, genes)
        return
    per_tube = (
        meta.group_by("Tube_id")
        .agg(
            pl.col("Donor_id").first(), pl.col("Age").first(), pl.col("Sex").first(), pl.col("Batch").first(),
            pl.col("Age").n_unique().alias("n_ages"),
        )
        .filter(pl.col("Tube_id").is_in(tubes))
    )  # fmt: skip
    order = pl.DataFrame({"Tube_id": tubes, "n_cells": n_cells.astype(np.int64),
                          "n_umis": np.rint(sums.sum(axis=1)).astype(np.int64)})  # fmt: skip
    df = order.join(per_tube, on="Tube_id", how="left")
    samples = conform(
        df.with_columns(
            sample_id=pl.lit("terekhova2023:") + pl.col("Tube_id"),
            source=pl.lit(NAME),
            study_id=pl.lit("terekhova2023"),
            donor_id=pl.lit("terekhova2023:") + pl.col("Donor_id"),
            age_years=pl.col("Age").cast(pl.Float64),
            age_min=pl.col("Age").cast(pl.Float64),
            age_max=pl.col("Age").cast(pl.Float64),
            age_type=pl.lit("exact"),
            age_raw=pl.lit("Age: ") + pl.col("Age").cast(pl.String),
            sex=pl.col("Sex").str.to_lowercase(),
            tissue=pl.lit("blood"),
            cell_type=pl.lit("PBMC"),
            disease=pl.lit("normal"),
            is_healthy=pl.lit(True),
            is_tumor=pl.lit(False),
            is_cell_line=pl.lit(False),
            platform=pl.lit("10x Genomics"),
            library_type=pl.lit(f"10x; counts from {mode}{' (recovered from log1p-CP10k)' if recover else ''}"),
            assay=pl.lit("10x"),
            suspension_type=pl.lit("cell"),
            is_pseudobulk=pl.lit(True),
            read_count_total=pl.col("n_umis"),
            cohort=pl.lit("terekhova2023"),
            exclude_reason=pl.when((pl.col("n_cells") < MIN_CELLS) | (pl.col("n_umis") < MIN_UMIS)).then(
                pl.lit("low_material")
            ),
        )
    )
    ids = samples["sample_id"].to_list()
    col = {f"terekhova2023:{t}": i for i, t in enumerate(tubes)}
    write_parts(out, genes, ((b, np.rint(sums[[col[s] for s in b]].T)) for b in blocks_of(ids)))
    pl.DataFrame({"gene_id": genes}).with_columns(
        symbol=pl.lit(None, pl.String), biotype=pl.lit(None, pl.String)
    ).write_parquet(out / "genes.parquet")
    samples.write_parquet(out / "samples.parquet")
    (out / "README.json").write_text(
        json.dumps({"source": NAME, "matrix": mode, "recovered_from_log1p_cp10k": bool(recover),
                    "samples": samples.height, "min_cells": MIN_CELLS, "min_umis": MIN_UMIS}, indent=2)
    )  # fmt: skip
    log.info("Terekhova pseudobulk: %d tubes from %d donors", samples.height, df["Donor_id"].n_unique())


def _write_cell_type(meta, keys: list[str], n_cells, sums, genes: list[str]) -> None:
    """Tube × cluster pseudobulk in the Census cell-type layout (``composition.analyse`` reads it)."""
    import numpy as np
    import polars as pl

    from rna_age.paths import input_dir

    per_tube = meta.group_by("Tube_id").agg(pl.col("Donor_id").first(), pl.col("Age").first(), pl.col("Sex").first())
    g = pl.DataFrame(
        {"key": keys, "n_cells": n_cells.astype(np.int64), "n_umis": np.rint(sums.sum(axis=1)).astype(np.int64)}
    )
    g = g.with_columns(pl.col("key").str.split_exact("\x1f", 1).struct.rename_fields(["Tube_id", "cell_type"])).unnest(
        "key"
    )
    g = g.join(per_tube, on="Tube_id", how="left").with_columns(
        donor_id=pl.col("Tube_id"),  # a tube is one donor visit; composition groups by donor:tissue
        tissue=pl.lit("blood"),
        development_stage=pl.col("Age").cast(pl.Int64).cast(pl.String) + "-year-old stage",
        disease=pl.lit("normal"),
        dataset_id=pl.lit("terekhova2023"),
        sample_id=pl.lit("terekhova2023:") + pl.col("Tube_id") + ":" + pl.col("cell_type"),
    )
    out = input_dir(NAME) / "by_dataset_celltype" / "terekhova2023"
    out.mkdir(parents=True, exist_ok=True)
    g.write_parquet(out / "groups.parquet")
    counts = pl.DataFrame({"gene_id": genes}).with_columns(
        [pl.Series(sid, np.rint(sums[i]).astype(np.uint32)) for i, sid in enumerate(g["sample_id"].to_list())]
    )
    counts.write_parquet(out / "counts.parquet", compression="zstd")
    log.info("Terekhova by cell type: %d tube × cluster groups, %d genes → %s", g.height, len(genes), out)
