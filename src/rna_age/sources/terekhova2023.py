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


def parse() -> None:
    raise NotImplementedError("terekhova2023 parse: inspect the unpacked archive first (see STATUS.md)")
