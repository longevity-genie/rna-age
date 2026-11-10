"""PRIMO benchmark data (Hugging Face dataset ``PRIMOmics/primo``), bulk RNA, at the revision the leaderboard
scores (``DATA_REVISION`` of primo-eval). Fetched with our downloader, verified against the Hub's LFS sha256,
into ``data/download/primo/data/`` with the same layout as ``primo download`` (``datasets.yaml``,
``dNNN/expression.h5ad``, ``public_tasks/tNNN/*``), so the official ``primo evaluate`` can read it.

PRIMO is not an age benchmark: we use it to test whether our age-trained models' sample embeddings help or hurt
patient-level tasks (clinical score, endotype, perturbation, prognosis, treatment outcome).
"""

from __future__ import annotations

import json
import logging

import httpx
import yaml

from rna_age.download import Remote
from rna_age.download import fetch as _fetch
from rna_age.paths import download_dir

log = logging.getLogger(__name__)

NAME = "primo"
REPO = "PRIMOmics/primo"
DATA_REVISION = "4b5aaaf9225bfaec61565c8c17232652845bb07f"  # primo-eval data commit of the current board
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/{DATA_REVISION}"


def data_dir():
    d = download_dir(NAME) / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d


def fetch(connections: int = 4, modality: str = "bulk RNA") -> None:
    d = data_dir()
    with httpx.Client(timeout=120, follow_redirects=True) as c:
        tree = c.get(f"https://huggingface.co/api/datasets/{REPO}/tree/{DATA_REVISION}", params={"recursive": "true"})
        tree.raise_for_status()
        files = {x["path"]: x for x in tree.json() if x["type"] == "file"}
        (d.parent / "hub_tree.json").write_text(json.dumps(list(files.values()), indent=1))
        manifest = c.get(f"{BASE}/datasets.yaml").raise_for_status().text
    (d / "datasets.yaml").write_text(manifest)
    entries = [e for e in yaml.safe_load(manifest)["datasets"] if e["modality"] == modality]
    wanted = [e["path"] for e in entries] + [p for p in files if p.startswith("public_tasks/")]
    wanted += [p for p in ("README.md", "ATTRIBUTION.md") if p in files]
    wanted += [p for p in files if p.endswith("LICENCE.md") and p.split("/")[0] in {e["id"] for e in entries}]
    for path in wanted:
        f = files.get(path)
        if f is None:
            log.warning("PRIMO: %s not on the Hub at %s", path, DATA_REVISION[:8])
            continue
        sha = (f.get("lfs") or {}).get("oid")
        sub = d / path.rsplit("/", 1)[0] if "/" in path else d
        _fetch(
            Remote(url=f"{BASE}/{path}", name=path.rsplit("/", 1)[-1], size=int(f["size"]),
                   checksum=f"sha256:{sha}" if sha else None),
            sub, connections=connections,
        )  # fmt: skip
    log.info("PRIMO: %d %s datasets + public tasks at %s", len(entries), modality, DATA_REVISION[:8])


def parse() -> None:
    log.info("PRIMO has no parse step: embeddings are produced by `rna-age primo-embed`")
