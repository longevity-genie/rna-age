"""Data layout. Everything lives under ``data/`` at the repository root (or ``$RNA_AGE_DATA``).

- ``data/download/<source>/``: raw bytes exactly as fetched, plus a ``<file>.meta.json`` sidecar each.
- ``data/input/<source>/``: parsed per-source tables (``samples.parquet``, ``genes.parquet``, ``counts/``).
- ``data/interim/``: merged tables, splits, embeddings.
- ``data/output/``: results; ``data/output/summary/*.csv`` is tracked in git.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("RNA_AGE_DATA", REPO_ROOT / "data"))
DOWNLOAD = DATA / "download"
INPUT = DATA / "input"
INTERIM = DATA / "interim"
OUTPUT = DATA / "output"
LOGS = DATA / "logs"


def download_dir(source: str) -> Path:
    path = DOWNLOAD / source
    path.mkdir(parents=True, exist_ok=True)
    return path


def input_dir(source: str) -> Path:
    path = INPUT / source
    path.mkdir(parents=True, exist_ok=True)
    return path
