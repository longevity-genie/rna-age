"""One module per data source. Each exposes ``NAME``, ``fetch(connections)`` and ``parse()``."""

from __future__ import annotations

import importlib
from types import ModuleType

# Cancer cohorts (TCGA, TARGET) are out of scope for now (Anton, 2026-11-10); tumour samples inside mixed
# sources are kept but flagged (is_tumor) and excluded from training and evaluation.
SOURCES = ("gtex", "archs4", "ncbi_geo", "bgee", "geo_cohorts", "census_pseudobulk", "terekhova2023")
# published-clock controls (no samples.parquet; read by merge / evaluation)
CONTROLS = ("longevitybench", "clockbase", "primo")


def get(name: str) -> ModuleType:
    name = name.replace("-", "_")
    if name not in SOURCES + CONTROLS:
        raise KeyError(f"unknown source {name!r}; known: {', '.join(SOURCES + CONTROLS)}")
    return importlib.import_module(f"rna_age.sources.{name}")
