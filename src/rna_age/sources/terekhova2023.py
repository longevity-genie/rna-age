"""Terekhova et al. 2023 (Immunity): single-cell atlas of human blood during healthy aging.

166 healthy donors aged 25–85, 317 samples (some longitudinal), ~2 M PBMC. Not in CELLxGENE. Files live on
Synapse ``syn49637038`` (processed: ``GEX_HTO_processed`` = syn50542388, ``raw_counts_h5ad.tar.gz`` =
syn56693935); the AWS open-data mirror bucket denies anonymous listing, and Synapse refuses anonymous
downloads (403). A (free) Synapse account token is required: set ``SYNAPSE_AUTH_TOKEN``.

Planned: fetch ``raw_counts_h5ad.tar.gz`` with the Synapse REST API (file handle → presigned S3 URL →
segmented fetch), then pseudobulk per sample (donor × timepoint) like ``census_pseudobulk``.
"""

from __future__ import annotations

import os

NAME = "terekhova2023"
SYNAPSE_RAW_COUNTS = "syn56693935"


def fetch(connections: int = 4) -> None:
    if not os.environ.get("SYNAPSE_AUTH_TOKEN"):
        raise SystemExit(
            "terekhova2023 needs a Synapse token: Synapse refuses anonymous downloads of syn49637038 and the AWS "
            "mirror denies anonymous listing. Create a free Synapse account, make a personal access token with "
            "download scope, and export SYNAPSE_AUTH_TOKEN (e.g. in slurm/env.local.sh)."
        )
    raise NotImplementedError("Synapse download not implemented yet (token present): see module docstring")


def parse() -> None:
    raise NotImplementedError("terekhova2023: nothing fetched yet (Synapse token required)")
