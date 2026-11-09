"""ARCHS4 human gene counts (Maayan lab), one HDF5 with all GEO/SRA human samples.

The file is ``human_gene_v2.latest.h5`` on the Maayan lab S3; no versioned name exists for this release,
so the S3 ETag recorded in the sidecar pins what we fetched.
"""

from __future__ import annotations

from rna_age.download import Remote
from rna_age.download import fetch as _fetch
from rna_age.paths import download_dir

NAME = "archs4"
URL = "https://s3.dev.maayanlab.cloud/archs4/files/human_gene_v2.latest.h5"
H5 = Remote(
    url=URL,
    name="human_gene_v2.latest.h5",
    size=64_432_723_113,
    checksum="s3etag:9b8fcd0486471467ef00169bc4ce1511-7681",
)


def fetch(connections: int = 8) -> None:
    _fetch(H5, download_dir(NAME), connections=connections)


def h5_path():
    return download_dir(NAME) / H5.name
