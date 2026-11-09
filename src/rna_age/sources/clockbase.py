"""ClockBase (Gladyshev lab): precomputed clock predictions for public GEO samples, per-GSE JSON API only.

For human RNA-seq series it returns two tAge outputs (``Human Chronological Age Clock``, ``Human Hazard Clock``)
computed on ARCHS4 counts, centred on each dataset/tissue's controls: **relative values, not years**. They are
a cross-check for our tAge wrapper (same model family, same input source), joined by GSM.

We only query the series of our held-out (val/test) ARCHS4 / NCBI samples, one request at a time with a pause:
``data/download/clockbase/<GSE>_clocks.json`` → ``data/input/clockbase/predictions.parquet``.
"""

from __future__ import annotations

import json
import logging
import time

import httpx
import polars as pl

from rna_age.paths import INTERIM, download_dir, input_dir

log = logging.getLogger(__name__)

NAME = "clockbase"
API = "https://clockbase.gladyshevlab.org/api"
PAUSE = 1.0


def fetch(connections: int = 1, max_series: int = 2000) -> None:
    d = download_dir(NAME)
    splits = pl.read_parquet(INTERIM / "samples_all.parquet", columns=["source", "study_id", "split"])
    series = (
        splits.filter(pl.col("source").is_in(["archs4", "ncbi_geo"]) & pl.col("split").is_in(["val", "test"]))
        ["study_id"].unique().sort().to_list()
    )  # fmt: skip
    log.info("ClockBase: %d held-out GEO series to query", len(series))
    with httpx.Client(timeout=120, follow_redirects=True) as c:
        for gse in series[:max_series]:
            path = d / f"{gse}_clocks.json"
            if path.exists():
                continue
            r = c.get(f"{API}/gse/{gse}/clocks")
            if r.status_code == 404:
                path.write_text("null")
            elif r.status_code == 200:
                path.write_bytes(r.content)
            else:
                log.warning("ClockBase %s: HTTP %d", gse, r.status_code)
            time.sleep(PAUSE)


def parse() -> None:
    rows = []
    for path in sorted(download_dir(NAME).glob("GSE*_clocks.json")):
        data = json.loads(path.read_text())
        if not data:
            continue
        gse = path.name.split("_", 1)[0]
        records = data if isinstance(data, list) else data.get("data") or data.get("clocks") or []
        for rec in records:
            sid = rec.get("sample_id") or rec.get("gsm")
            for k, v in rec.items():
                if k in ("sample_id", "gsm") or not isinstance(v, (int, float)):
                    continue
                rows.append({"study_id": gse, "sample_id": sid, "clock": k, "value": float(v)})
    out = pl.DataFrame(rows, schema={"study_id": pl.String, "sample_id": pl.String, "clock": pl.String,
                                     "value": pl.Float64})  # fmt: skip
    out.write_parquet(input_dir(NAME) / "predictions.parquet")
    log.info("ClockBase: %d predictions for %d samples", out.height, out["sample_id"].n_unique())
