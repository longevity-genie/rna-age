"""LongevityBench (Galkin, Gladyshev, Zhavoronkov et al., Cell 2026): Hugging Face ``insilicomedicine/longebench``.

Its transcriptomic tasks are GTEx-only, posed to language models as text prompts (ranked gene symbols with
log1p TPM), labelled with GTEx's public 10-year bins: LB-0014 ``gtex_age_multiclass`` and LB-0018
``gtex_age_pairwise`` (accuracy), plus LB-0022 ternary in ``extra``. We extract the GTEx sample IDs and labels so
(1) GTEx samples used by the benchmark can be flagged / evaluated as an external test with our own model on the
same samples, and (2) our predictions can be scored with the benchmark's bin-accuracy metric.
Licence: CC-BY-NC-4.0 (source datasets keep their own terms).

Output: ``data/input/longevitybench/gtex_tasks.parquet`` (lb_id, task, split_file, sample_id, label_raw).
"""

from __future__ import annotations

import json
import logging
import re

import polars as pl

from rna_age.download import Remote
from rna_age.download import fetch as _fetch
from rna_age.paths import download_dir, input_dir

log = logging.getLogger(__name__)

NAME = "longevitybench"
BASE = "https://huggingface.co/datasets/insilicomedicine/longebench/resolve/main"
FILES = {
    "benchmark": ("benchmark/eval-00000-of-00001.parquet", 91_418_832),
    "extra": ("extra/eval-00000-of-00001.parquet", 46_436_287),
    "mini": ("mini/eval-00000-of-00001.parquet", 6_920_300),
}
GTEX_ID = re.compile(r"GTEX-[A-Z0-9]+-[0-9A-Z]+-SM-[A-Z0-9]+")


def fetch(connections: int = 4) -> None:
    for key, (path, size) in FILES.items():
        _fetch(Remote(url=f"{BASE}/{path}", name=f"longebench_{key}.parquet", size=size), download_dir(NAME))


def parse() -> None:
    rows = []
    for key in FILES:
        df = pl.read_parquet(download_dir(NAME) / f"longebench_{key}.parquet", columns=["lb_id", "task", "metadata"])
        gtex = df.filter(pl.col("task").str.contains("(?i)gtex"))
        for lb_id, task, meta in gtex.iter_rows():
            text = meta if isinstance(meta, str) else json.dumps(meta)
            follow = json.loads(text).get("follow_up", "") if text.startswith("{") else text
            label = re.search(r"Actual age:\s*([^;\"]+)", str(follow))
            for sid in sorted(set(GTEX_ID.findall(text))):
                rows.append(
                    {"lb_id": lb_id, "task": task, "split_file": key, "sample_id": sid,
                     "label_raw": label.group(1).strip() if label else None}
                )  # fmt: skip
    out = pl.DataFrame(
        rows, schema={"lb_id": pl.String, "task": pl.String, "split_file": pl.String, "sample_id": pl.String,
                      "label_raw": pl.String},
    ).unique()  # fmt: skip
    out.write_parquet(input_dir(NAME) / "gtex_tasks.parquet")
    log.info(
        "LongevityBench: %d GTEx sample-task rows, %d distinct samples, tasks %s",
        out.height, out["sample_id"].n_unique(), sorted(out["task"].unique().to_list()),
    )  # fmt: skip
