"""Published transcriptomic clocks as controls: ``rna-age tage SOURCE --split test``.

tAge outputs are *relative*: each stratum (dataset × tissue) is centred on its own samples, so a prediction is
an age difference from the stratum's reference. We therefore compare against the true age difference from the
stratum mean (``age_delta``), and report results with and without samples from tAge's training series
(``in_tage_training``). Our own model is scored the same way in addition to absolute error.

Writes ``data/interim/tage/<source>_<split>/{counts,samples,predictions}.parquet`` and
``data/output/clocks/tage_<source>_<split>.parquet`` (sample_id, stratum, age, age_delta, clock columns).
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

import numpy as np
import polars as pl

from rna_age.counts import read_samples
from rna_age.paths import DATA, INPUT, INTERIM, OUTPUT, REPO_ROOT

log = logging.getLogger(__name__)

TAGE_ENV = Path(os.environ.get("TAGE_ENV", DATA / "envs" / "tage"))
CLOCK_DIR = DATA / "download" / "models" / "tage"
MIN_STRATUM = 6  # centring needs a few samples per stratum


def prepare(source: str, split: str = "test", max_samples: int | None = None) -> Path:
    m = pl.read_parquet(INTERIM / "samples_all.parquet")
    sel = m.filter(
        (pl.col("source") == source) & (pl.col("split") == split) & (pl.col("age_type") == "exact")
        if source != "gtex"
        else (pl.col("source") == source) & (pl.col("split") == split)
    ).with_columns(
        # centring unit = dataset × tissue; all of GTEx is one dataset (each donor is its own study_id)
        stratum=pl.when(pl.col("source") == "gtex").then(pl.lit("GTEx")).otherwise(pl.col("study_id"))
        + "|"
        + pl.col("tissue").fill_null("NA")
    )
    if max_samples and sel.height > max_samples:
        sel = sel.sample(max_samples, seed=0)
    sizes = sel.group_by("stratum").len()
    sel = sel.join(sizes.filter(pl.col("len") >= MIN_STRATUM).select("stratum"), on="stratum")
    sel = sel.with_columns(age_delta=pl.col("age_mid") - pl.col("age_mid").mean().over("stratum"))
    out = INTERIM / "tage" / f"{source}_{split}"
    out.mkdir(parents=True, exist_ok=True)
    genes, mat = read_samples(INPUT / source, sel["sample_id"].to_list())
    counts = pl.DataFrame({"gene_id": genes}).with_columns(
        [pl.Series(s, mat[i].astype(np.int64)) for i, s in enumerate(sel["sample_id"].to_list())]
    )
    counts.write_parquet(out / "counts.parquet")
    sel.select(
        "sample_id", "stratum", "study_id", "tissue", "age_mid", "age_delta", "in_tage_training", "sex"
    ).write_parquet(out / "samples.parquet")
    log.info("tAge input %s: %d samples in %d strata", out, sel.height, sel["stratum"].n_unique())
    return out


def run(source: str, split: str = "test", max_samples: int | None = None) -> pl.DataFrame:
    inp = prepare(source, split, max_samples)
    CLOCK_DIR.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, RETICULATE_PYTHON=str(TAGE_ENV / "bin" / "python"))
    cmd = [str(TAGE_ENV / "bin" / "Rscript"), str(REPO_ROOT / "scripts" / "tage_predict.R"), str(inp), str(CLOCK_DIR)]
    log.info("running %s", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)
    pred = pl.read_parquet(inp / "predictions.parquet")
    samples = pl.read_parquet(inp / "samples.parquet")
    res = samples.join(pred, on="sample_id", how="left")
    out = OUTPUT / "clocks"
    out.mkdir(parents=True, exist_ok=True)
    res.write_parquet(out / f"tage_{source}_{split}.parquet")
    for col in [c for c in pred.columns if "tAge" in c or "Chronoage" in c]:
        for name, sub in (("all", res), ("not_in_training", res.filter(~pl.col("in_tage_training")))):
            sub = sub.drop_nulls([col, "age_delta"])
            if sub.height > 2:
                r = np.corrcoef(sub[col].to_numpy(), sub["age_delta"].to_numpy())[0, 1]
                log.info("tAge %s %s/%s %s: n=%d, r(delta)=%.3f", col, source, split, name, sub.height, r)
    return res
