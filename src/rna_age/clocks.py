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
    rows = []
    for col in [c for c in pred.columns if "tAge" in c or "Chronoage" in c]:
        for name, sub in (("all", res), ("not_in_training", res.filter(~pl.col("in_tage_training")))):
            sub = sub.drop_nulls([col, "age_delta"])
            if sub.height > 2:
                r = float(np.corrcoef(sub[col].to_numpy(), sub["age_delta"].to_numpy())[0, 1])
                rows.append({"source": source, "split": split, "model": col, "subset": name, "n": sub.height,
                             "r_delta": r, "strata": sub["stratum"].n_unique()})  # fmt: skip
                log.info("tAge %s %s/%s %s: n=%d, r(delta)=%.3f", col, source, split, name, sub.height, r)
    pl.DataFrame(rows).write_parquet(out / f"tage_{source}_{split}_summary.parquet")
    return res


def compare_run(run_id: str, source: str, split: str = "test") -> pl.DataFrame:
    """Score one of our runs on the tAge samples/strata: r of (pred − stratum mean pred) with age_delta."""
    t = pl.read_parquet(OUTPUT / "clocks" / f"tage_{source}_{split}.parquet")
    tcols = [c for c in t.columns if "Chronoage" in c or "tAge" in c]
    t = t.drop_nulls(tcols[:1])  # only samples tAge scored, so both are compared on identical samples
    # recompute the age delta on exactly these samples (strata may have lost samples tAge could not score)
    t = t.with_columns(age_delta=pl.col("age_mid") - pl.col("age_mid").mean().over("stratum"))
    p = pl.read_parquet(OUTPUT / "runs" / run_id / "predictions.parquet").select("sample_id", "pred")
    j = t.join(p, on="sample_id").with_columns(pred_delta=pl.col("pred") - pl.col("pred").mean().over("stratum"))
    rows = []
    for name, sub in (("all", j), ("not_in_training", j.filter(~pl.col("in_tage_training")))):
        if sub.height > 2:
            for model, col in [(run_id, "pred_delta")] + [(c.removesuffix(".pkl"), c) for c in tcols]:
                r = float(np.corrcoef(sub[col].to_numpy(), sub["age_delta"].to_numpy())[0, 1])
                rows.append({"source": source, "split": split, "model": model, "subset": name, "n": sub.height,
                             "r_delta": r, "strata": sub["stratum"].n_unique()})  # fmt: skip
    out = pl.DataFrame(rows)
    out.write_parquet(OUTPUT / "clocks" / f"tage_{source}_{split}_vs_{run_id}_summary.parquet")
    return out


def compare_clockbase(run_id: str, splits: tuple[str, ...] = ("val", "test")) -> pl.DataFrame:
    """ClockBase's precomputed (relative, control-centred) RNA clocks vs one of our runs on the same GSMs.

    Metric: Pearson r of the within-series deviation of each prediction with the within-series deviation of
    chronological age (series = GEO study; ClockBase centres per dataset/tissue, so only deviations compare).
    """
    cb = pl.read_parquet(INPUT / "clockbase" / "predictions.parquet")
    wide = cb.pivot(on="clock", index="sample_id", values="value", aggregate_function="first")
    clocks = [c for c in wide.columns if c != "sample_id"]
    p = pl.read_parquet(OUTPUT / "runs" / run_id / "predictions.parquet").filter(
        pl.col("split").is_in(list(splits)) & (pl.col("age_type") == "exact")
    )
    j = p.join(wide, on="sample_id", how="inner").drop_nulls(clocks[:1])
    sizes = j.group_by("study_id").len().filter(pl.col("len") >= 6)
    j = j.join(sizes.select("study_id"), on="study_id")
    j = j.with_columns(age_delta=pl.col("age_years") - pl.col("age_years").mean().over("study_id"))
    rows = []
    for model, col in [(run_id, "pred")] + [(c, c) for c in clocks]:
        d = j.with_columns(_d=pl.col(col) - pl.col(col).mean().over("study_id")).drop_nulls("_d")
        if d.height > 2:
            r = float(np.corrcoef(d["_d"].to_numpy(), d["age_delta"].to_numpy())[0, 1])
            rows.append({"model": model, "n": d.height, "studies": d["study_id"].n_unique(), "r_delta": r})
    out = pl.DataFrame(rows)
    out.write_parquet(OUTPUT / "clocks" / f"clockbase_vs_{run_id}_summary.parquet")
    return out
