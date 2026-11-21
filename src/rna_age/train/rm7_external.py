"""RM7: NCBI-generated GEO counts as an external test of the headline clock and a pipeline cross-check.

The NCBI series never enter the age table, a split or any training (lead's decision, 2026-11-21). Two readings:

- **External test** (``not_in_archs4`` series, 278 with kept samples): studies ARCHS4 does not hold, so no study of
  them is in any split; samples whose GSM is in our sample table anyway are dropped. Reported: MAE, median absolute
  error, r, with a 95 % CI from resampling studies, for all kept exact-age samples and the healthy ones; against the
  headline's own test (7.46 y) and RM26's recount3 new studies (11.2 y).
- **Pipeline cross-check** (``crosscheck`` series, which ARCHS4 holds): the same GSM scored from NCBI counts and
  from ARCHS4 counts (the run's ``predictions.parquet``, val / test / transfer samples only; train samples have no
  out-of-sample prediction), the difference per sample (mean, SD, r between the two).

Predictions: the headline ``20261111-0747-ens-mlp3t2-plus-atlasft2`` is the mean of two ensembles, the 20k-HVG MLPs
(no weights are saved, so they are retrained with the same recipe and ``--external --external-source ncbi_geo``, which
scores these samples with each run's own genes and scaler; the reruns' test MAE is checked against the originals) and
the Atlas fine-tune pair (saved checkpoints, :func:`predict_atlas`, GPU).
"""

from __future__ import annotations

import logging

import numpy as np
import polars as pl

from rna_age.paths import INPUT, INTERIM, OUTPUT

log = logging.getLogger(__name__)

SOURCE = "ncbi_geo"
TARGETS = INTERIM / "ncbi_geo_targets.parquet"
PRED_DIR = INTERIM / "ncbi_geo_predictions"
ATLAS_RUNS = ("20261110-0555-atlas-ft-u4-all", "20261110-0533-atlas-ft-u6-all")
ATLAS_ENSEMBLE = "20261110-1329-ens-atlasft-u4u6"


def kept_samples() -> pl.DataFrame:
    """Kept NCBI samples with their target group (``crosscheck`` / ``not_in_archs4``) by series."""
    s = pl.read_parquet(INPUT / SOURCE / "samples.parquet").filter(pl.col("exclude_reason").is_null())
    t = pl.read_parquet(TARGETS).rename({"gse": "study_id", "reason": "target"})
    t = t.sort("target").unique("study_id", keep="first")  # a series listed under both: crosscheck wins (ARCHS4 holds it)
    return s.join(t, on="study_id", how="left")


def predict_atlas(runs: tuple[str, ...] = ATLAS_RUNS, batch: int = 2000) -> pl.DataFrame:
    """Each Atlas fine-tune scores every kept NCBI sample (GPU); writes ``<PRED_DIR>/atlas.parquet``
    (``sample_id``, ``run``, ``pred``)."""
    from rna_age.train.bulk_controls import atlas_inputs
    from rna_age.train.features import gene_ids, load_counts
    from rna_age.train.finetune import predict_checkpoint

    ids_all = kept_samples()["sample_id"].to_list()
    genes = gene_ids(SOURCE)
    parts = []
    for a in range(0, len(ids_all), batch):
        ids, X = atlas_inputs(ids_all[a : a + batch], genes, load_counts(SOURCE, ids_all[a : a + batch], genes))
        for r in runs:
            pred = predict_checkpoint(INTERIM / "checkpoints" / r / "best.pt", X)
            parts.append(pl.DataFrame({"sample_id": ids, "run": r, "pred": pred.astype(np.float64)}))
        log.info("atlas: %d / %d samples", min(a + batch, len(ids_all)), len(ids_all))
    out = pl.concat(parts)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    out.write_parquet(PRED_DIR / "atlas.parquet", compression="zstd")
    return out


def mlp_predictions(runs: list[str]) -> pl.DataFrame:
    """Mean of the MLP reruns' ``predictions_external_ncbi_geo.parquet``."""
    parts = [pl.read_parquet(OUTPUT / "runs" / r / f"predictions_external_{SOURCE}.parquet").with_columns(run=pl.lit(r))
             for r in runs]  # fmt: skip
    return pl.concat(parts).group_by("sample_id").agg(pl.col("pred").mean().alias("pred_mlp"), pl.len().alias("n_mlp"))


def headline_predictions(mlp_runs: list[str]) -> pl.DataFrame:
    """Per sample: the MLP ensemble, the Atlas pair and the headline (mean of the two ensembles, as its config)."""
    atlas = pl.read_parquet(PRED_DIR / "atlas.parquet").group_by("sample_id").agg(pl.col("pred").mean().alias("pred_atlas"))
    return mlp_predictions(mlp_runs).join(atlas, on="sample_id", how="inner").with_columns(
        pred=(pl.col("pred_mlp") + pl.col("pred_atlas")) / 2
    )


def study_bootstrap(err: np.ndarray, age: np.ndarray, pred: np.ndarray, study: np.ndarray, n_boot: int = 2000,
                    seed: int = 0) -> dict[str, float]:  # fmt: skip
    """MAE, median AE and r with 95 % CIs from resampling studies."""
    rng = np.random.default_rng(seed)
    us = np.unique(study)
    rows_of = {s: np.flatnonzero(study == s) for s in us}
    boots = []
    for _ in range(n_boot):
        idx = np.concatenate([rows_of[s] for s in rng.choice(us, len(us))])
        boots.append((err[idx].mean(), np.median(err[idx]), np.corrcoef(age[idx], pred[idx])[0, 1]))
    b = np.array(boots)
    lo, hi = np.nanpercentile(b, 2.5, axis=0), np.nanpercentile(b, 97.5, axis=0)
    return {"mae": float(err.mean()), "mae_lo": float(lo[0]), "mae_hi": float(hi[0]),
            "medae": float(np.median(err)), "medae_lo": float(lo[1]), "medae_hi": float(hi[1]),
            "r": float(np.corrcoef(age, pred)[0, 1]), "r_lo": float(lo[2]), "r_hi": float(hi[2])}  # fmt: skip


def external_test(pred: pl.DataFrame, n_boot: int = 2000) -> pl.DataFrame:
    """Accuracy on the not-in-ARCHS4 series, per model column and subset."""
    known = set(pl.read_parquet(INTERIM / "samples_all.parquet", columns=["sample_id"])["sample_id"].to_list())
    s = kept_samples().filter((pl.col("target") == "not_in_archs4") & (pl.col("age_type") == "exact"))
    s = s.filter(~pl.col("sample_id").is_in(list(known))).join(pred, on="sample_id", how="inner")
    rows = []
    for subset, d in (("all", s), ("healthy", s.filter(pl.col("is_healthy").fill_null(False)))):
        if d.height < 20:
            continue
        for col in ("pred", "pred_mlp", "pred_atlas"):
            age, p = d["age_years"].to_numpy(), d[col].to_numpy()
            m = study_bootstrap(np.abs(p - age), age, p, d["study_id"].to_numpy(), n_boot)
            rows.append({"reading": "external_test", "subset": subset, "model": col, "n": d.height,
                         "n_studies": d["study_id"].n_unique(), **m})  # fmt: skip
    return pl.DataFrame(rows)


def archs4_predictions(mlp_runs: list[str]) -> dict[str, pl.DataFrame]:
    """ARCHS4-count predictions from the same models that scored the NCBI counts: the MLP runs' own
    ``predictions.parquet`` (averaged), the Atlas ensemble's, and their mean as the headline."""
    cols = ["sample_id", "split", "age_years", "pred"]
    mlp = (pl.concat([pl.read_parquet(OUTPUT / "runs" / r / "predictions.parquet").select(cols) for r in mlp_runs])
           .group_by("sample_id", "split", "age_years").agg(pl.col("pred").mean()))  # fmt: skip
    atlas = pl.read_parquet(OUTPUT / "runs" / ATLAS_ENSEMBLE / "predictions.parquet").select(cols)
    head = mlp.join(atlas.select("sample_id", pl.col("pred").alias("pa")), on="sample_id", how="inner").with_columns(
        pred=(pl.col("pred") + pl.col("pa")) / 2
    ).drop("pa")
    return {"pred": head, "pred_mlp": mlp, "pred_atlas": atlas}


def crosscheck(pred: pl.DataFrame, mlp_runs: list[str]) -> pl.DataFrame:
    """NCBI-count vs ARCHS4-count predictions of the same GSMs by the same models (out-of-sample ARCHS4 predictions
    only: val / test / transfer)."""
    rows = []
    for col, a in archs4_predictions(mlp_runs).items():
        a = a.rename({"pred": "pred_archs4"})
        d = pred.select("sample_id", pl.col(col).alias("pred_ncbi")).join(a, on="sample_id", how="inner")
        if d.is_empty():
            continue
        diff = (d["pred_ncbi"] - d["pred_archs4"]).to_numpy()
        ok = d["age_years"].is_not_null().to_numpy()
        age = d["age_years"].to_numpy()
        rows.append({"reading": "crosscheck", "model": col, "n": d.height, "mean_diff": float(diff.mean()),
                     "sd_diff": float(diff.std()), "r_between": float(np.corrcoef(d["pred_ncbi"], d["pred_archs4"])[0, 1]),
                     "mae_ncbi": float(np.abs(d["pred_ncbi"].to_numpy()[ok] - age[ok]).mean()),
                     "mae_archs4": float(np.abs(d["pred_archs4"].to_numpy()[ok] - age[ok]).mean())})  # fmt: skip
    return pl.DataFrame(rows)


def run(mlp_runs: list[str], n_boot: int = 2000, write: bool = True) -> tuple[pl.DataFrame, pl.DataFrame]:
    pred = headline_predictions(mlp_runs)
    ext, cc = external_test(pred, n_boot), crosscheck(pred, mlp_runs)
    if write:
        ext.write_parquet(OUTPUT / "summary" / "rm7_external_test.parquet", compression="zstd")
        cc.write_parquet(OUTPUT / "summary" / "rm7_crosscheck.parquet", compression="zstd")
    return ext, cc
