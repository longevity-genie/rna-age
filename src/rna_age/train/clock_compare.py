"""RM18: the bulk clocks and the single-cell clocks on the same people.

Step 1, accuracy: every single-cell cohort's donors, scored by the single-cell clock (its best model of
``sc_tabular``, cross-validated by person) and by the bulk clocks (their predictions on the same donors' whole-sample
pseudobulk, the ``transfer`` split, which no bulk model trained on). Per cohort and clock: MAE, bias (mean of pred -
age), Pearson r; the paired bootstrap of the MAE difference; and how far the two clocks agree on who is old for their
age (Spearman of their age-adjusted gaps).
"""

from __future__ import annotations

import numpy as np
import polars as pl

from rna_age.paths import OUTPUT

BULK_RUNS = {"bulk": "20261111-0747-ens-mlp3t2-plus-atlasft2", "health": "20261112-0022-ens-atlasft-rank1-3s"}


def bulk_on_pseudobulk(run_id: str) -> pl.DataFrame:
    """(ds8, donor, cohort, pred) of a bulk run on the pseudobulk samples; ``donor`` as ``sc_tabular`` keys it."""
    p = pl.read_parquet(OUTPUT / "runs" / run_id / "predictions.parquet").filter(pl.col("is_pseudobulk"))
    parts = pl.col("sample_id").str.split(":")
    return p.select(
        ds8=parts.list.first().str.slice(0, 8),
        donor=parts.list.slice(1).list.join(":"),
        cohort=pl.col("cohort"),
        pred=pl.col("pred"),
    )


def sc_best(path) -> tuple[str, pl.DataFrame]:
    """The single-cell clock of one cohort: its lowest-MAE model on the whole-sample representation."""
    s = pl.read_parquet(path).filter(pl.col("representation") == "natural")
    m = s.group_by("model").agg((pl.col("pred") - pl.col("age")).abs().mean().alias("mae")).sort("mae")["model"][0]
    return m, s.filter(pl.col("model") == m)


def _gap(pred: np.ndarray, age: np.ndarray) -> np.ndarray:
    b = np.polyfit(age, pred, 1)
    return pred - np.polyval(b, age)


def compare_cohort(d: pl.DataFrame, clocks: list[str], n_boot: int = 2000, seed: int = 0) -> list[dict]:
    """One row per bulk clock against ``sc``: both clocks' accuracy on the same donors, MAE difference with a
    bootstrap 95 % CI (positive = the bulk clock worse), and the Spearman of their age-adjusted gaps."""
    from scipy.stats import spearmanr

    a, sc = d["age"].to_numpy(), d["sc"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), (n_boot, len(a)))
    rows = []
    for c in clocks:
        x = d[c].to_numpy()
        e_sc, e_x = np.abs(sc - a), np.abs(x - a)
        diff = (e_x[idx] - e_sc[idx]).mean(1)
        rows.append({
            "clock": c, "n": len(a),
            "mae_sc": float(e_sc.mean()), "mae_bulk": float(e_x.mean()),
            "bias_sc": float((sc - a).mean()), "bias_bulk": float((x - a).mean()),
            "r_sc": float(np.corrcoef(sc, a)[0, 1]), "r_bulk": float(np.corrcoef(x, a)[0, 1]),
            "mae_diff": float(e_x.mean() - e_sc.mean()),
            "mae_diff_lo": float(np.quantile(diff, 0.025)), "mae_diff_hi": float(np.quantile(diff, 0.975)),
            "gap_rho": float(spearmanr(_gap(sc, a), _gap(x, a))[0]),
        })  # fmt: skip
    return rows


def same_donors(runs: dict[str, str] = BULK_RUNS) -> pl.DataFrame:
    bulk = None
    for name, run in runs.items():
        b = bulk_on_pseudobulk(run).rename({"pred": name})
        bulk = b if bulk is None else bulk.join(b.drop("cohort"), on=["ds8", "donor"], how="left")
    rows = []
    for path in sorted((OUTPUT / "summary").glob("sc_tabular_*.parquet")):
        if path.stem == "sc_tabular_compare":
            continue
        model, s = sc_best(path)
        ds8 = s["dataset_id"][0][:8]
        s = s.select("donor", "age", sc="pred").with_columns(ds8=pl.lit(ds8))
        d = s.join(bulk, on=["ds8", "donor"])
        if d.is_empty():  # sources keyed by person alone in the bulk table (terekhova2023: "A01", not "A01:blood")
            d = s.with_columns(donor=pl.col("donor").str.replace(r":[^:]*$", "")).join(bulk, on=["ds8", "donor"])
        d = d.drop_nulls(list(runs))
        for r in compare_cohort(d, list(runs)):
            rows.append({"dataset": ds8, "cohort": d["cohort"][0], "sc_model": model, "n_sc": s.height, **r})
    out = pl.DataFrame(rows)
    out.write_parquet(OUTPUT / "summary" / "clock_compare_same_donors.parquet")
    return out
