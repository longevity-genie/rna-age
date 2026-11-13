"""RM30's evaluations of a trained expression-episode model, each through an existing code path:

- ``sc``: donor age on the seven single-cell cohorts with D7's folds (:func:`rna_age.train.sc_tabular.cross_validate`,
  the model behind :class:`~rna_age.train.expr_icl_model.ExprICLRegressor`), against the stored
  ``pseudobulk-tabicl-2000g`` predictions with the person-level paired bootstrap of
  :func:`rna_age.train.fm_embed.compare`. Bar (1): beats it beyond the CI in ≥ 4 of 7 cohorts.
- ``bulk``: test (and val) MAE with RM25's nearest-study prompt (:func:`rna_age.train.tabfm.nearest_study_context`) on
  the panel genes; a run with the usual ``predictions.parquet`` (:func:`rna_age.train.runs.finish_run`). Bar (2):
  within 1 y of 9.1.
- ``health``: :func:`rna_age.train.health.evaluate_health` on that bulk run. Bar (3): gap ≥ +2.9 y.
- ``perturb``: held-out knockdowns (never in a training episode): per knockdown × donor × condition, the Pearson
  over readout genes of predicted vs observed change, against the mean change of the training knockdowns (the
  mean baseline), paired bootstrap over genes. Features: the knockdown descriptor only (what is known of a
  knockdown not yet run).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import polars as pl

from rna_age.paths import OUTPUT
from rna_age.train.expr_icl import CACHE, EVAL_COHORTS, EpisodeSampler, descriptors, is_heldout_gene, load_corpus
from rna_age.train.expr_icl_model import ExprICL, ExprICLRegressor, _device, load_trained, predict_episode

log = logging.getLogger(__name__)

MODEL_NAME = "expr-icl"


def _run_dir(run_id: str) -> Path:
    return OUTPUT / "runs" / run_id


def eval_sc(
    run_id: str,
    model: ExprICL | None = None,
    cohorts: tuple[str, ...] = EVAL_COHORTS,
    n_genes: int = 500,
    n_estimators: int = 4,
    device: str = "auto",
    seed: int = 0,
    out_dir: Path | None = None,
) -> pl.DataFrame:
    """Out-of-fold donor ages per cohort (natural pseudobulk; blood cohorts also CD4 T cells), compared with the
    stored pseudobulk clocks. Writes ``eval_sc_oof.parquet`` and ``eval_sc.parquet`` in the run directory."""
    from rna_age.train import sc_tabular
    from rna_age.train.composition import _cpm, donor_profiles
    from rna_age.train.fm_embed import BLOOD, CD4, compare

    dev = _device(device)
    model = (model or load_trained(run_id)).to(dev)
    sc_tabular.ESTIMATORS[MODEL_NAME] = lambda s: ExprICLRegressor(model, dev, max_features=n_genes,
                                                                   n_estimators=n_estimators, seed=s)  # fmt: skip
    rows = []
    try:
        for ds in cohorts:
            dp = donor_profiles(ds, "broad" if ds in BLOOD else "cell_type")
            donors, age, groups, N, P = np.array(dp["donors"]), dp["age"], dp["groups"], dp["N"], dp["P"]
            reps = [("natural", np.ones(len(donors), dtype=bool), P.sum(1))]
            if ds in BLOOD and CD4 in dp["cix"]:
                k = dp["cix"][CD4]
                reps.append((f"class:{CD4}", N[:, k] >= 20, P[:, k]))
            for rep, ok, M in reps:
                X = np.log1p(_cpm(M[ok]))
                pred = sc_tabular.cross_validate(X, age[ok], groups[ok], (MODEL_NAME,), n_genes, seed)[MODEL_NAME]
                rows += [{"cohort": ds[:8], "representation": rep, "model": MODEL_NAME, "donor": d, "group": int(g),
                          "age": float(a), "pred": float(v)}
                         for d, g, a, v in zip(donors[ok], groups[ok], age[ok], pred, strict=True)]  # fmt: skip
                log.info("%s %s: MAE %.2f", ds[:8], rep, float(np.abs(pred - age[ok]).mean()))
            del dp, P
    finally:
        sc_tabular.ESTIMATORS.pop(MODEL_NAME, None)
    oof = pl.DataFrame(rows)
    out = compare(oof)
    d = out_dir or _run_dir(run_id)
    d.mkdir(parents=True, exist_ok=True)
    oof.write_parquet(d / "eval_sc_oof.parquet")
    out.write_parquet(d / "eval_sc.parquet")
    return out


def eval_bulk(
    run_id: str,
    model: ExprICL | None = None,
    splits: tuple[str, ...] = ("val", "test"),
    n_context: int = 4000,
    cap_per_study: int = 200,
    max_features: int = 500,
    n_estimators: int = 1,
    device: str = "auto",
    seed: int = 0,
    cache: Path = CACHE,
) -> dict:
    """One prompt per query study from the nearest training studies (RM25), panel genes; returns the bulk run's
    ``finish_run`` row (its ``run_id`` is what ``health`` reads). Context: training samples, healthy or unknown,
    exact ages or GTEx bin midpoints."""
    from sklearn.decomposition import PCA

    from rna_age.schema import age_mid
    from rna_age.train.baselines import PRED_COLS
    from rna_age.train.features import sample_table
    from rna_age.train.runs import finish_run, new_run
    from rna_age.train.tabfm import nearest_study_context

    dev = _device(device)
    model = (model or load_trained(run_id)).to(dev)
    corpus = load_corpus(cache, parts=("bulk",))
    s = sample_table().filter(~pl.col("is_pseudobulk"))
    allx = (
        corpus.bulk.select("sample_id")
        .with_row_index("_r")
        .join(s.with_columns(target=age_mid(s)), on="sample_id", how="inner")
    )
    X = corpus.bulk_X[allx["_r"].to_numpy()]
    y = allx["target"].to_numpy().astype(np.float64)
    ok_age = ((allx["age_type"] == "exact") | (allx["source"] == "gtex")).to_numpy() & ~np.isnan(y)
    pool = np.flatnonzero(
        (allx["split"] == "train").to_numpy() & allx["is_healthy"].fill_null(True).to_numpy() & ok_age
    )
    mu, sd = X[pool].mean(0), X[pool].std(0) + 1e-6
    Zc = PCA(50, svd_solver="randomized", random_state=seed).fit((X[pool] - mu) / sd).transform((X - mu) / sd)
    study = allx["study_id"].fill_null("").to_numpy().astype(str)
    q_rows = np.flatnonzero(np.isin(allx["split"].to_numpy(), list(splits)))
    out = np.full(allx.height, np.nan)
    studies = np.unique(study[q_rows])
    for j, st in enumerate(studies):
        q = q_rows[study[q_rows] == st]
        c = nearest_study_context(Zc, study, pool, q, n_context, cap_per_study, seed)
        est = ExprICLRegressor(model, dev, max_features=max_features, n_estimators=n_estimators, seed=seed)
        out[q] = est.fit(X[c], y[c]).predict(X[q])
        if j % 20 == 0:
            log.info("study %d / %d (%s, %d queries)", j + 1, len(studies), st, len(q))
    keep = np.zeros(allx.height, dtype=bool)
    keep[q_rows] = True
    pred = (
        allx.select([c for c in PRED_COLS if c in allx.columns])
        .with_columns(pred=pl.Series(out))
        .filter(pl.Series(keep))
    )
    bulk_dir = new_run(f"expricl-bulk-{run_id[:13]}", {
        "model": f"expr-icl in context ({run_id})", "features": f"panel {len(corpus.genes)} genes, top {max_features}",
        "train_set": "nearest-study prompt (RM25)", "n_context": n_context, "cap_per_study": cap_per_study,
        "splits": list(splits), "source_run": run_id})  # fmt: skip
    row = finish_run(bulk_dir, pred)
    (_run_dir(run_id) / "eval_bulk.json").write_text(json.dumps({"bulk_run_id": bulk_dir.name}, indent=2))
    return row


def eval_health(run_id: str, bulk_run_id: str | None = None, permutations: int = 0) -> pl.DataFrame:
    """D6's within-study case − control gap on the bulk run's predictions."""
    from rna_age.train.health import evaluate_health

    if bulk_run_id is None:
        bulk_run_id = json.loads((_run_dir(run_id) / "eval_bulk.json").read_text())["bulk_run_id"]
    return evaluate_health(bulk_run_id, permutations=permutations)


def _paired_ci(diff: np.ndarray, groups: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    from rna_age.train.fm_embed import paired_bootstrap

    _, lo, hi = paired_bootstrap(diff, np.zeros_like(diff), groups, n_boot, seed)
    return lo, hi


def eval_perturb(
    run_id: str,
    model: ExprICL | None = None,
    n_readouts: int = 100,
    device: str = "auto",
    seed: int = 0,
    cache: Path = CACHE,
    out_dir: Path | None = None,
) -> pl.DataFrame:
    """Per donor × condition and readout gene: context = training knockdowns, query = held-out knockdowns,
    features = the descriptor (own baseline expression, hashed id). Scored per held-out knockdown row as the
    Pearson over readouts of predicted vs observed change; the mean-change baseline is scored the same way."""
    dev = _device(device)
    model = (model or load_trained(run_id)).to(dev)
    c = load_corpus(cache, parts=("screen",))
    if not c.pert.height:
        raise ValueError("no screen table prepared")
    D = descriptors(c.pert)
    gid = c.pert["perturbed_gene_id"].to_numpy().astype(str)
    held = np.array([is_heldout_gene(g) for g in gid])
    key = (c.pert["donor_id"] + "|" + c.pert["culture_condition"]).to_numpy().astype(str)
    readouts = np.argsort(-c.pert_X[~held].var(0))[:n_readouts]
    rows = []
    for k in np.unique(key):
        ctx, qry = np.flatnonzero((key == k) & ~held), np.flatnonzero((key == k) & held)
        if len(ctx) < 10 or not len(qry):
            continue
        P = np.zeros((len(qry), len(readouts)))
        for j, g in enumerate(readouts):
            e = EpisodeSampler.episode(D[ctx], c.pert_X[ctx, g], D[qry], c.pert_X[qry, g], "perturbation", "pert")
            P[:, j] = predict_episode(model, e, dev)
        obs = c.pert_X[np.ix_(qry, readouts)]
        base = c.pert_X[np.ix_(ctx, readouts)].mean(0)
        for i, r in enumerate(qry):
            o = obs[i]
            rm = np.corrcoef(P[i], o)[0, 1] if np.std(P[i]) > 0 and np.std(o) > 0 else np.nan
            rb = np.corrcoef(base, o)[0, 1] if np.std(o) > 0 else np.nan
            rows.append({"donor_condition": k, "perturbed_gene_id": gid[r], "r_model": float(rm), "r_mean": float(rb),
                         "mae_model": float(np.abs(P[i] - o).mean()), "mae_mean": float(np.abs(base - o).mean())})  # fmt: skip
        log.info("%s: %d held-out knockdowns, %d context", k, len(qry), len(ctx))
    per = pl.DataFrame(rows)
    ok = per.drop_nans(["r_model", "r_mean"])
    genes = ok["perturbed_gene_id"].to_numpy().astype(str)
    summary = {"knockdowns": ok.height, "genes": int(len(np.unique(genes))), "readouts": int(len(readouts))}
    for m in ("r", "mae"):
        diff = (ok[f"{m}_model"] - ok[f"{m}_mean"]).to_numpy()
        lo, hi = _paired_ci(diff, np.unique(genes, return_inverse=True)[1], seed=seed) if len(diff) else (np.nan,) * 2
        summary |= {f"{m}_model": float(ok[f"{m}_model"].mean()), f"{m}_mean_baseline": float(ok[f"{m}_mean"].mean()),
                    f"{m}_diff": float(diff.mean()) if len(diff) else float("nan"), f"{m}_lo": lo, f"{m}_hi": hi}  # fmt: skip
    out = pl.DataFrame([summary])
    d = out_dir or _run_dir(run_id)
    d.mkdir(parents=True, exist_ok=True)
    per.write_parquet(d / "eval_perturb_rows.parquet")
    out.write_parquet(d / "eval_perturb.parquet")
    return out
