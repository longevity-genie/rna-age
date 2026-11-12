"""RM20: a multitask network (donor age + knockdown Δ, optional age-shift ranking) on the CD4 T Perturb-seq panel.

Splits: target genes ~80 / 20 % (never on both sides) and one donor × condition context held out of Δ training
altogether; donor age by 5-fold CV grouped by person. Each arm trains one network per age fold; Δ̂ and shifts on the
held-out genes are averaged over the five. Arms: ``age`` (Δ head off), ``age_delta``, ``age_delta_rank``, ``placebo``
(the rank term on coefficients shuffled across genes within condition). References: the mean Δ of training knockdowns
(global, per context), ridge on one-hot(donor) + one-hot(condition), the ridge clock's own shifts (RM17) on the same
held-out genes, and ridge age CV on the same panel.
Output: one row per arm × metric × scope (``data/output/summary/rm20_multitask_cd4.parquet``).
"""

from __future__ import annotations

import logging
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl

from rna_age.train.multitask_data import Rm20Data
from rna_age.train.multitask_eval import delta_metrics, ridge_donor_condition, shift_rho
from rna_age.train.multitask_model import Config, Task, predict, train

log = logging.getLogger(__name__)

ARMS = ("age", "age_delta", "age_delta_rank", "placebo")


def coef_table(raw: pl.DataFrame) -> pl.DataFrame:
    return raw.filter(pl.col("celltype") != "K562").select(
        condition=pl.col("celltype"), gene=pl.col("regulator"), coef_mean=pl.col("coef_mean")
    )


def shuffle_coef(coef: pl.DataFrame, seed: int) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    parts = [d.with_columns(coef_mean=pl.Series(rng.permutation(d["coef_mean"].to_numpy())))
             for _, d in coef.group_by("condition", maintain_order=True)]  # fmt: skip
    return pl.concat(parts)


def split(data: Rm20Data, test_frac: float, holdout_context: str | None, seed: int):
    rng = np.random.default_rng(seed)
    genes = np.array(sorted(data.units["gene"].unique().to_list()))
    held = set(rng.choice(genes, int(round(test_frac * len(genes))), replace=False).tolist())
    hold = holdout_context or data.contexts[int(rng.integers(len(data.contexts)))]
    is_held = data.units["gene"].is_in(list(held)).to_numpy()
    in_hold = (data.units["context"] == hold).to_numpy()
    return {"train": ~is_held & ~in_hold, "eval": is_held, "heldout_ctx": in_hold, "hold": hold, "n_held": len(held)}


def _rows(arm: str, scope: str, metrics: dict[str, float], n: int) -> list[dict]:
    return [{"arm": arm, "metric": k, "scope": scope, "value": float(v), "n": n} for k, v in metrics.items()]


def run(
    data: Rm20Data,
    coef_raw: pl.DataFrame,
    cfg: Config,
    arms: tuple[str, ...] = ARMS,
    w_rank: float = 0.5,
    test_frac: float = 0.2,
    holdout_context: str | None = None,
    folds: int = 5,
    seed: int = 0,
    device: str = "auto",
    ridge_clock_genes: pl.DataFrame | None = None,
) -> pl.DataFrame:
    import torch
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import GroupKFold

    dev = torch.device(
        "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
    )
    sp = split(data, test_frac, holdout_context, seed)
    u = data.units
    tr, ev = sp["train"], sp["eval"]
    log.info("split: %d train units, %d eval units (%d held-out genes), held-out context %s, device %s",
             tr.sum(), ev.sum(), sp["n_held"], sp["hold"], dev)  # fmt: skip
    ctx = u["ctx"].to_numpy()
    nctx = len(data.contexts)
    glob = data.delta[tr].mean(0)
    mean_ctx = np.tile(glob, (nctx, 1))
    for c in range(nctx):
        m = tr & (ctx == c)
        if m.any():
            mean_ctx[c] = data.delta[m].mean(0)
    feat = np.stack([data.gene_feat[g] for g in u["gene"].to_list()])
    coef = coef_table(coef_raw)
    conds = sorted(u["condition"].unique().to_list())
    cond_ix = np.array([conds.index(c) for c in u["condition"].to_list()])

    def coef_of(cf: pl.DataFrame) -> np.ndarray:
        j = u.select("gene", "condition").join(cf, on=["gene", "condition"], how="left", maintain_order="left")
        return j["coef_mean"].fill_null(np.nan).to_numpy().astype(np.float32)

    coef_true, coef_placebo = coef_of(coef), coef_of(shuffle_coef(coef, seed + 1))
    T = lambda a: torch.as_tensor(np.array(a), device=dev)  # noqa: E731
    controls, ctx_t, feat_t, delta_t = T(data.controls), T(ctx), T(feat), T(data.delta)
    mean_t, cond_t = T(mean_ctx.astype(np.float32)), T(cond_ix)
    ev_ix = np.flatnonzero(ev)
    tr_ix = T(np.flatnonzero(tr))
    ev_t = T(ev_ix)
    ref_ev = mean_ctx[ctx[ev_ix]]
    true_ev = data.delta[ev_ix]
    seen = ~sp["heldout_ctx"][ev_ix]
    strength = np.linalg.norm(true_ev - ref_ev, axis=1)
    strong = seen & (strength >= np.quantile(strength[seen], 0.9)) if seen.any() else seen
    scopes = {"seen_ctx": seen, "heldout_ctx": ~seen, "seen_ctx_strong": strong}
    ev_units = u[ev_ix]
    rows: list[dict] = [{"arm": "_split", "metric": "heldout_context:" + sp["hold"], "scope": "info", "value": 0.0,
                         "n": int(sp["n_held"])}]  # fmt: skip

    def delta_rows(arm: str, pred: np.ndarray):
        for s, m in scopes.items():
            if m.any():
                rows.extend(_rows(arm, s, delta_metrics(pred[m], true_ev[m], ref_ev[m]), int(m.sum())))

    # References for Δ
    delta_rows("mean_delta_global", np.tile(glob, (len(ev_ix), 1)))
    delta_rows("mean_delta_context", ref_ev)
    delta_rows("ridge_donor_condition", ridge_donor_condition(u.filter(pl.Series(tr)), data.delta[tr], ev_units))
    # References for age and shift
    folds_ix = list(GroupKFold(n_splits=folds).split(data.X_age, data.age, data.groups))
    oof = np.zeros_like(data.age)
    for a, b in folds_ix:
        oof[b] = RidgeCV(alphas=np.logspace(-1, 5, 13)).fit(data.X_age[a], data.age[a]).predict(data.X_age[b])
    rows.extend(_rows("ridge_age_panel", "cv", {"age_mae": np.abs(oof - data.age).mean(),
                                                "age_r": np.corrcoef(oof, data.age)[0, 1]}, len(oof)))  # fmt: skip
    held_genes = set(ev_units["gene"].to_list())
    if ridge_clock_genes is not None:
        rc = ridge_clock_genes.filter(pl.col("perturbed_gene_name").is_in(list(held_genes)))
        rc = rc.select(gene="perturbed_gene_name", condition="culture_condition", shift="shift")
        cf = coef.join(rc, on=["gene", "condition"])
        rows.extend(_rows("ridge_clock_rm17", "heldout_genes", shift_rho(cf.select("gene", "condition"),
                          cf["shift"].to_numpy(), coef), cf.height))  # fmt: skip

    for arm in arms:
        acfg = replace(cfg, w_delta=0.0 if arm == "age" else cfg.w_delta,
                       w_rank=w_rank if arm in ("age_delta_rank", "placebo") else 0.0)  # fmt: skip
        cf = T(coef_placebo if arm == "placebo" else coef_true)
        with_delta = acfg.w_delta > 0
        oof = np.zeros_like(data.age)
        acc: dict[str, np.ndarray] = {}
        t0 = time.time()
        for f, (a, b) in enumerate(folds_ix):
            mu, sd = data.age[a].mean(), data.age[a].std()
            task = Task(
                X_age=T(data.X_age[a]), y_age=T(((data.age[a] - mu) / sd).astype(np.float32)), controls=controls,
                ctx=ctx_t[tr_ix], feat=feat_t[tr_ix], delta=delta_t[tr_ix], mean_delta=mean_t, coef=cf[tr_ix],
                cond=cond_t[tr_ix],
            )  # fmt: skip
            net = train(task, replace(acfg, seed=cfg.seed + f), dev)
            p = predict(net, T(data.X_age[b]), controls, ctx_t[ev_t], feat_t[ev_t], delta_t[ev_t], with_delta)
            oof[b] = p.pop("age") * sd + mu
            for k, v in p.items():
                v = v * sd if k.startswith("shift") else v
                acc[k] = acc.get(k, 0) + v / len(folds_ix)
            log.info("%s fold %d done (%.0f s so far)", arm, f, time.time() - t0)
            del net, task
        rows.extend(_rows(arm, "cv", {"age_mae": np.abs(oof - data.age).mean(),
                                      "age_r": np.corrcoef(oof, data.age)[0, 1]}, len(oof)))  # fmt: skip
        rows.extend(_rows(arm, "heldout_genes_measured", shift_rho(ev_units, acc["shift_meas"], coef), len(ev_ix)))
        if with_delta:
            delta_rows(arm, acc["dhat"])
            rows.extend(_rows(arm, "heldout_genes", shift_rho(ev_units, acc["shift_pred"], coef), len(ev_ix)))
        rows.extend(_rows(arm, "info", {"seconds": time.time() - t0, "steps": acfg.steps}, folds))
    return pl.DataFrame(rows)


def write(df: pl.DataFrame, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out, compression="zstd")
    return out
