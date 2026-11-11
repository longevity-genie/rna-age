"""Rung 3 probes on frozen embeddings: ridge and a small MLP on Atlas CLS, alone or with HVG log-CPM
("+HVG", the PRIMO recipe). Same training set, splits, model selection (val exact MAE) and run tracking as
the baselines, so the numbers are directly comparable.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import polars as pl
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from rna_age.schema import age_mid
from rna_age.train.baselines import PRED_COLS
from rna_age.train.embed import load_embeddings
from rna_age.train.features import load_matrix, log_cpm, sample_table, select_hvg, shared_genes
from rna_age.train.runs import finish_run, new_run

log = logging.getLogger(__name__)


def _mlp(
    X,
    y,
    train,
    val,
    seed=0,
    epochs=200,
    delta=5.0,
    hidden=(512, 128),
    dropout=0.2,
    exact=None,
    lo=None,
    hi=None,
    weights=None,
    val_y=None,
    inv=None,
    X_extra=None,
    extra_out=None,
):
    """MLP regressor with Huber loss on exact targets and, where ``exact`` is False, an interval hinge on
    [lo, hi) (all in the target's space); optional per-sample weights; early stopping on val MAE in years."""
    import torch

    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xt = torch.as_tensor(X, device=dev)
    yt = torch.as_tensor(np.nan_to_num(y), dtype=torch.float32, device=dev)
    n = len(y)
    ex = torch.as_tensor(np.ones(n, bool) if exact is None else exact, device=dev)
    lot = torch.as_tensor(np.zeros(n) if lo is None else np.nan_to_num(lo), dtype=torch.float32, device=dev)
    hit = torch.as_tensor(np.zeros(n) if hi is None else np.nan_to_num(hi), dtype=torch.float32, device=dev)
    wt = torch.as_tensor(np.ones(n) if weights is None else weights, dtype=torch.float32, device=dev)
    layers, d = [], X.shape[1]
    for h in hidden:
        layers += [torch.nn.Linear(d, h), torch.nn.GELU(), torch.nn.Dropout(dropout)]
        d = h
    net = torch.nn.Sequential(*layers, torch.nn.Linear(d, 1)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    tr, va = np.flatnonzero(train), np.flatnonzero(val)
    vy = torch.as_tensor(val_y if val_y is not None else y[va], dtype=torch.float32, device=dev)
    inv = inv or (lambda a: a)
    best, best_state, bad = np.inf, None, 0
    for epoch in range(epochs):
        net.train()
        for b in np.array_split(np.random.default_rng(seed * 1000 + epoch).permutation(tr), max(1, len(tr) // 512)):
            pred = net(Xt[b]).squeeze(-1)
            exact_l = torch.nn.functional.huber_loss(pred, yt[b], delta=delta, reduction="none")
            interval_l = torch.relu(lot[b] - pred) + torch.relu(pred - hit[b])
            loss = (torch.where(ex[b], exact_l, interval_l) * wt[b]).sum() / wt[b].sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            pv = net(Xt[va]).squeeze(-1).float().cpu().numpy()
        mae = float(np.mean(np.abs(inv(pv) - vy.cpu().numpy())))
        if mae < best - 0.01:
            best, best_state, bad = mae, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 15:
                break
    net.load_state_dict(best_state)
    with torch.no_grad():
        if X_extra is not None and extra_out is not None:  # samples outside the table (train/external.py)
            extra_out.append(net(torch.as_tensor(X_extra, device=dev)).squeeze(-1).float().cpu().numpy())
        return net(Xt).squeeze(-1).float().cpu().numpy(), best


def run_probe(
    features: str = "cls",
    model: str = "ridge",
    n_hvg: int = 1200,
    gtex_midpoints: bool = True,
    transform: str = "none",
    seed: int = 0,
    interval: bool = False,
    study_weights: bool = False,
    hidden: tuple[int, ...] = (512, 128),
    dropout: float = 0.2,
    train_set: str = "healthy",
    restrict_to: str = "",
    external: bool = False,
) -> dict:
    from rna_age.train.baselines import horvath, horvath_inv

    t0 = time.time()
    s = sample_table()
    s = s.with_columns(target=age_mid(s))
    emb = load_embeddings("latest" if features.startswith("latest") else "atlas")
    if restrict_to == "latest" and not features.startswith("latest"):  # same samples as the Latest runs
        s = s.join(load_embeddings("latest").select("sample_id"), on="sample_id", how="inner")
    s = s.join(emb, on="sample_id", how="inner")
    s = s.with_columns(split=pl.when(pl.col("is_pseudobulk")).then(pl.lit("transfer")).otherwise(pl.col("split")))
    prefix = {"latest": "s", "latestpanel": "p"}.get(features, "e")
    ecols = [c for c in emb.columns if c.startswith(prefix) and c[1:].isdigit()]
    X = s.select(ecols).to_numpy().astype(np.float32)
    if features == "panel":  # the Atlas input panel itself (log2 TPM+1), no model: separates panel from encoder
        from rna_age.train.finetune import load_inputs

        ids, xin = load_inputs()
        pos = {sid: i for i, sid in enumerate(ids)}
        X = xin[[pos[sid] for sid in s["sample_id"].to_list()]].astype(np.float32)
    train = (
        (s["split"] == "train").to_numpy()
        & ((s["is_healthy"].fill_null(True)).to_numpy() | (train_set == "all"))
        & ((s["age_type"] == "exact").to_numpy() | (gtex_midpoints & (s["source"] == "gtex").to_numpy()))
    )
    if features in ("cls+hvg", "hvg", "tpm"):
        genes = shared_genes()
        if features == "tpm":  # BulkFormer-style gene-length TPM on genes with a released length
            from rna_age.train.features import gene_lengths, log_tpm

            lens = gene_lengths(genes)
            genes = [g for g, ln in zip(genes, lens, strict=True) if ln == ln and ln > 0]
            lens = lens[np.isfinite(lens) & (lens > 0)]
        counts = load_matrix(s.select("sample_id", "source"), genes)
        lc = log_tpm(counts, lens, umi_rows=s["is_pseudobulk"].to_numpy()) if features == "tpm" else log_cpm(counts)
        del counts
        hvg = select_hvg(lc[train][: min(20000, int(train.sum()))], genes, n_hvg)
        gi = [genes.index(g) for g in hvg]
        if external and features == "hvg":
            from rna_age.train.external import log_cpm_matrix

            ext_ids, ext_lc = log_cpm_matrix(genes)
            X_ext = ext_lc[:, gi]
            del ext_lc
        X = lc[:, gi] if features in ("hvg", "tpm") else np.hstack([X, lc[:, gi]])
        del lc
    y_years = s["target"].to_numpy()
    fwd, inv = (horvath, horvath_inv) if transform == "horvath" else (lambda a: a, lambda a: a)
    y = fwd(np.nan_to_num(y_years, nan=30.0))
    val = (s["split"] == "val").to_numpy() & (s["age_type"] == "exact").to_numpy()
    scaler = StandardScaler().fit(X[train])
    Xs = scaler.transform(X).astype(np.float32)
    ext_out: list = []
    if external and features != "hvg":
        raise ValueError("external scoring needs --features hvg")
    if model == "ridge":
        best = None
        for a in (1, 10, 100, 1e3, 1e4, 1e5, 1e6):
            est = Ridge(alpha=a).fit(Xs[train], y[train])
            mae = float(np.mean(np.abs(inv(est.predict(Xs[val])) - y_years[val])))
            log.info("ridge alpha=%g val MAE %.2f", a, mae)
            if best is None or mae < best[0]:
                best = (mae, a, est)
        pred, params = inv(best[2].predict(Xs)), {"alpha": best[1]}
    else:
        exact = (s["age_type"] == "exact").to_numpy()
        if interval:  # GTEx / other bins: hinge on [min, max+1) instead of the midpoint target
            train = train | ((s["split"] == "train").to_numpy() & (s["age_type"] == "binned").to_numpy())
        lo = fwd(np.nan_to_num(s["age_min"].to_numpy(), nan=0.0))
        hi = fwd(np.nan_to_num(s["age_max"].to_numpy(), nan=0.0) + 1)
        w = None
        if study_weights:  # each study counts ~sqrt(n) instead of n
            n_study = s.group_by("study_id").len()
            cnt = s.join(n_study, on="study_id", how="left")["len"].to_numpy().astype(float)
            w = 1 / np.sqrt(cnt)
        pred, _ = _mlp(
            Xs, y, train, val, seed=seed, delta=0.25 if transform == "horvath" else 5.0, hidden=hidden,
            dropout=dropout, exact=exact if interval else None, lo=lo, hi=hi, weights=w, val_y=y_years[val], inv=inv,
            X_extra=scaler.transform(X_ext).astype(np.float32) if external else None, extra_out=ext_out,
        )  # fmt: skip
        pred = inv(pred)
        params = {"val_mae": float(np.mean(np.abs(pred[val] - y_years[val]))), "seed": seed}
    out = s.select([c for c in PRED_COLS if c in s.columns]).with_columns(pred=pl.Series(pred.astype(np.float64)))
    run_dir = new_run(
        (f"{'tpm' if features == 'tpm' else ''}hvg{n_hvg}-{model}" if features in ("hvg", "tpm") else f"{'' if features.startswith('latest') else 'atlas-'}{features.replace('+', '-')}-{model}")
        + ("-horvath" if transform == "horvath" else "") + ("-interval" if interval else "")
        + ("-studyw" if study_weights else "") + ("" if hidden == (512, 128) else "-h" + "x".join(map(str, hidden)))
        + ("-trainall" if train_set == "all" else "") + ("-onlatest" if restrict_to == "latest" else "")
        + (f"-seed{seed}" if seed else ""),
        {"model": f"{model} probe", "features": "Latest pooled sample embedding" if features == "latest" else "Latest gene embeddings pooled over panel" if features == "latestpanel" else "Atlas panel log2TPM (no encoder)" if features == "panel" else f"logCPM HVG {n_hvg}" if features == "hvg" else f"log2 TPM HVG {n_hvg}" if features == "tpm" else f"Atlas CLS{' + HVG ' + str(n_hvg) if features == 'cls+hvg' else ''}",
         "params": params, "target_transform": transform, "interval_loss": interval, "study_weights": study_weights,
         "hidden": list(hidden), "dropout": dropout, "train_set": ("bulk train, all health states" if train_set == "all" else "bulk train, healthy-or-unknown") + ", exact" + (" + GTEx midpoints" if gtex_midpoints else ""),
         "n_train": int(train.sum())},
    )  # fmt: skip
    if ext_out:
        pl.DataFrame({"sample_id": ext_ids, "pred": inv(ext_out[0]).astype(np.float64)}).write_parquet(
            run_dir / "predictions_external.parquet"
        )
    row = finish_run(run_dir, out, {"seconds": time.time() - t0})
    log.info("run %s: %s", run_dir.name, {k: v for k, v in row.items() if k.startswith("test_")})
    return row
