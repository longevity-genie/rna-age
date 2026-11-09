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


def _mlp(X, y, train, val, seed=0, epochs=200):
    import torch

    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xt = torch.as_tensor(X, device=dev)
    yt = torch.as_tensor(y, dtype=torch.float32, device=dev)
    net = torch.nn.Sequential(
        torch.nn.Linear(X.shape[1], 512), torch.nn.GELU(), torch.nn.Dropout(0.2),
        torch.nn.Linear(512, 128), torch.nn.GELU(), torch.nn.Linear(128, 1),
    ).to(dev)  # fmt: skip
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    tr, va = np.flatnonzero(train), np.flatnonzero(val)
    best, best_state, bad = np.inf, None, 0
    for _ in range(epochs):
        net.train()
        for b in np.array_split(np.random.default_rng(_).permutation(tr), max(1, len(tr) // 512)):
            loss = torch.nn.functional.huber_loss(net(Xt[b]).squeeze(-1), yt[b], delta=5.0)
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            mae = (net(Xt[va]).squeeze(-1) - yt[va]).abs().mean().item()
        if mae < best - 0.01:
            best, best_state, bad = mae, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 15:
                break
    net.load_state_dict(best_state)
    with torch.no_grad():
        return net(Xt).squeeze(-1).float().cpu().numpy(), best


def run_probe(features: str = "cls", model: str = "ridge", n_hvg: int = 1200, gtex_midpoints: bool = True) -> dict:
    t0 = time.time()
    s = sample_table()
    s = s.with_columns(target=age_mid(s))
    emb = load_embeddings("atlas")
    s = s.join(emb, on="sample_id", how="inner")
    s = s.with_columns(split=pl.when(pl.col("is_pseudobulk")).then(pl.lit("transfer")).otherwise(pl.col("split")))
    ecols = [c for c in emb.columns if c.startswith("e")]
    X = s.select(ecols).to_numpy().astype(np.float32)
    train = (
        (s["split"] == "train").to_numpy()
        & (s["is_healthy"].fill_null(True)).to_numpy()
        & ((s["age_type"] == "exact").to_numpy() | (gtex_midpoints & (s["source"] == "gtex").to_numpy()))
    )
    if features == "cls+hvg":
        genes = shared_genes()
        counts = load_matrix(s.select("sample_id", "source"), genes)
        lc = log_cpm(counts)
        del counts
        hvg = select_hvg(lc[train][: min(20000, int(train.sum()))], genes, n_hvg)
        gi = [genes.index(g) for g in hvg]
        X = np.hstack([X, lc[:, gi]])
        del lc
    y = s["target"].to_numpy()
    val = (s["split"] == "val").to_numpy() & (s["age_type"] == "exact").to_numpy()
    Xs = StandardScaler().fit(X[train]).transform(X).astype(np.float32)
    if model == "ridge":
        best = None
        for a in (1, 10, 100, 1e3, 1e4, 1e5, 1e6):
            est = Ridge(alpha=a).fit(Xs[train], y[train])
            mae = float(np.mean(np.abs(est.predict(Xs[val]) - y[val])))
            log.info("ridge alpha=%g val MAE %.2f", a, mae)
            if best is None or mae < best[0]:
                best = (mae, a, est)
        pred, params = best[2].predict(Xs), {"alpha": best[1]}
    else:
        pred, val_mae = _mlp(Xs, y, train, val)
        params = {"val_mae": val_mae}
    out = s.select([c for c in PRED_COLS if c in s.columns]).with_columns(pred=pl.Series(pred.astype(np.float64)))
    run_dir = new_run(
        f"atlas-{features.replace('+', '-')}-{model}",
        {"model": f"{model} probe", "features": f"Atlas CLS{' + HVG ' + str(n_hvg) if features == 'cls+hvg' else ''}",
         "params": params, "train_set": "bulk train, healthy-or-unknown, exact" + (" + GTEx midpoints" if gtex_midpoints else ""),
         "n_train": int(train.sum())},
    )  # fmt: skip
    row = finish_run(run_dir, out, {"seconds": time.time() - t0})
    log.info("run %s: %s", run_dir.name, {k: v for k, v in row.items() if k.startswith("test_")})
    return row
