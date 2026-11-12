"""Tabular foundation models (in-context learners) as age regressors, on the same samples, splits and features as
the baselines.

- Features: ``hvg-pca`` = log-CPM of the train-chosen HVGs (as the MLP), standardised, PCA fitted on train rows
  (these models are pretrained on ≤ a few hundred features: TabPFN-3 uses at most 200 per estimator, TabICLv2 was
  pretrained on ≤ 100); ``atlas`` = BulkFormer-Atlas CLS (768-d), PCA'd the same way.
- Context: the "training set" of an in-context model is its prompt. Each study contributes at most
  ``cap_per_study`` samples (big ARCHS4 series do not dominate, as in fine-tuning), up to ``n_context`` rows.
  Healthy-or-unknown train samples with exact ages, plus GTEx at bin midpoints (as the baselines).
- No fitting of weights and no tuning on val: one configuration per run; val/test/transfer are scored as usual.

The model packages are not project dependencies: jobs run with ``uv run --with <package>`` so the project
environment (and jobs running from it) stays untouched.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import polars as pl

from rna_age.paths import INTERIM
from rna_age.train.baselines import PRED_COLS, hvg_design
from rna_age.train.runs import finish_run, new_run

log = logging.getLogger(__name__)

MODELS = ("tabicl", "tabfm", "geneicl", "geneicl-fast", "mitra2", "limix2", "tabpfn", "ridge")


def _estimator(model: str, seed: int):
    if model == "tabicl":
        from tabicl import TabICLRegressor

        return TabICLRegressor(random_state=seed, device="cuda", batch_size=2)
    if model == "tabfm":  # TabFM 1.1 weights (Hub) through the 1.0.1 package's loader (same module, decoder config)
        from huggingface_hub import snapshot_download
        from tabfm import TabFMRegressor
        from tabfm.src.pytorch import tabfm_v1_0_0 as tabfm_pt

        ckpt = snapshot_download("google/tabfm-1.1.0-pytorch", allow_patterns=["regression/*", "config.json"])
        net = tabfm_pt.load(model_type="regression", checkpoint_path=f"{ckpt}/regression", device="cuda")
        return TabFMRegressor(model=net, random_state=seed)
    if model.startswith("geneicl"):  # transcriptome in-context model: log1p-CPM genes in, its own PCA inside
        from geneicl import GeneICL

        return GeneICL(mode="regression", inference="fast" if model.endswith("fast") else "normal", device="cuda")
    if model == "mitra2":  # Mitra-v2: 50-step fine-tune, 8-fold bagged (AutoGluon); every predict re-runs the bag
        return _Mitra(seed)
    if model == "limix2":
        return _LimiX()
    if model == "tabpfn":
        from tabpfn import TabPFNRegressor

        return TabPFNRegressor(device="cuda", random_state=seed, ignore_pretraining_limits=True)
    if model == "ridge":  # control: same features and context rows, linear readout
        from sklearn.linear_model import RidgeCV

        return RidgeCV(alphas=np.logspace(-2, 4, 13))
    raise ValueError(f"unknown model {model!r}; expected one of {MODELS}")


class _Mitra:
    """Mitra-v2 regressor (``autogluon/mitra-regressor-2``) behind the fit/predict interface of the others."""

    def __init__(self, seed: int):
        from huggingface_hub import snapshot_download
        from mitra_finetune import MitraFinetune

        self.m = MitraFinetune(checkpoint_dir=snapshot_download("autogluon/mitra-regressor-2"),
                               problem_type="regression", random_state=seed)  # fmt: skip

    @staticmethod
    def _df(X: np.ndarray):
        import pandas as pd

        return pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])

    def fit(self, X: np.ndarray, y: np.ndarray):
        import pandas as pd

        self.m.fit(self._df(X), pd.Series(y, name="age"))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.m.predict(self._df(X)), dtype=np.float64)


class _LimiX:
    """LimiX-2 (``stable-ai/LimiX-2``, inference code from the cloned repo) behind fit/predict."""

    def __init__(self):
        import os

        import torch
        from huggingface_hub import hf_hub_download
        from inference.predictor import LimiXPredictor

        for k, v in {"RANK": "0", "WORLD_SIZE": "1", "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29511"}.items():
            os.environ.setdefault(k, v)
        repo = INTERIM.parent / "download" / "tfm" / "LimiX"
        self.m = LimiXPredictor(device=torch.device("cuda"),
                                model_path=hf_hub_download("stable-ai/LimiX-2", "LimiX-2.ckpt"),
                                inference_config=str(repo / "config" / "reg_default_noretrieval_v2.json"))  # fmt: skip

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X, self.y = X, y
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.m.predict(self.X, self.y, X, task_type="Regression"), dtype=np.float64).ravel()


def context_rows(allx: pl.DataFrame, is_train: np.ndarray, cap_per_study: int, n_context: int, seed: int) -> np.ndarray:
    """Indices of the prompt: train rows, at most ``cap_per_study`` per study, at most ``n_context`` in all."""
    idx = pl.DataFrame({"i": np.flatnonzero(is_train), "study": allx["study_id"].to_numpy()[is_train]})
    capped = (
        idx.sample(fraction=1.0, shuffle=True, seed=seed).group_by("study", maintain_order=True).head(cap_per_study)
    )
    rows = capped["i"].to_numpy()
    if len(rows) > n_context:
        rows = np.random.default_rng(seed).choice(rows, n_context, replace=False)
    return np.sort(rows)


def study_centroids(Z: np.ndarray, study: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(study ids, mean of ``Z`` per study) over ``rows``."""
    ids, inv = np.unique(study[rows].astype(str), return_inverse=True)
    C = np.zeros((len(ids), Z.shape[1]))
    np.add.at(C, inv, Z[rows])
    return ids, C / np.bincount(inv)[:, None]


def nearest_study_context(
    Z: np.ndarray, study: np.ndarray, pool: np.ndarray, query: np.ndarray, n_context: int, cap_per_study: int, seed: int
) -> np.ndarray:
    """RM25: the prompt for one query study: training studies in order of distance between their mean profile and the
    query's (in ``Z``), at most ``cap_per_study`` samples each, until ``n_context`` rows."""
    ids, C = study_centroids(Z, study, pool)
    q = Z[query].mean(0)
    order = np.argsort(((C - q) ** 2).sum(1))
    rng = np.random.default_rng(seed)
    pool_study = study[pool].astype(str)
    rows: list[np.ndarray] = []
    n = 0
    for k in order:
        r = pool[pool_study == ids[k]]
        if len(r) > cap_per_study:
            r = rng.choice(r, cap_per_study, replace=False)
        rows.append(r)
        n += len(r)
        if n >= n_context:
            break
    return np.sort(np.concatenate(rows)[:n_context])  # nearest first, so the cut drops the farthest study's tail


def _features(kind: str, allx: pl.DataFrame, X: np.ndarray, is_train: np.ndarray, k: int, seed: int):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    if kind == "atlas":
        from rna_age.train.embed import load_embeddings

        emb = load_embeddings("atlas")
        cols = [c for c in emb.columns if c.startswith("e") and c[1:].isdigit()]
        joined = allx.select("sample_id").join(emb, on="sample_id", how="left")
        X = joined.select(cols).to_numpy().astype(np.float32)
    elif kind == "logcpm":  # raw log1p-CPM of the HVGs, no scaling or PCA (GeneICL reduces itself)
        return X, np.ones(len(X), dtype=bool)
    elif kind == "hvg":  # the HVGs themselves, standardised on train (RM25: as TabICLv2 reads single-cell donors)
        have = ~np.isnan(X).any(axis=1)
        mu, sd = X[is_train & have].mean(0), X[is_train & have].std(0) + 1e-6
        return ((np.where(have[:, None], X, 0) - mu) / sd).astype(np.float32), have
    elif kind != "hvg-pca":
        raise ValueError(f"unknown features {kind!r}")
    have = ~np.isnan(X).any(axis=1)
    fit = is_train & have
    Xs = StandardScaler().fit(X[fit]).transform(np.where(have[:, None], X, 0)).astype(np.float32)
    pca = PCA(n_components=min(k, Xs.shape[1]), svd_solver="randomized", random_state=seed).fit(Xs[fit])
    log.info("%s: PCA %d components explain %.3f of train variance", kind, pca.n_components_,
             pca.explained_variance_ratio_.sum())  # fmt: skip
    return pca.transform(Xs).astype(np.float32), have


def run_tabfm(
    model: str = "tabicl",
    features: str = "hvg-pca",
    n_genes: int = 20000,
    n_components: int = 200,
    n_context: int = 10000,
    cap_per_study: int = 200,
    chunk: int = 2048,
    seed: int = 0,
    name: str | None = None,
    context: str = "random",
) -> dict:
    t0 = time.time()
    allx, X, is_train, _, genes, n_hv = hvg_design(n_genes)
    Z, have = _features(features, allx, X, is_train, n_components, seed)
    del X
    y = allx["target"].to_numpy()
    ctx = context_rows(allx, is_train & have & ~np.isnan(y), cap_per_study, n_context, seed)
    n_studies = len(np.unique(allx["study_id"].to_numpy()[ctx].astype(str)))
    log.info("%s on %s: context %d rows from %d studies", model, features, len(ctx), n_studies)
    keep = have & (allx["split"].to_numpy() != "train")  # score val/test/transfer only
    out = np.full(allx.height, np.nan)
    rows = np.flatnonzero(keep)
    if context == "nearest-study":  # one prompt per query study, from the training studies most like it
        from sklearn.decomposition import PCA

        pool = np.flatnonzero(is_train & have & ~np.isnan(y))
        Zc = PCA(50, svd_solver="randomized", random_state=seed).fit(Z[pool]).transform(Z)
        study = allx["study_id"].to_numpy().astype(str)
        groups = {s: rows[study[rows] == s] for s in np.unique(study[rows])}
        for j, (s, q) in enumerate(groups.items()):
            c = nearest_study_context(Zc, study, pool, q, n_context, cap_per_study, seed)
            est = _estimator(model, seed).fit(Z[c], y[c])
            for i in range(0, len(q), chunk):
                out[q[i : i + chunk]] = est.predict(Z[q[i : i + chunk]])
            if j % 20 == 0:
                log.info("study %d / %d (%s, %d queries, context %d)", j + 1, len(groups), s, len(q), len(c))
    else:
        if context != "random":
            raise ValueError(f"unknown context {context!r}")
        est = _estimator(model, seed).fit(Z[ctx], y[ctx])
        if model == "mitra2":
            chunk = len(rows)  # one predict = one bagged fine-tune
        for i in range(0, len(rows), chunk):
            part = rows[i : i + chunk]
            out[part] = est.predict(Z[part])
            log.info("predicted %d / %d", min(i + chunk, len(rows)), len(rows))
    pred = (
        allx.select([c for c in PRED_COLS if c in allx.columns])
        .with_columns(pred=pl.Series(out))
        .filter(pl.Series(keep))
    )
    run_dir = new_run(
        name or f"tfm-{model}-{features}{'' if features in ('logcpm', 'hvg') else n_components}-ctx{n_context // 1000}k"
        f"{'-near' if context == 'nearest-study' else ''}",
        {"model": model, "features": f"{features} ({n_components} PCs of logCPM HVG {n_genes})"
         if features == "hvg-pca" else f"logCPM HVG {n_genes}" if features in ("logcpm", "hvg") else f"{features} ({n_components} PCs)",
         "n_context": int(len(ctx)), "n_context_studies": n_studies, "cap_per_study": cap_per_study, "seed": seed, "hvg_sample": n_hv,
         "n_shared_genes": len(genes), "context_set": "train, healthy-or-unknown, exact + GTEx midpoints",
         "context": context,
         "tuning": "none (zero-shot in-context)"},
    )  # fmt: skip
    row = finish_run(run_dir, pred, {"seconds": time.time() - t0})
    log.info("run %s: %s", run_dir.name, {k: v for k, v in row.items() if k.startswith(("val_", "test_"))})
    return row
