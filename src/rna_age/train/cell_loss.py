"""RM35 step 1: losses that make a cell-level age head learn age instead of shrinking to the mean.

RM32's per-cell clock asked every cell (or 15-cell pseudocell) to equal its donor's age; most units carry little age,
so the cheapest answer is the mean, and slope and r both fell. Here a unit head reads raw panel expression (no
pretrained encoder: RM28's encoder was itself 1.8 y behind summed expression) and the donor's age is pooled over its
units. Arms (``loss``):

- ``mse``: donor squared error only, units mean-pooled. The arm-free control of this code (no past run fits: RM32's
  reference read RM28's encoder, not raw expression).
- ``nll``: each unit also predicts a log-variance; the donor's age is the precision-weighted mean of its units, and a
  per-unit Gaussian NLL against the donor's age is added (weight ``w``), so units with little signal are allowed a
  wide variance instead of pulling toward the mean (heteroscedastic regression, Kendall & Gal 2017).
- ``ccc``: donor squared error + ``w`` × (1 − Lin's concordance) over the batch's donors, which penalises a slope
  below 1 as well as poor ranking.
- ``rank``: donor squared error + ``w`` × a pairwise logistic loss on every pair of the batch's donors ≥ 5 y apart.

Read-out on RM14's folds and donors (``fm_embed.clock_folds``), compared with the pseudobulk clocks by
``fm_embed.compare``; rows ``cell-<loss>[-w<w>]-pc<k>`` go to ``data/output/summary/sc_pretrain_{oof,clock}.parquet``
and the slope per cohort to ``cell_loss_slope.parquet``.
"""

from __future__ import annotations

import copy
import logging
import time

import numpy as np
import polars as pl
import torch
from torch import nn

from rna_age.train.sc_pretrain import (
    EVAL_COHORTS,
    RUNS,
    SUMMARY,
    _device,
    _inner_split,
    _merge_summary,
    cohort_cells,
    pseudocells,
)

log = logging.getLogger(__name__)

LOSSES = ("mse", "nll", "ccc", "rank")
RANK_GAP = 5.0  # years: pairs closer than this carry no order


class UnitHead(nn.Module):
    """Unit (cell or pseudocell) → (age, log-variance) in standardised units, from standardised panel expression."""

    def __init__(self, n_genes: int, hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        self.body = nn.Sequential(nn.Dropout(dropout), nn.Linear(n_genes, hidden), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, 64), nn.GELU())  # fmt: skip
        self.mu = nn.Linear(64, 1)
        self.logvar = nn.Linear(64, 1)
        nn.init.zeros_(self.logvar.weight)
        nn.init.zeros_(self.logvar.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.body(x)
        return self.mu(h).squeeze(-1), self.logvar(h).squeeze(-1).clamp(-6.0, 6.0)


def pool_units(mu: torch.Tensor, logvar: torch.Tensor, seg: torch.Tensor, n: int, weighted: bool) -> torch.Tensor:
    """Donor ages from unit ages; ``seg`` gives each unit's donor position 0..n-1. ``weighted``: precision-weighted
    mean (weights exp(−logvar)), else the plain mean."""
    w = torch.exp(-logvar) if weighted else torch.ones_like(mu)
    num = torch.zeros(n, device=mu.device, dtype=mu.dtype).index_add_(0, seg, w * mu)
    den = torch.zeros(n, device=mu.device, dtype=mu.dtype).index_add_(0, seg, w)
    return num / den


def ccc(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Lin's concordance correlation coefficient."""
    pm, ym = pred.mean(), y.mean()
    cov = ((pred - pm) * (y - ym)).mean()
    return 2 * cov / (pred.var(unbiased=False) + y.var(unbiased=False) + (pm - ym) ** 2 + 1e-8)


def rank_loss(pred: torch.Tensor, y: torch.Tensor, gap: float) -> torch.Tensor:
    """Mean logistic loss over pairs with ``y_i − y_j ≥ gap`` (standardised units) that ``pred`` should order."""
    d_y = y[:, None] - y[None, :]
    pairs = d_y >= gap
    if not pairs.any():
        return pred.sum() * 0.0
    d_p = (pred[:, None] - pred[None, :])[pairs]
    return nn.functional.softplus(-d_p).mean()


def batch_loss(head: UnitHead, x: torch.Tensor, seg: torch.Tensor, y: torch.Tensor, loss: str, w: float,
               rank_gap: float) -> torch.Tensor:  # fmt: skip
    """``x``: the batch's units; ``seg``: unit → donor position; ``y``: donor ages (standardised)."""
    mu, logvar = head(x)
    don = pool_units(mu, logvar, seg, len(y), weighted=loss == "nll")
    out = ((don - y) ** 2).mean()
    if loss == "nll" and w > 0:
        t = y[seg]
        out = out + w * (0.5 * (logvar + (mu - t) ** 2 * torch.exp(-logvar))).mean()
    elif loss == "ccc" and w > 0:
        out = out + w * (1 - ccc(don, y))
    elif loss == "rank" and w > 0:
        out = out + w * rank_loss(don, y, rank_gap)
    return out


def fit_fold(xu: np.ndarray, unit_donor: np.ndarray, train: list[str], age_tr: np.ndarray, val: list[str],
             age_val: np.ndarray, test: list[str], loss: str = "mse", w: float = 0.0, epochs: int = 60,
             lr: float = 1e-3, donors_per_batch: int = 64, seed: int = 0, device=None) -> np.ndarray:  # fmt: skip
    """Train a :class:`UnitHead` on ``train`` donors' units (genes standardised on them), keep the epoch with the
    lowest MAE on ``val`` donors, return predictions (years) for ``test`` donors."""
    if loss not in LOSSES:
        raise ValueError(f"loss must be one of {LOSSES}, got {loss!r}")
    dev = device or torch.device("cpu")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    tr_units = np.isin(unit_donor, train)
    m = xu[tr_units].astype(np.float32).mean(0)
    s = xu[tr_units].astype(np.float32).std(0) + 1e-3
    X = torch.from_numpy(((xu.astype(np.float32) - m) / s)).to(dev)
    units = {d: np.flatnonzero(unit_donor == d) for d in set(train) | set(val) | set(test)}
    mu_a, sd_a = float(age_tr.mean()), float(age_tr.std() + 1e-6)
    y_tr = (age_tr - mu_a) / sd_a
    head = UnitHead(X.shape[1]).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.01)

    def gather(donors: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        idx = [units[d] for d in donors]
        seg = np.concatenate([np.full(len(u), k) for k, u in enumerate(idx)])
        return torch.from_numpy(np.concatenate(idx)).to(dev), torch.from_numpy(seg).to(dev)

    @torch.no_grad()
    def predict(donors: list[str]) -> np.ndarray:
        head.eval()
        ui, seg = gather(donors)
        mu, logvar = head(X[ui])
        out = pool_units(mu, logvar, seg, len(donors), weighted=loss == "nll").cpu().numpy() * sd_a + mu_a
        head.train()
        return out

    best, best_state = np.inf, copy.deepcopy(head.state_dict())
    gap = RANK_GAP / sd_a
    for _ in range(epochs):
        for b in np.array_split(rng.permutation(len(train)), max(1, len(train) // donors_per_batch)):
            ui, seg = gather([train[i] for i in b])
            yb = torch.tensor(y_tr[b], dtype=torch.float32, device=dev)
            lv = batch_loss(head, X[ui], seg, yb, loss, w, gap)
            opt.zero_grad(set_to_none=True)
            lv.backward()
            opt.step()
        mae = float(np.abs(predict(val) - age_val).mean()) if val else 0.0
        if mae < best:
            best, best_state = mae, copy.deepcopy(head.state_dict())
    head.load_state_dict(best_state)
    return predict(test)


def model_name(loss: str, w: float, pseudocell: int) -> str:
    return f"cell-{loss}" + (f"-w{w:g}" if loss != "mse" else "") + (f"-pc{pseudocell}" if pseudocell > 1 else "-cells")


def slopes(oof: pl.DataFrame) -> pl.DataFrame:
    """OLS slope of predicted on true age and r, per cohort × representation × model."""
    rows = []
    for (c, rep, mdl), d in oof.group_by("cohort", "representation", "model", maintain_order=True):
        a, p = d["age"].to_numpy(), d["pred"].to_numpy()
        rows.append({"cohort": c, "representation": rep, "model": mdl, "n": len(a),
                     "slope": float(np.polyfit(a, p, 1)[0]), "r": float(np.corrcoef(a, p)[0, 1]),
                     "mae": float(np.abs(p - a).mean())})  # fmt: skip
    return pl.DataFrame(rows)


def run(loss: str = "mse", w: float = 0.0, cohorts: tuple[str, ...] = EVAL_COHORTS, pseudocell: int = 15,
        epochs: int = 60, lr: float = 1e-3, donors_per_batch: int = 64, inner_val: float = 0.15, cap: int = 200,
        seed: int = 0, max_donors: int | None = None, device: str | None = None, n_boot: int = 2000,
        pool: str = "default", write: bool = True) -> pl.DataFrame:  # fmt: skip
    """One arm over the cohorts on RM14's folds; returns the comparison with the pseudobulk clocks."""
    from sklearn.model_selection import GroupKFold

    from rna_age.train.fm_embed import clock_folds, compare

    dev = _device(device)
    name = model_name(loss, w, pseudocell)
    rows = []
    for c in cohorts:
        t0 = time.time()
        cc = cohort_cells(c, pool, cap, seed, max_donors)
        dp, reps = clock_folds(c)
        donors_all, age_all, groups_all = np.array(dp["donors"]), dp["age"], dp["groups"]
        for rep, subset, ok in reps:
            flag = cc["in_all"].astype(bool) if subset == "all" else cc["in_cd4"].astype(bool)
            ok = ok & np.isin(donors_all, list(set(cc["donor"][flag].tolist())))
            donors, age, groups = donors_all[ok], age_all[ok], groups_all[ok]
            if pseudocell > 1:
                xu, unit_donor = pseudocells(cc["x"], cc["donor"], flag, pseudocell, seed)
            else:
                xu, unit_donor = cc["x"][flag], cc["donor"][flag]
            pred = np.full(len(donors), np.nan)
            for k, (tr, te) in enumerate(GroupKFold(n_splits=5).split(donors, age, groups)):
                iv = _inner_split(groups[tr], inner_val, seed + k)
                fit, val = tr[~iv], tr[iv]
                pred[te] = fit_fold(xu, unit_donor, donors[fit].tolist(), age[fit], donors[val].tolist(), age[val],
                                    donors[te].tolist(), loss, w, epochs, lr, donors_per_batch, seed + k, dev)  # fmt: skip
            rows += [{"cohort": c[:8], "representation": rep, "model": name, "donor": d, "group": int(g),
                      "age": float(a), "pred": float(p)} for d, g, a, p in zip(donors, groups, age, pred, strict=True)]  # fmt: skip
            log.info("%s %s %s MAE %.2f (%d donors, %.0fs)", c[:8], rep, name, float(np.abs(pred - age).mean()),
                     len(donors), time.time() - t0)  # fmt: skip
    oof = pl.DataFrame(rows).with_columns(run_id=pl.lit(f"rm35-{name}"))
    cmp = compare(oof.drop("run_id"), n_boot=n_boot).with_columns(run_id=pl.lit(f"rm35-{name}"))
    if write:
        (RUNS / f"rm35-{name}").mkdir(parents=True, exist_ok=True)
        oof.write_parquet(RUNS / f"rm35-{name}" / "oof.parquet", compression="zstd")
        _merge_summary(oof, SUMMARY / "sc_pretrain_oof.parquet")
        _merge_summary(cmp, SUMMARY / "sc_pretrain_clock.parquet")
        _merge_summary(slopes(oof.drop("run_id")), SUMMARY / "cell_loss_slope.parquet", keys=("model",))
    return cmp
