"""RM20 network: a shared trunk on the gene panel with an age head and a knockdown-Δ head, and its losses.

- age(x) = linear(x) + linear(trunk(x)), on standardised years;
- Δ(control, gene) = MLP([trunk(control), embed(gene features)]), on the standardised panel;
- Δ loss: DEG-weighted MSE (Diversity by Design): per knockdown, each gene weighted by |Δ − mean Δ of the training
  knockdowns of its context| (its deviation from the mean perturbed profile), normalised to mean 1, floor 0.1;
- rank loss (optional): pairwise logistic loss so that age(control + Δ̂) − age(control) orders training knockdowns of
  one condition as the study's aging coefficients do.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

log = logging.getLogger(__name__)


@dataclass
class Config:
    hidden: int = 512
    emb: int = 128
    dropout: float = 0.1
    steps: int = 3000
    batch: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-4
    w_delta: float = 1.0  # 0 = Δ head off (age-only arm)
    w_rank: float = 0.0  # 0 = no ranking term
    tau: float = 0.05  # temperature of the rank loss, standardised years
    seed: int = 0


class Net(nn.Module):
    def __init__(self, n_genes: int, n_feat: int, cfg: Config):
        super().__init__()
        h = cfg.hidden
        self.trunk = nn.Sequential(nn.Linear(n_genes, h), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(h, h), nn.GELU())
        self.age_lin = nn.Linear(n_genes, 1)
        self.age_mlp = nn.Linear(h, 1)
        self.embed = nn.Sequential(nn.Linear(n_feat, 2 * cfg.emb), nn.GELU(), nn.Linear(2 * cfg.emb, cfg.emb))
        self.delta_head = nn.Sequential(
            nn.Linear(h + cfg.emb, h), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(h, h), nn.GELU(), nn.Linear(h, n_genes)
        )

    def age(self, x: torch.Tensor) -> torch.Tensor:
        return (self.age_lin(x) + self.age_mlp(self.trunk(x))).squeeze(-1)

    def delta(self, control: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        return self.delta_head(torch.cat([self.trunk(control), self.embed(feat)], -1))


def deg_weights(delta: torch.Tensor, mean_delta: torch.Tensor, floor: float = 0.1) -> torch.Tensor:
    a = (delta - mean_delta).abs()
    w = a / (a.mean(-1, keepdim=True) + 1e-8)
    return w.clamp_min(floor)


def rank_loss(shift: torch.Tensor, coef: torch.Tensor, cond: torch.Tensor, tau: float) -> torch.Tensor:
    """Pairwise logistic loss over pairs of one condition with different coefficients."""
    same = cond[:, None] == cond[None, :]
    sign = torch.sign(coef[:, None] - coef[None, :])
    m = same & (sign != 0)
    if not m.any():
        return shift.sum() * 0
    d = (shift[:, None] - shift[None, :]) / tau
    return nn.functional.softplus(-sign * d)[m].mean()


@dataclass
class Task:
    """Tensors of one training run (already on the device)."""

    X_age: torch.Tensor  # train donors × genes
    y_age: torch.Tensor  # standardised
    controls: torch.Tensor  # contexts × genes
    ctx: torch.Tensor  # train units → context
    feat: torch.Tensor  # train units × features
    delta: torch.Tensor  # train units × genes
    mean_delta: torch.Tensor  # contexts × genes (mean Δ of the training units of each context)
    coef: torch.Tensor  # train units; NaN where the study has no coefficient
    cond: torch.Tensor  # train units → condition index


def train(task: Task, cfg: Config, device: torch.device) -> Net:
    torch.manual_seed(cfg.seed)
    net = Net(task.X_age.shape[1], task.feat.shape[1], cfg).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=cfg.steps, pct_start=0.1)
    g = torch.Generator(device="cpu").manual_seed(cfg.seed)
    n_units = len(task.ctx)
    has_coef = ~torch.isnan(task.coef)
    for step in range(cfg.steps):
        net.train()
        loss_age = nn.functional.mse_loss(net.age(task.X_age), task.y_age)
        loss = loss_age
        ld = lr_ = torch.zeros((), device=device)
        if cfg.w_delta > 0 and n_units:
            i = torch.randint(0, n_units, (min(cfg.batch, n_units),), generator=g).to(device)
            c = task.ctx[i]
            ctrl = task.controls[c]
            pred = net.delta(ctrl, task.feat[i])
            w = deg_weights(task.delta[i], task.mean_delta[c])
            ld = (w * (pred - task.delta[i]) ** 2).mean()
            loss = loss + cfg.w_delta * ld
            if cfg.w_rank > 0:
                k = has_coef[i]
                if k.sum() > 1:
                    shift = net.age(ctrl[k] + pred[k]) - net.age(ctrl[k])
                    lr_ = rank_loss(shift, task.coef[i][k], task.cond[i][k], cfg.tau)
                    loss = loss + cfg.w_rank * lr_
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % max(1, cfg.steps // 10) == 0 or step == cfg.steps - 1:
            log.info("step %d: age %.4f delta %.4f rank %.4f", step, loss_age.item(), ld.item(), lr_.item())
    net.eval()
    return net


@torch.no_grad()
def predict(net: Net, X_age: torch.Tensor, controls: torch.Tensor, ctx: torch.Tensor, feat: torch.Tensor,
            delta_true: torch.Tensor, with_delta: bool, chunk: int = 4096) -> dict[str, np.ndarray]:  # fmt: skip
    """Ages (standardised) of ``X_age``; for units: Δ̂ (if ``with_delta``), predicted and measured age shifts."""
    out: dict[str, list] = {"dhat": [], "shift_pred": [], "shift_meas": []}
    base = net.age(controls)
    for a in range(0, len(ctx), chunk):
        c = ctx[a : a + chunk]
        ctrl = controls[c]
        out["shift_meas"].append((net.age(ctrl + delta_true[a : a + chunk]) - base[c]).cpu())
        if with_delta:
            d = net.delta(ctrl, feat[a : a + chunk])
            out["dhat"].append(d.cpu())
            out["shift_pred"].append((net.age(ctrl + d) - base[c]).cpu())
    res = {k: torch.cat(v).numpy() for k, v in out.items() if v}
    res["age"] = net.age(X_age).cpu().numpy()
    return res
