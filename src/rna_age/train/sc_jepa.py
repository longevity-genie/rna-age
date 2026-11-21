"""RM35 step 2: a SIGReg-JEPA encoder on single cells (falsification run).

Same pool, gene panel, encoder and step budget as RM28's masking arm (``sc_pretrain``), but no reconstruction: each
cell is seen through two random gene views (each hides a fraction of the panel). The encoder embeds both; a small
predictor maps the context view's embedding to the target view's, and SIGReg (LeJEPA, Balestriero & LeCun 2025) keeps
the embeddings near an isotropic Gaussian, which is what stops collapse without a stop-gradient or an EMA teacher:

    loss = (1 - λ) · ||pred(z_ctx) - z_tgt||² + λ · SIGReg(z_ctx ∪ z_tgt)

SIGReg is the sliced Epps-Pulley statistic: project the batch's embeddings onto random unit directions and compare
each projection's empirical characteristic function with N(0, 1)'s on a grid of t, under a Gaussian window.

The checkpoint holds an ``ScMAE`` state (its decoder and perturbation head unused), so ``sc_pretrain``'s
``load_encoder``, ``embed_cohorts``, ``clock`` and ``finetune`` read it unchanged; the projector and predictor are
saved under ``heads``. Reused control: RM28's masking arm (run 20261113-1942-scpre-mask-s0, ee227b2).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch import nn

from rna_age.train.sc_pretrain import ResBlock, ScMAE, _device, load_pool_matrix, sample_mask, split_donors

log = logging.getLogger(__name__)


def sigreg(z: torch.Tensor, n_dirs: int = 256, n_t: int = 17, t_max: float = 3.0,
           generator: torch.Generator | None = None) -> torch.Tensor:  # fmt: skip
    """Sliced Epps-Pulley statistic of ``z`` (N, D) against an isotropic standard Gaussian; 0 for a perfect fit.

    The characteristic function of N(0, 1) is real and even, so t runs over [0, t_max] and the imaginary part is
    compared with 0. Scaled by N as in LeJEPA, so the term does not vanish with the batch size."""
    n, d = z.shape
    a = torch.randn(d, n_dirs, device=z.device, dtype=z.dtype, generator=generator)
    a = a / a.norm(dim=0, keepdim=True)
    p = z @ a  # (N, K)
    t = torch.linspace(0.0, t_max, n_t, device=z.device, dtype=z.dtype)
    xt = p.unsqueeze(-1) * t  # (N, K, T)
    re, im = torch.cos(xt).mean(0), torch.sin(xt).mean(0)  # (K, T)
    phi = torch.exp(-0.5 * t**2)
    err = ((re - phi) ** 2 + im**2) * phi  # Gaussian window
    return torch.trapezoid(err, t, dim=-1).mean() * n


class JepaHeads(nn.Module):
    """Projector (the space SIGReg and the prediction loss act in) and predictor (context → target embedding)."""

    def __init__(self, emb: int, proj: int = 128, hidden: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(emb, hidden), nn.GELU(), nn.Linear(hidden, proj))
        self.pred = nn.Sequential(nn.Linear(proj, hidden), nn.GELU(), ResBlock(hidden, dropout), nn.LayerNorm(hidden),
                                  nn.Linear(hidden, proj))  # fmt: skip


def jepa_loss(model: ScMAE, heads: JepaHeads, x: torch.Tensor, m_ctx: torch.Tensor, m_tgt: torch.Tensor, lam: float,
              generator: torch.Generator | None = None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:  # fmt: skip
    z_ctx = heads.proj(model.encode(x, m_ctx))
    z_tgt = heads.proj(model.encode(x, m_tgt))
    pred = ((heads.pred(z_ctx) - z_tgt) ** 2).sum(-1).mean()
    reg = sigreg(torch.cat([z_ctx, z_tgt]).float(), generator=generator)
    return (1 - lam) * pred + lam * reg, {"pred": pred.detach(), "sigreg": reg.detach()}


def effective_rank(z: torch.Tensor) -> float:
    """exp(entropy of the normalised singular values): ~D for isotropic embeddings, ~1 for collapsed ones."""
    s = torch.linalg.svdvals(z.float() - z.float().mean(0))
    p = s / s.sum()
    return float(torch.exp(-(p * torch.log(p.clamp_min(1e-12))).sum()))


def pretrain_jepa(epochs: int = 30, seed: int = 0, pool: str = "default", batch_size: int = 1024, lr: float = 1e-3,
                  weight_decay: float = 0.01, mask_lo: float = 0.3, mask_hi: float = 0.6, lam: float = 0.05,
                  proj: int = 128, val_frac: float = 0.05, val_cells: int = 50_000, hidden: int = 1024, emb: int = 256,
                  depth: int = 2, dropout: float = 0.1, max_cells: int | None = None, max_steps: int | None = None,
                  device: str | None = None, name: str | None = None) -> dict:  # fmt: skip
    """Train the SIGReg-JEPA encoder; RM28's loop (pool on the card, AdamW, warm-up + cosine, per-epoch checkpoints).
    ``best.pt`` is the epoch with the lowest validation loss. ``failed`` when the embeddings collapsed (effective rank
    of the validation embeddings below 2 at the end)."""
    from rna_age.train.runs import has_native_bf16, new_run

    dev = _device(device)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    table, X = load_pool_matrix(pool, max_cells, seed)
    is_val = split_donors(table, val_frac, seed)
    tr_idx, va_idx = np.flatnonzero(~is_val), np.flatnonzero(is_val)
    if len(va_idx) > val_cells:
        va_idx = np.sort(rng.choice(va_idx, val_cells, replace=False))
    Xt = torch.from_numpy(X).to(dev)
    n_genes = X.shape[1]
    config = {"model": "sc_pretrain", "arm": "jepa", "pool": pool, "epochs": epochs, "seed": seed,
              "batch_size": batch_size, "lr": lr, "weight_decay": weight_decay, "mask": [mask_lo, mask_hi],
              "sigreg_lambda": lam, "proj": proj, "hidden": hidden, "emb": emb, "depth": depth, "dropout": dropout,
              "n_genes": n_genes, "train_cells": len(tr_idx), "val_cells": len(va_idx), "max_cells": max_cells,
              "device": str(dev), "control": "20261113-1942-scpre-mask-s0"}  # fmt: skip
    run_dir = new_run(name or f"scpre-jepa-s{seed}", config)
    cfg = json.loads((run_dir / "config.json").read_text())
    ckpt = Path(cfg["checkpoint_dir"])
    ckpt.mkdir(parents=True, exist_ok=True)
    log.info("run %s: %d train / %d val cells × %d genes on %s (%.0fs)", cfg["run_id"], len(tr_idx), len(va_idx),
             n_genes, dev, time.time() - t0)  # fmt: skip
    model = ScMAE(n_genes, hidden, emb, depth, dropout).to(dev)
    heads = JepaHeads(emb, proj, hidden, dropout).to(dev)
    params = list(model.parameters()) + list(heads.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    steps_per_epoch = -(-len(tr_idx) // batch_size)
    if max_steps:
        steps_per_epoch = min(steps_per_epoch, max_steps)
    total = steps_per_epoch * epochs
    warm = min(1000, max(1, total // 20))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + np.cos(np.pi * min(1.0, s / max(1, total))))
    )
    bf16 = dev.type == "cuda" and has_native_bf16()
    amp = dict(device_type="cuda", dtype=torch.bfloat16) if bf16 else dict(device_type=dev.type, enabled=False)
    gval = torch.Generator(device=dev).manual_seed(seed + 12345)
    va_t = torch.from_numpy(va_idx).to(dev)
    chunk = 4096
    va_masks = [(sample_mask(min(chunk, len(va_idx) - a), n_genes, mask_lo, mask_hi, gval, dev),
                 sample_mask(min(chunk, len(va_idx) - a), n_genes, mask_lo, mask_hi, gval, dev))
                for a in range(0, len(va_idx), chunk)]  # fmt: skip
    rows, status, best = [], {"run_id": cfg["run_id"], "failed": False, "reason": None}, np.inf
    for epoch in range(1, epochs + 1):
        model.train(), heads.train()
        te = time.time()
        perm = torch.from_numpy(rng.permutation(tr_idx)).to(dev)
        tot = {"loss": 0.0, "pred": 0.0, "sigreg": 0.0}
        for step in range(steps_per_epoch):
            x = Xt[perm[step * batch_size : (step + 1) * batch_size]].float()
            m1 = sample_mask(len(x), n_genes, mask_lo, mask_hi, device=dev)
            m2 = sample_mask(len(x), n_genes, mask_lo, mask_hi, device=dev)
            with torch.autocast(**amp):
                loss, parts = jepa_loss(model, heads, x, m1, m2, lam)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            tot["loss"] += float(loss.detach())
            for k, v in parts.items():
                tot[k] += float(v)
            if step % 200 == 0:
                log.info("epoch %d step %d/%d: loss %.4f pred %.4f sigreg %.4f", epoch, step, steps_per_epoch,
                         float(loss.detach()), float(parts["pred"]), float(parts["sigreg"]))  # fmt: skip
        model.eval(), heads.eval()
        gsig = torch.Generator(device=dev).manual_seed(seed + 999)
        with torch.no_grad(), torch.autocast(**amp):
            vl, vp, vs, nv, embs = 0.0, 0.0, 0.0, 0, []
            for k, a in enumerate(range(0, len(va_idx), chunk)):
                x = Xt[va_t[a : a + chunk]].float()
                loss, parts = jepa_loss(model, heads, x, *va_masks[k], lam, generator=gsig)
                w = len(x)
                vl, vp, vs, nv = vl + float(loss) * w, vp + float(parts["pred"]) * w, vs + float(parts["sigreg"]) * w, nv + w
                if k < 4:
                    embs.append(model.encode(x).float())
        erank = effective_rank(torch.cat(embs))
        row = {"epoch": epoch, **{f"train_{k}": v / steps_per_epoch for k, v in tot.items()}, "val_loss": vl / nv,
               "val_pred": vp / nv, "val_sigreg": vs / nv, "val_effective_rank": erank, "seconds": time.time() - te}  # fmt: skip
        rows.append(row)
        pl.DataFrame(rows).write_parquet(run_dir / "metrics.parquet")
        log.info("epoch %d: %s", epoch, {k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()})
        state = {"model": model.state_dict(), "heads": heads.state_dict(), "config": cfg, "epoch": epoch}
        torch.save(state, ckpt / "last.pt")
        if row["val_loss"] < best:
            best = row["val_loss"]
            torch.save(state, ckpt / "best.pt")
    if rows[-1]["val_effective_rank"] < 2:
        status |= {"failed": True, "reason": f"collapsed: effective rank {rows[-1]['val_effective_rank']:.2f}"}
    status |= {"epochs_run": len(rows), "best_score": best, "best_by": "val_loss", "final": rows[-1],
               "seconds": round(time.time() - t0)}  # fmt: skip
    (run_dir / "status.json").write_text(json.dumps(status, indent=2, default=float))
    return status
