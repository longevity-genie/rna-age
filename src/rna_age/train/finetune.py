"""Rung 4: fine-tune BulkFormer-Atlas for age (GPU, fp16 autocast + GradScaler; resumable).

- Inputs: the saved 4,096-gene panel inputs from the embedding jobs (``*.inputs.npz``), so training does not
  re-normalise count matrices.
- Model: Atlas encoder, CLS → LayerNorm → MLP head → age. Only the last ``unfreeze`` blocks, the output norm
  and the head train (``unfreeze=0`` = head only); gradient checkpointing on the trainable blocks.
- Loss: Huber (δ = 5 y) on exact ages + interval hinge for binned ages (GTEx: zero inside ``[min, max+1)``,
  linear outside), so bins train without inventing ages. Exact and binned batches are mixed.
- Sampling: each study contributes at most ``cap_per_study`` samples per epoch (big ARCHS4 series do not
  dominate).
- Early stopping on validation exact-age MAE; checkpoint ``last.pt`` / ``best.pt`` under
  ``data/interim/checkpoints/<run>/``; a requeued job resumes from ``last.pt``.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import polars as pl

from rna_age.paths import INTERIM
from rna_age.schema import age_mid
from rna_age.train.baselines import PRED_COLS
from rna_age.train.embed import EMB, MODEL_DIR, PANEL
from rna_age.train.runs import finish_run, new_run

log = logging.getLogger(__name__)


def load_inputs() -> tuple[list[str], np.ndarray]:
    ids, xs = [], []
    for p in sorted((EMB / "atlas").rglob("*.inputs.npz")):
        z = np.load(p, allow_pickle=False)
        ids += z["sample_id"].tolist()
        xs.append(z["x"])
    return ids, np.concatenate(xs)


def train(
    unfreeze: int = 2,
    epochs: int = 6,
    lr_head: float = 1e-3,
    lr_body: float = 2e-5,
    batch: int = 4,
    accum: int = 8,
    cap_per_study: int = 200,
    max_train: int = 30000,
    run_name: str | None = None,
    resume: str | None = None,
    train_set: str = "healthy",
    seed: int = 0,
) -> dict:
    import torch
    from bulkformer import load_atlas

    from rna_age.train.atlas_patch import patch_atlas_attention

    patch_atlas_attention()

    t0 = time.time()
    torch.manual_seed(seed)
    ids, X = load_inputs()
    pos = {s: i for i, s in enumerate(ids)}
    m = pl.read_parquet(INTERIM / "samples_all.parquet").filter(pl.col("sample_id").is_in(ids))
    m = m.filter(pl.col("split").is_not_null())
    m = m.with_columns(target=age_mid(m))
    m = m.with_columns(split=pl.when(pl.col("is_pseudobulk")).then(pl.lit("transfer")).otherwise(pl.col("split")))
    tr = m.filter(
        (pl.col("split") == "train") & (pl.col("is_healthy").fill_null(True) | (train_set == "all"))  # healthy or unknown (or all)
        & pl.col("age_type").is_in(["exact", "binned"])
    )  # fmt: skip
    rng = np.random.default_rng(seed)
    va = m.filter((pl.col("split") == "val") & (pl.col("age_type") == "exact"))
    va = va.sample(min(3000, va.height), seed=0)

    panel = pl.read_parquet(PANEL)["index"].to_numpy()
    model, _, _ = load_atlas(str(MODEL_DIR / "atlas"), device="cuda")
    dev = next(model.parameters()).device
    for p_ in model.parameters():
        p_.requires_grad = False
    blocks = list(model.blocks)
    trainable = blocks[len(blocks) - unfreeze :] if unfreeze else []
    for b in trainable:
        for p_ in b.parameters():
            p_.requires_grad = True
    for p_ in model.output_norm.parameters():
        p_.requires_grad = unfreeze > 0
    model.gradient_checkpointing = unfreeze > 0
    dim = model.output_norm.weight.shape[0]
    head = torch.nn.Sequential(
        torch.nn.LayerNorm(dim), torch.nn.Linear(dim, 256), torch.nn.GELU(), torch.nn.Dropout(0.1),
        torch.nn.Linear(256, 1),
    ).to(dev)  # fmt: skip
    groups = [{"params": head.parameters(), "lr": lr_head}]
    body = [p_ for p_ in model.parameters() if p_.requires_grad]
    if body:
        groups.append({"params": body, "lr": lr_body})
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda")
    gene_ids = torch.as_tensor(np.r_[model.num_genes, panel], dtype=torch.long, device=dev)

    def forward(rows: np.ndarray, train_mode: bool):
        x = torch.as_tensor(X[rows].astype(np.float32), device=dev)
        x = torch.cat([x.new_zeros(len(x), 1), x], dim=1)
        gid = gene_ids[None].expand(len(x), -1)
        model.train(train_mode and unfreeze > 0)
        head.train(train_mode)
        with torch.autocast("cuda", dtype=torch.float16):
            h = model.encode(gid, x, torch.ones_like(gid, dtype=torch.bool))[:, 0]
            return head(h.float()).squeeze(-1).float()

    def loss_fn(pred, kind, y, lo, hi):
        exact = torch.nn.functional.huber_loss(pred, y, reduction="none", delta=5.0)
        interval = torch.relu(lo - pred) + torch.relu(pred - (hi + 1))
        return torch.where(kind, exact, interval).mean()

    name = run_name or f"atlas-ft-u{unfreeze}"
    ckpt_dir = Path(resume) if resume else None
    if ckpt_dir is None:  # a requeued job resumes its own unfinished run (same name, no predictions yet)
        for cand in sorted((INTERIM / "checkpoints").glob(f"*-{name}"), reverse=True):
            run = INTERIM.parent / "output" / "runs" / cand.name
            if (cand / "last.pt").exists() and not (run / "predictions.parquet").exists():
                ckpt_dir = cand
                break
    run_dir = None
    start_epoch, best = 0, np.inf
    if ckpt_dir and (ckpt_dir / "last.pt").exists():
        state = torch.load(ckpt_dir / "last.pt", map_location=dev, weights_only=False)
        head.load_state_dict(state["head"])
        model.load_state_dict(state["model"], strict=False)
        opt.load_state_dict(state["opt"])
        start_epoch, best = state["epoch"] + 1, state["best"]
        run_dir = Path(state["run_dir"])
        log.info("resumed from epoch %d (best val MAE %.2f)", start_epoch, best)
    if run_dir is None:
        run_dir = new_run(
            name,
            {"model": f"Atlas fine-tune (last {unfreeze} blocks + head)", "features": "Atlas panel 4096",
             "train_set": "bulk train, healthy-or-unknown, exact + GTEx intervals", "unfreeze": unfreeze,
             "epochs": epochs, "lr_head": lr_head, "lr_body": lr_body, "batch": batch, "accum": accum,
             "cap_per_study": cap_per_study, "max_train": max_train, "precision": "fp16 autocast",
             "health_filter": train_set, "seed": seed},
        )  # fmt: skip
        ckpt_dir = INTERIM / "checkpoints" / run_dir.name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    va_rows = np.array([pos[s] for s in va["sample_id"].to_list()])
    va_y = va["age_years"].to_numpy()

    def predict(rows: np.ndarray, bs: int = 16) -> np.ndarray:
        out = []
        with torch.no_grad():
            for i in range(0, len(rows), bs):
                out.append(forward(rows[i : i + bs], False).cpu().numpy())
        return np.concatenate(out) if out else np.array([])

    bad = 0
    for epoch in range(start_epoch, epochs):
        ep = tr.with_columns(_r=pl.col("sample_id").hash(seed=epoch + 1000 * seed)).sort("_r")
        ep = ep.with_columns(_k=pl.int_range(pl.len()).over("study_id")).filter(pl.col("_k") < cap_per_study)
        ep = ep.head(max_train)
        rows = np.array([pos[s] for s in ep["sample_id"].to_list()])
        kind = torch.as_tensor((ep["age_type"] == "exact").to_numpy(), device=dev)
        y = torch.as_tensor(ep["target"].to_numpy(), dtype=torch.float32, device=dev)
        lo = torch.as_tensor(ep["age_min"].fill_null(0).to_numpy(), dtype=torch.float32, device=dev)
        hi = torch.as_tensor(ep["age_max"].fill_null(0).to_numpy(), dtype=torch.float32, device=dev)
        perm = rng.permutation(len(rows))
        te = time.time()
        for step, i in enumerate(range(0, len(perm), batch)):
            b = perm[i : i + batch]
            loss = loss_fn(forward(rows[b], True), kind[b], y[b], lo[b], hi[b]) / accum
            scaler.scale(loss).backward()
            if (step + 1) % accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(list(head.parameters()) + body, 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            if step % 500 == 0:
                log.info("epoch %d step %d/%d loss %.3f (%.0fs)", epoch, step, len(perm) // batch,
                         loss.item() * accum, time.time() - te)  # fmt: skip
        val_mae = float(np.mean(np.abs(predict(va_rows) - va_y)))
        log.info("epoch %d: val exact MAE %.2f (%.0fs)", epoch, val_mae, time.time() - te)
        state = {"head": head.state_dict(), "opt": opt.state_dict(), "epoch": epoch, "best": min(best, val_mae),
                 "run_dir": str(run_dir),
                 "model": {k: v for k, v in model.state_dict().items() if any(k.startswith(f"blocks.{len(blocks) - j - 1}.") for j in range(unfreeze)) or k.startswith("output_norm")}}  # fmt: skip
        torch.save(state, ckpt_dir / "last.pt")
        if val_mae < best:
            best, bad = val_mae, 0
            torch.save(state, ckpt_dir / "best.pt")
        else:
            bad += 1
            if bad >= 2:
                break
    best_state = torch.load(ckpt_dir / "best.pt", map_location=dev, weights_only=False)
    head.load_state_dict(best_state["head"])
    model.load_state_dict(best_state["model"], strict=False)
    # evaluation samples only (val, test, transfer): predicting the 67k training samples costs ~2 GPU-hours
    ev = m.filter(pl.col("split") != "train")
    ev_rows = np.array([pos[s] for s in ev["sample_id"].to_list()])
    pred = predict(ev_rows, bs=8)
    out = ev.select([c for c in PRED_COLS if c in ev.columns]).with_columns(pred=pl.Series(pred.astype(np.float64)))
    (run_dir / "checkpoint.json").write_text(json.dumps({"checkpoint_dir": str(ckpt_dir)}))
    row = finish_run(run_dir, out, {"seconds": time.time() - t0, "best_val_mae": best,
                                    "host_gpu": os.environ.get("CUDA_VISIBLE_DEVICES")})  # fmt: skip
    log.info("run %s: %s", run_dir.name, {k: v for k, v in row.items() if k.startswith("test_")})
    return row
