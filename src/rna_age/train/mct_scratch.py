"""RM37 step (c): MCT's architecture pretrained from scratch on our own single-cell pool.

Pool (``build_pool``, a CPU job): every non-evaluation Census h5ad (Tabula Sapiens' blood-only subset dropped once the
whole atlas is on disk); primary cells of donors aged ≥ 18 whose disease does not match the repo's cancer pattern
(diseased donors stay: the objective never sees age or disease); donors sharing an id with an evaluation donor
removed. A sample is donor × tissue, keeping ≤ ``factor`` × the slot count cells per MCT type (``mct.pool_cells``),
stored once per dataset as log1p CP10k on MCT's vocabulary (CSR, float16 values, token ids as column indices) under
``data/interim/mct/pool/<dataset[:8]>/``, cells sorted by sample.

Pretraining (``pretrain``, one GPU): MCT's encoder randomly initialised, pre-norm by default (``norm_first``; MCT's own
12 post-norm layers did not learn at all from scratch here, not even to copy a visible id), MCT's masked gene identity (15 % of tokens:
70 % id → [MASK], 10 % expression masked, 10 % random id, 10 % unchanged; cross-entropy over the vocabulary), one bag
per forward, ``accumulate`` bags per optimiser step, AdamW with linear warm-up then cosine. Validation masked-id loss
and accuracy on fixed bags of held-out samples; best and last checkpoints in MCT's Lightning layout
(``model.encoder.*``, ``model.gene_id_classifier.*``, ``hyper_parameters.config``) so ``mct.load_encoder`` reads them.
"""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import polars as pl
import scipy.sparse as sp
import torch
from torch import nn

from rna_age.train import mct
from rna_age.train.sc_pretrain import ADULT, H5ADS, eval_donors, exclude_eval, pool_datasets, read_obs

log = logging.getLogger(__name__)

POOL = mct.MCT_DIR / "pool"
TS_ALL = "53d208b0-2cfd-4366-9866-c3c6114081bc"  # Tabula Sapiens - All Cells
TS_BLOOD = "983d5ec9-40e8-4512-9e65-a572a9c486cb"  # its blood-only subset (same donors)
MASK_ID = 1


# ------------------------------------------------------------------ pool


def scratch_datasets(h5ad_dir: Path = H5ADS) -> list[str]:
    """Non-evaluation h5ads on disk; Tabula Sapiens' blood subset dropped when the whole atlas is there."""
    ds = pool_datasets(h5ad_dir)
    return [d for d in ds if not (d == TS_BLOOD and TS_ALL in ds)]


def eligible(obs: pl.DataFrame, min_age: float = ADULT) -> pl.DataFrame:
    """Primary cells of adult donors without a cancer diagnosis, with ``sample`` = donor × tissue and ``age``."""
    from rna_age.sources.bgee import stage_age
    from rna_age.sources.census_pseudobulk import CANCER

    stages = {s: stage_age(s) for s in obs["development_stage"].unique().to_list()}
    age = [stages[s][1] if stages[s][1] is not None else stages[s][2] for s in obs["development_stage"].to_list()]
    return obs.with_columns(age=pl.Series(age, dtype=pl.Float64)).filter(
        pl.col("is_primary_data")
        & pl.col("donor_id").is_not_null()
        & (pl.col("age") >= min_age)
        & ~pl.col("disease").str.contains(CANCER)
    ).with_columns(sample=pl.col("donor_id") + ":" + pl.col("tissue"))


def dataset_pool(ds: str, index: mct.Index, evals: pl.DataFrame, factor: int = 2, seed: int = 0,
                 out_root: Path = POOL, max_samples: int | None = None) -> dict:  # fmt: skip
    """Write one dataset's pool (see the module docstring); returns its summary row."""
    from rna_age.train.fm_embed import read_rows
    from rna_age.train.sc_pretrain import file_gene_ids, log_cp10k

    t0 = time.time()
    h5ad = H5ADS / f"{ds}.h5ad"
    obs = read_obs(h5ad)
    e = eligible(obs).with_columns(dataset_id=pl.lit(ds))
    e, overlap = exclude_eval(e, evals)
    if max_samples:
        keep = sorted(e["sample"].unique().to_list())[:max_samples]
        e = e.filter(pl.col("sample").is_in(keep))
    summary = {"dataset_id": ds, "cells_file": obs.height, "cells_eligible": e.height,
               "samples": e["sample"].n_unique(), "donors": e["donor_id"].n_unique(),
               "donors_removed_eval_overlap": overlap.height}  # fmt: skip
    if not e.height:
        return summary | {"cells_pool": 0, "seconds": time.time() - t0}
    cells = e.select("row", pl.col("sample").alias("donor"), pl.col("cell_type").alias("ct"), "age", "disease",
                     "tissue")  # fmt: skip
    sel = mct.pool_cells(cells, index, "", factor, np.random.default_rng(seed))
    cols, toks = mct.vocab_columns(file_gene_ids(h5ad), index, symbols=False)
    parts = []
    for _, X in read_rows(h5ad, sel["row"].to_numpy()):
        Y = log_cp10k(X)[:, cols].tocsr()
        parts.append(sp.csr_matrix((Y.data.astype(np.float32), toks[Y.indices].astype(np.int32), Y.indptr),
                                   shape=(Y.shape[0], len(index.gene2idx))))  # fmt: skip
    X = sp.vstack(parts).tocsr()
    order = np.argsort(sel["donor"].to_numpy(), kind="stable")
    X, sel = X[order], sel[order.tolist()]
    names = list(index.order)
    tix = {n: i for i, n in enumerate(names)}
    d = out_root / ds[:8]
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "data.npy", X.data.astype(np.float16))
    np.save(d / "indices.npy", X.indices.astype(np.int32))
    np.save(d / "indptr.npy", X.indptr.astype(np.int64))
    pl.DataFrame({"slot": np.array([tix[s] for s in sel["slot"].to_list()], dtype=np.int16)}).write_parquet(
        d / "cells.parquet", compression="zstd")  # fmt: skip
    samples = (
        sel.with_row_index("i")
        .group_by("donor", maintain_order=True)
        .agg(pl.col("i").min().alias("start"), (pl.col("i").max() + 1).alias("end"), pl.col("age").first(),
             pl.col("disease").first(), pl.col("tissue").first())  # fmt: skip
        .rename({"donor": "sample"})
        .with_columns(dataset_id=pl.lit(ds))
    )
    samples.write_parquet(d / "samples.parquet", compression="zstd")
    summary |= {"cells_pool": X.shape[0], "nnz": int(X.nnz), "seconds": time.time() - t0}
    (d / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info("%s: %s", ds[:8], summary)
    return summary


def build_pool(datasets: tuple[str, ...] | None = None, factor: int = 2, seed: int = 0, index_dir: Path = mct.INDEX,
               out_root: Path = POOL, max_samples: int | None = None, skip_done: bool = True) -> pl.DataFrame:  # fmt: skip
    from rna_age.train.sc_pretrain import EVAL_COHORTS

    datasets = tuple(datasets) if datasets else tuple(scratch_datasets())
    bad = set(datasets) & set(EVAL_COHORTS)
    if bad:
        raise ValueError(f"evaluation cohorts cannot be pretraining data: {sorted(bad)}")
    index, evals = mct.Index.load(index_dir), eval_donors()
    rows = []
    for ds in datasets:
        done = out_root / ds[:8] / "summary.json"
        rows.append(json.loads(done.read_text()) if skip_done and done.exists()
                    else dataset_pool(ds, index, evals, factor, seed, out_root, max_samples))  # fmt: skip
    out = pl.DataFrame(rows)
    out.write_parquet(out_root / "pool_summary.parquet", compression="zstd")
    return out


class Pool:
    """The written pool, read into memory (``in_memory``, the default: memory-mapped reads from the network filesystem
    stalled a 2080 Ti run at ~1 % GPU for 90 min); ``bag(i, rng, max_genes)`` gives one MCT bag of sample ``i``."""

    def __init__(self, root: Path = POOL, index: mct.Index | None = None, in_memory: bool = True,
                 min_cells: int = 50):  # fmt: skip
        self.index = index or mct.Index.load()
        self.names = list(self.index.order)
        self.n_vocab = len(self.index.gene2idx)
        self.parts, frames = [], []
        for k, d in enumerate(sorted(p for p in root.iterdir() if (p / "samples.parquet").exists())):
            self.parts.append({
                "data": np.load(d / "data.npy", mmap_mode=None if in_memory else "r"),
                "indices": np.load(d / "indices.npy", mmap_mode=None if in_memory else "r"),
                "indptr": np.load(d / "indptr.npy"),
                "slot": pl.read_parquet(d / "cells.parquet")["slot"].to_numpy().astype(np.int64),
            })  # fmt: skip
            frames.append(pl.read_parquet(d / "samples.parquet").with_columns(part=pl.lit(k)))
        if not frames:
            raise FileNotFoundError(f"no pool under {root}")
        s = pl.concat(frames, how="vertical_relaxed")
        self.samples = s.filter((pl.col("end") - pl.col("start")) >= min_cells)  # tiny samples give near-empty bags
        self.dropped = s.height - self.samples.height

    def __len__(self) -> int:
        return self.samples.height

    def bag(self, i: int, rng: np.random.Generator, max_genes: int) -> tuple[np.ndarray, np.ndarray]:
        s = self.samples.row(i, named=True)
        p = self.parts[s["part"]]
        a, b = s["start"], s["end"]
        lo, hi = int(p["indptr"][a]), int(p["indptr"][b])
        X = sp.csr_matrix((np.asarray(p["data"][lo:hi], dtype=np.float32), np.asarray(p["indices"][lo:hi]),
                           p["indptr"][a:b + 1] - lo), shape=(b - a, self.n_vocab))  # fmt: skip
        dc = mct.DonorCells(X, np.arange(self.n_vocab), [s["sample"]], [np.arange(b - a)], [p["slot"][a:b]],
                            self.names, self.index.order)  # fmt: skip
        return mct.bag_tensor(dc, 0, rng, max_genes)


# ------------------------------------------------------------------ pretraining


def mask_tokens(tok: torch.Tensor, n_vocab: int, gen: torch.Generator, p: float = 0.15
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # fmt: skip
    """MCT's masking: (input ids, expression-masked tokens, selected tokens to predict)."""
    sel = torch.rand(tok.shape, generator=gen) < p
    if not sel.any():
        sel[..., int(torch.randint(tok.shape[-1], (1,), generator=gen))] = True
    u = torch.rand(tok.shape, generator=gen)
    ids = tok.clone()
    ids[sel & (u < 0.7)] = MASK_ID
    rnd = sel & (u >= 0.8) & (u < 0.9)
    ids[rnd] = torch.randint(mct.N_SPECIAL, n_vocab, (int(rnd.sum()),), generator=gen)
    return ids, sel & (u >= 0.7) & (u < 0.8), sel


class MctPretrain(nn.Module):
    def __init__(self, n_vocab: int, n_input: int = mct.BAG, emb: int = 64, hidden: int = 256, n_heads: int = 4,
                 n_layers: int = 12, dropout: float = 0.2, transformer_dropout: float = 0.1,
                 norm_first: bool = True):  # fmt: skip
        super().__init__()
        self.encoder = mct.MctEncoder(n_vocab, n_input, emb, hidden, n_heads, n_layers, 0, dropout,
                                      transformer_dropout, norm_first=norm_first)  # fmt: skip
        self.gene_id_classifier = mct.MctMGMHead(emb, n_vocab)

    def forward(self, x, ids, expr_mask=None):
        return self.gene_id_classifier(self.encoder(x, ids, expr_mask=expr_mask))


def mgm_loss(model: MctPretrain, x: np.ndarray, tok: np.ndarray, n_vocab: int, gen: torch.Generator, device: str,
             amp: torch.dtype | None) -> tuple[torch.Tensor, float]:  # fmt: skip
    t = torch.from_numpy(tok)[None]
    ids, emask, sel = mask_tokens(t, n_vocab, gen)
    with torch.autocast(device_type="cuda", dtype=amp or torch.float16, enabled=amp is not None):
        logits = model(torch.from_numpy(x)[None].to(device), ids.to(device), emask.to(device))
    s = sel.to(device)
    lg, y = logits[s].float(), t.to(device)[s]
    return nn.functional.cross_entropy(lg, y), float((lg.argmax(-1) == y).float().mean())


def save_checkpoint(path: Path, model: MctPretrain, config: dict, step: int) -> None:
    sd = {f"model.{k}": v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({"state_dict": sd, "hyper_parameters": {"config": config}, "global_step": step}, path)


def pretrain(steps: int = 20_000, accumulate: int = 8, lr: float = 1e-3, warmup: int = 1000, max_genes: int = 4096,
             emb: int = 64, n_layers: int = 12, n_heads: int = 4, hidden: int = 256, val_share: float = 0.02,
             n_val_bags: int = 64, eval_every: int = 1000, seed: int = 0, device: str = "cuda",
             pool_root: Path = POOL, name: str | None = None, index_dir: Path = mct.INDEX,
             min_cells: int = 50, norm_first: bool = True) -> dict:  # fmt: skip
    from rna_age.train.runs import has_native_bf16, new_run

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    t_load = time.time()
    pool = Pool(pool_root, mct.Index.load(index_dir), min_cells=min_cells)
    log.info("pool: %d samples from %d datasets (%d with < 50 cells dropped) in %.0fs", len(pool), len(pool.parts),
             pool.dropped, time.time() - t_load)  # fmt: skip
    n = len(pool)
    perm = rng.permutation(n)
    n_val = max(1, int(round(val_share * n)))
    va, tr = perm[:n_val], perm[n_val:]
    n_in = sum(b - a for a, b in pool.index.order.values())
    config = {"model": "mct-scratch", "arm": "rm37-c", "steps": steps, "accumulate": accumulate, "lr": lr,
              "warmup": warmup, "max_genes": max_genes, "gene_emb_dim": emb, "n_layers": n_layers, "n_heads": n_heads,
              "hidden_dim": hidden, "n_genes": pool.n_vocab, "n_input": n_in, "dropout": 0.2,
              "transformer_dropout": 0.1, "padding_token_idx": 0, "seed": seed, "min_cells": min_cells, "norm_first": norm_first, "samples_train": len(tr),
              "samples_val": len(va), "datasets": pool.samples["dataset_id"].unique().sort().to_list()}  # fmt: skip
    run_dir = new_run(name or f"mct-scratch-s{seed}", config)
    cfg = json.loads((run_dir / "config.json").read_text())
    ckpt = Path(cfg["checkpoint_dir"])
    ckpt.mkdir(parents=True, exist_ok=True)
    model = MctPretrain(pool.n_vocab, n_in, emb, hidden, n_heads, n_layers, norm_first=norm_first).to(device)
    on_gpu = device.startswith("cuda")
    amp = (torch.bfloat16 if has_native_bf16() else torch.float16) if on_gpu else None
    scaler = torch.amp.GradScaler(enabled=amp == torch.float16)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warmup if s < warmup else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, steps - warmup))))  # fmt: skip
    vrng = np.random.default_rng(seed + 1)
    t_val = time.time()
    val_bags = [pool.bag(int(va[i % len(va)]), vrng, max_genes) for i in range(n_val_bags)]
    log.info("%d validation bags in %.1fs", n_val_bags, time.time() - t_val)
    log.info("run %s: %d train / %d val samples, %d parameters, amp %s", cfg["run_id"], len(tr), len(va),
             sum(p.numel() for p in model.parameters()), amp)  # fmt: skip
    rows, best, t0 = [], np.inf, time.time()
    tl, ta = [], []
    for step in range(1, steps + 1):
        model.train()
        for _ in range(accumulate):
            x, tok = pool.bag(int(tr[rng.integers(len(tr))]), rng, max_genes)
            loss, acc = mgm_loss(model, x, tok, pool.n_vocab, gen, device, amp)
            scaler.scale(loss / accumulate).backward()
            tl.append(float(loss.detach()))
            ta.append(acc)
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        sched.step()
        if step in (1, 10, 100):
            log.info("step %d: %.2fs per step so far", step, (time.time() - t0) / step)
        if step == 500 or step % eval_every == 0 or step == steps:
            model.eval()
            vgen = torch.Generator().manual_seed(seed + 2)
            with torch.no_grad():
                v = [mgm_loss(model, x, tok, pool.n_vocab, vgen, device, amp) for x, tok in val_bags]
            vl, vacc = float(np.mean([float(a) for a, _ in v])), float(np.mean([b for _, b in v]))
            row = {"step": step, "train_loss": float(np.mean(tl)), "train_acc": float(np.mean(ta)), "val_loss": vl,
                   "val_acc": vacc, "lr": sched.get_last_lr()[0], "seconds": time.time() - t0}  # fmt: skip
            rows.append(row)
            tl, ta = [], []
            log.info("%s", row)
            pl.DataFrame(rows).write_parquet(run_dir / "metrics.parquet")
            save_checkpoint(ckpt / "last.ckpt", model, config, step)
            if vl < best:
                best = vl
                save_checkpoint(ckpt / "best.ckpt", model, config, step)
    status = {"run_id": cfg["run_id"], "failed": False, "best_val_loss": best, "final": rows[-1],
              "best_checkpoint": str(ckpt / "best.ckpt"), "seconds_per_step": (time.time() - t0) / steps}  # fmt: skip
    (run_dir / "status.json").write_text(json.dumps(status, indent=2))
    return status
