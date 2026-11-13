"""Rung 5 (docs/DECISIONS.md D2, D3): a bidirectional KronLSTM over expression tokens, trained from scratch.

Two encodings of one sample, both built from statistics of the training split only:

- ``rank`` (Geneformer's rank-value encoding): CPM divided by each gene's median non-zero CPM over training
  samples, genes sorted by that score, the top ``length`` gene ids as tokens (0 pads a sample with fewer
  expressed genes). Library size and absolute level are gone; only the order is left.
- ``genome``: the HVGs (chosen on training samples) along the genome (the GTEx gene table's GTF order), each
  token a gene embedding plus a projection of its standardised log-CPM. ``genome-shuffled`` is the same genes in
  one fixed random order: the ablation that says whether the genome order matters.

The model: gene embedding (+ value) → ``n_layers`` bidirectional KronLSTM blocks (kron-fm's ``kron_core``, read from a kron-fm clone at run time:
``$RNA_AGE_KRON_FM``, else ``../kron-fm`` beside this repository. Not a package dependency: kron-fm's own uv
sources pin torch to another CUDA build, which would move this project's torch. Its kernels need CUDA; the tests
swap in its CPU stand-in) → mean over real tokens →
RMSNorm → MLP head → age in years. Loss as the MLP baseline: Huber (δ 5 y) on exact ages and GTEx midpoints,
training samples of all health states. Early stopping on validation exact-age MAE; ``last.pt`` / ``best.pt``
under ``data/interim/checkpoints/<run>/``; a requeued job resumes its own unfinished run.

The ablation ``rank_mlp`` (D3) trains the baseline MLP on the rank encoding as a dense vector (each gene's score
``1 - position / length`` when in the top ``length``, else 0), so the encoder is separated from the encoding.
"""

from __future__ import annotations

import logging
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

from rna_age.paths import INPUT, INTERIM, REPO_ROOT
from rna_age.schema import age_mid
from rna_age.train.baselines import PRED_COLS
from rna_age.train.features import load_matrix, log_cpm, sample_table, select_hvg, shared_genes
from rna_age.train.runs import finish_run, new_run

log = logging.getLogger(__name__)

ENCODINGS = ("rank", "genome", "genome-shuffled")


# ---------------------------------------------------------------- tokenizers (numpy, testable on the CPU)


def cpm(counts: np.ndarray) -> np.ndarray:
    lib = counts.sum(axis=1, keepdims=True)
    return (counts / np.where(lib > 0, lib, 1) * 1e6).astype(np.float32)


def nonzero_medians(train_cpm: np.ndarray) -> np.ndarray:
    """Per gene, the median of its non-zero CPM over the given (training) rows; 1 where a gene is never seen,
    so an unseen gene neither wins nor divides by zero."""
    x = np.where(train_cpm > 0, train_cpm, np.nan)
    import warnings

    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)  # an all-zero gene: its median is NaN, set to 1 below
        med = np.nanmedian(x, axis=0)
    return np.where(np.isfinite(med) & (med > 0), med, 1.0).astype(np.float32)


def rank_tokens(sample_cpm: np.ndarray, medians: np.ndarray, length: int, chunk: int = 2048) -> np.ndarray:
    """``[n, length]`` int32 gene tokens (gene index + 1), highest normalised expression first; 0 pads.
    Ties are broken by gene index, so the encoding is deterministic."""
    n, g = sample_cpm.shape
    out = np.zeros((n, length), dtype=np.int32)
    k = min(length, g)
    for a in range(0, n, chunk):
        score = sample_cpm[a : a + chunk] / medians
        # stable sort of -score: equal scores keep gene order
        order = np.argsort(-score, axis=1, kind="stable")[:, :k]
        top = np.take_along_axis(score, order, axis=1)
        tok = (order + 1).astype(np.int32)
        tok[top <= 0] = 0  # unexpressed genes are padding, never tokens
        out[a : a + chunk, :k] = tok
    return out


def rank_dense(tokens: np.ndarray, n_genes: int) -> np.ndarray:
    """The rank encoding as a dense ``[n, n_genes]`` vector for the MLP ablation: ``1 - position / length``."""
    n, length = tokens.shape
    out = np.zeros((n, n_genes), dtype=np.float32)
    score = (1 - np.arange(length) / length).astype(np.float32)
    rows, pos = np.nonzero(tokens)
    out[rows, tokens[rows, pos] - 1] = score[pos]
    return out


def genome_order(genes: list[str]) -> list[int]:
    """Indices of ``genes`` in genome order (the GTEx gene table is in GTF order); genes it lacks go last, sorted."""
    ref = pl.read_parquet(INPUT / "gtex" / "genes.parquet", columns=["gene_id"])["gene_id"].to_list()
    rank = {g: i for i, g in enumerate(ref)}
    return sorted(range(len(genes)), key=lambda i: (rank.get(genes[i], len(ref)), genes[i]))


# ---------------------------------------------------------------- the model (torch; kron_core on demand)


def kron_fm_src() -> Path:
    """kron-fm's ``src`` on ``sys.path`` (``$RNA_AGE_KRON_FM`` or ``../kron-fm``), so ``kron_core`` imports."""
    root = Path(os.environ.get("RNA_AGE_KRON_FM", REPO_ROOT.parent / "kron-fm"))
    src = root / "src"
    if not (src / "kron_core").is_dir():
        raise FileNotFoundError(
            f"no kron-fm clone at {root}: clone github.com/antonkulaga/kron-fm there or set RNA_AGE_KRON_FM"
        )
    if str(src) not in sys.path:
        sys.path.append(str(src))
    return src


def mask_id(n_genes: int) -> int:
    """The [MASK] token: one past the last gene (genes are 1..n_genes, 0 pads)."""
    return n_genes + 1


def mask_tokens(ids: np.ndarray, frac: float, rng: np.random.Generator, mask: int) -> tuple[np.ndarray, np.ndarray]:
    """Masked-gene targets (RM9): ``frac`` of each row's real tokens (at least one) become ``mask``; returns the
    masked ids and the target ids (0 = no loss at that position). Padding is never masked."""
    masked, target = ids.copy(), np.zeros_like(ids)
    real = ids > 0
    pick = (rng.random(ids.shape) < frac) & real
    for r in np.flatnonzero(real.any(1) & ~pick.any(1)):  # every row with tokens contributes at least one
        pick[r, rng.choice(np.flatnonzero(real[r]))] = True
    target[pick] = ids[pick]
    masked[pick] = mask
    return masked, target


def build_model(n_genes: int, d_model: int, n_layers: int, n_heads: int, dropout: float, values: bool):
    import torch

    kron_fm_src()
    from kron_core.bidirectional import BiKronLSTMBlock
    from kron_core.segments import cu_seqlens_from_doc_ids

    class KronAge(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.n_genes = n_genes
            self.gene = torch.nn.Embedding(n_genes + 2, d_model, padding_idx=0)  # 0 pad, 1..n genes, n+1 [MASK]
            self.value = torch.nn.Linear(1, d_model) if values else None
            self.blocks = torch.nn.ModuleList(
                BiKronLSTMBlock(d_model, n_heads, 4 * d_model, dropout, i) for i in range(n_layers)
            )
            self.norm = torch.nn.RMSNorm(d_model, eps=1e-5)
            self.head = torch.nn.Sequential(
                torch.nn.Linear(d_model, d_model), torch.nn.GELU(), torch.nn.Linear(d_model, 1)
            )

        def encode(self, ids: torch.Tensor, vals: torch.Tensor | None = None) -> torch.Tensor:
            """``[B, T]`` tokens -> ``[B, T, D]`` states."""
            batch, steps = ids.shape
            h = self.gene(ids)
            if self.value is not None and vals is not None:
                h = h + self.value(vals.unsqueeze(-1))
            # each row's real tokens are one segment and its padding another, so no real token reads padding
            row = torch.arange(batch, device=ids.device).unsqueeze(1).expand(batch, steps)
            cu = cu_seqlens_from_doc_ids(torch.where(ids > 0, row, -1 - row))
            for blk in self.blocks:
                h = blk(h, cu)
            return h

        def forward(self, ids: torch.Tensor, vals: torch.Tensor | None = None) -> torch.Tensor:
            h = self.encode(ids, vals)
            real = (ids > 0).unsqueeze(-1).to(h.dtype)
            pooled = (h * real).sum(1) / real.sum(1).clamp(min=1)
            return self.head(self.norm(pooled)).squeeze(-1)

    return KronAge()


# ---------------------------------------------------------------- data


def _design(train_set: str = "all") -> tuple[pl.DataFrame, np.ndarray, np.ndarray]:
    """Samples, the training mask and the validation mask, exactly as the 20k-HVG MLP baseline builds them."""
    s = sample_table()
    s = s.with_columns(target=age_mid(s))
    s = s.with_columns(split=pl.when(pl.col("is_pseudobulk")).then(pl.lit("transfer")).otherwise(pl.col("split")))
    train = (
        (s["split"] == "train").to_numpy()
        & ((s["is_healthy"].fill_null(True)).to_numpy() | (train_set == "all"))
        & ((s["age_type"] == "exact").to_numpy() | (s["source"] == "gtex").to_numpy())
    )
    val = (s["split"] == "val").to_numpy() & (s["age_type"] == "exact").to_numpy()
    return s, train, val


def encode(s: pl.DataFrame, train: np.ndarray, encoding: str, length: int, n_hvg: int, seed: int = 0) -> dict:
    """Token ids (and values) for every sample, with every statistic taken from training rows only."""
    genes = shared_genes()
    counts = load_matrix(s.select("sample_id", "source"), genes)
    sub = np.flatnonzero(train)[: min(20000, int(train.sum()))]
    if encoding == "rank":
        c = cpm(counts)
        del counts
        med = nonzero_medians(c[sub])
        ids = rank_tokens(c, med, length)
        return {"ids": ids, "vals": None, "genes": genes, "n_genes": len(genes)}
    lc = log_cpm(counts)
    del counts
    hvg = select_hvg(lc[sub], genes, n_hvg)
    gi = [genes.index(g) for g in hvg]
    order = genome_order(hvg)
    if encoding == "genome-shuffled":
        order = list(np.random.default_rng(seed + 12345).permutation(len(hvg)))
    cols = np.array(gi)[order]
    x = lc[:, cols]
    del lc
    mu, sd = x[sub].mean(0), x[sub].std(0) + 1e-6
    vals = ((x - mu) / sd).astype(np.float16)
    ids = np.broadcast_to(np.arange(1, len(cols) + 1, dtype=np.int32), (len(x), len(cols)))
    return {"ids": ids, "vals": vals, "genes": [hvg[i] for i in order], "n_genes": len(cols)}


# ---------------------------------------------------------------- training


def micro_batches(rows: np.ndarray, micro: int) -> list[np.ndarray]:
    """``rows`` split into consecutive pieces of at most ``micro`` (0 = one piece)."""
    if micro <= 0 or micro >= len(rows):
        return [rows]
    return [rows[a : a + micro] for a in range(0, len(rows), micro)]


def _resume_dir(name: str) -> Path | None:
    for cand in sorted((INTERIM / "checkpoints").glob(f"*-{name}"), reverse=True):
        run = INTERIM.parent / "output" / "runs" / cand.name
        if (cand / "last.pt").exists() and not (run / "predictions.parquet").exists():
            return cand
    return None


def train(
    encoding: str = "rank",
    length: int = 2048,
    n_hvg: int = 20000,
    d_model: int = 256,
    n_layers: int = 4,
    n_heads: int = 4,
    dropout: float = 0.1,
    epochs: int = 30,
    patience: int = 4,
    batch: int = 32,
    micro: int = 0,
    lr: float = 3e-4,
    seed: int = 0,
    train_set: str = "all",
    run_name: str | None = None,
    init: str | None = None,
) -> dict:
    """``init``: a ``kron-pretrain`` run id whose encoder starts the model (its encoding, length and shape win).
    ``micro``: rows per forward pass (0 = ``batch``); gradients accumulate to one step per ``batch`` rows, so a
    long context fits a small card at the same effective batch and schedule."""
    import json

    import torch

    if init:
        pre = json.loads((PRETRAIN_RUNS / init / "config.json").read_text())
        encoding, length = "rank", pre["length"]
        d_model, n_layers, n_heads = pre["d_model"], pre["n_layers"], pre["n_heads"]
    if encoding not in ENCODINGS:
        raise ValueError(f"encoding must be one of {ENCODINGS}, got {encoding!r}")
    t0 = time.time()
    torch.manual_seed(seed)
    dev = "cuda"
    s, tr_mask, va_mask = _design(train_set)
    enc = encode(s, tr_mask, encoding, length, n_hvg, seed)
    ids, vals = enc["ids"], enc["vals"]
    y = np.nan_to_num(s["target"].to_numpy(), nan=30.0).astype(np.float32)
    va_y = s["target"].to_numpy()[va_mask]
    tr_rows, va_rows = np.flatnonzero(tr_mask), np.flatnonzero(va_mask)

    model = build_model(enc["n_genes"], d_model, n_layers, n_heads, dropout, values=vals is not None).to(dev)
    if init:
        enc_state = torch.load(INTERIM / "checkpoints" / init / "encoder.pt", map_location=dev)
        missing, unexpected = model.load_state_dict(enc_state, strict=False)
        if unexpected or any(not k.startswith(("head.", "norm.")) for k in missing):
            raise ValueError(f"pretrained encoder does not fit: missing {missing}, unexpected {unexpected}")
        log.info("encoder from pretraining run %s", init)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    steps_per_epoch = math.ceil(len(tr_rows) / batch)
    total, warm = steps_per_epoch * epochs, min(1000, steps_per_epoch)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda st: min(1.0, (st + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(st, total) / total))
    )

    seq = length if encoding == "rank" else enc["n_genes"]
    name = run_name or (
        f"kron-{encoding}-{seq}-d{d_model}x{n_layers}" + ("-pre" if init else "") + (f"-seed{seed}" if seed else "")
    )
    ckpt_dir, run_dir, start, best, bad = _resume_dir(name), None, 0, np.inf, 0
    if ckpt_dir is not None:
        st = torch.load(ckpt_dir / "last.pt", map_location=dev, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        start, best, bad, run_dir = st["epoch"] + 1, st["best"], st["bad"], Path(st["run_dir"])
        log.info("resumed %s at epoch %d (best val MAE %.2f)", name, start, best)
    if run_dir is None:
        run_dir = new_run(
            name,
            {"model": f"bidirectional KronLSTM {n_layers}x{d_model} ({n_heads} heads), from scratch",
             "features": (f"rank tokens, top {length} genes by CPM / train non-zero median" if encoding == "rank"
                          else f"{enc['n_genes']} HVG log-CPM tokens in {'genome' if encoding == 'genome' else 'shuffled'} order"),
             "train_set": "bulk train, all health states, exact + GTEx midpoints" if train_set == "all"
                          else "bulk train, healthy-or-unknown, exact + GTEx midpoints",
             "encoding": encoding, "length": seq, "d_model": d_model, "n_layers": n_layers, "n_heads": n_heads,
             "dropout": dropout, "epochs": epochs, "patience": patience, "batch": batch, "micro": micro or batch, "lr": lr, "seed": seed,
             "n_params": n_params, "n_train": int(len(tr_rows)), "precision": "bf16 autocast", "init": init},
        )  # fmt: skip
        ckpt_dir = INTERIM / "checkpoints" / run_dir.name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    def batch_tensors(rows: np.ndarray):
        i = torch.as_tensor(np.ascontiguousarray(ids[rows]), device=dev)
        v = None if vals is None else torch.as_tensor(vals[rows], device=dev).float()
        return i, v

    def predict(rows: np.ndarray, bs: int = 64) -> np.ndarray:
        model.eval()
        out = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for a in range(0, len(rows), bs):
                out.append(model(*batch_tensors(rows[a : a + bs])).float().cpu().numpy())
        return np.concatenate(out) if out else np.array([], dtype=np.float32)

    for epoch in range(start, epochs):
        model.train()
        perm = np.random.default_rng(seed * 1000 + epoch).permutation(tr_rows)
        te = time.time()
        for a in range(0, len(perm), batch):
            rows = perm[a : a + batch]
            opt.zero_grad(set_to_none=True)
            for part in micro_batches(rows, micro):
                i, v = batch_tensors(part)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pred = model(i, v).float()
                loss = torch.nn.functional.huber_loss(pred, torch.as_tensor(y[part], device=dev), delta=5.0)
                (loss * len(part) / len(rows)).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        mae = float(np.mean(np.abs(predict(va_rows) - va_y)))
        log.info("epoch %d: val MAE %.2f (%.0f s)", epoch, mae, time.time() - te)
        if mae < best - 0.01:
            best, bad = mae, 0
            torch.save(model.state_dict(), ckpt_dir / "best.pt")
        else:
            bad += 1
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": epoch, "best": best, "bad": bad, "run_dir": str(run_dir)}, ckpt_dir / "last.pt")  # fmt: skip
        if bad >= patience:
            break

    model.load_state_dict(torch.load(ckpt_dir / "best.pt", map_location=dev))
    pred = predict(np.arange(s.height))
    out = s.select([c for c in PRED_COLS if c in s.columns]).with_columns(pred=pl.Series(pred.astype(np.float64)))
    return finish_run(run_dir, out, {"seconds": time.time() - t0, "n_params": n_params})


def train_rank_mlp(length: int = 2048, seed: int = 0, train_set: str = "all") -> dict:
    """D3's ablation: the baseline MLP on the rank encoding as a dense vector, same samples and splits."""
    from sklearn.preprocessing import StandardScaler

    from rna_age.train.probes import _mlp

    t0 = time.time()
    s, tr, va = _design(train_set)
    enc = encode(s, tr, "rank", length, 0, seed)
    X = rank_dense(enc["ids"], enc["n_genes"])
    keep = X[tr].any(axis=0)  # genes that ever reach the top in training
    X = StandardScaler().fit(X[tr][:, keep]).transform(X[:, keep]).astype(np.float32)
    y_years = s["target"].to_numpy()
    pred, best = _mlp(X, np.nan_to_num(y_years, nan=30.0), tr, va, seed=seed, val_y=y_years[va])
    out = s.select([c for c in PRED_COLS if c in s.columns]).with_columns(pred=pl.Series(pred.astype(np.float64)))
    run_dir = new_run(
        f"rank{length}-mlp-trainall" + (f"-seed{seed}" if seed else ""),
        {"model": "mlp probe (ablation of the KronLSTM rank arm)",
         "features": f"rank encoding, top {length} genes, dense ({int(keep.sum())} genes ever in the top)",
         "train_set": "bulk train, all health states, exact + GTEx midpoints", "seed": seed, "val_mae": best,
         "n_train": int(tr.sum())},
    )  # fmt: skip
    return finish_run(run_dir, out, {"seconds": time.time() - t0})


# ---------------------------------------------------------------- masked-gene pretraining (RM9, D3)

PRETRAIN_RUNS = INTERIM.parent / "output" / "pretrain"


def pretrain(
    length: int = 2048,
    d_model: int = 256,
    n_layers: int = 4,
    n_heads: int = 4,
    dropout: float = 0.1,
    mask_frac: float = 0.15,
    epochs: int = 20,
    patience: int = 3,
    batch: int = 32,
    micro: int = 0,
    lr: float = 3e-4,
    monitor_frac: float = 0.05,
    seed: int = 0,
    run_name: str | None = None,
) -> dict:
    """Masked-gene prediction over rank tokens of training-split samples only (D3: never a validation or test
    study, ages unused). A share of training *studies* (``monitor_frac``) is held out to stop on; the encoder
    (everything but the age head) is saved as ``data/interim/checkpoints/<run>/encoder.pt`` for ``kron --init``.
    The tokenizer statistics are the age runs' own (same training rows), so the vocabulary and order agree."""
    import json
    from datetime import datetime

    import torch

    t0 = time.time()
    torch.manual_seed(seed)
    dev = "cuda"
    s, tr_mask, _ = _design("all")
    enc = encode(s, tr_mask, "rank", length, 0, seed)
    ids, n_genes = enc["ids"], enc["n_genes"]
    studies = np.array(sorted(set(s.filter(pl.Series(tr_mask))["study_id"].to_list())))
    rng = np.random.default_rng(seed)
    held = set(rng.choice(studies, max(1, int(len(studies) * monitor_frac)), replace=False).tolist())
    in_held = s["study_id"].is_in(list(held)).to_numpy()
    fit_rows, mon_rows = np.flatnonzero(tr_mask & ~in_held), np.flatnonzero(tr_mask & in_held)
    mask = mask_id(n_genes)

    model = build_model(n_genes, d_model, n_layers, n_heads, dropout, values=False).to(dev)
    lm = torch.nn.Linear(d_model, n_genes + 1).to(dev)  # predicts the gene id (0 unused) at masked positions
    params = list(model.parameters()) + list(lm.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-2)
    steps = math.ceil(len(fit_rows) / batch) * epochs
    warm = min(1000, steps // 10 + 1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda st: min(1.0, (st + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(st, steps) / steps))
    )
    name = run_name or f"kron-pre-rank-{length}-d{d_model}x{n_layers}" + (f"-seed{seed}" if seed else "")
    ckpt = None
    for cand in sorted((INTERIM / "checkpoints").glob(f"*-{name}"), reverse=True):
        if (cand / "last.pt").exists() and not (cand / "encoder.pt").exists():
            ckpt = cand
            break
    start, best, bad = 0, np.inf, 0
    if ckpt is not None:
        st = torch.load(ckpt / "last.pt", map_location=dev, weights_only=False)
        model.load_state_dict(st["model"])
        lm.load_state_dict(st["lm"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        start, best, bad = st["epoch"] + 1, st["best"], st["bad"]
        run_id = ckpt.name
    else:
        run_id = f"{datetime.now().strftime('%Y%m%d-%H%M')}-{name}"
        ckpt = INTERIM / "checkpoints" / run_id
    run_dir = PRETRAIN_RUNS / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(
        {"run_id": run_id, "length": length, "d_model": d_model, "n_layers": n_layers, "n_heads": n_heads,
         "dropout": dropout, "mask_frac": mask_frac, "epochs": epochs, "batch": batch, "micro": micro or batch, "lr": lr, "seed": seed,
         "n_genes": n_genes, "n_fit": int(len(fit_rows)), "n_monitor": int(len(mon_rows)),
         "monitor_studies": len(held), "corpus": "bulk train split, all health states (ages unused)",
         "checkpoint_dir": str(ckpt)}, indent=2))  # fmt: skip

    def step_loss(rows: np.ndarray, r: np.random.Generator) -> tuple[torch.Tensor, int, int]:
        m, tgt = mask_tokens(ids[rows], mask_frac, r, mask)
        it = torch.as_tensor(m, device=dev)
        tt = torch.as_tensor(tgt, device=dev, dtype=torch.long)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = model.encode(it)
        sel = tt > 0
        logits = lm(h[sel].float())
        loss = torch.nn.functional.cross_entropy(logits, tt[sel])
        return loss, int((logits.argmax(-1) == tt[sel]).sum()), int(sel.sum())

    hist_file = run_dir / "history.parquet"
    history = pl.read_parquet(hist_file).to_dicts() if start and hist_file.exists() else []
    for epoch in range(start, epochs):
        model.train()
        perm = np.random.default_rng(seed * 1000 + epoch).permutation(fit_rows)
        r = np.random.default_rng(seed * 7919 + epoch)
        for a in range(0, len(perm), batch):
            rows = perm[a : a + batch]
            opt.zero_grad(set_to_none=True)
            for part in micro_batches(rows, micro):
                loss, _, _ = step_loss(part, r)
                (loss * len(part) / len(rows)).backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
        model.eval()
        tot, hit, n = 0.0, 0, 0
        r = np.random.default_rng(12345)  # the same masks every epoch, so the monitor loss is comparable
        with torch.no_grad():
            for a in range(0, len(mon_rows), 64):
                loss, h_, n_ = step_loss(mon_rows[a : a + 64], r)
                tot, hit, n = tot + float(loss) * n_, hit + h_, n + n_
        mon = tot / max(n, 1)
        history.append({"epoch": epoch, "monitor_loss": mon, "monitor_acc": hit / max(n, 1)})
        log.info("pretrain epoch %d: monitor loss %.3f, top-1 %.3f", epoch, mon, hit / max(n, 1))
        if mon < best - 1e-3:
            best, bad = mon, 0
            torch.save({k: v for k, v in model.state_dict().items() if not k.startswith(("head.", "norm."))},
                       ckpt / "best_encoder.pt")  # fmt: skip
        else:
            bad += 1
        torch.save({"model": model.state_dict(), "lm": lm.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": epoch, "best": best, "bad": bad}, ckpt / "last.pt")  # fmt: skip
        pl.DataFrame(history).write_parquet(hist_file)
        if bad >= patience:
            break
    (ckpt / "best_encoder.pt").rename(ckpt / "encoder.pt")
    out = {"run_id": run_id, "best_monitor_loss": best, "seconds": time.time() - t0}
    (run_dir / "result.json").write_text(json.dumps(out, indent=2))
    return out


# ---------------------------------------------------------------- the paired bar (D3)


def compare(runs_a: list[str], runs_b: list[str], n_boot: int = 2000, seed: int = 0) -> dict:
    """Mean validation exact-age MAE of runs A minus runs B (each arm averaged over its seeds, per sample), with a
    95 % CI from resampling validation studies. Negative means A is better. Also per source."""
    from rna_age.train.runs import RUNS

    def mean_pred(runs: list[str]) -> pl.DataFrame:
        parts = [pl.read_parquet(RUNS / r / "predictions.parquet", columns=["sample_id", "pred"]) for r in runs]
        return pl.concat(parts).group_by("sample_id").agg(pl.col("pred").mean())

    base = pl.read_parquet(RUNS / runs_a[0] / "predictions.parquet")
    base = base.filter((pl.col("split") == "val") & (pl.col("age_type") == "exact"))
    # only samples eligible in the current table, so runs from before a re-merge compare on the same set
    base = base.join(sample_table().select("sample_id"), on="sample_id", how="semi")
    d = base.select("sample_id", "study_id", "source", "age_years")
    d = d.join(mean_pred(runs_a).rename({"pred": "a"}), on="sample_id").join(
        mean_pred(runs_b).rename({"pred": "b"}), on="sample_id"
    )
    d = d.with_columns(diff=(pl.col("a") - pl.col("age_years")).abs() - (pl.col("b") - pl.col("age_years")).abs())
    return paired_ci(d, n_boot, seed)


def paired_ci(d: pl.DataFrame, n_boot: int = 2000, seed: int = 0) -> dict:
    """``d`` has ``study_id``, ``source`` and ``diff`` (|error A| - |error B|) per sample: the mean difference and
    its 95 % CI by resampling studies with replacement, overall and per source."""
    rng = np.random.default_rng(seed)

    def one(t: pl.DataFrame) -> dict:
        per = t.group_by("study_id").agg(pl.col("diff").sum().alias("s"), pl.len().alias("n")).sort("study_id")
        s_, n_ = per["s"].to_numpy(), per["n"].to_numpy()
        boots = []
        for _ in range(n_boot):
            k = rng.integers(0, len(s_), len(s_))
            boots.append(s_[k].sum() / n_[k].sum())
        lo, hi = np.percentile(boots, [2.5, 97.5])
        return {"diff": float(s_.sum() / n_.sum()), "lo": float(lo), "hi": float(hi), "n": int(n_.sum()),
                "studies": len(s_)}  # fmt: skip

    out = {"all": one(d)}
    for (src,), t in d.sort("source").group_by("source", maintain_order=True):  # fixed order: one rng
        out[str(src)] = one(t)
    return out
