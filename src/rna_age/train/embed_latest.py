"""BulkFormer-Latest frozen embeddings (768-d): full coding transcriptome in (19,973 positions, raw counts →
``prepare_latest_input``), two sample vectors out:

- ``s*``: the model's own pooled sample embedding (mean over all observed genes, ``extract_latest``);
- ``p*``: mean of the contextual gene embeddings over the Atlas panel genes present in Latest's vocabulary
  (panel chosen on training samples only), as the input guide recommends for HVG-style pooling.

fp16 autocast on pre-Ampere GPUs (bf16 is emulated there), checked against fp32 on the first chunk. Evaluation
samples (test, val, transfer) first, then training samples in a fixed random order; only samples without an
embedding are processed, so jobs resume and can be stopped at any time.
Output: ``data/interim/embeddings/latest/<source>/shard-*.parquet``.
"""

from __future__ import annotations

import logging
import os
import time

import numpy as np
import polars as pl

from rna_age.paths import INTERIM
from rna_age.train.embed import EMB, MODEL_DIR, PANEL, _frame

log = logging.getLogger(__name__)


def _encode(model, x: np.ndarray, sel: np.ndarray, precision: str) -> tuple[np.ndarray, np.ndarray]:
    import torch

    dev = next(model.parameters()).device
    s_out, p_out = [], []
    ctx = torch.autocast("cuda", dtype=torch.float16) if precision == "fp16" else torch.autocast("cuda", enabled=False)
    with torch.inference_mode(), ctx:
        for i in range(len(x)):
            xt = torch.as_tensor(x[i : i + 1], device=dev)
            genes = model(xt, return_gene_embeddings=True)["gene_embeddings"].float()
            valid = xt.ne(-10)
            s_out.append(((genes * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True)).cpu().numpy())
            vsel = valid[:, sel]
            p_out.append(
                ((genes[:, sel] * vsel.unsqueeze(-1)).sum(1) / vsel.sum(1, keepdim=True).clamp(min=1)).cpu().numpy()
            )
    return np.concatenate(s_out), np.concatenate(p_out)


def embed(sources: list[str], shard: int = 0, n_shards: int = 1, chunk: int = 64, max_train: int = 15000) -> None:
    import torch
    from bulkformer import load_latest, prepare_latest_input

    model, vocab = load_latest(str(MODEL_DIR / "latest"), device="cuda")
    precision = os.environ.get("RNA_AGE_PRECISION") or (
        "bf16" if torch.cuda.get_device_capability(0)[0] >= 8 else "fp16"
    )
    panel_genes = set(pl.read_parquet(PANEL)["gene_id"].to_list())
    sel = np.array([i for i, g in enumerate(vocab.ensembl_gene.to_numpy()) if g in panel_genes], dtype=np.int64)
    log.info(
        "Latest on %s, precision %s; %d panel genes in Latest vocabulary",
        torch.cuda.get_device_name(0),
        precision,
        len(sel),
    )
    m = pl.read_parquet(INTERIM / "samples_all.parquet").filter(pl.col("split").is_not_null())
    for source in sources:
        sdf = (
            m.filter(pl.col("source") == source)
            .with_columns(
                _order=pl.col("split").replace_strict({"test": 0, "val": 1, "train": 2}, default=0),
                _rand=pl.col("sample_id").hash(seed=11),
            )
            .sort("_order", "_rand")
        )
        # cap training samples: Latest costs ~20k tokens per sample
        sdf = pl.concat([sdf.filter(pl.col("_order") != 2), sdf.filter(pl.col("_order") == 2).head(max_train)])
        out_dir = EMB / "latest" / source
        out_dir.mkdir(parents=True, exist_ok=True)
        done: set[str] = set()
        for f in out_dir.glob("*.parquet"):
            done.update(pl.read_parquet(f, columns=["sample_id"])["sample_id"].to_list())
        # shard the full fixed ordering first, then skip what is done, so shards started at different times
        # never overlap or leave gaps
        ids = [i for i in sdf["sample_id"].to_list()[shard::n_shards] if i not in done]
        tag = f"n{n_shards}-s{shard:02d}-{len(done)}"
        log.info("%s: %d done, %d to do in this shard", source, len(done), len(ids))
        for c, start in enumerate(range(0, len(ids), chunk)):
            t0 = time.time()
            part = ids[start : start + chunk]
            frame = _frame(source, part)
            frame = frame.loc[frame.sum(axis=1) > 0]
            part = list(frame.index)
            x, _ = prepare_latest_input(frame, vocab, input_scale="counts")
            s_emb, p_emb = _encode(model, x, sel, precision)
            if c == 0 and precision == "fp16":
                s32, _ = _encode(model, x[:1], sel, "fp32")
                cos = float((s32[0] * s_emb[0]).sum() / np.linalg.norm(s32[0]) / np.linalg.norm(s_emb[0]))
                log.info("fp16 vs fp32 Latest sample-embedding cosine: %.5f", cos)
            df = pl.DataFrame({"sample_id": part}).with_columns(
                [pl.Series(f"s{i:03d}", s_emb[:, i]) for i in range(s_emb.shape[1])]
                + [pl.Series(f"p{i:03d}", p_emb[:, i]) for i in range(p_emb.shape[1])]
            )
            df.write_parquet(out_dir / f"shard-{tag}-{c:04d}.parquet")
            log.info("%s: %d/%d samples (%.1f s/chunk)", source, start + len(part), len(ids), time.time() - t0)
