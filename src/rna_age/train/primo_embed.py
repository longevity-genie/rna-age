"""Embed the PRIMO bulk datasets with BulkFormer-Atlas (frozen or one of our age fine-tunes).

Protocol follows the BulkFormer authors' PRIMO entry (benchmarks/primo/README.md at cec8b90) so that the frozen
model reproduces their setting: ``prepare_atlas_input`` on the full matrix (counts → length-normalised TPM on
the 41,623-gene axis, log2), one fixed panel of 4,096 positive-variance coding genes per dataset chosen without
labels on the whole dataset (transductive, as theirs; disclosed), native CLS before the contrastive projection.
Differences: fp16 autocast instead of bf16 (our GPUs emulate bf16; fp16 vs fp32 cosine ≈ 1), and gene IDs are
mapped NCBI symbol → Ensembl through ARCHS4's gene table (coverage logged per dataset).

Fine-tuned models: our checkpoint's unfrozen blocks + output norm are loaded on top of the released weights; the
embedding is the same CLS vector (the age head is not used).

Output: ``data/output/primo/submission_<name>.parquet`` (``dataset_id``, ``sample_id``, ``e0000``…), the format
``primo evaluate`` / ``primo submit`` read. Nothing is submitted from here.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import polars as pl
import yaml

from rna_age.paths import INTERIM, OUTPUT
from rna_age.sources.primo import data_dir
from rna_age.train.embed import MODEL_DIR, _encode_fp16

log = logging.getLogger(__name__)


def _frame(path: Path, smap: dict[str, str]):
    import anndata as ad
    import pandas as pd

    a = ad.read_h5ad(path)
    X = a.X.toarray() if hasattr(a.X, "toarray") else np.asarray(a.X)
    sym = a.var["gene_symbols"].astype(str).to_numpy()
    ens = np.array([smap.get(s, "") for s in sym])
    keep = ens != ""
    frame = pd.DataFrame(X[:, keep].astype(np.float32), index=a.obs_names.astype(str), columns=ens[keep])
    frame = frame.T.groupby(level=0).sum().T  # several NCBI genes on one Ensembl ID
    return frame, float(keep.mean())


def embed(model_name: str = "atlas", checkpoint: str | None = None, batch_size: int = 4, n_genes: int = 4096) -> Path:
    import torch
    from bulkformer import load_atlas, prepare_atlas_input, select_variable_genes

    from rna_age.train.atlas_patch import patch_atlas_attention
    from rna_age.train.features import symbol_map

    patch_atlas_attention()
    model, vocab, norm = load_atlas(str(MODEL_DIR / "atlas"), device="cuda")
    if checkpoint:
        ck = Path(checkpoint)
        ck = ck if ck.suffix == ".pt" else INTERIM / "checkpoints" / checkpoint / "best.pt"
        state = torch.load(ck, map_location="cuda", weights_only=False)
        missing = model.load_state_dict(state["model"], strict=False)
        log.info("loaded %d fine-tuned tensors from %s (%d unchanged)", len(state["model"]), ck,
                 len(missing.missing_keys))  # fmt: skip
    model.eval()
    smap = symbol_map()
    entries = [e for e in yaml.safe_load((data_dir() / "datasets.yaml").read_text())["datasets"]
               if e["modality"] == "bulk RNA"]  # fmt: skip
    blocks, report = [], []
    for e in entries:
        t0 = time.time()
        frame, cov = _frame(data_dir() / e["path"], smap)
        x, _ = prepare_atlas_input(frame, vocab, norm, input_scale="counts")
        k = min(n_genes, int(((x >= 0).all(0) & (x.var(0) > 0)).sum()))
        panel = select_variable_genes(x, k)
        emb = _encode_fp16(model, x, panel, batch_size)
        df = pl.DataFrame({"dataset_id": [e["id"]] * len(frame), "sample_id": list(frame.index)}).with_columns(
            [pl.Series(f"e{i:04d}", emb[:, i].astype(np.float64)) for i in range(emb.shape[1])]
        )
        blocks.append(df)
        report.append({"dataset_id": e["id"], "samples": len(frame), "symbol_coverage": cov, "panel_genes": k,
                       "input_normalization": e.get("input_normalization", "raw counts"), "seconds": time.time() - t0})  # fmt: skip
        log.info("PRIMO %s: %d samples, symbol coverage %.3f, panel %d (%.0fs)", e["id"], len(frame), cov, k,
                 time.time() - t0)  # fmt: skip
    out_dir = OUTPUT / "primo"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"submission_{model_name}.parquet"
    pl.concat(blocks).write_parquet(path)
    pl.DataFrame(report).write_parquet(out_dir / f"embedding_report_{model_name}.parquet")
    log.info("wrote %s", path)
    return path
