"""Rung 3: frozen BulkFormer-Atlas CLS embeddings (1,024-d) for every sample with a split.

Follows BulkFormer's input guide (docs/input-guide.md at cec8b90):
- raw counts of **all** of a source's genes go through ``prepare_atlas_input`` (normalisation on the
  41,623-gene axis with the released gene lengths → log2(TPM+1)), unavailable genes = -10;
- one fixed panel of 4,096 coding genes, chosen **on training samples only** with ``select_variable_genes``
  over genes observed in every source (the model needs the panel observed in every sample), saved with the
  embeddings: ``data/interim/embeddings/atlas_panel.parquet``;
- native CLS before the contrastive projection (``extract_atlas``), bf16 when the GPU supports it, else our
  own fp16-autocast loop over ``model.encode`` (V100 has no bf16) or fp32 (``RNA_AGE_PRECISION``).

Output: ``data/interim/embeddings/atlas/<source>/shard-NNN.parquet`` (``sample_id`` + ``e0000..e1023``).
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import numpy as np
import polars as pl

from rna_age.counts import read_samples
from rna_age.paths import DATA, INPUT, INTERIM

log = logging.getLogger(__name__)

MODEL_DIR = DATA / "download" / "models" / "bulkformer"
EMB = INTERIM / "embeddings"
PANEL = EMB / "atlas_panel.parquet"


def _frame(source: str, ids: list[str]):
    """samples × genes pandas frame of raw counts (pandas only at BulkFormer's boundary)."""
    import pandas as pd

    from rna_age.train.features import symbol_map

    genes, mat = read_samples(INPUT / source, ids)
    if not genes[0].startswith("ENSG"):
        smap = symbol_map()
        keep = [i for i, g in enumerate(genes) if g in smap]
        ens = [smap[genes[i]] for i in keep]
        frame = pd.DataFrame(mat[:, keep].astype(np.float32), index=ids, columns=ens)
        return frame.T.groupby(level=0).sum().T  # several symbols → one Ensembl ID
    return pd.DataFrame(mat.astype(np.float32), index=ids, columns=genes)


def _vocab():
    import pandas as pd

    d = MODEL_DIR / "atlas"
    return pd.read_csv(d / "gene_vocab.csv"), pd.read_csv(d / "normalization_vocab.csv")


def make_panel(n_genes: int = 4096, per_source: int = 1500) -> np.ndarray:
    from bulkformer import prepare_atlas_input, select_variable_genes

    vocab, norm = _vocab()
    m = pl.read_parquet(INTERIM / "samples_all.parquet")
    train = m.filter((pl.col("split") == "train") & ~pl.col("is_pseudobulk") & (pl.col("is_healthy") != False))  # noqa: E712
    blocks = []
    for (source,), sdf in train.group_by("source"):
        ids = sdf.sample(min(per_source, sdf.height), seed=0)["sample_id"].to_list()
        x, _ = prepare_atlas_input(_frame(source, ids), vocab, norm, input_scale="counts")
        blocks.append(x)
        log.info("panel input %s: %d samples", source, len(ids))
    x = np.vstack(blocks)
    idx = select_variable_genes(x, n_genes)
    panel = pl.DataFrame({"index": idx.astype(np.int64), "gene_id": vocab.ensembl_gene.to_numpy()[idx]})
    EMB.mkdir(parents=True, exist_ok=True)
    panel.write_parquet(PANEL)
    log.info("Atlas panel: %d genes from %d training samples", len(idx), len(x))
    return idx


def _encode_fp16(model, values: np.ndarray, selected: np.ndarray, batch_size: int) -> np.ndarray:
    import torch

    device = next(model.parameters()).device
    ids = torch.as_tensor(np.r_[model.num_genes, selected], dtype=torch.long, device=device)
    out = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for start in range(0, len(values), batch_size):
            x = torch.as_tensor(values[start : start + batch_size, selected], device=device)
            x = torch.cat([x.new_zeros(len(x), 1), x], dim=1)
            gid = ids[None].expand(len(x), -1)
            hidden = model.encode(gid, x, torch.ones_like(gid, dtype=torch.bool))
            out.append(hidden[:, 0].float().cpu().numpy())
    return np.concatenate(out)


def embed(sources: list[str], shard: int = 0, n_shards: int = 1, chunk: int = 256, batch_size: int = 8) -> None:
    import torch
    from bulkformer import extract_atlas, load_atlas, prepare_atlas_input

    from rna_age.train.atlas_patch import patch_atlas_attention

    patch_atlas_attention()

    panel = pl.read_parquet(PANEL)["index"].to_numpy() if PANEL.exists() else make_panel()
    model, vocab, norm = load_atlas(str(MODEL_DIR / "atlas"), device="cuda")
    # fp16 by default: on pre-Ampere cards bf16 is reported as supported but emulated (slower)
    precision = os.environ.get("RNA_AGE_PRECISION") or (
        "bf16" if torch.cuda.get_device_capability(0)[0] >= 8 else "fp16"
    )
    log.info("Atlas on %s, precision %s, batch %d", torch.cuda.get_device_name(0), precision, batch_size)
    m = pl.read_parquet(INTERIM / "samples_all.parquet").filter(pl.col("split").is_not_null())
    for source in sources:
        # evaluation samples first (val/test), then train in a fixed random order, so a partial run is usable
        sdf = m.filter(pl.col("source") == source).with_columns(
            _order=pl.col("split").replace_strict({"test": 0, "val": 1, "train": 2}, default=3),
            _rand=pl.col("sample_id").hash(seed=7),
        )
        ids_all = sdf.sort("_order", "_rand")["sample_id"].to_list()
        out_dir = EMB / "atlas" / source
        out_dir.mkdir(parents=True, exist_ok=True)
        # only samples without an embedding yet (sources grow when parsers are rerun); file names carry the
        # shard layout so runs with different layouts never collide
        done: set[str] = set()
        for f in out_dir.glob("*.parquet"):
            done.update(pl.read_parquet(f, columns=["sample_id"])["sample_id"].to_list())
        ids_all = [i for i in ids_all if i not in done]
        ids = ids_all[shard::n_shards]
        tag = f"n{n_shards}-s{shard:02d}-{len(done)}"
        log.info("%s: %d already embedded, %d to do in this shard", source, len(done), len(ids))
        for c, start in enumerate(range(0, len(ids), chunk)):
            path = out_dir / f"shard-{tag}-{c:04d}.parquet"
            if path.exists() and path.with_suffix(".inputs.npz").exists():
                continue
            t0 = time.time()
            part = ids[start : start + chunk]
            frame = _frame(source, part)
            matched = frame.columns.isin(norm.ensembl_gene)
            empty = frame.loc[:, matched].sum(axis=1) <= 0
            if empty.any():  # all-zero over the normalisation genes: cannot be normalised, skip (logged)
                log.warning("%s: skipping %d samples with zero matched library: %s", source, int(empty.sum()),
                            list(frame.index[empty])[:5])  # fmt: skip
                frame = frame.loc[~empty]
                part = list(frame.index)
            x, cov = prepare_atlas_input(frame, vocab, norm, input_scale="counts")
            # panel genes a source lacks (pseudobulk gene sets) count as not detected (0), recorded per sample
            sub = x[:, panel]
            missing = (sub < 0).mean(axis=1)
            x[:, panel] = np.where(sub < 0, 0.0, sub)
            if precision == "fp16":
                emb = _encode_fp16(model, x, panel, batch_size)
                if c == 0:  # fp16 sanity check against fp32 on a few samples
                    ref = extract_atlas(model, x[:2], panel, batch_size=1, precision="fp32", progress=False)
                    cos = (ref * emb[:2]).sum(1) / np.linalg.norm(ref, axis=1) / np.linalg.norm(emb[:2], axis=1)
                    log.info("fp16 vs fp32 CLS cosine: %s", np.round(cos, 5).tolist())
            else:
                emb = extract_atlas(model, x, panel, batch_size=batch_size, precision=precision, progress=False)
            df = pl.DataFrame({"sample_id": part, "panel_missing": missing}).with_columns(
                [pl.Series(f"e{i:04d}", emb[:, i]) for i in range(emb.shape[1])]
            )
            df.write_parquet(path)
            # the panel inputs themselves (log2 TPM+1, fp16) for fine-tuning without re-normalising counts
            np.savez(path.with_suffix(".inputs.npz"), sample_id=np.array(part), x=x[:, panel].astype(np.float16))
            log.info("%s: %d/%d samples (%.1f s/chunk)", source, start + len(part), len(ids), time.time() - t0)


def load_embeddings(model: str = "atlas") -> pl.DataFrame:
    paths = sorted(Path(EMB / model).rglob("shard-*.parquet"))
    return pl.concat([pl.read_parquet(p) for p in paths]).unique("sample_id")
