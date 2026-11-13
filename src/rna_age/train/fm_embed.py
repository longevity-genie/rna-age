"""RM22: frozen single-cell foundation-model embeddings as donor-age clock features (Geneformer V2-104M).

Why Geneformer and not Tahoe-x1 (the first choice): Tahoe-x1-70M ships with ``attn_impl: flash`` and ``amp_bf16``
on the MosaicML Composer / llm-foundry stack; our GPUs (11 GB, pre-Ampere) have neither bf16 nor FlashAttention-2.
Geneformer V2 (``ctheodoris/Geneformer``, Apache 2.0) is a plain BERT that runs in fp16 under ``transformers``.

Input format, as Geneformer's ``TranscriptomeTokenizer`` (V2) builds it: raw counts of each cell, Ensembl ids
collapsed through ``ensembl_mapping_dict_gc104M`` (duplicates summed), genes outside the token dictionary dropped,
``count / n_counts * 10,000 / gene's non-zero median``, non-zero genes ranked by that value (descending), truncated to
4,094 and framed by ``<cls>`` … ``<eos>``. Embedding as its ``EmbExtractor`` defaults (``emb_layer=-1``: the
second-to-last layer): the ``<cls>`` vector (``cls``, declared primary) and the mean over gene tokens (``mean``).

Per cohort (``embed``): a fixed-seed sample of primary cells per donor (the donor key of ``donor_profiles``):
≤ ``cap`` cells stratified by class (proportional allocation), and for blood ≤ ``cap`` CD4 T cells; the union is
embedded once and averaged to one vector per donor (``all``) and per donor × CD4 T (``CD4 T``).
``clock``: ridge and TabICLv2 on those vectors with ``sc_tabular.cross_validate`` (the same GroupKFold(5) by person,
the same donor order and, for CD4 T, the same donors as RM14's ``class:CD4 T``), compared with RM14/RM16's
pseudobulk predictions for the same donors (TabICLv2 2,000 genes and the best of ridge/enet/gbm) by a paired
bootstrap over persons.
"""

from __future__ import annotations

import logging
import pickle
import re
import time
from pathlib import Path

import h5py
import numpy as np
import polars as pl
import scipy.sparse as sp

from rna_age.paths import DOWNLOAD, INTERIM, OUTPUT

log = logging.getLogger(__name__)

REPO_ID = "ctheodoris/Geneformer"
MODEL_DIR = "Geneformer-V2-104M"
DICTS = {
    "token": "geneformer/token_dictionary_gc104M.pkl",
    "median": "geneformer/gene_median_dictionary_gc104M.pkl",
    "mapping": "geneformer/ensembl_mapping_dict_gc104M.pkl",
    "name": "geneformer/gene_name_id_dict_gc104M.pkl",
}
MAX_LEN = 4096
TARGET_SUM = 10_000
BLOOD = {
    "terekhova2023",
    "3faad104-2ab8-4434-816d-474d8d2641db",
    "c838aec3-03ef-4398-b882-0e3912abfff0",
    "218acb0f-9f2f-4f76-b90b-15a4b7c7f629",
}
CD4 = "CD4 T"
EMB_DIR = INTERIM / "fm_embed"


# ------------------------------------------------------------------ vocabulary


class _SafeUnpickler(pickle.Unpickler):
    """Downloaded pickles are untrusted: only plain containers and numpy scalars may be rebuilt."""

    ALLOWED = {
        ("numpy.core.multiarray", "scalar"),
        ("numpy._core.multiarray", "scalar"),
        ("numpy", "dtype"),
    }

    def find_class(self, module: str, name: str):
        if (module, name) in self.ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"refusing to load {module}.{name}")


def _load_dict(path: Path) -> dict:
    with open(path, "rb") as f:
        return _SafeUnpickler(f).load()


def load_vocab() -> dict[str, dict]:
    from huggingface_hub import hf_hub_download

    return {k: _load_dict(Path(hf_hub_download(REPO_ID, v))) for k, v in DICTS.items()}


def gene_map(var_ids: list[str], vocab: dict[str, dict]) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    """Columns of the data → model genes: a 0/1 matrix (n_var × n_genes, duplicates summed), each gene's token id
    and non-zero median. Symbols (no ``ENSG``) are looked up in the name → id dictionary first."""
    token, median, mapping, name = vocab["token"], vocab["median"], vocab["mapping"], vocab["name"]
    mapped = []
    for v in var_ids:
        e = re.sub(r"\.\d+$", "", v).upper() if v.upper().startswith("ENSG") else name.get(v)
        m = mapping.get(e) if e else None
        mapped.append(m if m in token and m in median else None)
    genes = sorted({m for m in mapped if m is not None})
    gix = {g: i for i, g in enumerate(genes)}
    rows = [i for i, m in enumerate(mapped) if m is not None]
    cols = [gix[mapped[i]] for i in rows]
    G = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(var_ids), len(genes)))
    return (G, np.array([token[g] for g in genes], dtype=np.int32),
            np.array([median[g] for g in genes], dtype=np.float64))  # fmt: skip


def tokenize(X: sp.csr_matrix, G: sp.csr_matrix, tokens: np.ndarray, medians: np.ndarray, cls: int, eos: int,
             max_len: int = MAX_LEN) -> list[np.ndarray]:  # fmt: skip
    """Rank-value encoding of each row of raw counts ``X`` (cells × data columns), Geneformer V2 style."""
    n_counts = np.asarray(X.sum(axis=1)).ravel()
    n_counts[n_counts == 0] = 1
    M = sp.csr_matrix(X @ G)
    M = sp.csr_matrix(sp.diags(TARGET_SUM / n_counts) @ M @ sp.diags(1 / medians))
    M.eliminate_zeros()
    M.sort_indices()
    out = []
    for i in range(M.shape[0]):
        a, b = M.indptr[i], M.indptr[i + 1]
        order = np.argsort(-M.data[a:b], kind="stable")
        ranked = tokens[M.indices[a:b][order]][: max_len - 2]
        out.append(np.concatenate([[cls], ranked, [eos]]).astype(np.int32))
    return out


# ------------------------------------------------------------------ cells


def sample_cells(cells: pl.DataFrame, cap: int, cd4_cap: int, seed: int = 0) -> pl.DataFrame:
    """``cells``: row, donor, cls. Per donor ≤ ``cap`` cells, class-proportional (largest remainder), flagged
    ``in_all``; ≤ ``cd4_cap`` CD4 T cells flagged ``in_cd4`` (0 = none). One row per distinct cell."""
    rng = np.random.default_rng(seed)
    pick_all, pick_cd4 = [], []
    for (_,), d in cells.sort("row").group_by("donor", maintain_order=True):
        rows, cls = d["row"].to_numpy(), d["cls"].to_numpy()
        if len(rows) <= cap:
            pick_all.append(rows)
        else:
            names, counts = np.unique(cls, return_counts=True)
            quota = counts / counts.sum() * cap
            take = np.floor(quota).astype(int)
            for j in np.argsort(-(quota - take))[: cap - take.sum()]:
                take[j] += 1
            for nm, k in zip(names, take, strict=True):
                if k:
                    pick_all.append(rng.choice(rows[cls == nm], k, replace=False))
        if cd4_cap:
            r = rows[cls == CD4]
            pick_cd4.append(r if len(r) <= cd4_cap else rng.choice(r, cd4_cap, replace=False))
    a = np.concatenate(pick_all) if pick_all else np.array([], dtype=np.int64)
    c = np.concatenate(pick_cd4) if pick_cd4 else np.array([], dtype=np.int64)
    sel = pl.DataFrame({"row": np.union1d(a, c).astype(np.int64)})
    return (
        sel.with_columns(in_all=pl.col("row").is_in(a.tolist()), in_cd4=pl.col("row").is_in(c.tolist()))
        .join(cells, on="row", how="left")
        .sort("row")
    )


def _cohort_cells(dataset_id: str) -> tuple[Path, pl.DataFrame, bool]:
    """(h5ad, cells table: row, donor, cls, age_ok; whether X holds log1p-CP10k)."""
    from rna_age.sources.bgee import stage_age
    from rna_age.sources.census_pseudobulk import _obs_column
    from rna_age.train.composition import broad_class

    blood = dataset_id in BLOOD
    if dataset_id == "terekhova2023":
        root = INTERIM / "terekhova2023" / "unpacked" / "all_pbmcs"
        h5ad = root / "all_pbmcs_rna.h5ad"
        with h5py.File(h5ad, "r") as f:
            barcodes = _obs_column(f["obs"], f["obs"].attrs.get("_index", "_index")).astype(str)
        meta = pl.read_csv(root / "all_pbmcs_metadata.csv", infer_schema_length=10000,
                           columns=["", "Tube_id", "Cluster_names"]).rename({"": "barcode"})  # fmt: skip
        cells = (
            pl.DataFrame({"barcode": barcodes})
            .with_row_index("row")
            .join(meta, on="barcode", how="left")
            .filter(pl.col("Tube_id").is_not_null() & pl.col("Cluster_names").is_not_null())
            .select(pl.col("row").cast(pl.Int64), donor=pl.col("Tube_id") + ":blood", ct=pl.col("Cluster_names"))
        )
        log_cp10k = True
    else:
        h5ad = DOWNLOAD / "census_pseudobulk" / "h5ads" / f"{dataset_id}.h5ad"
        with h5py.File(h5ad, "r") as f:
            cols = {c: _obs_column(f["obs"], c) for c in ("donor_id", "tissue", "cell_type", "development_stage")}
            n = len(cols["donor_id"])
            primary = (_obs_column(f["obs"], "is_primary_data") if "is_primary_data" in f["obs"]
                       else np.ones(n, dtype=bool)).astype(bool)  # fmt: skip
        stages = {s: stage_age(s)[1] for s in set(cols["development_stage"].astype(str))}
        ok = primary & np.array([stages[str(s)] is not None for s in cols["development_stage"]])
        cells = pl.DataFrame({
            "row": np.arange(n, dtype=np.int64),
            "donor": [f"{d}:{t}" for d, t in zip(cols["donor_id"].astype(str), cols["tissue"].astype(str), strict=True)],
            "ct": cols["cell_type"].astype(str),
        }).filter(pl.Series(ok))  # fmt: skip
        log_cp10k = False
    mapping = {c: broad_class(c) if blood else c for c in cells["ct"].unique().to_list()}
    return h5ad, cells.with_columns(cls=pl.col("ct").replace_strict(mapping)).drop("ct"), log_cp10k


def read_rows(h5ad: Path, rows: np.ndarray, window: int = 200_000):
    """Yield (rows, CSR of those rows) in windows of the file's row order (one pass over the compressed matrix)."""
    from rna_age.sources.census_pseudobulk import _var_ids

    with h5py.File(h5ad, "r") as f:
        raw = "raw" in f and "X" in f["raw"]
        X = f["raw/X"] if raw else f["X"]
        n_var = len(_var_ids(f, raw))
        indptr = X["indptr"][()]
        rows = np.sort(rows)
        for lo in range(0, len(indptr) - 1, window):
            r = rows[(rows >= lo) & (rows < lo + window)]
            if not len(r):
                continue
            a, b = int(indptr[r[0]]), int(indptr[r[-1] + 1])
            data, ind = X["data"][a:b], X["indices"][a:b]
            starts, ends = indptr[r] - a, indptr[r + 1] - a
            take = np.concatenate([np.arange(s, e) for s, e in zip(starts, ends, strict=True)])
            new_ptr = np.concatenate([[0], np.cumsum(ends - starts)])
            yield r, sp.csr_matrix((data[take].astype(np.float64), ind[take], new_ptr), shape=(len(r), n_var))


def var_ids(h5ad: Path) -> list[str]:
    from rna_age.sources.census_pseudobulk import _var_ids

    with h5py.File(h5ad, "r") as f:
        return _var_ids(f, "raw" in f and "X" in f["raw"])


# ------------------------------------------------------------------ model


def load_model(device: str = "cuda"):
    import torch
    from huggingface_hub import snapshot_download
    from transformers import BertModel

    path = snapshot_download(REPO_ID, allow_patterns=[f"{MODEL_DIR}/*"])
    model = BertModel.from_pretrained(f"{path}/{MODEL_DIR}", add_pooling_layer=False, attn_implementation="sdpa")
    return model.to(device).eval(), torch


def embed(token_lists: list[np.ndarray], model, torch, budget: int = 32_768, pad: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """``cls`` and gene-mean vectors of the second-to-last layer, fp16 autocast, batches of ≤ ``budget`` tokens."""
    n = len(token_lists)
    lens = np.array([len(t) for t in token_lists])
    order = np.argsort(-lens, kind="stable")
    d = model.config.hidden_size
    out_cls, out_mean = np.zeros((n, d), np.float32), np.zeros((n, d), np.float32)
    i, t0, last = 0, time.time(), time.time()
    while i < n:
        L = int(lens[order[i]])
        idx = order[i : i + max(1, budget // L)]
        ids = np.full((len(idx), L), pad, dtype=np.int64)
        for k, j in enumerate(idx):
            ids[k, : lens[j]] = token_lists[j]
        ids_t = torch.from_numpy(ids).cuda()
        mask = (torch.arange(L, device="cuda")[None, :] < torch.from_numpy(lens[idx]).cuda()[:, None]).long()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            h = model(input_ids=ids_t, attention_mask=mask, output_hidden_states=True).hidden_states[-2].float()
        gene = mask.clone()
        gene[:, 0] = 0
        gene[torch.arange(len(idx)), torch.from_numpy(lens[idx] - 1).cuda()] = 0  # the <eos> position
        out_cls[idx] = h[:, 0].cpu().numpy()
        out_mean[idx] = ((h * gene[..., None]).sum(1) / gene.sum(1, keepdim=True)).cpu().numpy()
        i += len(idx)
        if time.time() - last > 60 or i >= n:
            last = time.time()
            log.info("embedded %d / %d cells (%.1f cells/s)", i, n, i / (time.time() - t0))
    return out_cls, out_mean


def run_embed(dataset_id: str, cap: int = 200, cd4_cap: int = 200, seed: int = 0, budget: int = 32_768) -> Path:
    """Per donor (and donor × CD4 T for blood): mean Geneformer embeddings of a capped cell sample."""
    t0 = time.time()
    h5ad, cells, log_cp10k = _cohort_cells(dataset_id)
    sel = sample_cells(cells, cap, cd4_cap if dataset_id in BLOOD else 0, seed)
    log.info("%s: %d donors, %d cells sampled of %d (%.0fs)", dataset_id[:8], sel["donor"].n_unique(), sel.height,
             cells.height, time.time() - t0)  # fmt: skip
    vocab = load_vocab()
    G, tokens, medians = gene_map(var_ids(h5ad), vocab)
    log.info("%d of the data's genes map to %d model genes", G.nnz, G.shape[1])
    cls_id, eos_id = vocab["token"]["<cls>"], vocab["token"]["<eos>"]
    toks: dict[int, np.ndarray] = {}
    for r, X in read_rows(h5ad, sel["row"].to_numpy()):
        if log_cp10k:  # Terekhova's matrix is log1p-CP10k: expm1 is proportional to the counts of each cell
            X.data = np.expm1(X.data)
        toks.update(zip(r.tolist(), tokenize(X, G, tokens, medians, cls_id, eos_id), strict=True))
        log.info("tokenised %d / %d cells (%.0fs)", len(toks), sel.height, time.time() - t0)
    sel = sel.filter(pl.col("row").is_in(list(toks)))
    lens = np.array([len(toks[r]) for r in sel["row"].to_list()])
    log.info("tokens per cell: median %d, 90th pct %d, truncated %.1f%%", np.median(lens), np.percentile(lens, 90),
             100 * np.mean(lens >= MAX_LEN))  # fmt: skip
    model, torch = load_model()
    t1 = time.time()
    e_cls, e_mean = embed([toks[r] for r in sel["row"].to_list()], model, torch, budget=budget)
    rate = sel.height / (time.time() - t1)
    log.info("%s: %d cells embedded at %.1f cells/s", dataset_id[:8], sel.height, rate)
    rows = []
    for subset, flag in (("all", "in_all"), (CD4, "in_cd4")):
        part = sel.with_row_index("k").filter(pl.col(flag))
        if not part.height:
            continue
        for (donor,), d in part.group_by("donor", maintain_order=True):
            k = d["k"].to_numpy()
            rows.append({"donor": donor, "subset": subset, "n_cells": len(k),
                         **{f"cls_{j}": float(v) for j, v in enumerate(e_cls[k].mean(0))},
                         **{f"mean_{j}": float(v) for j, v in enumerate(e_mean[k].mean(0))}})  # fmt: skip
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    path = EMB_DIR / f"geneformer_{dataset_id[:8]}.parquet"
    pl.DataFrame(rows).write_parquet(path, compression="zstd")
    (EMB_DIR / f"geneformer_{dataset_id[:8]}.rate.txt").write_text(
        f"cells={sel.height} cells_per_s={rate:.2f} total_s={time.time() - t0:.0f}\n"
    )
    return path


# ------------------------------------------------------------------ clocks


def clock_folds(dataset_id: str) -> tuple[dict, list[tuple[str, str, np.ndarray]]]:
    """``donor_profiles`` of a cohort (donors, ages, CV groups) and its representations: (name, embedding subset,
    donor mask), RM14's donors for each."""
    from rna_age.train.composition import donor_profiles

    dp = donor_profiles(dataset_id, "broad" if dataset_id in BLOOD else "cell_type")
    reps = [("natural", "all", np.ones(len(dp["donors"]), dtype=bool))]
    if dataset_id in BLOOD and CD4 in dp["cix"]:
        reps.append((f"class:{CD4}", CD4, dp["N"][:, dp["cix"][CD4]] >= 20))  # sc_tabular's min_cells
    return dp, reps


def clock_oof(dataset_id: str, emb: pl.DataFrame, feats: tuple[str, ...], name, models: tuple[str, ...] = ("ridge", "tabicl"),
              seed: int = 0) -> pl.DataFrame:  # fmt: skip
    """Out-of-fold age predictions on per-donor embeddings (``emb``: donor, subset, ``<feat>_<j>`` columns), the same
    folds and donors as RM14 (sc_tabular); ``name(feat, model)`` gives the model column."""
    from rna_age.train.sc_tabular import cross_validate

    dp, reps = clock_folds(dataset_id)
    donors, age, groups = np.array(dp["donors"]), dp["age"], dp["groups"]
    rows = []
    for rep, subset, ok in reps:
        e = pl.DataFrame({"donor": donors[ok]}).join(emb.filter(pl.col("subset") == subset), on="donor", how="left")
        for feat in feats:
            X = e.select([c for c in emb.columns if c.startswith(f"{feat}_")]).to_numpy()
            if np.isnan(X).any():
                raise ValueError(f"{dataset_id[:8]} {rep}: {np.isnan(X).any(1).sum()} donors without an embedding")
            pred = cross_validate(X, age[ok], groups[ok], models, n_genes=X.shape[1], seed=seed)
            for m, p in pred.items():
                rows += [{"cohort": dataset_id[:8], "representation": rep, "model": name(feat, m),
                          "donor": d, "group": int(g), "age": float(a), "pred": float(v)}
                         for d, g, a, v in zip(donors[ok], groups[ok], age[ok], p, strict=True)]  # fmt: skip
            log.info("%s %s %s: %s", dataset_id[:8], rep, feat,
                     {m: round(float(np.abs(p - age[ok]).mean()), 2) for m, p in pred.items()})  # fmt: skip
    return pl.DataFrame(rows)


def run_clock(dataset_id: str, models: tuple[str, ...] = ("ridge", "tabicl"), seed: int = 0) -> pl.DataFrame:
    """Out-of-fold age predictions on the Geneformer embeddings, the same folds and donors as RM14 (sc_tabular)."""
    emb = pl.read_parquet(EMB_DIR / f"geneformer_{dataset_id[:8]}.parquet")
    out = clock_oof(dataset_id, emb, ("cls", "mean"), lambda f, m: f"geneformer-{f}-{m}", models, seed)
    path = OUTPUT / "summary" / "fm_embedding_oof.parquet"
    if path.exists():
        out = pl.concat([pl.read_parquet(path).filter(pl.col("cohort") != dataset_id[:8]), out], how="vertical_relaxed")
    out.sort("cohort", "representation", "model", "donor").write_parquet(path, compression="zstd")
    return out


def paired_bootstrap(e: np.ndarray, e_ref: np.ndarray, groups: np.ndarray, n_boot: int = 2000, seed: int = 0):
    """Mean of ``e - e_ref`` with a 95 % CI from resampling ``groups`` (persons) with replacement."""
    rng = np.random.default_rng(seed)
    diff = e - e_ref
    ug, inv = np.unique(groups, return_inverse=True)
    s, c = np.bincount(inv, diff, len(ug)), np.bincount(inv, minlength=len(ug))
    boots = []
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, len(ug), len(ug)), minlength=len(ug))
        boots.append((w * s).sum() / (w * c).sum())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(diff.mean()), float(lo), float(hi)


def compare(oof: pl.DataFrame, sc_dir: Path | None = None, n_boot: int = 2000) -> pl.DataFrame:
    """Each embedding clock against the pseudobulk clocks on the same donors: ``tabicl-2000g`` and the best (lowest
    MAE) of ridge/enet/gbm per cohort and representation; paired bootstrap over persons."""
    sc_dir = sc_dir or OUTPUT / "summary"
    rows = []
    for (cohort, rep), d in oof.group_by("cohort", "representation", maintain_order=True):
        sc = pl.read_parquet(sc_dir / f"sc_tabular_{cohort}.parquet").filter(pl.col("representation") == rep)
        mae = {m: float((g["pred"] - g["age"]).abs().mean()) for (m,), g in sc.group_by("model")}
        base = min(("ridge", "enet", "gbm"), key=lambda m: mae.get(m, np.inf))
        refs = {"pseudobulk-tabicl-2000g": "tabicl-2000g", f"pseudobulk-{base}": base}
        for (m,), dm in d.group_by("model", maintain_order=True):
            dm = dm.sort("donor")
            e = (dm["pred"] - dm["age"]).abs().to_numpy()
            for ref_name, ref_model in refs.items():
                r = dm.select("donor").join(sc.filter(pl.col("model") == ref_model).select("donor", "age", "pred"),
                                            on="donor", how="left")  # fmt: skip
                if r["pred"].null_count():
                    raise ValueError(f"{cohort} {rep}: {ref_model} lacks {r['pred'].null_count()} donors")
                e_ref = (r["pred"] - r["age"]).abs().to_numpy()
                diff, lo, hi = paired_bootstrap(e, e_ref, dm["group"].to_numpy(), n_boot)
                rows.append({"cohort": cohort, "representation": rep, "model": m, "n": len(e),
                             "mae": float(e.mean()), "r": float(np.corrcoef(dm["pred"], dm["age"])[0, 1]),
                             "ref": ref_name, "ref_mae": float(e_ref.mean()), "diff": diff, "lo": lo, "hi": hi,
                             "beats_ref": hi < 0})  # fmt: skip
    return pl.DataFrame(rows)


def write_summary() -> pl.DataFrame:
    oof = pl.read_parquet(OUTPUT / "summary" / "fm_embedding_oof.parquet")
    out = compare(oof)
    out.write_parquet(OUTPUT / "summary" / "fm_embedding_clock.parquet", compression="zstd")
    return out
