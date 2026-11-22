"""RM37: MCT (Multicellular Transformer) for donor age, a frozen read-out and an age fine-tune.

MCT (Wiatrak, Andersson et al., bioRxiv 10.64898/2026.11.14.766251; github.com/almaan/mct, Apache-2.0) reads a sample
as a bag of 1,024 cells packed into fixed slots per coarse cell type. Its tokens are genes: each carries its log1p
CP10k expression across the bag's cells (a 1,024-vector), embedded by a 2-layer MLP and added to a gene-id embedding,
then a 12-layer encoder (width 64). Genes expressed in more than 1 % of the bag's cells are kept, at most 12k.

The encoder is re-implemented here with MCT's parameter names (``MctEncoder``; the original is
``cell_cell_interactions.modeling.modeling.MulticellularTransformerEncoder``, Apache-2.0), so the checkpoint loads into
our environment without MCT's pinned dependencies (torch 2.3, Lightning 2.3). The checkpoint is a downloaded pickle:
``read_checkpoint`` reads it with torch's weights-only unpickler, MCT's own classes rebuilt as inert stubs.

Inputs, under ``data/interim/mct/`` (git-ignored, put there by the runner):
- ``multicellular_transformer.ckpt``: the Lightning checkpoint;
- ``index/``: ``ensemble2idx.json`` (Ensembl id → token), ``gene_symbol2idx.json``, ``cell_type_input_order.json``
  (54 coarse types → slot ranges over 1,024), ``cell_type_map.json`` (CELLxGENE label → coarse type), all from
  MCT's ``tutorials/data``.

Step (a), ``readout``: 4 bags per donor; per bag a 64-vector per gene token; averaged over bags. Features: the mean
over genes (``mctmean``), and per-gene vectors for genes in ≥ 90 % of the cohort's donors with 500 columns picked by
|r| with age inside each training fold (``mctgene``). Ridge and TabICLv2, RM14's folds.
Step (b), ``finetune``: the encoder's top ``n_train_layers`` layers and a linear head on the mean over gene tokens,
MSE on standardised age, one fresh bag per training donor per epoch (≤ ``max_genes`` genes), test = mean over 4 bags.
"""

from __future__ import annotations

import importlib
import json
import logging
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import scipy.sparse as sp
import torch
from sklearn.model_selection import GroupKFold
from torch import nn

from rna_age.paths import INTERIM
from rna_age.train.sc_pretrain import EVAL_COHORTS, SUMMARY, TEREKHOVA, _merge_summary

log = logging.getLogger(__name__)

MCT_DIR = INTERIM / "mct"
CKPT = MCT_DIR / "multicellular_transformer.ckpt"
INDEX = MCT_DIR / "index"
N_SPECIAL = 2  # [PAD], [MASK]
BAG = 1024
OTHER = "Other"
IN_CORPUS = {"3faad104", "c838aec3", "218acb0f"}  # OneK1K (sc_eqtl), AIDA, lupus: in MCT's pretraining list
REFERENCE = "recipe-stack_agesel-tabicl"  # RM35's best single-cell clock, 5.23 y

# Terekhova's own cluster names (no CELLxGENE labels) → MCT's coarse types
TEREKHOVA_TYPES = {
    "CD4+ T cells": "CD4-positive T cells",
    "TRAV1-2- CD8+ T cells": "CD8-positive T cells",
    "NK cells": "NK cells",
    "B cells": "B cells",
    "Myeloid cells": "Monocyte",
    "gd T cells": "Gamma-delta T cells",
    "MAIT cells": "T cells",
    "DN T cells": "T cells",
    "Progenitor cells": "Progenitor and stem cells",
}


# ------------------------------------------------------------------ checkpoint


class _Stub:
    """An inert stand-in for a class pickled in the checkpoint (MCT's config dataclass): it keeps the state, runs
    nothing."""

    def __setstate__(self, state):
        self.__dict__.update(state if isinstance(state, dict) else {"_state": state})


def _stub(qualified: str) -> type:
    mod, _, name = qualified.rpartition(".")
    parts = mod.split(".")
    for i in range(1, len(parts) + 1):
        sys.modules.setdefault(".".join(parts[:i]), types.ModuleType(".".join(parts[:i])))
    m = sys.modules[mod]
    if not isinstance(getattr(m, name, None), type):
        setattr(m, name, type(name, (_Stub,), {"__module__": mod, "__qualname__": name}))
    return getattr(m, name)


SAFE_BUILTINS = {"builtins.set", "builtins.frozenset", "builtins.slice", "builtins.complex"}


def _real(qualified: str):
    mod, _, name = qualified.rpartition(".")
    return getattr(importlib.import_module(mod), name)


def read_checkpoint(path: Path) -> dict:
    """The checkpoint dict, read with torch's weights-only unpickler: numpy scalars and dtypes are allowed as
    themselves, every other foreign global (MCT's and Lightning's classes) becomes an inert ``_Stub``."""
    allow = [type(np.dtype(t)) for t in ("f2", "f4", "f8", "i1", "i2", "i4", "i8", "u1", "u2", "u4", "u8", "?")]
    for g in torch.serialization.get_unsafe_globals_in_checkpoint(str(path)):
        if g.startswith("numpy.") or g in SAFE_BUILTINS:
            allow.append(_real(g))
        elif g.startswith(("builtins.", "torch.", "collections.", "copyreg.", "functools.", "os.", "subprocess.")):
            raise ValueError(f"checkpoint needs {g}, which the weights-only reader does not allow")
        else:
            allow.append(_stub(g))
    with torch.serialization.safe_globals(allow):
        return torch.load(str(path), map_location="cpu", weights_only=True)


def config_of(ck: dict) -> dict:
    """MCT's config as a dict (from the Lightning ``hyper_parameters``), empty when absent."""
    cfg = (ck.get("hyper_parameters") or {}).get("config")
    return dict(getattr(cfg, "__dict__", {}) or (cfg if isinstance(cfg, dict) else {}))


# ------------------------------------------------------------------ model


class MctEncoder(nn.Module):
    """MCT's encoder with its parameter names; no tissue token (MCT's ``embed`` passes none without a tissue key)."""

    def __init__(self, n_genes: int, n_input: int = BAG, emb: int = 64, hidden: int = 256, n_heads: int = 4,
                 n_layers: int = 12, n_tissues: int = 0, dropout: float = 0.2, transformer_dropout: float = 0.1,
                 pad: int = 0):  # fmt: skip
        super().__init__()
        self.gene_id_emb = nn.Embedding(n_genes, emb, padding_idx=pad)
        if n_tissues > 0:
            self.tissue_emb = nn.Embedding(n_tissues, emb, padding_idx=pad)
        self.gene_expression_encoder = nn.Sequential(
            nn.Linear(n_input, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, emb)
        )
        self.gene_expression_mask_emb = nn.Embedding(1, emb)
        self.layer_norm = nn.LayerNorm(emb)
        self.dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(emb, n_heads, emb * 4, transformer_dropout, activation="gelu",
                                           batch_first=True)  # fmt: skip
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)

    def forward(self, x: torch.Tensor, gene_ids: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        """``x``: batch × genes × cells (log1p CP10k), ``gene_ids``: batch × genes; returns batch × genes × emb."""
        h = self.gene_id_emb(gene_ids) + self.gene_expression_encoder(x)
        return self.encoder(self.dropout(self.layer_norm(h)), src_key_padding_mask=pad_mask)


def load_encoder(path: Path = CKPT) -> tuple[MctEncoder, dict]:
    """The pretrained encoder (shapes from the weights, heads from the config) and the config."""
    ck = read_checkpoint(path)
    cfg = config_of(ck)
    sd = ck.get("state_dict", ck)
    enc = {k.removeprefix("model.encoder."): v for k, v in sd.items() if k.startswith("model.encoder.")}
    if not enc:
        raise ValueError(f"{path}: no model.encoder.* weights")
    n_genes, emb = enc["gene_id_emb.weight"].shape
    hidden, n_input = enc["gene_expression_encoder.0.weight"].shape
    n_layers = 1 + max(int(k.split(".")[2]) for k in enc if k.startswith("encoder.layers."))
    n_tissues = enc["tissue_emb.weight"].shape[0] if "tissue_emb.weight" in enc else 0
    model = MctEncoder(n_genes, n_input, emb, hidden, int(cfg.get("n_heads", 4)), n_layers, n_tissues,
                       float(cfg.get("dropout", 0.2)), float(cfg.get("transformer_dropout", 0.1)),
                       int(cfg.get("padding_token_idx", 0)))  # fmt: skip
    model.load_state_dict(enc, strict=True)
    return model, cfg


class MctAge(nn.Module):
    """The encoder, a mean over gene tokens and a linear age head (on standardised age)."""

    def __init__(self, encoder: MctEncoder):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.layer_norm.normalized_shape[0], 1)

    def forward(self, x, gene_ids, pad_mask=None):
        h = self.encoder(x, gene_ids, pad_mask)
        if pad_mask is None:
            pooled = h.mean(1)
        else:
            keep = (~pad_mask).unsqueeze(-1).to(h.dtype)
            pooled = (h * keep).sum(1) / keep.sum(1).clamp(min=1)
        return self.head(pooled).squeeze(-1)


def freeze_below(model: MctAge, n_train_layers: int) -> None:
    """Train the top ``n_train_layers`` encoder layers and the head; freeze the rest."""
    for p in model.encoder.parameters():
        p.requires_grad = False
    layers = model.encoder.encoder.layers
    for layer in layers[len(layers) - n_train_layers:]:
        for p in layer.parameters():
            p.requires_grad = True


# ------------------------------------------------------------------ index and cells


@dataclass
class Index:
    gene2idx: dict[str, int]
    symbol2idx: dict[str, int]
    order: dict[str, tuple[int, int]]
    ctmap: dict[str, str]

    @classmethod
    def load(cls, d: Path = INDEX) -> Index:
        r = lambda n: json.loads((d / n).read_text())  # noqa: E731
        order = {k: (int(a), int(b)) for k, (a, b) in r("cell_type_input_order.json").items()}
        return cls(r("ensemble2idx.json"), r("gene_symbol2idx.json"), order, r("cell_type_map.json"))

    def slot_type(self, label: str, cohort: str = "") -> str:
        """A cell label's coarse type: itself if it is one, else MCT's map, Terekhova's own table, else Other."""
        if cohort == TEREKHOVA:
            return TEREKHOVA_TYPES.get(label, OTHER)
        if label in self.order:
            return label
        t = self.ctmap.get(label, OTHER)
        return t if t in self.order else OTHER


@dataclass
class DonorCells:
    """A cohort's pooled cells on MCT's vocabulary: ``X`` (cells × columns, log1p CP10k, float32 CSR), ``gene_ids``
    (token per column), per donor the pool rows (``rows[d]``) and their slot type index (``types[d]``)."""

    X: sp.csr_matrix
    gene_ids: np.ndarray
    donors: list[str]
    rows: list[np.ndarray]
    types: list[np.ndarray]
    type_names: list[str]
    order: dict[str, tuple[int, int]]


def pool_cells(cells: pl.DataFrame, index: Index, cohort: str, factor: int, rng: np.random.Generator) -> pl.DataFrame:
    """Per donor and slot type at most ``factor`` × the type's slot count cells (all when fewer): a bag never uses
    more than the slot count, so the pool bounds memory and still lets bags differ."""
    t = cells.with_columns(slot=pl.col("ct").map_elements(lambda c: index.slot_type(c, cohort), return_dtype=pl.Utf8))
    keep = []
    for (_, slot), d in t.group_by("donor", "slot", maintain_order=True):
        rows = d["row"].to_numpy()
        a, b = index.order[slot]
        n = factor * (b - a)
        keep.append(rows if len(rows) <= n else rng.choice(rows, n, replace=False))
    sel = np.sort(np.concatenate(keep)) if keep else np.array([], dtype=np.int64)
    return t.filter(pl.col("row").is_in(sel.tolist())).sort("row")


def vocab_columns(h5ad_ids: list[str | None], index: Index, symbols: bool) -> tuple[np.ndarray, np.ndarray]:
    """(file columns, their tokens) for the file's genes in MCT's vocabulary (first occurrence of a token)."""
    table = index.symbol2idx if symbols else index.gene2idx
    cols, toks, seen = [], [], set()
    for i, g in enumerate(h5ad_ids):
        k = table.get(g) if g is not None else None
        if k is not None and k >= N_SPECIAL and k not in seen:
            seen.add(k)
            cols.append(i)
            toks.append(k)
    return np.array(cols, dtype=np.int64), np.array(toks, dtype=np.int64)


def donor_cells(cohort: str, donors: list[str], index: Index, factor: int = 4, seed: int = 0) -> DonorCells:
    """Read a cohort's pooled cells (``pool_cells``, RM14's donors) on MCT's vocabulary, log1p CP10k over all of
    the file's genes."""
    import h5py

    from rna_age.sources.census_pseudobulk import _var_ids
    from rna_age.train.fm_embed import _cohort_cells, read_rows
    from rna_age.train.sc_pretrain import log_cp10k

    h5ad, cells, log_input = _cohort_cells(cohort, keep_ct=True)
    cells = cells.filter(pl.col("donor").is_in(donors))
    sel = pool_cells(cells, index, cohort, factor, np.random.default_rng(seed))
    with h5py.File(h5ad, "r") as f:
        raw = "raw" in f and "X" in f["raw"]
        ids = [str(x) for x in _var_ids(f, raw)]
    symbols = cohort == TEREKHOVA
    if not symbols:
        ids = [x.split(".")[0] if x.startswith("ENSG") else None for x in ids]
    cols, toks = vocab_columns(ids, index, symbols)
    parts, got = [], []
    for r, X in read_rows(h5ad, sel["row"].to_numpy()):
        if log_input:
            X.data = np.expm1(X.data)
        parts.append(log_cp10k(X)[:, cols].astype(np.float32))
        got.append(r)
    X = sp.vstack(parts).tocsr() if parts else sp.csr_matrix((0, len(cols)), dtype=np.float32)
    order = np.concatenate(got) if got else np.array([], dtype=np.int64)
    if not np.array_equal(order, sel["row"].to_numpy()):
        raise ValueError(f"{cohort[:8]}: rows came back out of order")
    names = list(index.order)
    tix = {n: i for i, n in enumerate(names)}
    pos = {d: [] for d in donors}
    for i, d in enumerate(sel["donor"].to_list()):
        pos[d].append(i)
    slot = np.array([tix[s] for s in sel["slot"].to_list()], dtype=np.int64)
    rows = [np.array(pos[d], dtype=np.int64) for d in donors]
    return DonorCells(X, toks, list(donors), rows, [slot[r] for r in rows], names, dict(index.order))


def pack_bag(types_: np.ndarray, order: dict[str, tuple[int, int]], names: list[str],
             rng: np.random.Generator) -> np.ndarray:  # fmt: skip
    """MCT's stratified packing: for each coarse type its slot range filled with the donor's cells of that type
    (sampled without replacement when more, padded with -1 when fewer). Returns pool positions (or -1) per slot."""
    out = np.full(sum(b - a for a, b in order.values()), -1, dtype=np.int64)
    by = {t: np.flatnonzero(types_ == i) for i, t in enumerate(names)}
    for t, (a, b) in order.items():
        have = by.get(t, np.array([], dtype=np.int64))
        n = b - a
        pick = rng.choice(have, n, replace=False) if len(have) >= n else rng.permutation(have)
        out[a:a + len(pick)] = pick
    return out


def bag_tensor(dc: DonorCells, d: int, rng: np.random.Generator, max_genes: int, min_frac: float = 0.01
               ) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """One bag of donor ``d``: (genes × slots expression, gene tokens), genes expressed in more than
    ``int(min_frac × 1024)`` of the bag's cells, a random ``max_genes`` of them when more (MCT's rule)."""
    slots = pack_bag(dc.types[d], dc.order, dc.type_names, rng)
    filled = slots >= 0
    sub = dc.X[dc.rows[d][slots[filled]]]
    nexp = np.bincount(sub.indices, minlength=sub.shape[1])
    genes = np.flatnonzero(nexp > int(min_frac * len(slots)))
    if len(genes) > max_genes:
        genes = np.sort(rng.choice(genes, max_genes, replace=False))
    dense = np.zeros((len(slots), len(genes)), dtype=np.float32)
    dense[filled] = sub[:, genes].toarray()
    return dense.T, dc.gene_ids[genes]


# ------------------------------------------------------------------ step (a): frozen read-out


@torch.no_grad()
def embed_donors(model: MctEncoder, dc: DonorCells, n_bags: int = 4, max_genes: int = 12_000, seed: int = 0,
                 device: str = "cuda") -> tuple[np.ndarray, dict[int, np.ndarray], np.ndarray]:  # fmt: skip
    """Per donor the mean over bags of (i) the mean over gene tokens (donors × emb) and (ii) per token its vector;
    returns (means, {token: donors × emb with NaN where a donor lacks it}, token counts per donor)."""
    rng = np.random.default_rng(seed)
    model.eval().to(device)
    emb = model.layer_norm.normalized_shape[0]
    means = np.zeros((len(dc.donors), emb), dtype=np.float32)
    acc: dict[int, np.ndarray] = {}
    cnt: dict[int, np.ndarray] = {}
    n_tok = np.zeros(len(dc.donors), dtype=np.int64)
    for d in range(len(dc.donors)):
        for _ in range(n_bags):
            x, tok = bag_tensor(dc, d, rng, max_genes)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
                h = model(torch.from_numpy(x)[None].to(device), torch.from_numpy(tok)[None].to(device))[0]
            h = h.float().cpu().numpy()
            means[d] += h.mean(0) / n_bags
            for t, v in zip(tok.tolist(), h, strict=True):
                if t not in acc:
                    acc[t] = np.zeros((len(dc.donors), emb), dtype=np.float32)
                    cnt[t] = np.zeros(len(dc.donors), dtype=np.int32)
                acc[t][d] += v
                cnt[t][d] += 1
            n_tok[d] += len(tok)
        n_tok[d] //= n_bags
    per_gene = {}
    for t, a in acc.items():
        c = cnt[t]
        v = np.full_like(a, np.nan)
        v[c > 0] = a[c > 0] / c[c > 0, None]
        per_gene[t] = v
    return means, per_gene, n_tok


def gene_matrix(per_gene: dict[int, np.ndarray], n_donors: int, min_share: float = 0.9) -> tuple[np.ndarray, list[int]]:
    """Donors × (token × emb) for tokens present in ≥ ``min_share`` of donors; a missing donor gets the token's mean."""
    toks = sorted(t for t, v in per_gene.items() if (~np.isnan(v[:, 0])).mean() >= min_share)
    blocks = []
    for t in toks:
        v = per_gene[t].copy()
        miss = np.isnan(v[:, 0])
        v[miss] = np.nanmean(v, 0)
        blocks.append(v)
    X = np.hstack(blocks) if blocks else np.zeros((n_donors, 0), dtype=np.float32)
    return X, toks


def readout_cohort(cohort: str, model: MctEncoder, index: Index, n_bags: int = 4, n_genes: int = 500,
                   models: tuple[str, ...] = ("ridge", "tabicl"), seed: int = 0, device: str = "cuda",
                   factor: int = 4) -> pl.DataFrame:  # fmt: skip
    from rna_age.train.fm_embed import clock_folds
    from rna_age.train.sc_recipe import age_genes, standardise
    from rna_age.train.sc_tabular import _fit_predict

    dp, _ = clock_folds(cohort)
    donors, age, groups = list(dp["donors"]), dp["age"], dp["groups"]
    t0 = time.time()
    dc = donor_cells(cohort, donors, index, factor, seed)
    log.info("%s: %d donors, %d pooled cells, %d vocab genes (%.0fs)", cohort[:8], len(donors), dc.X.shape[0],
             dc.X.shape[1], time.time() - t0)  # fmt: skip
    means, per_gene, n_tok = embed_donors(model, dc, n_bags, seed=seed, device=device)
    G, toks = gene_matrix(per_gene, len(donors))
    log.info("%s: embedded (%.0fs), median %d tokens per bag, %d genes in ≥ 90 %% of donors", cohort[:8],
             time.time() - t0, int(np.median(n_tok)), len(toks))  # fmt: skip
    feats = {"mctmean": means, "mctgene": G}
    rows = []
    for name, X in feats.items():
        pred = {m: np.full(len(donors), np.nan) for m in models}
        for tr, te in GroupKFold(n_splits=5).split(X, age, groups):
            if name == "mctgene":
                cols = age_genes(X[tr], age[tr], n_genes)
                a, b = standardise(X[tr][:, cols], X[te][:, cols])
            else:
                a, b = standardise(X[tr], X[te])
            for m in models:
                pred[m][te] = _fit_predict(m, a, age[tr], b, seed)
        for m, p in pred.items():
            rows += [{"cohort": cohort[:8], "representation": "natural", "model": f"mct-{name}-{m}", "donor": d,
                      "group": int(g), "age": float(a), "pred": float(v)}
                     for d, g, a, v in zip(donors, groups, age, p, strict=True)]  # fmt: skip
    return pl.DataFrame(rows)


# ------------------------------------------------------------------ step (b): fine-tune


def finetune_fold(base: MctEncoder, dc: DonorCells, tr: np.ndarray, te: np.ndarray, age: np.ndarray,
                  epochs: int = 30, lr: float = 1e-4, n_train_layers: int = 4, max_genes: int = 4096,
                  n_test_bags: int = 4, accumulate: int = 8, seed: int = 0, device: str = "cuda") -> np.ndarray:  # fmt: skip
    import copy

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = MctAge(copy.deepcopy(base)).to(device)
    freeze_below(model, n_train_layers)
    mu, sd = float(age[tr].mean()), float(age[tr].std() + 1e-6)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler(enabled=device.startswith("cuda"))
    for ep in range(epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        losses = []
        for k, d in enumerate(rng.permutation(tr)):
            x, tok = bag_tensor(dc, int(d), rng, max_genes)
            y = torch.tensor([(age[d] - mu) / sd], device=device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
                p = model(torch.from_numpy(x)[None].to(device), torch.from_numpy(tok)[None].to(device))
            loss = nn.functional.mse_loss(p.float(), y) / accumulate
            scaler.scale(loss).backward()
            losses.append(float(loss.detach()) * accumulate)
            if (k + 1) % accumulate == 0 or k + 1 == len(tr):
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
        log.info("  epoch %d: train mse %.3f", ep + 1, float(np.mean(losses)))
    model.eval()
    out = np.zeros(len(te))
    with torch.no_grad():
        for i, d in enumerate(te):
            ps = []
            for _ in range(n_test_bags):
                x, tok = bag_tensor(dc, int(d), rng, max_genes)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
                    ps.append(float(model(torch.from_numpy(x)[None].to(device), torch.from_numpy(tok)[None].to(device))))
            out[i] = np.mean(ps) * sd + mu
    return out


def finetune_cohort(cohort: str, base: MctEncoder, index: Index, seed: int = 0, device: str = "cuda",
                    factor: int = 4, **kw) -> pl.DataFrame:  # fmt: skip
    from rna_age.train.fm_embed import clock_folds

    dp, _ = clock_folds(cohort)
    donors, age, groups = list(dp["donors"]), dp["age"], dp["groups"]
    dc = donor_cells(cohort, donors, index, factor, seed)
    pred = np.full(len(donors), np.nan)
    for f, (tr, te) in enumerate(GroupKFold(n_splits=5).split(donors, age, groups)):
        t0 = time.time()
        pred[te] = finetune_fold(base, dc, tr, te, age, seed=seed, device=device, **kw)
        log.info("%s fold %d: MAE %.2f (%.0fs)", cohort[:8], f, float(np.abs(pred[te] - age[te]).mean()),
                 time.time() - t0)  # fmt: skip
    return pl.DataFrame([{"cohort": cohort[:8], "representation": "natural", "model": "mct-finetune",
                          "donor": d, "group": int(g), "age": float(a), "pred": float(v)}
                         for d, g, a, v in zip(donors, groups, age, pred, strict=True)])  # fmt: skip


# ------------------------------------------------------------------ comparison and runs


def compare_reference(oof: pl.DataFrame, reference: str = REFERENCE, n_boot: int = 2000) -> pl.DataFrame:
    """Each model against RM35's best clock on the same donors (paired bootstrap over persons), plus the pseudobulk
    references (``fm_embed.compare``); ``in_corpus`` marks the cohorts MCT pretrained on."""
    from rna_age.train.fm_embed import compare, paired_bootstrap

    ref = pl.read_parquet(SUMMARY / "sc_pretrain_oof.parquet").filter(pl.col("model") == reference)
    rows = []
    for (cohort, m), d in oof.group_by("cohort", "model", maintain_order=True):
        r = d.select("donor", "group", "age", "pred").join(
            ref.filter(pl.col("cohort") == cohort).select("donor", pl.col("pred").alias("ref_pred")), on="donor",
            how="left")  # fmt: skip
        if r["ref_pred"].null_count():
            raise ValueError(f"{cohort}: {reference} lacks {r['ref_pred'].null_count()} donors")
        e = (r["pred"] - r["age"]).abs().to_numpy()
        e_ref = (r["ref_pred"] - r["age"]).abs().to_numpy()
        diff, lo, hi = paired_bootstrap(e, e_ref, r["group"].to_numpy(), n_boot)
        rows.append({"cohort": cohort, "representation": "natural", "model": m, "n": len(e), "mae": float(e.mean()),
                     "r": float(np.corrcoef(r["pred"], r["age"])[0, 1]), "ref": reference,
                     "ref_mae": float(e_ref.mean()), "diff": diff, "lo": lo, "hi": hi, "beats_ref": hi < 0})  # fmt: skip
    out = pl.concat([pl.DataFrame(rows), compare(oof, n_boot=n_boot)], how="vertical_relaxed")
    return out.with_columns(in_corpus=pl.col("cohort").is_in(list(IN_CORPUS)))


def bar(cmp: pl.DataFrame) -> pl.DataFrame:
    """RM37's bar per model and reference: mean diff ≤ −0.3 y and beyond the CI in ≥ 4 of 7; also per corpus split."""
    g = cmp.group_by("model", "ref").agg(
        pl.len().alias("cohorts"), pl.col("diff").mean().alias("mean_diff"), pl.col("beats_ref").sum().alias("beats"),
        pl.col("mae").mean().alias("mean_mae"),
        pl.col("diff").filter(pl.col("in_corpus")).mean().alias("diff_in_corpus"),
        pl.col("diff").filter(~pl.col("in_corpus")).mean().alias("diff_out_of_corpus"),
    )  # fmt: skip
    return g.with_columns(passes=(pl.col("mean_diff") <= -0.3) & (pl.col("beats") >= 4)).sort("ref", "model")


def run(step: str, cohorts: tuple[str, ...] = EVAL_COHORTS, ckpt: Path = CKPT, index_dir: Path = INDEX,
        seed: int = 0, device: str = "cuda", write: bool = True, **kw) -> pl.DataFrame:  # fmt: skip
    """``step`` ``readout`` or ``finetune`` over ``cohorts``; rows into ``sc_pretrain_{oof,clock}.parquet`` under run
    ``rm37-mct-<step>``; returns the bar table."""
    if step not in ("readout", "finetune"):
        raise ValueError(f"step must be readout or finetune, got {step!r}")
    index = Index.load(index_dir)
    model, cfg = load_encoder(ckpt)
    log.info("MCT: %d tokens, width %d, %d layers, input %d; config keys %d", model.gene_id_emb.num_embeddings,
             model.layer_norm.normalized_shape[0], len(model.encoder.layers),
             model.gene_expression_encoder[0].in_features, len(cfg))  # fmt: skip
    parts = []
    for c in cohorts:
        t0 = time.time()
        f = readout_cohort if step == "readout" else finetune_cohort
        o = f(c, model, index, seed=seed, device=device, **kw)
        parts.append(o)
        log.info("%s %s: %s (%.0fs)", c[:8], step,
                 {m: round(float((g["pred"] - g["age"]).abs().mean()), 2) for (m,), g in o.group_by("model")},
                 time.time() - t0)  # fmt: skip
    oof = pl.concat(parts)
    cmp = compare_reference(oof)
    if write:
        run_id = f"rm37-mct-{step}" + (f"-s{seed}" if seed else "")
        _merge_summary(oof.with_columns(run_id=pl.lit(run_id)), SUMMARY / "sc_pretrain_oof.parquet")
        _merge_summary(cmp.with_columns(run_id=pl.lit(run_id)), SUMMARY / "sc_pretrain_clock.parquet")
    return bar(cmp)
