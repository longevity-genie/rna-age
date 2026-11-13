"""RM28 arm ``universal``: the masked autoencoder's encoder shared by masked expression, the perturbation head, an
age head and a health head (maintainer's scope addition, 2026-11-13).

- Age head (``Linear(256, 1)`` on the embedding, ages standardised on the bulk training ages), trained on
  (a) pretraining-pool donors with an exact adult age, both as their pseudobulk on the panel (log1p of the mean
  CP10k of their cells) and as the mean of their cells' embeddings; the pool's validation donors and every
  evaluation cohort stay out (the pool never holds them: :func:`sc_pretrain.build_pool`); (b) bulk RNA-seq
  (ARCHS4/GTEx/Bgee) ``train`` samples with exact ages that are controls (``health.health_table``), on the same
  panel with the same transform (log1p CP10k, library over the shared bulk genes). Bulk ``val`` controls choose the
  checkpoint; ``test`` (and anything not ``train``) never enters a batch.
- Health head (D6's recipe, ``health.train_health``): the within-study ranking loss on (aging-linked case, control)
  pairs of training studies (``health._pairs``: the case's age gap should exceed the control's by ``margin``
  years) and a case-vs-control logit on training samples with exact ages.
- Loss: masked MSE + ``perturb_weight``·perturbation + ``age_weight``·(bulk + pool pseudobulk + pool cells, Huber on
  the standardised scale) + ``rank_weight``·ranking + ``health_weight``·BCE; one batch per task each step.

``universal_eval`` writes the bulk predictions (``predictions.parquet`` in the run dir, the shape of other runs,
through ``runs.finish_run``) and the within-study gap (``health.evaluate_health``).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch import nn

log = logging.getLogger(__name__)

BULK_SOURCES = ("archs4", "gtex", "bgee")
WEIGHTS = {"age": 1.0, "rank": 0.5, "health": 0.5}


class Heads(nn.Module):
    def __init__(self, emb: int, mu: float, sd: float):
        super().__init__()
        self.age = nn.Linear(emb, 1)
        self.health = nn.Linear(emb, 1)
        self.register_buffer("mu", torch.tensor(float(mu)))
        self.register_buffer("sd", torch.tensor(float(sd)))

    def age_std(self, z: torch.Tensor) -> torch.Tensor:
        return self.age(z).squeeze(-1)

    def age_years(self, z: torch.Tensor) -> torch.Tensor:
        return self.age_std(z) * self.sd + self.mu

    def logit(self, z: torch.Tensor) -> torch.Tensor:
        return self.health(z).squeeze(-1)


# ------------------------------------------------------------------ which rows train what


def bulk_table() -> pl.DataFrame:
    """Bulk samples of ``health.health_table`` (group, category), pseudobulk sources dropped."""
    from rna_age.train.health import health_table

    t = health_table()
    return t.filter(~pl.col("is_pseudobulk") & pl.col("source").is_in(list(BULK_SOURCES)))


def bulk_for_run(pool: str, max_bulk: int | None, seed: int) -> tuple[pl.DataFrame, np.ndarray]:
    """The bulk table (a fixed-seed subsample of ``max_bulk`` rows in smoke runs) and its panel matrix (cached)."""
    from rna_age.train.sc_pretrain import panel_genes, pool_dir

    bulk = bulk_table()
    if max_bulk and bulk.height > max_bulk:
        bulk = bulk.sample(max_bulk, seed=seed).sort("sample_id")
    cache = pool_dir(pool) / (f"bulk_{max_bulk}_s{seed}.npz" if max_bulk else "bulk.npz")
    return bulk, bulk_panel(bulk, panel_genes(pool)["gene_id"].to_list(), cache)


def bulk_indices(t: pl.DataFrame, aging_only: bool = True, seed: int = 0) -> dict[str, np.ndarray]:
    """Row indices of ``t`` per task: ``age_train`` (train, exact, control), ``age_val`` (val, exact, control),
    ``head`` (train, exact, control or case), ``pairs`` (case, control) within training studies. Only ``train``
    rows feed a training batch; ``age_val`` is for checkpoint choice."""
    from rna_age.train.health import AGING_LINKED, _pairs

    split = t["split"].to_numpy()
    train = split == "train"
    exact = (t["age_type"] == "exact").to_numpy()
    ctrl = (t["group"] == "control").to_numpy()
    case = (t["group"] == "case").to_numpy()
    if aging_only:
        case = case & t["category"].is_in(list(AGING_LINKED)).fill_null(False).to_numpy()
    out = {
        "age_train": np.flatnonzero(train & exact & ctrl),
        "age_val": np.flatnonzero((split == "val") & exact & ctrl),
        "head": np.flatnonzero(train & exact & (ctrl | case)),
        "pairs": _pairs(t, train, aging_only, seed=seed).astype(np.int64),
    }
    for k in ("age_train", "head"):
        assert (split[out[k]] == "train").all(), k
    assert len(out["pairs"]) == 0 or (split[out["pairs"].ravel()] == "train").all()
    return out


def pool_donors(pool: pl.DataFrame, is_val: np.ndarray) -> dict[str, list]:
    """Pool donors with an exact age, split into training and validation donors (whole donors), each with the
    matrix rows of its cells. Raises if an evaluation cohort is in the pool."""
    from rna_age.train.sc_pretrain import EVAL_COHORTS

    bad = set(pool["dataset_id"].unique().to_list()) & set(EVAL_COHORTS)
    if bad:
        raise ValueError(f"evaluation cohorts in the pretraining pool: {sorted(bad)}")
    p = pool.with_row_index("_i").with_columns(
        is_val=pl.Series(is_val), key=pl.col("dataset_id") + ":" + pl.col("donor_id")
    )
    p = p.filter(pl.col("age_exact"))
    g = p.group_by("key", maintain_order=True).agg(pl.col("_i"), pl.col("age").first(), pl.col("is_val").first())
    out: dict[str, list] = {}
    for part, flag in (("train", False), ("val", True)):
        s = g.filter(pl.col("is_val") == flag)
        out[part] = [s["key"].to_list(), s["age"].to_numpy(), [np.asarray(r) for r in s["_i"].to_list()]]
    return out


# ------------------------------------------------------------------ bulk on the panel


def bulk_panel(t: pl.DataFrame, panel: list[str], cache: Path | None = None, chunk: int = 8000) -> np.ndarray:
    """Bulk samples × panel, log1p CP10k with the library over the shared bulk genes (fp16), cached."""
    from rna_age.train.features import load_matrix, shared_genes

    if cache is not None and cache.exists():
        z = np.load(cache)
        if z["sample_id"].tolist() == t["sample_id"].to_list() and z["panel"].tolist() == panel:
            return z["x"]
    genes = shared_genes()
    pos = {g: i for i, g in enumerate(genes)}
    cols = np.array([pos.get(g, -1) for g in panel])
    log.info("bulk: %d of %d panel genes on the shared bulk axis", (cols >= 0).sum(), len(panel))
    out = np.zeros((t.height, len(panel)), dtype=np.float16)
    for a in range(0, t.height, chunk):
        c = load_matrix(t[a : a + chunk].select("sample_id", "source"), genes)
        lib = c.sum(1, keepdims=True)
        lib[lib == 0] = 1
        x = np.zeros((len(c), len(panel)), dtype=np.float32)
        x[:, cols >= 0] = np.log1p(c[:, cols[cols >= 0]] / lib * 1e4)
        out[a : a + chunk] = x.astype(np.float16)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(".tmp.npz")
        np.savez(tmp, sample_id=np.array(t["sample_id"].to_list()), panel=np.array(panel), x=out)
        tmp.rename(cache)
    return out


def donor_pseudobulk(X: torch.Tensor, rows: list[np.ndarray]) -> torch.Tensor:
    """Per donor: log1p of the mean CP10k of its cells (cells hold log1p CP10k)."""
    return torch.stack([torch.log1p(torch.expm1(X[torch.from_numpy(r).to(X.device)].float()).mean(0)) for r in rows])


# ------------------------------------------------------------------ the task


class UniversalTask:
    """Age and health losses beside the masked (and perturbation) losses of ``sc_pretrain.pretrain``."""

    def __init__(self, X: torch.Tensor, pool: pl.DataFrame, is_val: np.ndarray, bulk: pl.DataFrame,
                 bulk_x: np.ndarray, emb: int, device, weights: dict[str, float] | None = None, batch: int = 256,
                 donor_batch: int = 32, cell_donors: int = 8, cells_per_donor: int = 64, margin: float = 3.0,
                 seed: int = 0):  # fmt: skip
        self.w = WEIGHTS | (weights or {})
        self.X, self.dev = X, device
        self.idx = bulk_indices(bulk, seed=seed)
        self.age = torch.tensor(
            np.nan_to_num(bulk["age_years"].to_numpy(), nan=0.0), dtype=torch.float32, device=device
        )
        self.case = torch.tensor((bulk["group"] == "case").to_numpy().astype(np.float32), device=device)
        self.bx = torch.from_numpy(bulk_x).to(device)
        a = bulk["age_years"].to_numpy()[self.idx["age_train"]]
        self.heads = Heads(emb, float(a.mean()), float(a.std() + 1e-6)).to(device)
        self.donors = pool_donors(pool, is_val)
        self.pb = {k: donor_pseudobulk(X, v[2]) if len(v[2]) else torch.zeros(0, X.shape[1], device=device)
                   for k, v in self.donors.items()}  # fmt: skip
        self.pb_age = {k: torch.tensor(v[1], dtype=torch.float32, device=device) for k, v in self.donors.items()}
        self.batch, self.donor_batch, self.cell_donors, self.cpd = batch, donor_batch, cell_donors, cells_per_donor
        self.margin = margin
        self.rng = np.random.default_rng(seed + 7)
        log.info("universal: %d bulk age / %d head rows / %d pairs (train), %d bulk val; pool donors %d train / %d val",
                 len(self.idx["age_train"]), len(self.idx["head"]), len(self.idx["pairs"]), len(self.idx["age_val"]),
                 len(self.donors["train"][0]), len(self.donors["val"][0]))  # fmt: skip

    def parameters(self):
        return self.heads.parameters()

    def _huber(self, z: torch.Tensor, age: torch.Tensor) -> torch.Tensor:
        h = self.heads
        return nn.functional.huber_loss(h.age_std(z).float(), (age - h.mu) / h.sd, delta=float(5.0 / h.sd))

    def sample(self) -> dict[str, np.ndarray]:
        """One batch of row indices per task (training rows only)."""
        r, i = self.rng, self.idx
        nd = len(self.donors["train"][0])
        out = {"bulk_age": r.choice(i["age_train"], self.batch), "head": r.choice(i["head"], self.batch)}
        if len(i["pairs"]):
            out["pairs"] = i["pairs"][r.integers(0, len(i["pairs"]), self.batch)]
        if nd:
            out["pool_pb"] = r.integers(0, nd, self.donor_batch)
            out["pool_cells"] = r.integers(0, nd, self.cell_donors)
        return out

    def loss(self, model) -> dict[str, torch.Tensor]:
        b = self.sample()
        h = self.heads
        enc = model.encode
        terms = {}
        ib = torch.from_numpy(b["bulk_age"]).to(self.dev)
        age_terms = [self._huber(enc(self.bx[ib].float()), self.age[ib])]
        if "pool_pb" in b:
            d = torch.from_numpy(b["pool_pb"]).to(self.dev)
            age_terms.append(self._huber(enc(self.pb["train"][d]), self.pb_age["train"][d]))
            rows = self.donors["train"][2]
            z = []
            for k in b["pool_cells"]:
                r = rows[k]
                pick = r if len(r) <= self.cpd else self.rng.choice(r, self.cpd, replace=False)
                z.append(enc(self.X[torch.from_numpy(pick).to(self.dev)].float()).mean(0))
            age_terms.append(
                self._huber(torch.stack(z), self.pb_age["train"][torch.from_numpy(b["pool_cells"]).to(self.dev)])
            )
        terms["age"] = sum(age_terms) / len(age_terms)
        if "pairs" in b:
            p = torch.from_numpy(b["pairs"]).to(self.dev)
            gc = h.age_years(enc(self.bx[p[:, 0]].float())).float() - self.age[p[:, 0]]
            gk = h.age_years(enc(self.bx[p[:, 1]].float())).float() - self.age[p[:, 1]]
            terms["rank"] = torch.relu(self.margin - (gc - gk)).mean() / h.sd
        ih = torch.from_numpy(b["head"]).to(self.dev)
        terms["health"] = nn.functional.binary_cross_entropy_with_logits(h.logit(enc(self.bx[ih].float())).float(),
                                                                          self.case[ih])  # fmt: skip
        return terms

    def total(self, terms: dict[str, torch.Tensor]) -> torch.Tensor:
        return sum(self.w[k] * v for k, v in terms.items())

    @torch.no_grad()
    def evaluate(self, model) -> dict[str, float]:
        """Bulk val controls' MAE (checkpoint choice) and pool validation donors' pseudobulk MAE."""
        out = {}
        iv = self.idx["age_val"]
        if len(iv):
            pred = torch.cat([self.heads.age_years(model.encode(self.bx[torch.from_numpy(iv[a : a + 8192]).to(self.dev)].float())).float()
                              for a in range(0, len(iv), 8192)])  # fmt: skip
            out["bulk_val_mae"] = float((pred - self.age[torch.from_numpy(iv).to(self.dev)]).abs().mean())
        if len(self.pb["val"]):
            p = self.heads.age_years(model.encode(self.pb["val"])).float()
            out["pool_val_pb_mae"] = float((p - self.pb_age["val"]).abs().mean())
        return out


# ------------------------------------------------------------------ read-outs


def universal_eval(run_id: str, device: str | None = None, which: str = "best") -> pl.DataFrame:
    """Bulk predictions of the age head (every split; ``predictions.parquet`` via ``finish_run``) and the
    within-study health gap (``evaluate_health``) for ``pred`` and ``bio_age = pred + β·logit`` (β chosen on val
    as in D6: the largest within-study gap at ≤ 1 y control MAE cost)."""
    from rna_age.train.baselines import PRED_COLS
    from rna_age.train.health import evaluate_health
    from rna_age.train.runs import finish_run
    from rna_age.train.sc_pretrain import RUNS, _device, load_encoder

    dev = _device(device)
    model, cfg = load_encoder(run_id, dev, which)
    state = torch.load(Path(cfg["checkpoint_dir"]) / f"{which}.pt", map_location=dev, weights_only=False)
    heads = Heads(cfg["emb"], 0.0, 1.0).to(dev)
    heads.load_state_dict(state["heads"])
    heads.eval()
    t, x = bulk_for_run(cfg["pool"], cfg.get("max_bulk"), cfg["seed"])
    preds, logits = [], []
    with torch.no_grad():
        for a in range(0, len(x), 8192):
            z = model.encode(torch.from_numpy(x[a : a + 8192].astype(np.float32)).to(dev))
            preds.append(heads.age_years(z).float().cpu().numpy())
            logits.append(heads.logit(z).float().cpu().numpy())
    out = t.select([c for c in PRED_COLS if c in t.columns]).with_columns(
        pred=pl.Series(np.concatenate(preds).astype(np.float64)),
        health_logit=pl.Series(np.concatenate(logits).astype(np.float64)),
    )
    beta = _choose_beta(out.join(t.select("sample_id", "group"), on="sample_id"))
    out = out.with_columns(bio_age=pl.col("pred") + beta * pl.col("health_logit"))
    run_dir = RUNS / run_id
    row = finish_run(run_dir, out, {"beta": beta})
    res = pl.concat([evaluate_health(run_id, s) for s in ("pred", "bio_age")], how="diagonal_relaxed")
    res.write_parquet(run_dir / "health.parquet")
    log.info("bulk test MAE %.2f (val %.2f), beta %g", row.get("test_exact_mae") or np.nan,
             row.get("val_exact_mae") or np.nan, beta)  # fmt: skip
    return res


def _choose_beta(v: pl.DataFrame) -> float:
    """D6's β on val: the largest within-study case − control gap among β with ≤ 1 y more control MAE."""
    v = v.filter((pl.col("split") == "val") & (pl.col("age_type") == "exact"))
    cand = []
    for b in (0.0, 1.0, 2.0, 4.0, 8.0):
        g = v.with_columns(s=pl.col("pred") + b * pl.col("health_logit")).with_columns(
            gap=pl.col("s") - pl.col("age_years")
        )
        c = g.filter(pl.col("group") == "control")
        mae = float((c["s"] - c["age_years"]).abs().mean()) if c.height else np.inf
        cs = (
            g.filter(pl.col("group") == "case")
            .group_by("study_id")
            .agg(pl.col("gap").mean().alias("cg"), pl.len().alias("n"))
        )
        ks = (
            g.filter(pl.col("group") == "control")
            .group_by("study_id")
            .agg(pl.col("gap").mean().alias("kg"), pl.len().alias("m"))
        )
        j = cs.join(ks, on="study_id").filter((pl.col("n") >= 5) & (pl.col("m") >= 5))
        cand.append((b, mae, float((j["cg"] - j["kg"]).mean()) if j.height else 0.0))
    ok = [c for c in cand if c[1] <= cand[0][1] + 1.0]
    return max(ok, key=lambda c: c[2])[0] if ok else 0.0
