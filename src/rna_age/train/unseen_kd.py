"""RM21: can the age effect of a knockdown never seen be predicted from what the gene is?

Features: an ESM-2 embedding of the knocked-down gene's canonical protein (UniProt, reviewed human), mean-pooled over
residues (proteins longer than the model's window are cut into windows, each mean-pooled, averaged by length).
Non-coding targets have no protein and are dropped (counted).

Targets, per culture condition of the CD4 T Perturb-seq screen:
- ``clock``: our CD4 T clock's mean age shift per perturbed gene (``perturb_cd4_genes.parquet``, RM17);
- ``study``: the study's own aging coefficient per regulator (``coef_mean``; the file ``perturb-check`` reads).

Models, 5-fold CV over genes (a gene's row is in exactly one held-out fold):
- ``mean``: the training-fold mean;
- ``ridge``: RidgeCV on the standardised embedding;
- ``ridge_shuffled_emb``: the same ridge, every gene's embedding reassigned to a random other gene (the bar's control:
  embeddings with their geometry but no tie to the gene);
- ``tabicl``: TabICLv2 on the top PCs of the embedding (fitted on the training fold);
- ``*_label_shuffle``: the model trained on permuted targets, scored against the true ones (the null).

Scores on the pooled held-out predictions: Spearman and Pearson with a gene-bootstrap 95 % CI, skill over the mean
(1 − MSE / MSE_mean) and Δρ against ridge and against the shuffled-embedding ridge, each with a paired bootstrap CI.
The bar (ROADMAP RM21): a model beats the mean (skill CI above 0) AND the shuffled-embedding ridge (Δρ CI above 0).
"""

from __future__ import annotations

import gzip
import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import polars as pl

from rna_age.paths import INTERIM, OUTPUT, download_dir

log = logging.getLogger(__name__)

UNIPROT_URL = (
    "https://rest.uniprot.org/uniprotkb/stream?query=%28organism_id%3A9606%29+AND+%28reviewed%3Atrue%29"
    "&fields=accession%2Cgene_primary%2Cgene_synonym%2Csequence&format=tsv&compressed=true"
)
UNIPROT_FILE = "uniprot_human_reviewed.tsv.gz"
EMBEDDINGS = INTERIM / "gene_esm2.parquet"
RESULT = OUTPUT / "summary" / "unseen_knockdown.parquet"
CONDITIONS = ("Rest", "Stim8hr", "Stim48hr")


# ---------------------------------------------------------------- proteins


def fetch_uniprot() -> Path:
    from rna_age.download import Remote, fetch

    return fetch(Remote(UNIPROT_URL, UNIPROT_FILE), download_dir("uniprot"), connections=1)


def parse_uniprot(raw: bytes) -> pl.DataFrame:
    """gene symbol → (accession, sequence) from UniProt's TSV. A primary symbol wins; a synonym is used only when no
    entry has it as primary and exactly one entry lists it. Several entries with one primary: the longest."""
    t = pl.read_csv(raw, separator="\t", infer_schema_length=0).rename(
        {"Entry": "accession", "Gene Names (primary)": "primary", "Gene Names (synonym)": "synonym", "Sequence": "seq"}
    )
    prim = (
        t.filter(pl.col("primary").is_not_null())
        .with_columns(gene=pl.col("primary").str.split("; "))
        .explode("gene")
        .sort(pl.col("seq").str.len_chars(), descending=True)
        .unique("gene", keep="first")
        .select("gene", "accession", "seq")
    )
    syn = (
        t.filter(pl.col("synonym").is_not_null())
        .with_columns(gene=pl.col("synonym").str.split(" "))
        .explode("gene")
        .filter(pl.col("gene") != "")
        .group_by("gene")
        .agg(pl.col("accession").first(), pl.col("seq").first(), pl.len().alias("k"))
        .filter(pl.col("k") == 1)
        .drop("k")
        .join(prim.select("gene"), on="gene", how="anti")
    )
    return pl.concat([prim.with_columns(via=pl.lit("primary")), syn.with_columns(via=pl.lit("synonym"))])


def target_genes() -> list[str]:
    g = pl.read_parquet(OUTPUT / "summary" / "perturb_cd4_genes.parquet")
    c = _coefficients()
    return sorted(set(g["perturbed_gene_name"].drop_nulls()) | set(c["perturbed_gene_name"].drop_nulls()))


# ---------------------------------------------------------------- ESM-2


def pool_windows(n: int, window: int) -> list[tuple[int, int]]:
    """Residue windows [a, b) covering 0..n, each at most ``window`` long, near-equal."""
    k = max(1, -(-n // window))
    edges = np.linspace(0, n, k + 1).round().astype(int)
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:], strict=True)]


def esm2_embedder(model_name: str, device: str = "cuda", window: int = 1022) -> Callable[[str], np.ndarray]:
    """seq → mean-pooled last-layer embedding (special tokens excluded; long sequences by windows, length-weighted)."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    net = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16).to(device).eval()

    @torch.inference_mode()
    def embed(seq: str) -> np.ndarray:
        parts, weights = [], []
        for a, b in pool_windows(len(seq), window):
            ids = tok(seq[a:b], return_tensors="pt").to(device)
            h = net(**ids).last_hidden_state[0, 1:-1].float()  # drop <cls> and <eos>
            parts.append(h.mean(0).cpu().numpy())
            weights.append(b - a)
        return np.average(np.stack(parts), axis=0, weights=weights).astype(np.float32)

    return embed


def run_embed(model_name: str = "facebook/esm2_t33_650M_UR50D", device: str = "cuda") -> pl.DataFrame:
    import torch

    torch.cuda.init() if device == "cuda" else None
    raw = gzip.decompress((download_dir("uniprot") / UNIPROT_FILE).read_bytes())
    prot = parse_uniprot(raw)
    genes = target_genes()
    have = prot.filter(pl.col("gene").is_in(genes)).sort("gene")
    log.info("%d target genes; %d with a reviewed human protein (%d via a synonym); %d dropped",
             len(genes), have.height, (have["via"] == "synonym").sum(), len(genes) - have.height)  # fmt: skip
    embed = esm2_embedder(model_name, device)
    vecs = []
    for i, s in enumerate(have["seq"].to_list()):
        vecs.append(embed(s))
        if i % 500 == 0:
            log.info("embedded %d / %d", i, have.height)
    out = have.select("gene", "accession", "via", seq_len=pl.col("seq").str.len_chars()).with_columns(
        model=pl.lit(model_name), emb=pl.Series(vecs, dtype=pl.Array(pl.Float32, len(vecs[0])))
    )
    EMBEDDINGS.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(EMBEDDINGS, compression="zstd")
    return out


# ---------------------------------------------------------------- targets


def _coefficients() -> pl.DataFrame:
    from rna_age.train.perturb import COEFFICIENTS

    c = pl.read_csv(download_dir("cd4_perturbseq") / COEFFICIENTS)
    return c.filter(pl.col("celltype") != "K562").select(
        pl.col("celltype").alias("culture_condition"), pl.col("regulator").alias("perturbed_gene_name"),
        pl.col("coef_mean").alias("y"),
    )  # fmt: skip


def targets() -> pl.DataFrame:
    """culture_condition, perturbed_gene_name, y, target ∈ {clock, study}."""
    g = pl.read_parquet(OUTPUT / "summary" / "perturb_cd4_genes.parquet").select(
        "culture_condition", "perturbed_gene_name", pl.col("shift").alias("y")
    )
    return pl.concat([g.with_columns(target=pl.lit("clock")), _coefficients().with_columns(target=pl.lit("study"))])


# ---------------------------------------------------------------- CV


def folds(n: int, k: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).permutation(np.arange(n) % k)


def _ridge():
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 6, 15)))


class _PCAModel:
    """Standardise, PCA to ``k`` (fitted on the training fold), then ``inner`` (TabICL takes ≤ ~100 features)."""

    def __init__(self, inner, k: int):
        self.inner, self.k = inner, k

    def fit(self, X, y):
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        self.sc = StandardScaler().fit(X)
        self.pca = PCA(min(self.k, *X.shape), random_state=0).fit(self.sc.transform(X))
        self.inner.fit(self.pca.transform(self.sc.transform(X)), y)
        return self

    def predict(self, X):
        return self.inner.predict(self.pca.transform(self.sc.transform(X)))


def make_model(name: str, seed: int, n_pcs: int = 64):
    if name == "ridge":
        return _ridge()
    if name == "tabicl":
        from rna_age.train.tabfm import _estimator

        return _PCAModel(_estimator("tabicl", seed), n_pcs)
    if name == "pca_ridge":  # CPU stand-in for tabicl in tests
        from sklearn.linear_model import RidgeCV

        return _PCAModel(RidgeCV(alphas=np.logspace(-1, 6, 15)), n_pcs)
    raise ValueError(name)


def oof_predictions(X: np.ndarray, y: np.ndarray, models: tuple[str, ...], k: int = 5, seed: int = 0,
                    n_pcs: int = 64) -> dict[str, np.ndarray]:  # fmt: skip
    """Held-out predictions per model, every gene predicted once. Adds ``mean``, ``ridge_shuffled_emb`` and
    ``<model>_label_shuffle`` for each model. The label shuffle permutes targets among the training fold's genes
    only, so its intercept is the same fold mean as the ``mean`` baseline's (no held-out value leaks into it)."""
    rng = np.random.default_rng(seed + 1)
    f = folds(len(y), k, seed)
    X_shuf = X[rng.permutation(len(y))]  # each gene gets another gene's embedding, fixed across folds
    out: dict[str, np.ndarray] = {}
    plan: list[tuple[str, str | None, np.ndarray, bool]] = [("mean", None, X, False)]
    plan += [("ridge_shuffled_emb", "ridge", X_shuf, False)]
    for m in models:
        plan += [(m, m, X, False), (f"{m}_label_shuffle", m, X, True)]
    for name, m, Xm, shuffle in plan:
        p = np.empty(len(y))
        for i in range(k):
            tr, te = f != i, f == i
            ytr = rng.permutation(y[tr]) if shuffle else y[tr]
            if m is None:
                p[te] = ytr.mean()
            else:
                p[te] = make_model(m, seed, n_pcs).fit(Xm[tr], ytr).predict(Xm[te])
        out[name] = p
        log.info("%s done", name)
    return out


def _corr(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    from scipy.stats import rankdata

    if a.std() == 0 or b.std() == 0:
        return float("nan"), float("nan")
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1]), float(np.corrcoef(a, b)[0, 1])


def score(y: np.ndarray, preds: dict[str, np.ndarray], n_boot: int = 1000, seed: int = 0) -> pl.DataFrame:
    """Per model: ρ, r, skill over the mean, Δρ vs ridge and vs the shuffled-embedding ridge, gene-bootstrap CIs
    (the same resamples for every model, so the differences are paired)."""
    rng = np.random.default_rng(seed)
    idx = [rng.integers(0, len(y), len(y)) for _ in range(n_boot)]
    mse_mean = lambda ii: np.mean((y[ii] - preds["mean"][ii]) ** 2)  # noqa: E731

    def stats(p: np.ndarray, ii: np.ndarray) -> dict[str, float]:
        rho, r = _corr(p[ii], y[ii])
        out = {"spearman": rho, "pearson": r, "skill_vs_mean": 1 - np.mean((y[ii] - p[ii]) ** 2) / mse_mean(ii)}
        for ref in ("ridge", "ridge_shuffled_emb"):
            if ref in preds:
                out[f"d_rho_vs_{ref}"] = rho - _corr(preds[ref][ii], y[ii])[0]
        return out

    all_i = np.arange(len(y))
    rows = []
    for name, p in preds.items():
        point = stats(p, all_i)
        boots = [stats(p, ii) for ii in idx]
        row: dict = {"model": name, "n_genes": len(y)}
        for key, v in point.items():
            b = np.array([s[key] for s in boots], dtype=float)
            lo, hi = np.nanpercentile(b, [2.5, 97.5]) if np.isfinite(b).any() else (np.nan, np.nan)
            row |= {key: float(v), f"{key}_lo": float(lo), f"{key}_hi": float(hi)}
        rows.append(row)
    d = pl.DataFrame(rows)
    return d.with_columns(
        beats_bar=(pl.col("skill_vs_mean_lo") > 0) & (pl.col("d_rho_vs_ridge_shuffled_emb_lo") > 0)
        & ~pl.col("model").is_in(["mean", "ridge_shuffled_emb"]),
    )  # fmt: skip


def evaluate(emb: pl.DataFrame, tg: pl.DataFrame, models: tuple[str, ...], k: int = 5, seed: int = 0,
             n_boot: int = 1000, n_pcs: int = 64) -> pl.DataFrame:  # fmt: skip
    """Every (target, condition): join targets to embeddings by gene symbol, CV, score."""
    out = []
    for (target, cond), t in sorted(tg.group_by("target", "culture_condition"), key=lambda x: x[0]):
        t = t.filter(pl.col("y").is_finite()).unique("perturbed_gene_name", keep="first")
        j = t.join(emb, left_on="perturbed_gene_name", right_on="gene").sort("perturbed_gene_name")
        if j.height < 5 * k:
            log.warning("%s %s: %d genes, skipped", target, cond, j.height)
            continue
        X, y = j["emb"].to_numpy().astype(np.float64), j["y"].to_numpy().astype(np.float64)
        log.info("%s %s: %d genes with an embedding of %d", target, cond, j.height, t.height)
        preds = oof_predictions(X, y, models, k=k, seed=seed, n_pcs=n_pcs)
        s = score(y, preds, n_boot=n_boot, seed=seed)
        out.append(s.with_columns(target=pl.lit(target), culture_condition=pl.lit(cond),
                                  n_targets=pl.lit(t.height), n_dropped_no_protein=pl.lit(t.height - j.height)))  # fmt: skip
    first = ["target", "culture_condition", "model", "n_genes", "n_targets", "n_dropped_no_protein"]
    d = pl.concat(out)
    return d.select(first + [c for c in d.columns if c not in first])


def run_cv(models: tuple[str, ...] = ("ridge", "tabicl"), k: int = 5, seed: int = 0, n_boot: int = 1000,
           n_pcs: int = 64) -> pl.DataFrame:  # fmt: skip
    emb = pl.read_parquet(EMBEDDINGS).select("gene", "emb")
    d = evaluate(emb, targets(), models, k=k, seed=seed, n_boot=n_boot, n_pcs=n_pcs)
    d.write_parquet(RESULT, compression="zstd")
    return d
