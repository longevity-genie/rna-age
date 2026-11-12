"""RM23: MaxToki's in-silico knockdown screen in blood CD4 T cells against the genome-scale CD4 T CRISPRi screen's aging
effects and against our clock's knockdown shifts (RM17).

Cells: naive and central-memory CD4 T cells of the OneK1K-size cohort (``3faad104…``, primary cells), one cell per
donor, capped per cell type × sex × age decade. Each cell is MaxToki's rank value encoding: counts / total counts ×
10,000 / the gene's non-zero median (the model's own medians), protein-coding vocabulary genes, descending, between
``<bos>`` and ``<eos>``, truncated to ``max_genes``.

Prompts follow the paper's task 1, within one cell type and sex: ``[cell age a1] [a2-a1] [cell age a2] <boq> [query cell,
age a3] <eoq>`` (three different donors, a1 < a2, time in months), answered with the expected time token at ``<eoq>``.
Validation first: the unperturbed answer against the true a3 - a2, per RoPE factor (the checkpoint records 1.0; the
paper trained at 16,384 with RoPE scaling), the best factor used for the screen.

Screen (the paper's in-silico inhibition): each screen gene expressed in the query cell is removed from the query's rank
encoding, context unchanged; effect = predicted time (perturbed) - predicted time (unperturbed), > 0 = the knockdown reads
older (age-promoting). Per gene: mean over the prompts whose query expresses it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from rna_age.paths import INTERIM, OUTPUT, download_dir

log = logging.getLogger(__name__)

COHORT = "3faad104-2ab8-4434-816d-474d8d2641db"
CELL_TYPES = {"naive": "naive thymus-derived CD4-positive, alpha-beta T cell",
              "cm": "central memory CD4-positive, alpha-beta T cell"}  # fmt: skip
WORK = INTERIM / "maxtoki_cd4"


# ------------------------------------------------------------------ cells


def rank_encode(counts: np.ndarray, cols: np.ndarray, total: float, token: np.ndarray, median: np.ndarray,
                max_genes: int, bos: int, eos: int) -> list[int]:  # fmt: skip
    """One cell's rank value encoding. ``counts``/``cols``: its non-zero counts and their var columns; ``token`` and
    ``median``: per var column, the token id (-1 = not in the vocabulary) and the gene's median."""
    keep = token[cols] >= 0
    c, j = counts[keep], cols[keep]
    x = c / total * 10_000 / median[j]
    order = np.argsort(-x, kind="stable")[:max_genes]
    return [bos, *token[j[order]].tolist(), eos]


def sample_cells(per_bin: int = 40, seed: int = 0) -> pl.DataFrame:
    """One random cell per donor for each CD4 type, at most ``per_bin`` donors per type × sex × decade (20s … 80s; the
    teens join the 20s, the 90s the 80s). Columns: cell (row), donor_id, age, sex, ctype, decade."""
    import h5py

    from rna_age.sources.census_pseudobulk import _obs_column

    path = download_dir("census_pseudobulk") / "h5ads" / f"{COHORT}.h5ad"
    with h5py.File(path, "r") as f:
        cols = ("donor_id", "age", "sex", "cell_type", "is_primary_data")
        obs = pl.DataFrame({c: _obs_column(f["obs"], c) for c in cols}).with_row_index("cell")
    inv = {v: k for k, v in CELL_TYPES.items()}
    obs = obs.filter(pl.col("is_primary_data") & pl.col("cell_type").is_in(list(inv))).with_columns(
        ctype=pl.col("cell_type").replace_strict(inv), decade=(pl.col("age").clip(20, 89) // 10 * 10).cast(pl.Int32)
    )
    one = obs.sample(fraction=1.0, shuffle=True, seed=seed).unique(["donor_id", "ctype"], keep="first")
    pick = one.sample(fraction=1.0, shuffle=True, seed=seed + 1).with_columns(
        k=pl.int_range(pl.len()).over("ctype", "sex", "decade")
    )
    pick = pick.filter(pl.col("k") < per_bin).select("cell", "donor_id", "age", "sex", "ctype", "decade")
    return pick.sort("cell")


def encode_cells(cells: pl.DataFrame, vocab: dict[str, int], medians: dict[str, float], max_genes: int):
    """(token lists, gene → set of cells expressing it as tokens) for the sampled rows, read from the CSR h5ad."""
    import h5py

    path = download_dir("census_pseudobulk") / "h5ads" / f"{COHORT}.h5ad"
    with h5py.File(path, "r") as f:
        var = [x.decode() if isinstance(x, bytes) else str(x) for x in f["var"]["_index"][()]]
        token = np.array([vocab.get(g, -1) if g in medians else -1 for g in var])
        median = np.array([medians.get(g, 1.0) for g in var])
        X = f["X"]
        indptr = X["indptr"][()]
        out = []
        for r in cells["cell"].to_list():
            lo, hi = indptr[r], indptr[r + 1]
            c, j = X["data"][lo:hi].astype(np.float64), X["indices"][lo:hi]
            out.append(rank_encode(c, j, float(c.sum()), token, median, max_genes, vocab["<bos>"], vocab["<eos>"]))
    return out


# ------------------------------------------------------------------ prompts


@dataclass
class Prompt:
    context: list[int]
    query: list[int]  # the query cell's tokens, <bos> … <eos>
    t_true: float  # months from the last context cell to the query cell
    query_cell: int  # index into the cell table


def build_prompts(cells: pl.DataFrame, toks: list[list[int]], vocab: dict[str, int], n: int, seed: int) -> list[Prompt]:
    """``n`` prompts, each within one cell type × sex: two context cells (a1 < a2, different donors and decades when
    possible) and a query cell from a third donor; every query cell used at most once."""
    rng = np.random.default_rng(seed)
    c = cells.with_row_index("i")
    used: set[int] = set()
    out: list[Prompt] = []
    groups = [g for _, g in sorted(c.group_by("ctype", "sex"), key=lambda t: t[0])]
    tries = 0
    while len(out) < n and tries < 50 * n:
        tries += 1
        g = groups[rng.integers(len(groups))]
        free = g.filter(~pl.col("i").is_in(list(used)))
        if free.height == 0 or g.height < 3:
            continue
        q = free.row(int(rng.integers(free.height)), named=True)
        rest = g.filter(pl.col("donor_id") != q["donor_id"])
        a, b = (rest.row(int(k), named=True) for k in rng.choice(rest.height, 2, replace=False))
        if a["donor_id"] == b["donor_id"] or a["age"] == b["age"]:
            continue
        a, b = sorted((a, b), key=lambda r: r["age"])
        dt = 12 * (b["age"] - a["age"])
        context = toks[a["i"]] + [vocab[str(dt)]] + toks[b["i"]]
        out.append(Prompt(context, toks[q["i"]], 12.0 * (q["age"] - b["age"]), q["i"]))
        used.add(q["i"])
    return out


# ------------------------------------------------------------------ model calls


class TimeReader:
    """MaxToki's answer at ``<eoq>`` for a prompt context and a batch of query cells, with the context's key/value
    cache computed once per prompt."""

    def __init__(self, model, vocab: dict[str, int], batch: int = 16) -> None:
        import torch

        from rna_age.train.maxtoki import time_tokens

        self.model, self.batch, self.torch = model, batch, torch
        ids, vals = time_tokens(vocab)
        self.ids = torch.tensor(ids, device=model.device)
        self.vals = torch.tensor(vals, device=model.device, dtype=torch.float32)
        self.boq, self.eoq = vocab["<boq>"], vocab["<eoq>"]

    def expected_time(self, logits):
        """Headless decode: softmax over the vocabulary, restricted to time tokens, renormalised, expected value; also
        the probability mass on time tokens."""
        p = self.torch.softmax(logits.float(), dim=-1)
        pt = p[:, self.ids]
        mass = pt.sum(-1)
        return (pt / mass[:, None]) @ self.vals, mass

    def __call__(self, context: list[int], queries: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
        """Times and time-token masses for each query (all queries the same length within a call to batch them;
        different lengths are run in separate groups)."""
        torch = self.torch
        from transformers import DynamicCache

        dev = self.model.device
        with torch.inference_mode():
            pre = self.model(input_ids=torch.tensor([context], device=dev), use_cache=True)
            cache = pre.past_key_values
            kv = [(layer.keys, layer.values) for layer in cache.layers]
            t_out = np.zeros(len(queries))
            m_out = np.zeros(len(queries))
            order = sorted(range(len(queries)), key=lambda i: len(queries[i]))
            i = 0
            while i < len(order):
                L = len(queries[order[i]])
                grp = [order[i]]
                while i + len(grp) < len(order) and len(grp) < self.batch and len(queries[order[i + len(grp)]]) == L:
                    grp.append(order[i + len(grp)])
                B = len(grp)
                c = DynamicCache()
                for li, (k, v) in enumerate(kv):
                    c.update(k.expand(B, -1, -1, -1).contiguous(), v.expand(B, -1, -1, -1).contiguous(), li)
                ids = torch.tensor([[self.boq, *queries[j], self.eoq] for j in grp], device=dev)
                out = self.model(input_ids=ids, past_key_values=c, use_cache=True, logits_to_keep=1)
                t, m = self.expected_time(out.logits[:, -1])
                t_out[grp], m_out[grp] = t.cpu().numpy(), m.cpu().numpy()
                i += B
        return t_out, m_out


# ------------------------------------------------------------------ the screen's genes


def screen_table() -> pl.DataFrame:
    """Per CD4 T condition and regulator: the screen's aging coefficient and our clock's shift (RM17)."""
    from rna_age.train.perturb import COEFFICIENTS

    coef = pl.read_csv(download_dir("cd4_perturbseq") / COEFFICIENTS).filter(pl.col("celltype") != "K562")
    coef = coef.select(pl.col("celltype").alias("condition"), pl.col("regulator").alias("gene"), "coef_mean")
    clock = pl.read_parquet(OUTPUT / "summary" / "perturb_cd4_genes.parquet").select(
        pl.col("culture_condition").alias("condition"), pl.col("perturbed_gene_name").alias("gene"), "shift"
    )
    return coef.join(clock, on=["condition", "gene"], how="full", coalesce=True)


def symbol_to_ensembl() -> dict[str, str]:
    import h5py

    from rna_age.sources.census_pseudobulk import _obs_column

    path = download_dir("census_pseudobulk") / "h5ads" / f"{COHORT}.h5ad"
    with h5py.File(path, "r") as f:
        ens = [x.decode() if isinstance(x, bytes) else str(x) for x in f["var"]["_index"][()]]
        sym = _obs_column(f["var"], "feature_name")
    m: dict[str, str] = {}
    for s, e in zip(sym, ens, strict=True):
        m.setdefault(str(s), e)
    return m


def screen_genes(vocab: dict[str, int], max_genes: int | None, seed: int) -> pl.DataFrame:
    """The regulators the screen scored in CD4 T cells that are in MaxToki's vocabulary: gene, ensembl, token, priority
    (max |coef| over conditions). ``max_genes``: the top half of that budget by |coef|, the rest at random."""
    s = screen_table().filter(pl.col("coef_mean").is_not_null())
    g = s.group_by("gene").agg(pl.col("coef_mean").abs().max().alias("priority"))
    ens = symbol_to_ensembl()
    g = g.with_columns(ensembl=pl.col("gene").replace_strict(ens, default=None)).filter(pl.col("ensembl").is_not_null())
    g = g.with_columns(token=pl.col("ensembl").replace_strict(vocab, default=None, return_dtype=pl.Int64))
    g = g.filter(pl.col("token").is_not_null()).sort("priority", descending=True)
    if max_genes is not None and g.height > max_genes:
        top = g.head(max_genes // 2)
        rest = g.slice(max_genes // 2).sample(max_genes - top.height, seed=seed)
        g = pl.concat([top, rest])
    return g


# ------------------------------------------------------------------ statistics


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import rankdata

    ra, rb = rankdata(a), rankdata(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    d = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def compare(x: np.ndarray, y: np.ndarray, n_boot: int, n_perm: int, seed: int, top: float = 0.2) -> dict:
    """Spearman of x (MaxToki) against y (a reference) with a gene-bootstrap 95 % CI and a shuffled-gene null (x permuted
    over genes); top fifth of y by |y|: Spearman there, and the fraction of those genes whose sign of x matches y's,
    against the same fraction under the shuffle."""
    from scipy.stats import spearmanr

    rng = np.random.default_rng(seed)
    n = len(x)
    rho, p = spearmanr(x, y)
    boot = np.array([_spearman(x[i], y[i]) for i in (rng.integers(0, n, n) for _ in range(n_boot))])
    null = np.array([_spearman(rng.permutation(x), y) for _ in range(n_perm)])
    t = np.abs(y) >= np.quantile(np.abs(y), 1 - top)
    agree = float((np.sign(x[t]) == np.sign(y[t])).mean())
    null_agree = np.array([(np.sign(rng.permutation(x)[t]) == np.sign(y[t])).mean() for _ in range(n_perm)])
    rho_top = _spearman(x[t], y[t])
    return {"n_genes": n, "rho": float(rho), "p": float(p),
            "ci_lo": float(np.nanquantile(boot, 0.025)), "ci_hi": float(np.nanquantile(boot, 0.975)),
            "null_mean": float(null.mean()), "null_q975": float(np.quantile(null, 0.975)),
            "p_perm": float((1 + (np.abs(null) >= abs(rho)).sum()) / (1 + n_perm)),
            "n_top": int(t.sum()), "rho_top": rho_top, "sign_agree_top": agree,
            "sign_agree_top_null": float(null_agree.mean()),
            "p_sign_top": float((1 + (null_agree >= agree).sum()) / (1 + n_perm))}  # fmt: skip


def evaluate(effects: pl.DataFrame, min_prompts: int, n_boot: int = 2000, n_perm: int = 2000, seed: int = 0):
    """Per condition: MaxToki's per-gene effect against the screen's aging coefficient and against our clock's shift;
    with, for the screen rows, our clock's own Spearman on the same genes (the bar, matched)."""
    s = screen_table()
    e = effects.filter(pl.col("n_prompts") >= min_prompts)
    rows = []
    for cond in sorted(s["condition"].unique().to_list()):
        j = e.join(s.filter(pl.col("condition") == cond), on="gene")
        for ref in ("coef_mean", "shift"):
            d = j.filter(pl.col(ref).is_not_null() & pl.col("effect").is_not_null())
            if d.height < 10:
                continue
            r = compare(d["effect"].to_numpy(), d[ref].to_numpy(), n_boot, n_perm, seed)
            bar = d.filter(pl.col("shift").is_not_null()) if ref == "coef_mean" else None
            r["clock_rho_same_genes"] = (
                _spearman(bar["shift"].to_numpy(), bar["coef_mean"].to_numpy()) if bar is not None else None
            )
            rows.append({"condition": cond, "reference": "screen" if ref == "coef_mean" else "clock", **r})
    return pl.DataFrame(rows)


# ------------------------------------------------------------------ the run


def cell_cross_entropy(model, cells: list[list[int]]) -> float:
    """Mean next-gene cross-entropy over single cells (the stage-1 task): a check that the converted weights still model
    genes (uniform over the vocabulary's genes ≈ ln 20,271 ≈ 9.9)."""
    import torch

    losses = []
    with torch.inference_mode():
        for c in cells:
            ids = torch.tensor([c], device=model.device)
            losses.append(float(model(input_ids=ids, labels=ids).loss))
    return float(np.mean(losses))


def _validate(model_factory, prompts: list[Prompt], vocab, factors: tuple[float, ...], batch: int) -> pl.DataFrame:
    """Unperturbed answers against the true timelapse, per RoPE factor."""
    from scipy.stats import pearsonr, spearmanr

    rows = []
    t_true = np.array([p.t_true for p in prompts])
    for f in factors:
        model = model_factory(f)
        ce = cell_cross_entropy(model, [p.query for p in prompts[:20]])
        reader = TimeReader(model, vocab, batch)
        res = [reader(p.context, [p.query]) for p in prompts]
        t = np.array([r[0][0] for r in res])
        m = np.array([r[1][0] for r in res])
        base = np.median(t_true)  # a constant answer: the median true timelapse
        rows.append({"rope_factor": f, "n": len(prompts), "spearman": float(spearmanr(t, t_true)[0]),
                     "pearson": float(pearsonr(t, t_true)[0]), "mae_months": float(np.abs(t - t_true).mean()),
                     "mae_constant_months": float(np.abs(base - t_true).mean()), "slope": float(np.polyfit(t_true, t, 1)[0]),
                     "time_mass_mean": float(m.mean()), "cell_ce": ce})  # fmt: skip
        log.info("validation %s", rows[-1])
        del reader, model
    return pl.DataFrame(rows)


def run(n_prompts: int = 300, n_validate: int = 150, per_bin: int = 40, max_genes_cell: int = 1300,
        max_screen_genes: int | None = None, batch: int = 16, factors: str = "1,4,8", min_prompts: int = 5,
        seed: int = 0, time_budget_h: float | None = None) -> pl.DataFrame:  # fmt: skip
    import time

    import torch

    from rna_age.train.maxtoki import gene_medians, load_model, token_dictionary

    torch.cuda.init() if torch.cuda.is_available() else None
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    WORK.mkdir(parents=True, exist_ok=True)
    vocab, medians = token_dictionary(), gene_medians()
    cells = sample_cells(per_bin, seed)
    toks = encode_cells(cells, vocab, medians, max_genes_cell)
    log.info("cells: %d (%s); tokens per cell median %d", cells.height,
             dict(cells.group_by("ctype").len().iter_rows()), int(np.median([len(t) for t in toks])))  # fmt: skip
    prompts = build_prompts(cells, toks, vocab, n_validate + n_prompts, seed)
    val, scr = prompts[:n_validate], prompts[n_validate:]

    vpath = OUTPUT / "summary" / "maxtoki_cd4_validation.parquet"
    if vpath.exists():
        v = pl.read_parquet(vpath)
    else:
        fs = tuple(float(x) for x in factors.split(","))
        v = _validate(lambda f: load_model(dev, "float16" if dev == "cuda" else "float32", f), val, vocab, fs, batch)
        v.write_parquet(vpath)
    best = float(v.sort("spearman", descending=True)["rope_factor"][0])
    log.info("RoPE factor for the screen: %s", best)

    genes = screen_genes(vocab, max_screen_genes, seed)
    tok2gene = dict(zip(genes["token"].to_list(), genes["gene"].to_list(), strict=True))
    reader = TimeReader(load_model(dev, "float16" if dev == "cuda" else "float32", best), vocab, batch)
    parts = WORK / f"effects_rope{best:g}"
    parts.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_done = 0
    for k, p in enumerate(scr):
        part = parts / f"prompt-{k:05d}.parquet"
        if part.exists():
            continue
        if time_budget_h is not None and time.time() - t0 > time_budget_h * 3600:
            log.info("time budget reached after %d prompts this run", n_done)
            break
        q = p.query
        hits = [i for i, t in enumerate(q) if t in tok2gene]
        variants = [q] + [q[:i] + q[i + 1 :] for i in hits]
        t, m = reader(p.context, variants)
        pl.DataFrame({"prompt": [k] * len(hits), "gene": [tok2gene[q[i]] for i in hits],
                      "rank": [i for i in hits], "query_len": [len(q) - 2] * len(hits),
                      "t_base": [t[0]] * len(hits), "t_pert": t[1:], "delta": t[1:] - t[0],
                      "t_true": [p.t_true] * len(hits), "mass_base": [m[0]] * len(hits)},
                     schema_overrides={"prompt": pl.Int32, "rank": pl.Int32, "query_len": pl.Int32}).write_parquet(part)  # fmt: skip
        n_done += 1
        if n_done % 10 == 0:
            el = time.time() - t0
            log.info("%d prompts this run, %.1f s/prompt, %d deletions in the last", n_done, el / n_done, len(hits))
    return summarise(parts, min_prompts, seed)


def summarise(parts: Path, min_prompts: int = 5, seed: int = 0) -> pl.DataFrame:
    d = pl.read_parquet(sorted(parts.glob("prompt-*.parquet")))
    effects = (
        d.group_by("gene")
        .agg(
            effect=pl.col("delta").mean(),
            effect_sd=pl.col("delta").std(),
            n_prompts=pl.col("prompt").n_unique(),
            mean_rank=pl.col("rank").mean(),
        )
        .sort("effect")
    )
    effects.write_parquet(OUTPUT / "summary" / "maxtoki_cd4_genes.parquet")
    res = evaluate(effects, min_prompts, seed=seed).with_columns(
        n_prompts_run=pl.lit(d["prompt"].n_unique()), min_prompts=pl.lit(min_prompts), run=pl.lit(parts.name)
    )
    res.write_parquet(OUTPUT / "summary" / "maxtoki_cd4.parquet")
    return res
