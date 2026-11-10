"""Health-aware age models and their evaluation.

1. ``disease_category``: free-text disease labels → categories; generic labels ("case", "affected") stay
   uncategorised and are not used; control-like labels ("ctl", "ngt", "normo", "not ibd") count as controls.
2. ``evaluate_health(run)``: on held-out studies (val + test, bulk, exact ages), the age gap (pred − age, minus the
   residual trend on age fitted on controls) of cases minus controls *within the same study*, per category, with
   95 % bootstrap CIs over studies (≥ 5 cases and ≥ 5 controls per study). Also chronological MAE on controls.
3. ``train_health``: MLP on 20k log-CPM HVGs with
   - age loss (Huber) on controls / healthy-or-unknown samples only (so disease never shapes the age target),
   - optional within-study ranking loss: for (case, control) pairs from the same training study, the case's gap
     should exceed the control's by a margin (only for aging-linked categories),
   - optional disease head (logit) trained on case vs control; ``bio_age = age_pred + β·logit`` with β chosen on
     validation to maximise the within-study gap at fixed MAE cost (reported separately).
"""

from __future__ import annotations

import logging
import re
import time

import numpy as np
import polars as pl

from rna_age.paths import OUTPUT
from rna_age.schema import age_mid
from rna_age.train.baselines import PRED_COLS
from rna_age.train.features import load_matrix, log_cpm, sample_table, select_hvg, shared_genes
from rna_age.train.runs import finish_run, new_run

log = logging.getLogger(__name__)

CATEGORIES: list[tuple[str, str]] = [
    ("progeroid", r"progeria|hgps|werner|down syndrome|trisomy 21|cockayne"),
    (
        "neurodegenerative",
        r"alzheimer|\bad\b|parkinson|\bpd\b|\bals\b|amyotrophic|huntington|dementia|\bftd\b|"
        r"frontotemporal|lewy|neurodegener|\bmci\b",
    ),  # fmt: skip
    (
        "autoimmune_inflammatory",
        r"rheumatoid|\bra\b|lupus|\bsle\b|sj[oö]gren|psoria|crohn|\bcd\b|colitis|\buc\b|\bibd\b|"
        r"multiple sclerosis|\bms\b|systemic sclerosis|scleroderma|dermatomyositis|myositis|"
        r"spondyl|vasculitis|sarcoid|eczema|atopic|asthma|t1d|type 1 diab",
    ),  # fmt: skip
    ("fibrotic_lung", r"\bipf\b|pulmonary fibrosis|fibrotic|copd|emphysema|interstitial lung|\bild\b|pneumonitis"),
    (
        "metabolic",
        r"t2d|type 2 diab|diabet|obes|nafld|nash|steatohepatitis|steatosis|metabolic syndrome|insulin resist",
    ),
    (
        "cardiovascular",
        r"cardiomyopathy|heart failure|\bhf\b|atheroscl|coronary|myocardial infarction|hypertens|"
        r"atrial fibrillation|aortic stenosis",
    ),  # fmt: skip
    ("kidney", r"\bckd\b|chronic kidney|nephropathy|end.stage renal|esrd|hemodialysis|dialysis"),
    (
        "infection",
        r"covid|sars|sepsis|septic|influenza|\bhiv\b|tuberculosis|\btb\b|malaria|candid|bacterial|viral|"
        r"infection|ltbi",
    ),  # fmt: skip
    ("psychiatric", r"\bmdd\b|depress|schizophren|psychosis|bipolar|autism|\basd\b|ptsd|anxiety"),
]
AGING_LINKED = {"progeroid", "neurodegenerative", "autoimmune_inflammatory", "fibrotic_lung", "metabolic",
                "cardiovascular", "kidney"}  # fmt: skip
CONTROL_LIKE = r"^(?:ctl|ctrl|ngt|normo.*|not ibd|non[- ]?ibd|asymptomatic|hc|nc|healthy.*|normal.*|control.*)$"


def disease_category(label: str | None) -> str | None:
    if not label:
        return None
    s = label.lower().strip()
    if re.match(CONTROL_LIKE, s):
        return "control"
    for cat, rx in CATEGORIES:
        if re.search(rx, s):
            return cat
    return None


def health_table() -> pl.DataFrame:
    """Samples with split, plus ``group`` ∈ {control, case, other} and ``category`` for cases."""
    m = sample_table()
    cats = [disease_category(d) for d in m["disease"].to_list()]
    m = m.with_columns(category=pl.Series(cats, dtype=pl.String))
    return m.with_columns(
        group=pl.when(pl.col("is_healthy").fill_null(True) | (pl.col("category") == "control"))
        .then(pl.lit("control"))
        .when(pl.col("category").is_not_null())
        .then(pl.lit("case"))
        .otherwise(pl.lit("other"))
    )


def _boot_ci(x: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    if len(x) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def evaluate_health(run_id: str, score: str = "pred", min_per_group: int = 5) -> pl.DataFrame:
    p = pl.read_parquet(OUTPUT / "runs" / run_id / "predictions.parquet")
    h = health_table().select("sample_id", "group", "category")
    p = p.join(h, on="sample_id", how="inner").filter(
        pl.col("split").is_in(["val", "test"]) & (pl.col("age_type") == "exact") & ~pl.col("is_pseudobulk")
    )
    if score not in p.columns:
        raise KeyError(score)
    ctrl = p.filter(pl.col("group") == "control")
    coef = np.polyfit(ctrl["age_years"].to_numpy(), (ctrl[score] - ctrl["age_years"]).to_numpy(), 1)
    p = p.with_columns(gap=pl.col(score) - pl.col("age_years") - (coef[0] * pl.col("age_years") + coef[1]))
    rows = []
    cases = p.filter(pl.col("group") == "case")
    for cat in ["all_aging_linked", "all", *sorted(cases["category"].unique().drop_nulls().to_list())]:
        if cat == "all":
            c = cases
        elif cat == "all_aging_linked":
            c = cases.filter(pl.col("category").is_in(list(AGING_LINKED)))
        else:
            c = cases.filter(pl.col("category") == cat)
        cs = c.group_by("study_id").agg(case_gap=pl.col("gap").mean(), n_case=pl.len())
        ks = p.filter(pl.col("group") == "control").group_by("study_id").agg(ctrl_gap=pl.col("gap").mean(),
                                                                            n_ctrl=pl.len())  # fmt: skip
        j = cs.join(ks, on="study_id").filter((pl.col("n_case") >= min_per_group) & (pl.col("n_ctrl") >= min_per_group))
        if not j.height:
            continue
        d = (j["case_gap"] - j["ctrl_gap"]).to_numpy()
        lo, hi = _boot_ci(d)
        rows.append({"run_id": run_id, "score": score, "category": cat, "studies": j.height,
                     "cases": int(j["n_case"].sum()), "within_study_gap": float(d.mean()), "ci_low": lo, "ci_high": hi,
                     "frac_studies_positive": float((d > 0).mean())})  # fmt: skip
    out = pl.DataFrame(rows)
    test_ctrl = p.filter((pl.col("split") == "test") & (pl.col("group") == "control"))
    out = out.with_columns(control_test_mae=pl.lit(float((test_ctrl[score] - test_ctrl["age_years"]).abs().mean())))
    path = OUTPUT / "summary" / f"health_{run_id}_{score}.parquet"
    out.write_parquet(path)
    return out


def _pairs(t: pl.DataFrame, train_mask: np.ndarray, aging_only: bool, max_pairs: int = 200_000, seed: int = 0):
    """(case_idx, control_idx) pairs within the same training study."""
    rng = np.random.default_rng(seed)
    t = t.with_row_index("_i").filter(pl.Series(train_mask) & (pl.col("age_type") == "exact"))
    cases = t.filter(pl.col("group") == "case")
    if aging_only:
        cases = cases.filter(pl.col("category").is_in(list(AGING_LINKED)))
    ctrls = t.filter(pl.col("group") == "control")
    pairs = []
    for (sid,), cdf in cases.group_by("study_id"):
        k = ctrls.filter(pl.col("study_id") == sid)["_i"].to_numpy()
        if len(k) < 3:
            continue
        ci = cdf["_i"].to_numpy()
        n = min(len(ci) * 20, 5000)
        pairs.append(np.stack([rng.choice(ci, n), rng.choice(k, n)], axis=1))
    if not pairs:
        return np.zeros((0, 2), dtype=np.int64)
    p = np.concatenate(pairs)
    return p[rng.permutation(len(p))[:max_pairs]]


def train_health(
    rank_weight: float = 0.0,
    margin: float = 3.0,
    disease_head: bool = False,
    aging_only: bool = True,
    n_hvg: int = 20000,
    seed: int = 0,
    epochs: int = 200,
    name: str | None = None,
) -> dict:
    import torch

    t0 = time.time()
    t = health_table()
    t = t.with_columns(target=age_mid(t))
    t = t.filter(~pl.col("is_pseudobulk"))  # bulk only here; pseudobulk transfer is evaluated elsewhere
    split = t["split"].to_numpy()
    is_train = split == "train"
    exact = (t["age_type"] == "exact").to_numpy()
    gtex = (t["source"] == "gtex").to_numpy()
    ctrl = (t["group"] == "control").to_numpy()
    age_mask = is_train & ctrl & (exact | gtex)  # age loss only on controls (+ GTEx bins as midpoints)
    genes = shared_genes()
    counts = load_matrix(t.select("sample_id", "source"), genes)
    lc = log_cpm(counts)
    del counts
    hvg = select_hvg(lc[np.flatnonzero(age_mask)[:20000]], genes, n_hvg)
    X = lc[:, [genes.index(g) for g in hvg]]
    del lc
    mu, sd = X[age_mask].mean(0), X[age_mask].std(0) + 1e-6
    X = ((X - mu) / sd).astype(np.float32)
    y = np.nan_to_num(t["target"].to_numpy(), nan=30.0).astype(np.float32)
    case = (t["group"] == "case").to_numpy()
    if aging_only:
        case = case & t["category"].is_in(list(AGING_LINKED)).fill_null(False).to_numpy()
    head_mask = is_train & exact & (ctrl | case)
    pairs = _pairs(t, is_train, aging_only, seed=seed) if rank_weight > 0 else np.zeros((0, 2), np.int64)
    log.info("health MLP: %d age-training samples, %d case/control pairs, disease head %s", age_mask.sum(), len(pairs),
             disease_head)  # fmt: skip

    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xt = torch.as_tensor(X, device=dev)
    yt = torch.as_tensor(y, device=dev)
    lab = torch.as_tensor(case.astype(np.float32), device=dev)
    body = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 512), torch.nn.GELU(), torch.nn.Dropout(0.2),
                               torch.nn.Linear(512, 128), torch.nn.GELU()).to(dev)  # fmt: skip
    age_head = torch.nn.Linear(128, 1).to(dev)
    d_head = torch.nn.Linear(128, 1).to(dev)
    params = list(body.parameters()) + list(age_head.parameters()) + (list(d_head.parameters()) if disease_head else [])
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-2)
    tr = np.flatnonzero(age_mask)
    hm = np.flatnonzero(head_mask)
    val = (split == "val") & exact & ctrl
    va = np.flatnonzero(val)
    rng = np.random.default_rng(seed)

    def fwd(idx, train):
        body.train(train)
        z = body(Xt[idx])
        return age_head(z).squeeze(-1), d_head(z).squeeze(-1)

    best, best_state, bad = np.inf, None, 0
    for _epoch in range(epochs):
        order = rng.permutation(tr)
        nb = max(1, len(order) // 512)
        for b in np.array_split(order, nb):
            a, _ = fwd(b, True)
            loss = torch.nn.functional.huber_loss(a, yt[b], delta=5.0)
            if rank_weight > 0 and len(pairs):
                pp = pairs[rng.integers(0, len(pairs), 256)]
                ac, _ = fwd(pp[:, 0], True)
                ak, _ = fwd(pp[:, 1], True)
                gap = (ac - yt[pp[:, 0]]) - (ak - yt[pp[:, 1]])
                loss = loss + rank_weight * torch.relu(margin - gap).mean()
            if disease_head and len(hm):
                hb = hm[rng.integers(0, len(hm), 512)]
                _, lg = fwd(hb, True)
                loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(lg, lab[hb])
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            a, _ = fwd(va, False)
            mae = (a - yt[va]).abs().mean().item()
        if mae < best - 0.01:
            best, bad = mae, 0
            best_state = [{k: v.clone() for k, v in m.state_dict().items()} for m in (body, age_head, d_head)]
        else:
            bad += 1
            if bad >= 15:
                break
    for m, s in zip((body, age_head, d_head), best_state, strict=True):
        m.load_state_dict(s)
    with torch.no_grad():
        outs = [fwd(idx, False) for idx in np.array_split(np.arange(len(y)), max(1, len(y) // 4096))]
    pred = torch.cat([o[0] for o in outs]).cpu().numpy().astype(np.float64)
    logit = torch.cat([o[1] for o in outs]).cpu().numpy().astype(np.float64)
    out = t.select([c for c in PRED_COLS if c in t.columns]).with_columns(
        pred=pl.Series(pred), health_logit=pl.Series(logit)
    )
    beta = 0.0
    if disease_head:  # bio_age = pred + β·logit, β chosen on val to maximise the within-study gap per unit MAE cost
        cand = []
        vdf = out.join(t.select("sample_id", "group", "category"), on="sample_id").filter(
            (pl.col("split") == "val") & (pl.col("age_type") == "exact")
        )
        for b in (0.0, 1.0, 2.0, 4.0, 8.0):
            v = vdf.with_columns(s=pl.col("pred") + b * pl.col("health_logit"))
            c = v.filter(pl.col("group") == "control")
            mae = float((c["s"] - c["age_years"]).abs().mean())
            g = v.with_columns(gap=pl.col("s") - pl.col("age_years"))
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
            gap = float((j["cg"] - j["kg"]).mean()) if j.height else 0.0
            cand.append((b, mae, gap))
        base_mae = cand[0][1]
        ok = [c for c in cand if c[1] <= base_mae + 1.0]  # allow ≤ 1 y MAE cost on val controls
        beta = max(ok, key=lambda c: c[2])[0]
        out = out.with_columns(bio_age=pl.col("pred") + beta * pl.col("health_logit"))
    run_dir = new_run(
        name or f"health-mlp-rank{rank_weight:g}{'-head' if disease_head else ''}{'' if aging_only else '-alldis'}-s{seed}",
        {"model": "health-aware MLP", "features": f"logCPM HVG {n_hvg}", "rank_weight": rank_weight, "margin": margin,
         "disease_head": disease_head, "aging_linked_only": aging_only, "beta": beta, "n_pairs": int(len(pairs)),
         "train_set": "age loss on controls (bulk train); ranking/head on train-study cases"},
    )  # fmt: skip
    row = finish_run(run_dir, out, {"seconds": time.time() - t0, "beta": beta})
    for score in ["pred"] + (["bio_age"] if disease_head else []):
        res = evaluate_health(run_dir.name, score)
        res.write_parquet(run_dir / f"health_{score}.parquet")
        log.info("health eval %s %s:\n%s", run_dir.name, score, res.select("category", "studies", "within_study_gap",
                                                                          "ci_low", "ci_high"))  # fmt: skip
    return row
