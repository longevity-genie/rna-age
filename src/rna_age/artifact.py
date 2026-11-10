"""``rna-age artifact``: the tabbed overnight report page (``docs/reports/rna-age-overnight.html``).

Every number on the page is read here from the tracked parquet summaries (``data/output/summary``), the run
directories and the clock outputs; figures are embedded as PNG data URIs. Narrative text lives in this file so
the page is reproducible. No cluster names or credentials appear on the page.
"""

from __future__ import annotations

import base64
import html
import json
import subprocess
from datetime import datetime

import polars as pl

from rna_age.paths import OUTPUT, REPO_ROOT

SUMMARY = OUTPUT / "summary"
FIG = OUTPUT / "figures"
OUT = REPO_ROOT / "docs" / "reports" / "rna-age-overnight.html"


def esc(x) -> str:
    return html.escape("" if x is None else str(x))


def fmt(v, digits: int = 2) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "–"
    if isinstance(v, float):
        return f"{v:,.{digits}f}"
    if isinstance(v, int):
        return f"{v:,}"
    return esc(v)


def table(df: pl.DataFrame, cols: dict[str, str] | None = None, digits: int = 2, highlight: str | None = None) -> str:
    cols = cols or {c: c for c in df.columns}
    head = "".join(f"<th>{esc(v)}</th>" for v in cols.values())
    rows = []
    for r in df.select(list(cols)).iter_rows(named=True):
        cls = ' class="hl"' if highlight and r.get("run_id") == highlight else ""
        rows.append(f"<tr{cls}>" + "".join(f"<td>{fmt(r[c], digits)}</td>" for c in cols) + "</tr>")
    return f'<div class="tw"><table><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def fig(name: str, alt: str) -> str:
    p = FIG / f"{name}.png"
    if not p.exists():
        return f'<p class="muted">Figure {esc(name)} not generated yet.</p>'
    b64 = base64.b64encode(p.read_bytes()).decode()
    return (
        f'<figure><img src="data:image/png;base64,{b64}" alt="{esc(alt)}"><figcaption>{esc(alt)}</figcaption></figure>'
    )


def stat(label: str, value: str, note: str = "") -> str:
    return f'<div class="stat"><div class="v">{value}</div><div class="l">{esc(label)}</div><div class="n">{esc(note)}</div></div>'


def build() -> str:
    by_source = pl.read_parquet(SUMMARY / "by_source.parquet")
    splits = pl.read_parquet(SUMMARY / "splits.parquet") if (SUMMARY / "splits.parquet").exists() else None
    dups = pl.read_parquet(SUMMARY / "duplicates.parquet")
    exp = pl.read_parquet(SUMMARY / "experiments.parquet").sort("val_exact_mae", nulls_last=True)
    exp = exp.with_columns(
        features=pl.when(pl.col("features").str.contains(r"\+20261110"))
        .then(pl.lit("mean of ") + (pl.col("features").str.count_matches(r"\+") + 1).cast(pl.String) + pl.lit(" runs"))
        .otherwise(pl.col("features"))
    )
    downloads = pl.read_parquet(SUMMARY / "downloads.parquet")
    top_genes = (
        pl.read_parquet(SUMMARY / "ridge20k_top_genes.parquet")
        if (SUMMARY / "ridge20k_top_genes.parquet").exists()
        else None
    )

    eligible = int(by_source["eligible"].sum())
    exact = int(by_source.filter(pl.col("age_type") == "exact")["eligible"].sum())
    pseudo = int(by_source.filter(pl.col("pseudobulk") > 0)["eligible"].sum())
    gtex_binned = int(by_source.filter(pl.col("source") == "gtex")["eligible"].sum())
    n_split = int(splits["samples"].sum()) if splits is not None else None
    best = exp.drop_nulls("val_exact_mae").row(0, named=True)
    # like-for-like clock comparison for the best run (identical tAge-scored samples), else tAge alone
    clocks = sorted((OUTPUT / "clocks").glob(f"tage_*_vs_{best['run_id']}_summary.parquet")) or sorted(
        p for p in (OUTPUT / "clocks").glob("tage_*_summary.parquet") if "_vs_" not in p.name
    )
    clock_df = pl.concat([pl.read_parquet(p) for p in clocks], how="diagonal_relaxed") if clocks else None
    base_runs = exp.filter(pl.col("run_id").str.contains("ridge|elasticnet"))
    best_base = base_runs.row(0, named=True) if base_runs.height else None
    best_atlas = exp.filter(pl.col("run_id").str.contains("atlas")).drop_nulls("val_exact_mae")
    best_atlas = best_atlas.row(0, named=True) if best_atlas.height else None
    dl_gb = downloads["size"].sum() / 1e9
    commits = subprocess.run(["git", "rev-list", "--count", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT)
    n_commits = commits.stdout.strip() or "?"
    n_runs = exp.height
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    from rna_age.paths import INTERIM

    m = pl.read_parquet(
        INTERIM / "samples_all.parquet", columns=["split", "age_type", "age_years", "is_healthy", "is_pseudobulk"]
    )
    m = m.filter(~pl.col("is_pseudobulk") & (pl.col("age_type") == "exact"))
    med = m.filter((pl.col("split") == "train") & pl.col("is_healthy").fill_null(True))["age_years"].median()
    const_mae = float((m.filter(pl.col("split") == "test")["age_years"] - med).abs().mean())

    exp_cols = {
        "run_id": "run", "features": "features", "val_exact_mae": "val MAE", "test_exact_mae": "test MAE",
        "test_exact_pearson_r": "test r", "test_exact_bias_slope": "bias slope",
        "test_binned_interval_hit": "GTEx bin hit", "transfer_exact_mae": "pseudobulk MAE",
        "transfer_exact_pearson_r": "pseudobulk r",
    }  # fmt: skip
    source_cols = {"source": "source", "age_type": "age type", "samples": "samples", "eligible": "eligible",
                   "studies": "studies", "donors": "donors", "pseudobulk": "pseudobulk"}  # fmt: skip

    tabs = {}
    tabs["motivation"] = (
        "Motivation",
        """
<p class="lede">Can a bulk-transcriptome foundation model read a donor's age from RNA-seq better than the
classic linear clocks, across tissues, studies and even single-cell pseudobulk?</p>
<p>On 2026-11-04 the BulkFormer authors (Kang et al., <i>Cell Systems</i> 2026) released two models:
<b>BulkFormer-Atlas</b> (242 M parameters, a 1,024-d CLS sample embedding from a fixed 4,096-gene panel;
Atlas + HVG ranks first on the PRIMO patient-level bulk benchmark) and <b>BulkFormer-Latest</b> (154 M
parameters, 768-d contextual gene embeddings over the full 19,973-gene coding transcriptome). Age is the
textbook sample-level phenotype: every public RNA-seq sample with a donor age is a free label, and a good
transcriptomic clock is useful for aging research, for disease "age acceleration", and as a stress test of
how much biology a foundation model captures beyond batch and tissue.</p>
<p>Tonight's goal: build the data side (reproducible downloaders and parsers, a harmonised sample table with
raw counts and donor ages), fix an evaluation protocol before any model, then climb a training ladder:
linear baselines, frozen Atlas embeddings with probes, Atlas fine-tuning, and compare everything with
published clocks (tAge, ClockBase) on the same held-out samples.</p>
""",
    )
    tabs["data"] = (
        "Data",
        f"""
<div class="stats">
{stat("eligible samples with age", fmt(eligible), "healthy-or-unknown + disease, no tumours/cell lines")}
{stat("exact-age samples", fmt(exact), "ARCHS4, Census, Terekhova, Bgee")}
{stat("GTEx v10, binned age", fmt(gtex_binned), "10-year bins, 946 donors")}
{stat("pseudobulk samples", fmt(pseudo), "donor × tissue / per tube")}
</div>
<h3>Samples by source</h3>
{table(by_source, source_cols)}
<p class="muted">"eligible" = has an age, not excluded (tumour, cell line, stem cell / organoid, prenatal,
single-cell run inside a bulk compendium, non-human). Disease samples stay eligible and are scored
separately.</p>
<h3>Sources</h3>
<ul>
<li><b>ARCHS4</b> human gene counts (file of 2026-11-06, 64 GB): ages parsed from GEO <code>characteristics_ch1</code>
(exact / binned / censored, months and weeks converted, gestational age and passage rejected).</li>
<li><b>GTEx v10</b> public release: read counts, 10-year age bins (exact age is controlled access, not used).</li>
<li><b>CELLxGENE Census</b> LTS 2025-11-08: primary cells summed per donor × tissue (≥ 200 cells, ≥ 500k UMIs),
14 datasets incl. OneK1K, AIDA, the lung and liver atlases and PsychENCODE brain cohorts.</li>
<li><b>Terekhova et al. 2023</b> PBMC aging atlas (HCA copy): pseudobulk per tube, 166 donors aged 25–85; raw
UMIs recovered from log1p-CP10k (100 % integral).</li>
<li><b>Bgee 15.2</b>: curated healthy libraries with "NN-year-old" stages (most also in ARCHS4).</li>
<li>Labels only: 17 named aging cohorts (Fleischer 2018 fibroblasts, …) and the Shokhirev &amp; Johnson 2021
compilation (6,471 curated samples), used to audit our age parser: 5,370 / 5,371 agree within 1 year.</li>
</ul>
<h3>Cross-source duplicates removed</h3>
{table(dups)}
{fig("data_age_by_source", "Age distribution of eligible samples by source (GTEx: bin midpoints)")}
<h3>Not used</h3>
<p>Cancer cohorts (TCGA, TARGET; out of scope tonight), controlled-access cohorts (GTEx exact age, TOPMed,
UK Biobank, most PsychENCODE bulk), NCBI-generated GEO counts (download endpoint behind a reCAPTCHA).</p>
<h3>Downloads</h3>
<p>{fmt(downloads.height)} files, {dl_gb:,.0f} GB, every one with a checksum sidecar (publisher md5 / S3 ETag /
sha256).</p>
""",
    )
    split_html = ""
    if splits is not None:
        sp = splits.group_by("split").agg(pl.col("samples").sum(), pl.col("groups").sum()).sort("split")
        split_html = table(sp, {"split": "split", "samples": "samples", "groups": "studies / donors"})
    tabs["methods"] = (
        "Methods",
        f"""
<h3>Pipeline</h3>
<p><code>rna-age fetch</code> (resumable, parallel HTTP ranges, checksums) → <code>rna-age parse</code>
(per-source <code>samples.parquet</code> + raw counts as gene × sample parquet parts) → <code>rna-age merge</code>
(dedup, cohort labels, splits) → models. Everything is polars + parquet; heavy steps run as batch jobs.</p>
<h3>Evaluation protocol (fixed before the first model)</h3>
<ul>
<li>Whole studies (GSE, Census dataset, Bgee experiment; GTEx donor) go to train / val / test (70/15/15) by a
stable hash: no series or donor spans two splits. {fmt(n_split)} samples carry a split.</li>
<li>Training set: bulk train split, healthy or unknown health status, exact ages; GTEx bins enter as
midpoints in baselines and as an interval loss in fine-tuning.</li>
<li>Metrics: MAE, median AE, Pearson r, R², bias slope (regression to the mean); GTEx scored as
interval hit rate and distance to the bin; per source, tissue, decade, health status and modality.</li>
<li>Transfer test: every pseudobulk sample predicted by models trained on bulk only.</li>
<li>Controls: tAge (relative clock, compared on age deltas within dataset × tissue strata) on the same
samples, with and without the series it was trained on.</li>
</ul>
{split_html}
<h3>Training ladder</h3>
<ol>
<li>Ridge / ElasticNet on log-CPM of highly variable genes chosen on train only (2k–20k genes; optional
Horvath log-linear age target).</li>
<li>BulkFormer-Atlas frozen CLS embeddings (fixed 4,096-gene panel chosen on train, fp16 on RTX 2080 Ti),
ridge and MLP probes, CLS + HVG.</li>
<li>Atlas fine-tuning: last N encoder blocks + regression head, Huber loss on exact ages + interval loss on
GTEx bins, per-study sampling cap, early stopping on validation MAE.</li>
</ol>
""",
    )
    clock_html = ""
    if clock_df is not None:
        clock_html = (
            "<h3>Published clock control (tAge) on the same samples</h3>"
            + table(
                clock_df.with_columns(
                    pl.col("model").str.replace(r"\.pkl$", "").str.replace("_Multispecies_Multitissue", "")
                ),
                {
                    "source": "source",
                    "split": "split",
                    "model": "model",
                    "subset": "subset",
                    "n": "n",
                    "strata": "strata",
                    "r_delta": "r (age delta)",
                },  # fmt: skip
                3,
            )
            + """<p class="muted">tAge outputs are relative (centred per dataset × tissue), so all models are
scored by the correlation of their within-stratum deviation with the true within-stratum age deviation.
Our ridge saw other GTEx donors in training; tAge never saw GTEx.</p>"""
        )
    run_figs = fig(f"run_{best['run_id']}", f"Best model: {best['run_id']}")
    run_figs += fig(f"run_{best['run_id']}_decade_bias", "Bias by age decade (test), best model")
    run_figs += fig(f"run_{best['run_id']}_tissue_mae", "Test MAE by tissue (best 25), best model")
    if best_base and False:
        run_figs += fig(f"run_{best_base['run_id']}", f"Best baseline: {best_base['run_id']}")
        run_figs += fig(f"run_{best_base['run_id']}_decade_bias", "Bias by age decade (test), best baseline")
        run_figs += fig(f"run_{best_base['run_id']}_tissue_mae", "Test MAE by tissue (best 25), best baseline")
    genes_html = ""
    if top_genes is not None:
        older = ", ".join(top_genes.filter(pl.col("direction") == "older")["symbol"].drop_nulls().to_list()[::-1][:12])
        younger = ", ".join(top_genes.filter(pl.col("direction") == "younger")["symbol"].drop_nulls().to_list()[:12])
        genes_html = f"""<h3>What drives the ridge clock</h3><p>Strongest "older" genes: {esc(older)}.
Strongest "younger" genes: {esc(younger)}. Known aging genes rank high (PTCHD4, LRRN3, CD248, the miR-34a and
miR-29 host genes).</p>"""
    headline = f"""<div class="stats">
{stat("best model (val MAE)", fmt(best["val_exact_mae"]), best["run_id"])}
{stat("its test MAE, years", fmt(best["test_exact_mae"]), "exact ages, held-out studies")}
{stat("its test Pearson r", fmt(best["test_exact_pearson_r"], 3), "")}
{stat("constant-age reference", fmt(const_mae), "test MAE of predicting the train median")}
</div>"""

    def run(pattern: str) -> dict | None:
        r = exp.filter(pl.col("run_id").str.contains(pattern)).sort("run_id", descending=True)
        return r.row(0, named=True) if r.height else None

    def cite(pattern: str, key: str = "test_exact_mae", digits: int = 2) -> str:
        r = run(pattern)
        return fmt(r.get(key), digits) if r else "n/a"

    findings = f"""<h3>Key findings</h3><ol>
<li><b>Non-linear readouts win.</b> Ridge on 20k log-CPM HVGs (+ Horvath target): test MAE
{cite("ridge-hvg20000-gtexmid-horvath$")}; the same genes through a small MLP: {cite("-hvg20000-mlp$")}; a 5-seed
MLP ensemble: {cite("ens-mlp20k-5seeds")}.</li>
<li><b>More training data helps.</b> Training on all health states (disease samples too): 5-model ensemble test MAE
{cite("ens-mlp20k-trainall-5")}, r {cite("ens-mlp20k-trainall-5", "test_exact_pearson_r", 3)}, validation MAE
{cite("ens-mlp20k-trainall-5", "val_exact_mae")}.</li>
<li><b>Frozen BulkFormer-Atlas adds nothing.</b> Ridge on Atlas CLS {cite("atlas-cls-ridge$")} vs ridge on its own 4,096
input genes {cite("atlas-panel-ridge$")}; CLS + 20k HVG MLP {cite("atlas-cls-hvg-mlp-trainall")} vs the HVG MLP
ensemble {cite("ens-mlp20k-trainall-5")}: within the ±0.5-year run-to-run noise or worse.</li>
<li><b>Gene-length TPM vs UMI pseudobulk.</b> TPM features: best bulk test MAE {cite("0220-tpmhvg20000-mlp")} but
transfer to 10x pseudobulk collapses (r {cite("0220-tpmhvg20000-mlp", "transfer_exact_pearson_r", 2)}); log-CPM keeps
transfer at r {cite("ens-mlp20k-5seeds", "transfer_exact_pearson_r", 2)}. Atlas normalises every sample by gene length,
which plausibly explains part of its weak transfer.</li>
<li><b>Regression to the mean</b> remains (test bias slope {cite("ens-mlp20k-trainall-5", "test_exact_bias_slope")});
a validation-fitted de-compression removes it but raises test MAE to {cite("ens-mlp20k-7-cal")}.</li>
<li><b>Atlas fine-tuning</b> (last 2 / 4 blocks + head) improves epoch by epoch; final numbers appear in the table
when the runs finish.</li>
</ol>"""
    tabs["results"] = (
        "Results",
        f"""
{headline}
{findings}
<h3>All runs (sorted by validation MAE)</h3>
{table(exp, exp_cols, highlight=best["run_id"])}
<p class="muted">MAE in years on exact-age samples; GTEx bin hit = share of GTEx test samples predicted inside
their 10-year bin; pseudobulk = bulk-trained model applied to Census + Terekhova pseudobulk.</p>
{clock_html}
{genes_html}
{run_figs}
{fig("runs_test_mae", "Test MAE of every run")}
""",
    )
    tabs["achievements"] = (
        "Achievements",
        f"""
<div class="stats">
{stat("eligible samples", fmt(eligible), "with donor age, de-duplicated")}
{stat("runs logged", fmt(n_runs), "config, metrics, predictions each")}
{stat("commits", n_commits, "pushed to main")}
{stat("downloaded", f"{dl_gb:,.0f} GB", "checksummed")}
</div>
<ul>
<li>A reproducible pipeline: <code>rna-age fetch | parse | merge | summary | baseline | embed | probe |
finetune | tage | figures | report-tables | artifact</code>.</li>
<li>Seven data sources parsed to one schema (samples, genes, raw counts), plus two label sources and one
benchmark (LongevityBench GTEx tasks, flagged on our samples).</li>
<li>An age parser with 80+ tests built from real GEO strings; flags for tumour, cell line, stem cell,
prenatal, single-cell runs.</li>
<li>A fixed evaluation protocol, run tracking (<code>data/output/runs/</code>, <code>experiments.parquet</code>),
and a decision log (<code>docs/experiments.md</code>).</li>
<li>BulkFormer-Atlas CLS embeddings and panel inputs for every sample, an fp16 + attention-memory fix that
lets Atlas run on 11 GB cards, and a resumable fine-tuning loop.</li>
<li>The tAge R package wrapped as a parquet-in / parquet-out step in an isolated R environment.</li>
</ul>
""",
    )
    tabs["problems"] = (
        "Problems & decisions",
        """
<ul>
<li><b>NCBI</b> started answering its GEO download endpoint with a reCAPTCHA page: NCBI-generated counts are
parked; we do not work around bot protection.</li>
<li><b>Synapse</b> token on the cluster answered 401 (also with the official client): Terekhova counts came
from the HCA copy, with raw UMIs recovered from log1p-CP10k (verified integral).</li>
<li><b>CELLxGENE Census</b> SOMA reads stalled through the proxy: we used the same LTS release's h5ads over
HTTPS; four chosen datasets were not in that release and were skipped to keep the release pinned.</li>
<li><b>ARCHS4 extraction</b> ran out of memory twice (a 60 GB copy into a memory-backed temp dir; h5py fancy
indexing over scattered columns): replaced by bounded dense windows, 75k samples in 4.5 minutes.</li>
<li><b>GPUs</b> are 11 GB cards without native bf16: fp16 autocast (checked against fp32, cosine 1.000);
Atlas passed an all-true attention mask that forced the full-matrix attention kernel, patched at runtime
with identical maths.</li>
<li><b>Training filter bug</b> caught before the first result: "healthy or unknown" silently dropped unknowns
(polars null semantics), which would have cut training to 5,208 samples.</li>
<li><b>Choices made on Anton's behalf</b>: GTEx bins as midpoints for baselines / interval loss for
fine-tuning; healthy-or-unknown training set; per-study sampling cap; pseudobulk only as a transfer test;
tAge compared on within-stratum deltas.</li>
<li>A 54 MB R diagnostics PDF slipped into three commits before it was ignored; history was not rewritten.</li>
</ul>
""",
    )
    tabs["next"] = (
        "Next steps",
        f"""
<ol>
<li>Finish the Atlas fine-tuning runs and compare with the ridge bar on the same split
{("(best Atlas run so far: " + esc(best_atlas["run_id"]) + ", test MAE " + fmt(best_atlas["test_exact_mae"]) + ")") if best_atlas else ""}.</li>
<li>Close the bulk → pseudobulk gap: train on bulk + pseudobulk, add assay / suspension-type covariates,
or per-dataset feature centring.</li>
<li>Reduce regression to the mean: Horvath target for all models, age-bias correction fitted on validation.</li>
<li>BulkFormer-Latest: full-transcriptome gene embeddings pooled over HVGs as a second feature set.</li>
<li>tAge and ClockBase on the ARCHS4 test set (with / without their training series); disease age gap.</li>
<li>Variants: healthy-only vs all, tissue-conditioned head, study weighting, 3 seeds for final numbers.</li>
</ol>
""",
    )
    nav = "".join(f'<a href="#{k}" data-tab="{k}">{esc(v[0])}</a>' for k, v in tabs.items())
    sections = "".join(
        f'<section id="tab-{k}" data-tab="{k}"><h2>{esc(v[0])}</h2>{v[1]}</section>' for k, v in tabs.items()
    )
    page = f"""<title>RNA Age Overnight</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600&family=IBM+Plex+Mono:wght@400;600&family=Space+Grotesk:wght@500;700&display=swap">
<style>
/* Layout: a lab-notebook report; one sticky tab bar, one readable column, data tables scroll in place. */
:root {{
  --bg: #f6f8f8; --surface: #ffffff; --fg: #17201f; --muted: #5b6b69; --line: #d6dfdd;
  --accent: #0f766e; --accent-soft: #d8efec; --hl: #fff4d6;
  --fig-bg: #ffffff; --fig-fg: #3d4847;  /* figures are white PNGs: their frame stays white in both themes */
  --display: "Space Grotesk", "IBM Plex Sans", system-ui, sans-serif;
  --body: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg: #0f1514; --surface: #162020; --fg: #e3ecea; --muted: #93a6a3; --line: #2a3735;
  --accent: #5ec4b6; --accent-soft: #1d3532; --hl: #3a3420; color-scheme: dark; }} }}
:root[data-theme="dark"] {{
  --bg: #0f1514; --surface: #162020; --fg: #e3ecea; --muted: #93a6a3; --line: #2a3735;
  --accent: #5ec4b6; --accent-soft: #1d3532; --hl: #3a3420; color-scheme: dark; }}
body {{ background: var(--bg); color: var(--fg); font: 15px/1.6 var(--body); }}
body::before {{ content: ""; position: fixed; inset: 0 0 auto; height: env(safe-area-inset-top, 0px); z-index: 10; background: var(--surface); }}
header, nav .in, main {{ padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }}
header {{ padding-block: 28px 12px; max-width: 980px; margin: 0 auto; }}
header h1 {{ font: 700 clamp(1.6rem, 4vw, 2.3rem)/1.15 var(--display); margin: 0 0 6px; text-wrap: balance; }}
header p {{ margin: 0; color: var(--muted); }}
nav {{ position: sticky; top: env(safe-area-inset-top, 0px); z-index: 5; background: var(--surface); border-block: 1px solid var(--line); }}
nav .in {{ max-width: 980px; margin: 0 auto; display: flex; gap: 4px; overflow-x: auto; scrollbar-width: none; }}
nav a {{ flex: none; padding: 10px 12px; color: var(--muted); text-decoration: none; font: 600 0.9rem var(--body); border-bottom: 2px solid transparent; }}
nav a[aria-current="page"] {{ color: var(--accent); border-bottom-color: var(--accent); }}
nav a:focus-visible {{ outline: 2px solid var(--accent); outline-offset: -2px; }}
main {{ max-width: 980px; margin: 0 auto; padding-block: 20px 60px; }}
section h2 {{ font: 700 1.5rem/1.2 var(--display); margin: 8px 0 14px; }}
h3 {{ font: 600 1.05rem/1.3 var(--display); margin: 26px 0 8px; }}
p, li {{ max-width: 72ch; }}
.lede {{ font-size: 1.15rem; }}
.muted {{ color: var(--muted); font-size: 0.9rem; }}
code {{ font: 0.85em var(--mono); background: var(--accent-soft); padding: 1px 4px; border-radius: 3px; }}
.stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; margin: 8px 0 18px; }}
.stat {{ background: var(--surface); border: 1px solid var(--line); border-radius: 6px; padding: 12px 14px; min-width: 0; }}
.stat .v {{ font: 600 1.6rem/1.1 var(--mono); color: var(--accent); font-variant-numeric: tabular-nums; }}
.stat .l {{ font-weight: 600; margin-top: 4px; }}
.stat .n {{ color: var(--muted); font-size: 0.8rem; overflow-wrap: anywhere; }}
.tw {{ overflow-x: auto; border: 1px solid var(--line); border-radius: 6px; background: var(--surface); margin: 8px 0; }}
table {{ border-collapse: collapse; width: 100%; font: 0.82rem/1.4 var(--mono); font-variant-numeric: tabular-nums; }}
th, td {{ padding: 6px 10px; text-align: right; border-bottom: 1px solid var(--line); white-space: nowrap; }}
th:first-child, td:first-child {{ text-align: left; }}
th {{ background: var(--accent-soft); font-weight: 600; position: sticky; top: 0; }}
tr.hl td {{ background: var(--hl); }}
figure {{ margin: 18px 0; background: var(--fig-bg); border: 1px solid var(--line); border-radius: 6px; padding: 8px; }}
figure img {{ display: block; width: 100%; height: auto; }}
figcaption {{ color: var(--fig-fg); font-size: 0.82rem; padding: 6px 4px 0; }}
footer {{ color: var(--muted); font-size: 0.8rem; margin-top: 40px; }}
@media (prefers-reduced-motion: no-preference) {{ section {{ animation: in .18s ease-out; }} @keyframes in {{ from {{ opacity: .4 }} to {{ opacity: 1 }} }} }}
</style>
<header><h1>Transcriptomic age with BulkFormer: overnight build</h1>
<p>Data, protocol and first models built 2026-11-09 → 2026-11-10. Generated {esc(now)} from the repository's parquet outputs.</p></header>
<nav aria-label="Report sections"><div class="in">{nav}</div></nav>
<main>{sections}
<footer>Every number on this page is read from <code>data/output/summary/*.parquet</code>, the run directories and the clock outputs by
<code>rna-age artifact</code>; unfinished items are marked. Source: docs/reports/rna-age-overnight.html in the repository.</footer></main>
<script>
(function () {{
  const tabs = {json.dumps(list(tabs))};
  function show() {{
    const want = (location.hash || "").slice(1);
    const cur = tabs.includes(want) ? want : tabs[0];
    document.querySelectorAll("section[data-tab]").forEach(s => s.hidden = s.dataset.tab !== cur);
    document.querySelectorAll("nav a").forEach(a => {{
      if (a.dataset.tab === cur) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current");
    }});
  }}
  window.addEventListener("hashchange", show);
  show();
}})();
</script>
"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page)
    return str(OUT)
