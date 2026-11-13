"""``uv run rna-age …``: fetch and parse sources, print the summary."""

from __future__ import annotations

import logging

import typer
from rich.logging import RichHandler

from rna_age import sources

app = typer.Typer(no_args_is_help=True, add_completion=False)

# RM26's switches, shared by `baseline` and `probe` (defaults keep today's table, gene axis and training set)
SAMPLES_SUFFIX = typer.Option(
    "", "--samples-suffix", help="read data/interim/samples_all<S>.parquet (rna-age merge --suffix S)"
)
GENES_WITH = typer.Option(
    "", "--genes-with", help="comma-separated extra sources: gene axis = shared bulk genes ∩ theirs"
)
EXCLUDE_TRAIN = typer.Option("", "--exclude-train-studies", help="comma-separated study_ids dropped from training only")


def _csv(text: str) -> tuple[str, ...]:
    return tuple(t.strip() for t in text.split(",") if t.strip())


def _rm26_setup(samples_suffix: str) -> None:
    from rna_age.train.features import set_samples_suffix

    set_samples_suffix(samples_suffix)  # every sample_table() reader in this process reads the same table


@app.callback()
def _setup(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        datefmt="%H:%M:%S",
        handlers=[RichHandler(show_path=False, rich_tracebacks=True)],
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)


@app.command()
def fetch(
    source: str = typer.Argument(..., help=f"one of: {', '.join(sources.SOURCES)}"),
    connections: int = typer.Option(8, help="parallel HTTP range connections for large files"),
) -> None:
    """Download a source's raw files into data/download/<source>/ (resumable, idempotent)."""
    sources.get(source).fetch(connections=connections)


@app.command()
def parse(source: str = typer.Argument(..., help=f"one of: {', '.join(sources.SOURCES)}")) -> None:
    """Parse a source's raw files into data/input/<source>/."""
    sources.get(source).parse()


@app.command("census-pseudobulk")
def census_pseudobulk(
    dataset_id: str = typer.Argument(..., help="Census dataset_id (h5ad already fetched)"),
    by_cell_type: bool = typer.Option(False, help="donor × tissue × cell type groups (composition analyses)"),
) -> None:
    """Pseudobulk one Census dataset (donor × tissue raw count sums); run as a Slurm job."""
    from rna_age.sources.census_pseudobulk import pseudobulk

    pseudobulk(dataset_id, by_cell_type=by_cell_type)


@app.command("census-donor-obs")
def census_donor_obs(dataset_id: str = typer.Argument(..., help="Census dataset_id (h5ad already fetched)")) -> None:
    """Per-donor obs metadata of one Census dataset (columns constant within a donor) to data/input/.../donor_obs/."""
    from rna_age.sources.census_pseudobulk import donor_obs

    donor_obs(dataset_id)


@app.command()
def merge(
    suffix: str = typer.Option("", help="write samples_all<suffix>.parquet / splits<suffix>.parquet instead"),
) -> None:
    """Merge all sources into data/interim/samples_all.parquet with dedup, cohort labels and splits."""
    from rna_age.merge import merge as _merge

    _merge(suffix)


@app.command()
def tage(
    source: str = typer.Argument(..., help="source with counts, e.g. gtex or archs4"),
    split: str = typer.Option("test", help="train | val | test"),
    max_samples: int = typer.Option(0, help="subsample (0 = all)"),
) -> None:
    """Run the tAge clocks (isolated R env) on a split; relative predictions → data/output/clocks/."""
    from rna_age.clocks import run

    run(source, split, max_samples or None)


@app.command()
def baseline(
    model: str = typer.Option("ridge", help="ridge | elasticnet"),
    n_genes: int = typer.Option(5000, help="HVGs chosen on train"),
    gtex_midpoints: bool = typer.Option(True, help="train on GTEx bin midpoints too"),
    transform: str = typer.Option("none", help="none | horvath (log-linear age)"),
    with_pseudobulk: bool = typer.Option(False, help="also train on the pseudobulk train split"),
    samples_suffix: str = SAMPLES_SUFFIX,
    genes_with: str = GENES_WITH,
    exclude_train_studies: str = EXCLUDE_TRAIN,
) -> None:
    """Rung 1 baseline on log-CPM HVGs; writes a run under data/output/runs/."""
    from rna_age.train.baselines import run_baseline

    _rm26_setup(samples_suffix)
    run_baseline(
        model, n_genes, gtex_midpoints, transform=transform, with_pseudobulk=with_pseudobulk,
        samples_suffix=samples_suffix, genes_with=_csv(genes_with), exclude_studies=_csv(exclude_train_studies),
    )  # fmt: skip


@app.command()
def embed(
    sources: str = typer.Argument(..., help="comma-separated sources"),
    shard: int = typer.Option(0),
    n_shards: int = typer.Option(1),
    batch_size: int = typer.Option(8),
) -> None:
    """Frozen BulkFormer-Atlas CLS embeddings (GPU job) → data/interim/embeddings/atlas/."""
    from rna_age.train.embed import embed as _embed

    _embed(sources.split(","), shard, n_shards, batch_size=batch_size)


@app.command()
def probe(
    features: str = typer.Option("cls", help="cls | cls+hvg"),
    model: str = typer.Option("ridge", help="ridge | mlp"),
    n_hvg: int = typer.Option(1200),
    transform: str = typer.Option("none", help="none | horvath"),
    seed: int = typer.Option(0),
    interval: bool = typer.Option(False, help="interval loss for binned ages (MLP)"),
    study_weights: bool = typer.Option(False, help="weight samples by 1/sqrt(study size) (MLP)"),
    hidden: str = typer.Option("512,128", help="MLP hidden sizes"),
    dropout: float = typer.Option(0.2),
    train_set: str = typer.Option("healthy", help="healthy (healthy-or-unknown) | all (incl. disease)"),
    restrict_to: str = typer.Option("", help="latest: only samples with Latest embeddings"),
    external: bool = typer.Option(False, help="also score data/input/external (rna-age external); --features hvg"),
    samples_suffix: str = SAMPLES_SUFFIX,
    genes_with: str = GENES_WITH,
    exclude_train_studies: str = EXCLUDE_TRAIN,
) -> None:
    """Probe / MLP on Atlas CLS, panel genes or HVGs; writes a run under data/output/runs/."""
    from rna_age.train.probes import run_probe

    _rm26_setup(samples_suffix)
    run_probe(
        features, model, n_hvg, transform=transform, seed=seed, interval=interval, study_weights=study_weights,
        hidden=tuple(int(h) for h in hidden.split(",")), dropout=dropout, train_set=train_set,
        restrict_to=restrict_to, external=external, samples_suffix=samples_suffix, genes_with=_csv(genes_with),
        exclude_studies=_csv(exclude_train_studies),
    )  # fmt: skip


@app.command()
def finetune(
    unfreeze: int = typer.Option(2, help="last N Atlas blocks to train (0 = head only)"),
    epochs: int = typer.Option(6),
    max_train: int = typer.Option(30000, help="training samples per epoch (after the per-study cap)"),
    cap_per_study: int = typer.Option(200),
    batch: int = typer.Option(4),
    name: str = typer.Option("", help="run name (resumes an unfinished run of the same name)"),
    train_set: str = typer.Option("healthy", help="healthy (healthy-or-unknown) | all"),
    seed: int = typer.Option(0),
    rank_weight: float = typer.Option(0.0, help="within-study case/control ranking loss weight (0 = off)"),
    margin: float = typer.Option(3.0, help="ranking margin in years"),
) -> None:
    """Rung 4: fine-tune Atlas (GPU job, resumable)."""
    from rna_age.train.finetune import train

    train(unfreeze=unfreeze, epochs=epochs, max_train=max_train, cap_per_study=cap_per_study, batch=batch,
          run_name=name or None, train_set=train_set, seed=seed, rank_weight=rank_weight, margin=margin)  # fmt: skip


@app.command("atlas-panel")
def atlas_panel(n_genes: int = typer.Option(4096)) -> None:
    """Choose the fixed Atlas input panel on training samples only (CPU)."""
    from rna_age.train.embed import make_panel

    make_panel(n_genes)


@app.command()
def figures(
    run_id: str = typer.Argument("", help="run directory name; empty = data overview + run comparison"),
) -> None:
    """Write PNG/SVG figures to data/output/figures/."""
    from rna_age import figures as f

    if run_id:
        f.run_plots(run_id)
    else:
        f.data_overview()
        f.compare_runs()


@app.command("report-tables")
def report_tables() -> None:
    """Markdown tables from the parquet summaries → data/output/summary/report_tables.md."""
    from rna_age.report import write

    print(write())


@app.command("compare-tage")
def compare_tage(run_id: str, source: str = "gtex", split: str = "test") -> None:
    """Score a run on the tAge samples/strata (relative-age correlation)."""
    from rna_age.clocks import compare_run

    print(compare_run(run_id, source, split))


@app.command()
def artifact() -> None:
    """Build the tabbed report page docs/reports/rna-age-overnight.html from the parquet outputs."""
    from rna_age.artifact import build

    print(build())


@app.command()
def ensemble(run_ids: list[str] = typer.Argument(...), name: str = typer.Option("ensemble")) -> None:  # noqa: B008
    """Average the predictions of finished runs into a new run."""
    from rna_age.train.ensemble import ensemble as _ens

    row = _ens(run_ids, name)
    print({k: v for k, v in row.items() if k.startswith(("val_", "test_", "transfer_exact"))})


@app.command("age-gap")
def age_gap(run_id: str) -> None:
    """Age-adjusted disease age acceleration for a run (val + test, exact ages, bulk)."""
    from rna_age.train.agegap import age_gap as _gap

    with __import__("polars").Config(tbl_rows=40, fmt_str_lengths=50):
        print(_gap(run_id))


@app.command()
def calibrate(run_id: str) -> None:
    """Linear de-compression of a run's predictions fitted on validation (regression-to-the-mean fix)."""
    from rna_age.train.calibrate import calibrate as _cal

    row = _cal(run_id)
    print({k: v for k, v in row.items() if k.startswith(("val_exact", "test_exact", "transfer_exact", "test_binned"))})


@app.command("embed-latest")
def embed_latest(
    sources: str = typer.Argument(..., help="comma-separated sources"),
    shard: int = typer.Option(0),
    n_shards: int = typer.Option(1),
    max_train: int = typer.Option(15000, help="training samples per source (Latest is expensive)"),
) -> None:
    """Frozen BulkFormer-Latest embeddings (GPU job) → data/interim/embeddings/latest/."""
    from rna_age.train.embed_latest import embed

    embed(sources.split(","), shard, n_shards, max_train=max_train)


@app.command("compare-clockbase")
def compare_clockbase(run_id: str) -> None:
    """ClockBase's RNA clocks vs a run on the same held-out GSMs (within-series relative-age r)."""
    from rna_age.clocks import compare_clockbase as _cmp

    print(_cmp(run_id))


@app.command()
def blend(run_a: str, run_b: str, name: str = typer.Option("blend")) -> None:
    """Weighted mean of two runs; weight chosen on validation."""
    from rna_age.train.ensemble import blend as _blend

    row = _blend(run_a, run_b, name)
    print({k: v for k, v in row.items() if k.startswith(("val_exact", "test_exact", "transfer_exact", "blend"))})


@app.command("tabfm")
def tabfm_cmd(
    model: str = typer.Option("tabicl", help="tabicl | tabpfn (package via `uv run --with`)"),
    features: str = typer.Option("hvg-pca", help="hvg-pca | hvg | logcpm | atlas"),
    n_genes: int = typer.Option(20000, help="HVGs (RM25: 2000 with --features hvg)"),
    context: str = typer.Option("random", help="random | nearest-study (RM25: one prompt per query study)"),
    splits: str = typer.Option("val,test,transfer", help="splits to score, comma separated (never train)"),
    n_components: int = typer.Option(200),
    n_context: int = typer.Option(10000, help="prompt rows"),
    cap_per_study: int = typer.Option(200),
    chunk: int = typer.Option(2048, help="query rows per predict call (GPU memory)"),
    seed: int = typer.Option(0),
    name: str | None = typer.Option(None),
) -> None:
    """Tabular foundation model (in-context) as an age regressor on the standard splits."""
    from rna_age.train.tabfm import run_tabfm

    row = run_tabfm(model, features, n_genes=n_genes, n_components=n_components, n_context=n_context,
                    cap_per_study=cap_per_study, chunk=chunk, seed=seed, name=name, context=context,
                    splits=tuple(x for x in splits.split(",") if x != "train"))  # fmt: skip
    print({k: v for k, v in row.items() if k.startswith(("val_exact", "test_exact", "transfer_exact"))})


@app.command("deconv-gap")
def deconv_gap(run_id: str, aging_only: bool = typer.Option(False)) -> None:
    """Disease age gap in held-out blood studies, raw vs adjusted for deconvolved cell fractions."""
    from rna_age.train.deconv import gap_adjusted

    print(gap_adjusted(run_id, aging_only=aging_only))


@app.command("terekhova-celltype")
def terekhova_celltype(chunk: int = typer.Option(100_000)) -> None:
    """Terekhova 2023 PBMC pseudobulk per tube × published cluster, for the composition analyses (Slurm job)."""
    from rna_age.sources.terekhova2023 import parse as _parse

    _parse(chunk=chunk, by_cell_type=True)


@app.command("health-eval")
def health_eval(
    run_id: str,
    score: str = typer.Option("pred", help="pred | bio_age"),
    permutations: int = typer.Option(0, help="labels shuffled within study: the null's mean and one-sided p"),
) -> None:
    """Within-study case − control age gap per disease category (held-out studies, bootstrap CIs)."""
    from rna_age.train.health import evaluate_health

    out = evaluate_health(run_id, score, permutations=permutations)
    cols = ["category", "studies", "cases", "within_study_gap", "ci_low", "ci_high", "frac_studies_positive"]
    cols += ["null_mean", "null_p"] if permutations else []
    with __import__("polars").Config(tbl_rows=30, fmt_str_lengths=30, tbl_width_chars=160):
        print(out.select(*cols, "control_test_mae"))


@app.command("train-health")
def train_health_cmd(
    rank_weight: float = typer.Option(0.0, help="within-study ranking loss weight (0 = off)"),
    margin: float = typer.Option(3.0, help="ranking margin in years"),
    disease_head: bool = typer.Option(False, help="add a case-vs-control head; bio_age = pred + beta*logit"),
    all_diseases: bool = typer.Option(False, help="use every categorised disease, not only aging-linked ones"),
    seed: int = typer.Option(0),
    external: bool = typer.Option(False, help="also score data/input/external (rna-age external)"),
    age_pair_weight: float = typer.Option(
        0.0, help="RM27 age-order loss weight on same study x tissue pairs (0 = off)"
    ),
    age_pair_min: float = typer.Option(20.0, help="minimum age difference of an age-order pair, years"),
    age_pair_scale: float = typer.Option(5.0, help="logistic scale of the age-order loss, years"),
) -> None:
    """Health-aware MLP (age loss on controls + optional ranking loss / disease head / age-order loss)."""
    from rna_age.train.health import train_health

    train_health(rank_weight=rank_weight, margin=margin, disease_head=disease_head, aging_only=not all_diseases,
                 seed=seed, external=external, age_pair_weight=age_pair_weight, age_pair_min=age_pair_min,
                 age_pair_scale=age_pair_scale)  # fmt: skip


@app.command("external")
def external_cmd(series: str = typer.Argument(..., help="comma-separated GEO series")) -> None:
    """RM10: every ARCHS4 sample of these series to data/input/external/ (never in the age table or training)."""
    from rna_age.train.external import extract

    extract(series.split(","))


@app.command("biology-paired")
def biology_paired_cmd(
    run_ids: str = typer.Argument(..., help="comma-separated runs with predictions_external.parquet, averaged"),
    name: str = typer.Option("model", help="label for the averaged model in the output"),
    permutations: int = typer.Option(5000),
) -> None:
    """RM10: the registered before/after tests on external samples."""
    from rna_age.train.biology import run_paired_all

    with __import__("polars").Config(tbl_rows=40, fmt_str_lengths=40, tbl_width_chars=170):
        print(run_paired_all(run_ids.split(","), name, permutations))


@app.command()
def composition(
    dataset_id: str,
    mapping: str = typer.Option("broad", help="broad (blood classes) | cell_type (the dataset's own labels)"),
    null_seed: int | None = typer.Option(None, help="shuffle donors' ages with this seed (null check)"),
) -> None:
    """Composition vs within-cell-type age signal for one cell-type-pseudobulked Census dataset."""
    from rna_age.train.composition import analyse

    with __import__("polars").Config(tbl_rows=30, fmt_str_lengths=40):
        print(analyse(dataset_id, mapping=mapping, null_seed=null_seed))


@app.command("primo-embed")
def primo_embed(
    name: str = typer.Option("atlas", help="name of the output file"),
    checkpoint: str = typer.Option("", help="fine-tune run name or .pt path (empty = frozen Atlas)"),
) -> None:
    """Embed the PRIMO bulk datasets with Atlas (GPU job) → data/output/primo/submission_<name>.parquet."""
    from rna_age.train.primo_embed import embed

    embed(name, checkpoint or None)


@app.command()
def summary() -> None:
    """Write data/output/summary/*.parquet from data/input/*."""
    from rna_age.summary import write_summary

    write_summary()


@app.command()
def inventory(
    fetch: bool = typer.Option(True, help="fetch scBaseCount / recount3 / ENA metadata first (resumable)"),
) -> None:
    """Age-labelled human transcriptomes beyond ours, from metadata only → data/output/summary/age_inventory.parquet."""
    from rna_age.sources.inventory import inventory as run

    print(run(fetch=fetch))


def main() -> None:
    app()


@app.command("robustness")
def robustness_cmd(
    run_id: str,
    flag_ratio: float = typer.Option(1.5, help="flag a group whose MAE exceeds this x the overall test MAE"),
) -> None:
    """RM11: a run's test error by tissue class, platform and source; flags unsupported groups."""
    from rna_age.train.robustness import by_group

    with __import__("polars").Config(tbl_rows=80, fmt_str_lengths=30, tbl_width_chars=160):
        print(by_group(run_id, flag_ratio=flag_ratio).drop("run_id"))


@app.command("biology")
def biology_cmd(
    run_ids: str = typer.Argument(..., help="comma-separated run ids"),
    permutations: int = typer.Option(2000),
) -> None:
    """RM10: the pre-registered tests of the disease gap (severity, treatment, a negative control)."""
    from rna_age.train.biology import run_all

    with __import__("polars").Config(tbl_rows=80, fmt_str_lengths=40, tbl_width_chars=170):
        print(run_all(run_ids.split(","), permutations))


@app.command("sc-tabular")
def sc_tabular_cmd(
    dataset_id: str,
    models: str = typer.Option(
        "ridge,mlp,tabicl", help="comma-separated: ridge, enet, gbm, mlp, tabicl (--with tabicl)"
    ),
    reference: str = typer.Option("ridge", help="the model every other is compared against"),
    n_genes: int = typer.Option(500),
    mapping: str = typer.Option("broad", help="broad (blood classes) | cell_type (the dataset's own labels)"),
) -> None:
    """RM14 (D7): tabular foundation model vs ridge and an MLP as single-cell aging clocks (per donor, CV by person)."""
    from rna_age.train.sc_tabular import compare, run

    pred = run(dataset_id, tuple(models.split(",")), n_genes=n_genes, mapping=mapping)
    with __import__("polars").Config(tbl_rows=60, tbl_width_chars=160):
        print(compare(pred, reference=reference))


@app.command("fm-embed-clock")
def fm_embed_clock_cmd(
    dataset_id: str,
    stage: str = typer.Option("all", help="embed | clock | summary | all (embed+clock+summary)"),
    cap: int = typer.Option(200, help="cells per donor (class-stratified), and CD4 T cells per donor in blood"),
    budget: int = typer.Option(32768, help="tokens per GPU batch"),
) -> None:
    """RM22: Geneformer V2 cell embeddings averaged per donor → ridge/TabICLv2 age clocks on RM14's folds
    (`uv run --with tabicl --with transformers`)."""
    from rna_age.train import fm_embed

    if stage in ("embed", "all"):
        fm_embed.run_embed(dataset_id, cap=cap, cd4_cap=cap, budget=budget)
    if stage in ("clock", "all"):
        fm_embed.run_clock(dataset_id)
    if stage in ("summary", "clock", "all"):
        with __import__("polars").Config(tbl_rows=80, tbl_width_chars=200):
            print(fm_embed.write_summary())


@app.command("perturb-score")
def perturb_score_cmd(
    cohort: str = typer.Option("3faad104-2ab8-4434-816d-474d8d2641db", help="the clock's training cohort"),
    n_genes: int = typer.Option(2000),
) -> None:
    """RM17: score the CD4 T-cell Perturb-seq with the CD4 T clock (age shift vs non-targeting guides, per gene)."""
    from rna_age.train.perturb import run

    g, info = run(cohort, n_genes=n_genes)
    print(info)
    with __import__("polars").Config(tbl_rows=40, tbl_width_chars=160):
        print(g.head(40))


@app.command("perturb-check")
def perturb_check_cmd() -> None:
    """RM17, D8: do the per-gene shifts agree with the paper's own aging coefficients? Run before reading hits."""
    from rna_age.train.perturb import run_check

    with __import__("polars").Config(tbl_width_chars=160):
        print(run_check())


@app.command("unseen-kd-embed")
def unseen_kd_embed_cmd(
    model: str = typer.Option("facebook/esm2_t33_650M_UR50D", help="ESM-2 checkpoint on the Hub"),
    device: str = typer.Option("cuda"),
) -> None:
    """RM21 step 1: UniProt (reviewed human) proteins of the screen's target genes → ESM-2 mean-pooled embeddings
    (data/interim/gene_esm2.parquet). Run with ``uv run --with transformers``."""
    from rna_age.train.unseen_kd import fetch_uniprot, run_embed

    fetch_uniprot()
    out = run_embed(model, device)
    print(out.drop("emb").head())


@app.command("unseen-kd")
def unseen_kd_cmd(
    models: str = typer.Option("ridge,tabicl", help="comma-separated: ridge, tabicl, pca_ridge"),
    folds: int = typer.Option(5),
    seed: int = typer.Option(0),
    n_boot: int = typer.Option(1000),
    n_pcs: int = typer.Option(64, help="PCs of the embedding fed to tabicl / pca_ridge"),
) -> None:
    """RM21: predict a knockdown's age effect (clock shift; the study's coefficient) from the target gene's ESM-2
    embedding, 5-fold CV over genes, against the mean and a shuffled-embedding ridge."""
    from rna_age.train.unseen_kd import run_cv

    d = run_cv(tuple(models.split(",")), k=folds, seed=seed, n_boot=n_boot, n_pcs=n_pcs)
    cols = ["target", "culture_condition", "model", "n_genes", "spearman", "spearman_lo", "spearman_hi",
            "skill_vs_mean", "d_rho_vs_ridge_shuffled_emb", "beats_bar"]  # fmt: skip
    with __import__("polars").Config(tbl_rows=80, tbl_cols=12, tbl_width_chars=220):
        print(d.select(cols))


@app.command("rm20-multitask")
def rm20_multitask_cmd(
    cohort: str = typer.Option("3faad104-2ab8-4434-816d-474d8d2641db", help="the age task's cohort"),
    n_genes: int = typer.Option(2000),
    arms: str = typer.Option("age,age_delta,age_delta_rank,placebo"),
    steps: int = typer.Option(3000),
    batch: int = typer.Option(512),
    hidden: int = typer.Option(512),
    lr: float = typer.Option(1e-3),
    w_rank: float = typer.Option(0.5, help="weight of the age-shift ranking term (rank and placebo arms)"),
    holdout_context: str | None = typer.Option(None, help="donor|condition held out of Δ training (default: by seed)"),
    subset_genes: int | None = typer.Option(None, help="smoke run: only this many target genes"),
    ntc_per_context: int | None = typer.Option(None, help="smoke run: at most this many control profiles per context"),
    seed: int = typer.Option(0),
    device: str = typer.Option("auto"),
    out: str | None = typer.Option(
        None, help="output parquet (default data/output/summary/rm20_multitask_cd4.parquet)"
    ),
) -> None:
    """RM20: one network, donor age + knockdown Δ (+ age-shift ranking), on the CD4 T Perturb-seq panel."""
    from pathlib import Path

    import polars as pl

    from rna_age.paths import OUTPUT, download_dir
    from rna_age.train import multitask, multitask_data
    from rna_age.train.multitask_model import Config
    from rna_age.train.perturb import COEFFICIENTS, SCREEN

    d = download_dir("cd4_perturbseq")
    data = multitask_data.load(d / SCREEN, cohort, n_genes, subset_genes=subset_genes, ntc_per_context=ntc_per_context,
                               seed=seed)  # fmt: skip
    rc_path = OUTPUT / "summary" / "perturb_cd4_genes.parquet"
    df = multitask.run(
        data, pl.read_csv(d / COEFFICIENTS), Config(steps=steps, batch=batch, hidden=hidden, lr=lr, seed=seed),
        arms=tuple(arms.split(",")), w_rank=w_rank, holdout_context=holdout_context, seed=seed, device=device,
        ridge_clock_genes=pl.read_parquet(rc_path) if rc_path.exists() else None,
    )  # fmt: skip
    path = multitask.write(df, Path(out) if out else OUTPUT / "summary" / "rm20_multitask_cd4.parquet")
    with pl.Config(tbl_rows=200, tbl_width_chars=200):
        print(df)
    print(path)


@app.command("maxtoki-cd4")
def maxtoki_cd4_cmd(
    n_prompts: int = typer.Option(300, help="screen prompts (one query cell each)"),
    n_validate: int = typer.Option(150, help="prompts for the unperturbed timelapse check"),
    per_bin: int = typer.Option(40, help="donors per CD4 type × sex × decade"),
    max_screen_genes: int = typer.Option(0, help="0 = every screened gene; else top half by |coef| + random rest"),
    batch: int = typer.Option(16),
    factors: str = typer.Option("1,4,8", help="RoPE factors to validate; the best runs the screen"),
    min_prompts: int = typer.Option(5, help="a gene needs this many expressing query cells"),
    time_budget_h: float = typer.Option(0.0, help="stop after this many hours (0 = none); rerun resumes"),
    summarise_only: bool = typer.Option(False),
) -> None:
    """RM23: MaxToki-217M in-silico knockdown in blood CD4 T cells vs the CD4 CRISPRi screen and our clock (needs
    `uv run --with transformers`)."""
    from rna_age.train import maxtoki_cd4 as m

    if summarise_only:
        parts = sorted(m.WORK.glob("effects_rope*"))[-1]
        res = m.summarise(parts, min_prompts)
    else:
        res = m.run(n_prompts, n_validate, per_bin, max_screen_genes=max_screen_genes or None, batch=batch,
                    factors=factors, min_prompts=min_prompts, time_budget_h=time_budget_h or None)  # fmt: skip
    with __import__("polars").Config(tbl_rows=20, tbl_width_chars=220, tbl_cols=30):
        print(res)


@app.command("clock-compare")
def clock_compare_cmd() -> None:
    """RM18 step 1: bulk and single-cell clocks on the same donors of every single-cell cohort."""
    from rna_age.train.clock_compare import same_donors

    with __import__("polars").Config(tbl_rows=40, tbl_cols=20, tbl_width_chars=220):
        print(same_donors())


@app.command("clock-disease")
def clock_disease_cmd(n_perm: int = typer.Option(1000)) -> None:
    """RM18 step 2: lupus and COVID-19 patients vs controls, gap per clock (bulk; single-cell per cell class)."""
    from rna_age.train.disease_sc import run

    with __import__("polars").Config(tbl_rows=40, tbl_width_chars=200):
        print(run(n_perm=n_perm))


@app.command("perturb-hits")
def perturb_hits_cmd() -> None:
    """RM19: knockdowns that shift the clock beyond their cell loss, FDR 5 % in two or more conditions, same sign."""
    import polars as pl

    from rna_age.train.perturb import run_hits

    g = run_hits().filter(pl.col("replicated")).sort("perturbed_gene_name", "culture_condition")
    with pl.Config(tbl_rows=120, tbl_width_chars=160):
        print(g.select("perturbed_gene_name", "culture_condition", "shift", "residual", "fit", "q"))


@app.command("kron")
def kron_cmd(
    encoding: str = typer.Option("rank", help="rank | genome | genome-shuffled"),
    length: int = typer.Option(2048, help="rank tokens per sample"),
    n_hvg: int = typer.Option(20000, help="genes of the genome encodings"),
    d_model: int = typer.Option(256),
    n_layers: int = typer.Option(4),
    n_heads: int = typer.Option(4),
    dropout: float = typer.Option(0.1),
    epochs: int = typer.Option(30),
    patience: int = typer.Option(4),
    batch: int = typer.Option(32),
    micro: int = typer.Option(0, help="rows per forward pass (0 = batch); gradients accumulate to --batch"),
    lr: float = typer.Option(3e-4),
    seed: int = typer.Option(0),
    name: str | None = typer.Option(None, help="run name (resumes an unfinished run of the same name)"),
    init: str | None = typer.Option(None, help="a kron-pretrain run id: start from its encoder (RM9)"),
) -> None:
    """Rung 5 (D2, D3): a bidirectional KronLSTM age model from scratch (GPU job, resumable; needs a kron-fm clone, see train/kron.py)."""
    from rna_age.train.kron import train

    row = train(encoding=encoding, length=length, n_hvg=n_hvg, d_model=d_model, n_layers=n_layers, n_heads=n_heads,
                dropout=dropout, epochs=epochs, patience=patience, batch=batch, micro=micro, lr=lr, seed=seed, run_name=name,
                init=init)  # fmt: skip
    print({k: v for k, v in row.items() if k.startswith(("val_exact", "test_exact", "transfer_exact"))})


@app.command("kron-pretrain")
def kron_pretrain(
    length: int = typer.Option(2048),
    d_model: int = typer.Option(256),
    n_layers: int = typer.Option(4),
    n_heads: int = typer.Option(4),
    mask_frac: float = typer.Option(0.15),
    epochs: int = typer.Option(20),
    batch: int = typer.Option(32),
    micro: int = typer.Option(0, help="rows per forward pass (0 = batch); gradients accumulate to --batch"),
    lr: float = typer.Option(3e-4),
    seed: int = typer.Option(0),
    name: str | None = typer.Option(None, help="run name (resumes an unfinished run of the same name)"),
) -> None:
    """RM9 (D3): masked-gene pretraining of the KronLSTM on training-split rank tokens; then `kron --init RUN`."""
    from rna_age.train.kron import pretrain

    print(pretrain(length=length, d_model=d_model, n_layers=n_layers, n_heads=n_heads, mask_frac=mask_frac,
                   epochs=epochs, batch=batch, micro=micro, lr=lr, seed=seed, run_name=name))  # fmt: skip


@app.command("kron-rank-mlp")
def kron_rank_mlp(length: int = typer.Option(2048), seed: int = typer.Option(0)) -> None:
    """D3's ablation: the baseline MLP on the KronLSTM's rank encoding."""
    from rna_age.train.kron import train_rank_mlp

    row = train_rank_mlp(length=length, seed=seed)
    print({k: v for k, v in row.items() if k.startswith(("val_exact", "test_exact", "transfer_exact"))})


@app.command("rm26-compare")
def rm26_compare_cmd(
    control: str = typer.Option(..., help="comma-separated run ids of the control arm's seeds"),
    arm: str = typer.Option(..., help="comma-separated run ids of the recount3 arm's seeds"),
    headline: str = typer.Option("20261111-0747-ens-mlp3t2-plus-atlasft2", help="the headline run to compare against"),
    n_boot: int = typer.Option(2000),
    seed: int = typer.Option(0),
) -> None:
    """RM26's bar: seed-averaged MAE per arm on test and test_new, arm − control with a 95 % study-bootstrap CI,
    each arm vs the headline; writes data/output/summary/rm26_compare.parquet."""
    from rna_age.train.rm26 import compare

    out = compare(_csv(control), _csv(arm), headline, n_boot=n_boot, seed=seed)
    import polars as pl

    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200):
        print(out)


@app.command("kron-compare")
def kron_compare(
    runs_a: str = typer.Argument(..., help="comma-separated run ids (the challenger's seeds)"),
    runs_b: str = typer.Argument(..., help="comma-separated run ids (the reference's seeds)"),
) -> None:
    """D3's bar: validation MAE of A minus B (seed-averaged), 95 % CI over validation studies, per source."""
    from rna_age.train.kron import compare

    for k, v in compare(runs_a.split(","), runs_b.split(",")).items():
        print(f"{k}: {v['diff']:+.2f} y [{v['lo']:+.2f}, {v['hi']:+.2f}]  n={v['n']} studies={v['studies']}")


# ---------------------------------------------------------------- RM28: our own single-cell pretraining

COHORTS_OPT = typer.Option("", help="comma-separated evaluation cohorts (default: all 7)")


def _cohorts(text: str) -> tuple[str, ...]:
    from rna_age.train.sc_pretrain import EVAL_COHORTS

    return _csv(text) or EVAL_COHORTS


@app.command("sc-pretrain-pool")
def sc_pretrain_pool_cmd(
    name: str = typer.Option("default", help="pool name (data/interim/sc_pretrain/<name>)"),
    cap: int = typer.Option(2000, help="cells per donor"),
    hvg_cap: int = typer.Option(200, help="cells per donor for the HVG statistics"),
    n_genes: int = typer.Option(2000),
    seed: int = typer.Option(0),
    datasets: str = typer.Option("", help="comma-separated Census dataset ids (default: every non-evaluation h5ad)"),
    max_cells_per_dataset: int = typer.Option(0, help="smoke runs only"),
    perturb_max_rows: int = typer.Option(0, help="smoke runs only: one context, this many knockdown rows"),
) -> None:
    """RM28: pretraining pool (healthy adult primary cells, evaluation cohorts and their donors excluded), gene panel,
    panel matrices and Perturb-seq tensors (a CPU job: reads every pool h5ad and the 44 GB screen)."""
    from rna_age.train.sc_pretrain import build_pool

    print(build_pool(name, cap, hvg_cap, n_genes, seed, max_cells_per_dataset or None, _csv(datasets) or None,
                     perturb_max_rows or None))  # fmt: skip


@app.command("sc-pretrain")
def sc_pretrain_cmd(
    arm: str = typer.Option("mask", help="mask | mask+perturb | universal"),
    epochs: int = typer.Option(30),
    seed: int = typer.Option(0),
    pool: str = typer.Option("default"),
    batch_size: int = typer.Option(1024),
    lr: float = typer.Option(1e-3),
    perturb_weight: float = typer.Option(1.0, help="weight of the perturbation loss (arm mask+perturb)"),
    max_cells: int = typer.Option(0, help="smoke runs only"),
    max_steps: int = typer.Option(0, help="smoke runs only: steps per epoch"),
    device: str = typer.Option("", help="cuda | cpu (default: cuda when available)"),
    age_weight: float = typer.Option(1.0, help="universal: age head (pool donors + bulk train controls)"),
    rank_weight: float = typer.Option(0.5, help="universal: within-study case/control ranking"),
    health_weight: float = typer.Option(0.5, help="universal: case-vs-control logit"),
    max_bulk: int = typer.Option(0, help="universal, smoke runs only: subsample the bulk table"),
) -> None:
    """RM28: pretrain the masked autoencoder (GPU); exits 3 when epoch 1 is not below the mean-gene baseline."""
    from rna_age.train.sc_pretrain import pretrain

    s = pretrain(arm, epochs, seed, pool, batch_size, lr, perturb_weight=perturb_weight, max_cells=max_cells or None,
                 max_steps=max_steps or None, device=device or None, age_weight=age_weight, rank_weight=rank_weight,
                 health_weight=health_weight, max_bulk=max_bulk or None)  # fmt: skip
    print(s)
    if s["failed"]:
        raise typer.Exit(3)


@app.command("sc-pretrain-embed")
def sc_pretrain_embed_cmd(
    run_id: str,
    cohorts: str = COHORTS_OPT,
    cap: int = typer.Option(200, help="cells per donor (class-stratified), and CD4 T cells per donor in blood"),
    max_donors: int = typer.Option(0, help="smoke runs only"),
    device: str = typer.Option(""),
) -> None:
    """RM28: per-donor mean frozen embeddings of the evaluation cohorts (reads the cohort h5ads once per pool)."""
    from rna_age.train.sc_pretrain import embed_cohorts

    for p in embed_cohorts(run_id, _cohorts(cohorts), cap, max_donors=max_donors or None, device=device or None):
        print(p)


@app.command("sc-pretrain-clock")
def sc_pretrain_clock_cmd(
    run_id: str,
    cohorts: str = COHORTS_OPT,
    models: str = typer.Option("ridge,tabicl", help="`uv run --with tabicl` for tabicl"),
) -> None:
    """RM28: ridge/TabICLv2 on the donor-mean embeddings vs pseudobulk-tabicl-2000g (paired bootstrap); writes
    data/output/summary/sc_pretrain_clock.parquet and prints bar (1)."""
    import polars as pl

    from rna_age.train.sc_pretrain import bar, clock

    cmp = clock(run_id, _cohorts(cohorts), _csv(models))
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200):
        print(cmp)
        print(bar(cmp))


@app.command("sc-pretrain-finetune")
def sc_pretrain_finetune_cmd(
    run_id: str,
    cohorts: str = COHORTS_OPT,
    epochs: int = typer.Option(20),
    lr: float = typer.Option(1e-4),
    max_donors: int = typer.Option(0, help="smoke runs only"),
    device: str = typer.Option(""),
    pool: str = typer.Option("mean", help="RM32: mean or attn (attention over the donor's cells)"),
    cell_weight: float = typer.Option(0.0, help="RM32: weight of the per-cell age loss (0 = donor loss only)"),
    pseudocell: int = typer.Option(0, help="RM32: average groups of this many cells per donor into units (0 = cells)"),
) -> None:
    """RM28/RM32: fine-tune read-out (encoder + per-cell age head pooled per donor, inside each training fold, GPU)."""
    import polars as pl

    from rna_age.train.sc_pretrain import bar, finetune

    cmp = finetune(run_id, _cohorts(cohorts), epochs, lr, max_donors=max_donors or None, device=device or None,
                   pool=pool, cell_weight=cell_weight, pseudocell=pseudocell)  # fmt: skip
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200):
        print(cmp)
        print(bar(cmp))


@app.command("sc-pretrain-universal-eval")
def sc_pretrain_universal_eval_cmd(run_id: str, device: str = typer.Option("")) -> None:
    """RM28 arm universal: bulk predictions of the age head (run dir predictions.parquet, every split) and the
    within-study health gap (evaluate_health) for pred and bio_age."""
    import polars as pl

    from rna_age.train.sc_universal import universal_eval

    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200):
        print(universal_eval(run_id, device=device or None))


@app.command("expr-icl-prepare")
def expr_icl_prepare(
    what: str = typer.Option("panel,bulk,sc,screen", help="comma-separated parts, in this order"),
    n_genes: int = typer.Option(1000, help="gene panel size"),
    max_bulk: int | None = typer.Option(None, help="bulk samples (smoke runs; default all)"),
    sc_datasets: str = typer.Option("", help="comma-separated single-cell datasets (default all on disk)"),
    screen_max_rows: int | None = typer.Option(None, help="smoke runs: one donor × condition, this many profiles"),
    cache: str | None = typer.Option(None, help="table directory (default data/interim/expr_icl)"),
) -> None:
    """RM30: the gene panel and the prepared tables (bulk, single-cell donors, screen knockdowns) on it."""
    from pathlib import Path

    from rna_age.train import expr_icl as e

    d = Path(cache) if cache else e.CACHE
    parts = _csv(what)
    if "panel" in parts:
        e.build_panel(n_genes, cache=d)
    genes = e.read_panel(d)
    if "bulk" in parts:
        e.build_bulk(genes, max_samples=max_bulk, cache=d)
    if "sc" in parts:
        e.build_sc(genes, list(_csv(sc_datasets)) or None, cache=d)
    if "screen" in parts:
        e.build_screen(genes, max_rows=screen_max_rows, cache=d)


@app.command("expr-icl-train")
def expr_icl_train(
    steps: int = typer.Option(20000),
    lr: float = typer.Option(3e-5),
    warmup: int = typer.Option(500),
    micro_batch: int = typer.Option(4, help="episodes per forward (one task, one shape)"),
    accum: int = typer.Option(4, help="forwards per optimiser step"),
    task_weights: str = typer.Option("age=1,health=1,masked_gene=1,perturbation=1"),
    rows: str = typer.Option("64,512", help="episode rows min,max"),
    cols: str = typer.Option("100,500", help="episode columns min,max"),
    eval_every: int = typer.Option(500),
    eval_episodes: int = typer.Option(32, help="fixed held-out episodes per task"),
    ckpt_every: int = typer.Option(2000),
    seed: int = typer.Option(0),
    device: str = typer.Option("auto"),
    name: str = typer.Option("expricl"),
    cache: str | None = typer.Option(None, help="table directory (default data/interim/expr_icl)"),
    recompute: bool = typer.Option(False, help="gradient checkpointing inside TabICL (less memory, ~30 % slower)"),
    amp: bool = typer.Option(True, help="bf16 where supported, else fp16 with a loss scaler; off = fp32"),
    resume: str = typer.Option("", help="continue from this run's last.pt up to --steps in total (new run dir)"),
) -> None:
    """RM30: train TabICLv2 (from its released weights) on expression episodes; prints the run id."""
    from pathlib import Path

    from rna_age.train.expr_icl import CACHE
    from rna_age.train.expr_icl_model import TrainConfig, train

    w = {k: float(v) for k, v in (t.split("=") for t in _csv(task_weights))}
    r, c = (tuple(int(x) for x in _csv(s)) for s in (rows, cols))
    cfg = TrainConfig(steps=steps, lr=lr, warmup=warmup, micro_batch=micro_batch, accum=accum, task_weights=w,
                      n_rows=r, n_cols=c, eval_every=eval_every, eval_episodes=eval_episodes,
                      ckpt_every=ckpt_every, seed=seed, device=device, recompute=recompute, amp=amp)  # fmt: skip
    run_dir = train(cfg, name=name, cache=Path(cache) if cache else CACHE, resume=resume or None)
    print(run_dir.name)


@app.command("expr-icl-eval")
def expr_icl_eval(
    run_id: str,
    what: str = typer.Option("sc", help="sc | bulk | health | perturb"),
    n_genes: int = typer.Option(500, help="sc: genes per fold (D7: 500); bulk: panel columns used"),
    cohorts: str = typer.Option("", help="sc: comma-separated cohorts (default the seven)"),
    bulk_run: str | None = typer.Option(None, help="health: the bulk run (default the one eval bulk wrote)"),
    splits: str = typer.Option("val,test", help="bulk: scored splits"),
    n_context: int = typer.Option(4000, help="bulk: prompt rows"),
    n_readouts: int = typer.Option(100, help="perturb: readout genes"),
    permutations: int = typer.Option(0, help="health: within-study label shuffles"),
    device: str = typer.Option("auto"),
    cache: str | None = typer.Option(None, help="table directory (default data/interim/expr_icl)"),
) -> None:
    """RM30's evaluations of a trained run (see rna_age.train.expr_icl_eval)."""
    from pathlib import Path

    import polars as pl

    from rna_age.train import expr_icl_eval as ev
    from rna_age.train.expr_icl import CACHE, EVAL_COHORTS

    d = Path(cache) if cache else CACHE
    if what == "sc":
        out = ev.eval_sc(run_id, cohorts=_csv(cohorts) or EVAL_COHORTS, n_genes=n_genes, device=device)
    elif what == "bulk":
        row = ev.eval_bulk(run_id, splits=_csv(splits), n_context=n_context, max_features=n_genes, device=device,
                           cache=d)  # fmt: skip
        out = pl.DataFrame([{k: v for k, v in row.items() if k.startswith(("run_id", "val_exact", "test_exact"))}])
    elif what == "health":
        out = ev.eval_health(run_id, bulk_run, permutations=permutations)
    elif what == "perturb":
        out = ev.eval_perturb(run_id, n_readouts=n_readouts, device=device, cache=d)
    else:
        raise typer.BadParameter(f"--what must be sc, bulk, health or perturb, got {what!r}")
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200):
        print(out)


@app.command("sc-multitype")
def sc_multitype_cmd(
    cohorts: str = typer.Option(
        "3faad104-2ab8-4434-816d-474d8d2641db,c838aec3-03ef-4398-b882-0e3912abfff0,"
        "218acb0f-9f2f-4f76-b90b-15a4b7c7f629,terekhova2023,37a17b78-4864-4a42-b67b-31c00962795a,"
        "5c97eeeb-7e52-44b3-b010-b832b1f5424c,5e57cd50-8e42-42d6-940d-5c1660d06864",
        help="comma-separated cohorts (default: the 7 of D7)",
    ),
    models: str = typer.Option("ridge,tabicl", help="comma-separated; tabicl needs uv run --with tabicl"),
    k: int = typer.Option(5, help="cell classes per donor"),
) -> None:
    """RM31: several cell types per donor at once (concat, mean of type clocks) vs the whole-sample clock."""
    import polars as pl

    from rna_age.train.sc_multitype import run

    out = run(tuple(cohorts.split(",")), tuple(models.split(",")), k=k)
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200):
        print(out.select("dataset_id", "model", "arm", "n", "mae", "ref_mae", "diff", "lo", "hi", "beats_ref"))


@app.command("positive-controls-fetch")
def positive_controls_fetch(
    dataset_ids: list[str] = typer.Argument(..., help="CELLxGENE dataset ids (from positive_controls_usability)"),  # noqa: B008
    connections: int = typer.Option(4, help="parallel range connections per file"),
) -> None:
    """RM34 step 2: download whole h5ads of chosen positive-control datasets (Census release, else Discover);
    resumable and verified; then census-donor-obs / census-pseudobulk --by-cell-type as usual. Run as a job."""
    from rna_age.positive_controls import fetch_h5ads

    for p in fetch_h5ads(dataset_ids, connections=connections):
        print(p.name, p.stat().st_size)


@app.command("positive-controls-scan")
def positive_controls_scan(
    seed: str = typer.Option(
        "data/output/summary/positive_controls_seed.parquet", help="RM33's hand-searched seed list (check only)"
    ),
    workers: int = typer.Option(4, help="parallel h5ad obs readers (HTTP range requests)"),
    fetch: bool = typer.Option(True, help="read uncached CELLxGENE obs over HTTP (--no-fetch: cached only)"),
    max_mb: int = typer.Option(200, help="skip a dataset whose obs read exceeds this many MB"),
) -> None:
    """RM34 step 1: score every human study with a non-cancer condition and same-tissue healthy controls
    (CELLxGENE, ARCHS4 bulk, scBaseCount) for usability as a positive control of system ageing (metadata only)."""
    from pathlib import Path

    import polars as pl

    from rna_age.paths import REPO_ROOT
    from rna_age.positive_controls import run

    path = Path(seed) if Path(seed).is_absolute() else REPO_ROOT / seed
    out = run(path, workers=workers, fetch=fetch, max_mb=max_mb)
    with pl.Config(tbl_rows=40, tbl_cols=-1, tbl_width_chars=220, fmt_str_lengths=40):
        print(out.group_by("source", "verdict").len().sort("source", "verdict"))
        print(out.filter(pl.col("verdict") == "usable").select(
            "source", "dataset_id", "condition", "tissue", "n_case", "n_control", "n_pairs_matched", "batch_status"))
