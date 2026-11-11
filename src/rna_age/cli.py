"""``uv run rna-age …``: fetch and parse sources, print the summary."""

from __future__ import annotations

import logging

import typer
from rich.logging import RichHandler

from rna_age import sources

app = typer.Typer(no_args_is_help=True, add_completion=False)


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


@app.command()
def merge() -> None:
    """Merge all sources into data/interim/samples_all.parquet with dedup, cohort labels and splits."""
    from rna_age.merge import merge as _merge

    _merge()


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
) -> None:
    """Rung 1 baseline on log-CPM HVGs; writes a run under data/output/runs/."""
    from rna_age.train.baselines import run_baseline

    run_baseline(model, n_genes, gtex_midpoints, transform=transform, with_pseudobulk=with_pseudobulk)


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
) -> None:
    """Probe / MLP on Atlas CLS, panel genes or HVGs; writes a run under data/output/runs/."""
    from rna_age.train.probes import run_probe

    run_probe(
        features, model, n_hvg, transform=transform, seed=seed, interval=interval, study_weights=study_weights,
        hidden=tuple(int(h) for h in hidden.split(",")), dropout=dropout, train_set=train_set,
        restrict_to=restrict_to,
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
    features: str = typer.Option("hvg-pca", help="hvg-pca | atlas"),
    n_components: int = typer.Option(200),
    n_context: int = typer.Option(10000, help="prompt rows"),
    cap_per_study: int = typer.Option(200),
    chunk: int = typer.Option(2048, help="query rows per predict call (GPU memory)"),
    seed: int = typer.Option(0),
    name: str | None = typer.Option(None),
) -> None:
    """Tabular foundation model (in-context) as an age regressor on the standard splits."""
    from rna_age.train.tabfm import run_tabfm

    row = run_tabfm(model, features, n_components=n_components, n_context=n_context, cap_per_study=cap_per_study,
                    chunk=chunk, seed=seed, name=name)  # fmt: skip
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
) -> None:
    """Health-aware MLP (age loss on controls + optional ranking loss / disease head)."""
    from rna_age.train.health import train_health

    train_health(rank_weight=rank_weight, margin=margin, disease_head=disease_head, aging_only=not all_diseases,
                 seed=seed)  # fmt: skip


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


def main() -> None:
    app()


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


@app.command("kron-compare")
def kron_compare(
    runs_a: str = typer.Argument(..., help="comma-separated run ids (the challenger's seeds)"),
    runs_b: str = typer.Argument(..., help="comma-separated run ids (the reference's seeds)"),
) -> None:
    """D3's bar: validation MAE of A minus B (seed-averaged), 95 % CI over validation studies, per source."""
    from rna_age.train.kron import compare

    for k, v in compare(runs_a.split(","), runs_b.split(",")).items():
        print(f"{k}: {v['diff']:+.2f} y [{v['lo']:+.2f}, {v['hi']:+.2f}]  n={v['n']} studies={v['studies']}")
