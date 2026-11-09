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
def census_pseudobulk(dataset_id: str = typer.Argument(..., help="Census dataset_id (h5ad already fetched)")) -> None:
    """Pseudobulk one Census dataset (donor × tissue raw count sums); run as a Slurm job."""
    from rna_age.sources.census_pseudobulk import pseudobulk

    pseudobulk(dataset_id)


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
) -> None:
    """Rung 1 baseline on log-CPM HVGs; writes a run under data/output/runs/."""
    from rna_age.train.baselines import run_baseline

    run_baseline(model, n_genes, gtex_midpoints)


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
) -> None:
    """Rung 3 probe on frozen Atlas embeddings; writes a run under data/output/runs/."""
    from rna_age.train.probes import run_probe

    run_probe(features, model, n_hvg)


@app.command()
def finetune(
    unfreeze: int = typer.Option(2, help="last N Atlas blocks to train (0 = head only)"),
    epochs: int = typer.Option(6),
    max_train: int = typer.Option(30000, help="training samples per epoch (after the per-study cap)"),
    cap_per_study: int = typer.Option(200),
    batch: int = typer.Option(4),
    name: str = typer.Option("", help="run name (resumes an unfinished run of the same name)"),
) -> None:
    """Rung 4: fine-tune Atlas (GPU job, resumable)."""
    from rna_age.train.finetune import train

    train(unfreeze=unfreeze, epochs=epochs, max_train=max_train, cap_per_study=cap_per_study, batch=batch,
          run_name=name or None)  # fmt: skip


@app.command("atlas-panel")
def atlas_panel(n_genes: int = typer.Option(4096)) -> None:
    """Choose the fixed Atlas input panel on training samples only (CPU)."""
    from rna_age.train.embed import make_panel

    make_panel(n_genes)


@app.command()
def summary() -> None:
    """Write data/output/summary/*.parquet from data/input/*."""
    from rna_age.summary import write_summary

    write_summary()


def main() -> None:
    app()
