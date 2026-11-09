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
def summary() -> None:
    """Write data/output/summary/*.parquet from data/input/*."""
    from rna_age.summary import write_summary

    write_summary()


def main() -> None:
    app()
