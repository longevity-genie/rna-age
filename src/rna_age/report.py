"""Markdown tables for the report, built only from the parquet summaries and run outputs.

``rna-age report-tables`` writes ``data/output/summary/report_tables.md`` (tracked) with: data by source,
splits, duplicates, cohorts, experiments (best first), clock controls. The overnight report and the artifact
quote these tables, so every number there traces back to a parquet file.
"""

from __future__ import annotations

import polars as pl

from rna_age.paths import OUTPUT

SUMMARY = OUTPUT / "summary"


def md(df: pl.DataFrame, floats: int = 2) -> str:
    cols = df.columns
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.iter_rows():
        cells = []
        for v in row:
            if isinstance(v, float):
                cells.append("" if v != v else f"{v:,.{floats}f}")
            elif isinstance(v, int):
                cells.append(f"{v:,}")
            else:
                cells.append("" if v is None else str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def tables() -> dict[str, pl.DataFrame]:
    t: dict[str, pl.DataFrame] = {}
    if (SUMMARY / "by_source.parquet").exists():
        t["Samples by source and age type"] = pl.read_parquet(SUMMARY / "by_source.parquet")
    if (SUMMARY / "splits.parquet").exists():
        s = pl.read_parquet(SUMMARY / "splits.parquet")
        t["Splits (eligible, de-duplicated)"] = s.pivot(
            on="split", index=["source", "age_type"], values="samples", aggregate_function="sum"
        ).sort("source")
    if (SUMMARY / "duplicates.parquet").exists():
        t["Cross-source duplicates removed"] = pl.read_parquet(SUMMARY / "duplicates.parquet")
    if (SUMMARY / "experiments.parquet").exists():
        e = pl.read_parquet(SUMMARY / "experiments.parquet")
        keep = [c for c in ("run_id", "model", "features", "val_exact_mae", "test_exact_mae", "test_exact_pearson_r",
                            "test_exact_bias_slope", "test_binned_interval_hit", "test_binned_mae_mid",
                            "transfer_exact_mae", "transfer_exact_pearson_r") if c in e.columns]  # fmt: skip
        t["Experiments (sorted by validation MAE)"] = e.select(keep).sort("val_exact_mae", nulls_last=True)
    clocks = sorted((OUTPUT / "clocks").glob("tage_*_summary.parquet"))
    if clocks:
        t["tAge control (relative clock; r with age delta within stratum)"] = pl.concat(
            [pl.read_parquet(p) for p in clocks], how="diagonal_relaxed"
        )
    return t


def write() -> str:
    parts = ["# Report tables (generated from parquet; do not edit)\n"]
    for title, df in tables().items():
        parts += [f"## {title}\n", md(df), ""]
    text = "\n".join(parts)
    SUMMARY.mkdir(parents=True, exist_ok=True)
    (SUMMARY / "report_tables.md").write_text(text)
    return text
