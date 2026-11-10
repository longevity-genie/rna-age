# rna-age: notes for agents

Transcriptome-based age prediction with BulkFormer (Atlas CLS embeddings → age regressor). This repo holds
the data side: downloaders, parsers, harmonised sample tables, later embeddings and baselines.

## Stack
- uv project, Python ≥ 3.12, src layout (`src/rna_age/`), typer CLI `uv run rna-age …`, pytest.
- **polars for every table, parquet for every output** (zstd). No CSV/TSV outputs, no pandas except at a
  library's boundary (convert at the edge). JSON only for download sidecars/state.
- `pathlib` everywhere. Typed, small modules, no notebooks for pipeline code.

## Layout (`data/` is git-ignored except `data/output/summary/`)
- `data/download/<source>/`: raw bytes exactly as fetched; each file has `<file>.meta.json`
  (url, size, checksum, fetched_at). Never edit.
- `data/input/<source>/`: parser output: `samples.parquet` (schema in `src/rna_age/schema.py`),
  `genes.parquet`, `counts/part-NNNNN.parquet` (gene × ≤1000 samples, UInt32 raw counts, `gene_id` = Ensembl
  without version; `counts/index.json` maps sample → part). See `src/rna_age/counts.py`.
- `data/interim/`: merged tables, gene mapping, splits, embeddings. `data/output/`: results.
- Downloaders reproduce `data/download`, parsers reproduce `data/input` from it; nothing hand-edited.

## Commands
```bash
uv run rna-age fetch gtex            # resumable, idempotent; --connections N for parallel ranges
uv run rna-age parse gtex            # heavy parsers run as Slurm jobs, see below
uv run rna-age summary               # data/output/summary/*.parquet
uv run pytest -q
```

## Cluster
- Heavy work goes through `slurm/submit.sh cpu|gpu NAME COMMAND...` (e.g.
  `slurm/submit.sh cpu parse-gtex rna-age parse gtex`). Cluster names (partitions, account, nodes) live only
  in the git-ignored `slurm/env.local.sh` (template: `slurm/env.local.example.sh`). Never commit them.
- Internet only through the http(s) proxy from the environment. NCBI throttles bursts: one connection,
  pauses between files.
- Keep caches off `$HOME` (UV_CACHE_DIR etc. point to the large filesystem).

## Rules
- Age parsing lives in `src/rna_age/ages.py` with tests in `tests/test_ages.py`; add real strings you meet.
- Cancer is out of scope for now: tumour samples stay in mixed sources, flagged `is_tumor`, and are
  excluded from training/evaluation. Pseudobulk sources stay separate from bulk (`is_pseudobulk`).
- Splits group by study (GSE / project / GTEx donor): never one donor or one series on both sides.
- `STATUS.md` is the running log; keep it current.

## Seats
- Agent sessions here may hold seats, standing roles run by the seats kit (agent-seat-blueprints; pinned in
  `.claude/settings.json`). The seat table is `.blueprints.toml`; each seat's priming is under `docs/seats/`
  (`README.md` lists them): `lead` owns `docs/ROADMAP.md` and `docs/DECISIONS.md`, `runner` owns the jobs,
  `STATUS.md`, `docs/experiments.md` and `docs/reports/` (`docs/DECISIONS.md` D1).
