# rna-age status

_Last update: 2026-11-09 23:25 CET (overnight session)._

## Done
- Scaffold: uv package (Python 3.12, src layout), typer CLI `rna-age fetch|parse|summary`, polars/parquet,
  pytest (78 age-parser tests green), ruff. `CLAUDE.md`, `README.md`, Slurm wrapper (`slurm/submit.sh`;
  cluster names only in the git-ignored `slurm/env.local.sh`).
- `rna_age.download`: resumable, idempotent fetcher with parallel HTTP Range segments, `.part` + segment
  state, publisher checksum (GCS md5, S3 multipart ETag) or own sha256, `gzip -t`, `.meta.json` sidecars.
- `rna_age.ages`: age parser (exact / binned / censored, months/weeks/days, rejects gestational age,
  passage, time post infection, NA, adult, prenatal values) with tests.
- GTEx v10 fetched and verified (`data/download/gtex/`, 0.98 GB) and parsed (Slurm job 6423309, ~1 min):
  `data/input/gtex/` = 19,788 samples × 58,988 genes (1.4 GB parquet); 946 donors, 68 tissues, age bins
  20-29 … 70-79 (`binned`); 19,461 eligible (327 EBV lymphocyte samples excluded as cell line).
- `rna-age summary` → `data/output/summary/*.parquet` (tracked).
- Bgee 15.2 fetched (`data/download/bgee/`, 1.3 GB: library annotation + 40 non-GTEx experiments with
  `NN-year-old` / decade stages, ~790 exact-age libraries); parser `sources/bgee.py` (kallisto counts rounded).
- NCBI GEO module `sources/ncbi_geo.py` (throttled: 1 request at a time, 4 s pauses, 20-min back-off on
  403/429, ≤ 400 new files per run): esearch lists ~27.7k human series with NCBI counts; targets =
  30 largest age-annotated ARCHS4 series (cross-check) + series missing from ARCHS4 (metadata text first,
  counts only when the text has donor ages). Waits for ARCHS4 metadata.
- ARCHS4 parser written (`sources/archs4.py`, `annotate.py`: sex/tissue/disease/tumour/cell line/stem/prenatal/
  single-cell flags), waiting for the download.

## Running
- ARCHS4 download: `uv run rna-age fetch archs4 --connections 12`, log `data/logs/fetch-archs4.log`
  (64.4 GB at ~16 MB/s; started 22:54, expected ~23:55).
- The shared Slurm account is often at its 320-CPU cap (other users' jobs); small jobs (4 cores, 48 GB, 4 h)
  start once a few cores free up.

- Slurm job `parse-bgee` (6423354), log `slurm/logs/rna-age-parse-bgee-6423354.out`.

## Decisions
- Through the proxy one connection gets ~2–4 MB/s; parallel ranges stack (12 → ~16 MB/s).
- ARCHS4: `human_gene_v2.latest.h5` (2026-11-06; no versioned name for this release), pinned by its S3 ETag
  `9b8fcd0486471467ef00169bc4ce1511-7681`. Reading its metadata remotely through HTTP ranges was too slow
  (>10 min without finishing), so metadata is read from the local copy once downloaded.
- GTEx: `read_count_total` = sum of gene counts (comparable across sources), not SMRDTTL. All samples
  `is_healthy`; Hardy scale kept in `disease` (`hardy:N`). EBV-transformed lymphocytes = cell line
  (excluded); cultured fibroblasts kept (primary cultures, as in Fleischer 2018). Gene versions stripped,
  `_PAR_Y` rows dropped.
- Counts: gene × sample parquet parts (≤ 1000 samples each, UInt32), `counts/index.json` maps sample → part.
- Anton's updates of 2026-11-10 (override the original plan):
  1. **No CSV anywhere**: all tables, manifests, summaries, results are polars parquet (zstd); the tracked
     summary is `data/output/summary/*.parquet`. JSON only for download sidecars/state.
  2. **No cancer for now**: TCGA/TARGET dropped. Tumour samples in mixed sources stay flagged (`is_tumor`,
     `disease`) and are excluded from training/evaluation (`summary.eligible`).
  3. **Bulk beyond ARCHS4**: NCBI-generated GEO RNA-seq counts (`ncbi_geo`, throttled) for age-annotated
     series ARCHS4 lacks and as a cross-check; Bgee for clean healthy age labels; recount3 only if time is left.
  4. **Pseudobulk** (priority 2b, after GTEx + ARCHS4 are parsed): CELLxGENE Census (`census_pseudobulk`,
     LTS release, donor × tissue, ≥ 200 cells and ≥ 500k UMIs) and Terekhova 2023 (`terekhova2023`) if not in
     Census. Schema gained `is_pseudobulk`, `assay`, `suspension_type`, `n_cells`, `n_umis`.

  5. **Published transcriptomic clocks as controls** (new priority 4, after the summary, before the
     BulkFormer stretch): tAge (Gladyshev lab, Nature 2026; R package in an isolated conda-forge R env on
     /data, wrapped in a CLI step parquet → parquet), LongevityBench (Cell 2026; its transcriptomic-age
     task as an external test), ClockBase precomputed predictions joined by GSM/SRR. Record which test
     samples were in each clock's training data and report metrics with and without them.
- Census: tiledbsoma over the proxy stalled on the first read (>5 min for the datasets table), so
  `census_pseudobulk` uses the pinned LTS release's h5ad copies over HTTPS
  (`cell-census/2025-11-08/h5ads/<dataset_id>.h5ad`, S3 ETags) and the CELLxGENE Discover API for dataset
  metadata (development stages, donors) to choose datasets; pseudobulk with h5py in Slurm jobs.
- Bgee: GTEx inside Bgee (SRP012682) is skipped (same bins as our GTEx source).

## Next
1. ARCHS4: when the download finishes, read sample metadata (light), parse ages, choose eligible samples,
   extract their counts in a Slurm job.
3. `ncbi_geo` targets + throttled fetch; curated GEO cohorts (Fleischer 2018, Shokhirev & Johnson 2021
   series; a lookup is running) as `cohort` labels on ARCHS4/NCBI samples.
4. Census pseudobulk (`census_pseudobulk`), Terekhova 2023.
5. Splits by study, summary, overnight report.
6. Clock controls (tAge, LongevityBench, ClockBase); then the BulkFormer stretch.
