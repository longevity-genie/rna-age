# rna-age status

**Report page (private artifact, republished as results change):** https://claude.ai/artifact/TfY8RYwiVDLfm6pEwj81ic


_Last update: 2026-11-10 02:35 CET (overnight session)._

## Mode (Anton, 2026-11-10 ~00:10)
Fully autonomous overnight: data → baselines → BulkFormer embeddings/probes → fine-tuning, logging every run
(`data/output/runs/`, `docs/experiments.md`, `data/output/summary/experiments.parquet`); the loop does not
stop before Anton is back; 07:00 report, then keep updating; one tabbed artifact (link here once published).

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
- Bgee parsed (job 6423376): 1,253 libraries (exact + decade stages) from 40 experiments → `data/input/bgee/`.
- `geo_cohorts` (labels only, no counts): 17 verified aging cohorts in GEO (Fleischer 2018 GSE113957, 500FG,
  MAGNet heart, retina, brain, blood, muscle GESTALT, ...) + Shokhirev & Johnson 2021 compilation from
  Mendeley (6,471 samples, 3,060 in the paper's set, curated age/sex/condition) → `data/input/geo_cohorts/`.
  The cohort series are always NCBI cross-check targets.
- `census_pseudobulk`: module + `rna-age census-pseudobulk <dataset_id>` Slurm step tested on the dermal
  fibroblast atlas (108k cells → 186 donor × tissue groups in 8 s; 74 pass ≥ 200 cells / ≥ 500k UMIs).
- tAge R env build started (`scripts/setup_tage_env.sh`, micromamba conda-forge R 4.4 in `data/envs/tage`),
  log `data/logs/setup-tage.log`.
- Clock controls (priority 4) groundwork:
  - LongevityBench (HF `insilicomedicine/longebench`, CC-BY-NC) fetched + parsed: its transcriptomic tasks are
    GTEx-only, 10-year-bin labels, text prompts for LLMs (LB-0014 multiclass, LB-0018 pairwise, LB-0022
    ternary; accuracy). 1,193 GTEx sample IDs → `data/input/longevitybench/gtex_tasks.parquet`; 1,062 are in
    our GTEx v10 set and flagged `in_longevitybench` in the merge. Published per-task baselines not verified
    (Cell page blocked).
  - ClockBase: no bulk dump; per-GSE JSON API (`/api/gse/{GSE}/clocks`, keyed by GSM). Human RNA outputs are
    two tAge clocks on ARCHS4 counts, control-centred (relative, not years). `sources/clockbase.py` queries
    only our held-out series.
  - tAge: no human-only clock; human-applicable = Multispecies Multi-Tissue EN/BR Chronoage (scaled / YuGene;
    Zenodo 22166800, MGB non-commercial licence). Outputs are **relative to a reference group** centred per
    dataset × tissue, so we compare deltas or add back the reference mean age. Published CV MAE ≈ 5 % of max
    lifespan (≈ 6 y in humans), r ≈ 0.95 (pooled species). Human training data: 16 GEO series + MESA (no
    GTEx) → samples from those series are flagged `in_tage_training` in the merge.
- NCBI GEO module `sources/ncbi_geo.py` (throttled: 1 request at a time, 4 s pauses, 20-min back-off on
  403/429, ≤ 400 new files per run): esearch lists ~27.7k human series with NCBI counts; targets =
  30 largest age-annotated ARCHS4 series (cross-check) + series missing from ARCHS4 (metadata text first,
  counts only when the text has donor ages). Waits for ARCHS4 metadata.
- ARCHS4 parser written (`sources/archs4.py`, `annotate.py`: sex/tissue/disease/tumour/cell line/stem/prenatal/
  single-cell flags), waiting for the download.

## Running
- Atlas fine-tuning, last 2 blocks (val MAE 15.95 → 13.34 after 2 epochs) and last 4 blocks (14.14 after
  1 epoch); logs `slurm/logs/rna-age-ft-u*.out`.
- MLP seed sweep (seeds 2–4, log-CPM 20k HVG) for a 5-seed ensemble; study-weighted MLP.

## Results so far (details: `docs/experiments.md`, `data/output/summary/experiments.parquet`)
- Data: 99,559 eligible de-duplicated samples (ARCHS4 74,293 exact; GTEx 19,461 binned; Census 3,787 +
  Terekhova 317 pseudobulk); parsed ages vs Shokhirev 2021 curation: 5,370 / 5,371 within 1 year.
- Reference (constant train median): test MAE 18.2 y.
- Rung 1 best: ridge on 20k HVG log-CPM + Horvath target, test MAE 12.64, r 0.65.
- Rung 3: frozen Atlas CLS is worse than its own 4,096 input genes with a linear readout; **MLPs are the big
  win**: MLP on 20k log-CPM HVGs test MAE 10.43 (r 0.77); 3-model MLP ensemble **val 10.00, test 10.36,
  r 0.775, pseudobulk transfer MAE 14.0 / r 0.68**. Gene-length TPM inputs are best within bulk (test 10.24)
  but break 10x pseudobulk transfer. Run-to-run noise ≈ 0.5 y test MAE.
- tAge vs our MLP on identical test samples (within-stratum relative age r): ARCHS4 0.33 vs 0.66; GTEx 0.32
  vs 0.76.
- Disease age gap (age-adjusted): +0.49 y overall; within-study +0.40 ± 0.61 y (69 studies), not significant.
- Terekhova PBMC transfers well from bulk-only models (ridge MAE 8.1, r 0.88); Census datasets do not.

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
- Census datasets: tier 1 = 16 non-cancer datasets with many exact-age donors (~130 GB: OneK1K, AIDA v2,
  HLCA, PsychENCODE HBCC/RADC, lupus PBMC, COVID PBMC, Indonesia PBMC, fibroblasts, HLiCA, Immune Health
  Atlas, CellRef, HSPCs, microglia, ILD, kidney); tier 2 (~96 GB: MSSM brain 1,042 donors, Immunobiology of
  Aging PBMC, HBCA) after tier 1 (`RNA_AGE_CENSUS_TIER=2`). Cancer atlases skipped.
- Terekhova 2023 (update 2026-11-10 00:25): the Synapse token in `~/.synapseConfig` answers 401 from this
  machine (also with the official `synapseclient`), so the raw-count archive (syn56693935) is not fetched;
  following brief item 6 we use the HCA `all_pbmcs_rna.h5ad` and recover UMIs from log1p-CP10k if `X` is not
  integral (checked on the first chunk). Not blocked. Parse job 6423688.
- Terekhova 2023: not in CELLxGENE and Synapse/AWS refuse anonymous access, but the HCA Data Portal serves it
  openly (Anton, 2026-11-10): Azul catalog `dcp60`, project `896f377c-…`; `all_pbmcs.tar.gz` (17.1 GB) +
  metadata xlsx fetched through Azul's redirect, sha256-verified (`sources/terekhova2023.py`).
- **NCBI-generated GEO counts are blocked**: since 2026-11-09 ~23:55 NCBI's `/geo/download/` answers this
  client with a Google reCAPTCHA challenge page (HTML) instead of the file; the GEO text metadata endpoint
  still works. We do not work around bot protection. `ncbi_geo` is parked; its gap-filling would need Anton to
  ask NCBI / use an approved route (e.g. GEO FTP if NCBI publishes these counts there later).
- Controls: Anton confirmed tAge and ClockBase are the controls; LongevityBench is optional (kept, already
  fetched; it costs nothing).
- Bgee: GTEx inside Bgee (SRP012682) is skipped (same bins as our GTEx source).

## Next
1. ARCHS4: when the download finishes, read sample metadata (light), parse ages, choose eligible samples,
   extract their counts in a Slurm job.
3. `ncbi_geo` targets + throttled fetch; curated GEO cohorts (Fleischer 2018, Shokhirev & Johnson 2021
   series; a lookup is running) as `cohort` labels on ARCHS4/NCBI samples.
4. Census pseudobulk (`census_pseudobulk`), Terekhova 2023.
5. Splits by study, summary, overnight report.
6. Clock controls (tAge, LongevityBench, ClockBase); then the BulkFormer stretch.
