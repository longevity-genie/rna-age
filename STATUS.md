# rna-age status

**Report page (private artifact, republished as results change):** https://claude.ai/artifact/TfY8RYwiVDLfm6pEwj81ic


_Last update: 2026-11-12 00:55 CET (runner seat; the plan is docs/ROADMAP.md, the lead's)._

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
- 2026-11-12 age inventory (metadata only, `rna-age inventory` → `data/output/summary/age_inventory.parquet`):
  Census LTS all 1,086 human non-spatial datasets (obs via TileDB-SOMA with the S3 proxy set explicitly, 159 M
  cells in ~2 min), Discover datasets newer than the LTS (upper bounds), scBaseCount 2026-01-12 (35,263 SRX +
  ENA BioSample attributes), recount3 human SRA (277,781 experiments). New adult donors/samples passing the quality
  rules: Census 2,356 healthy (3,815 any health), scBaseCount 492 (1,656), recount3 bulk 4,430 (6,380).

## Running (jobs carry seat / what / eta in their Slurm comment)
- RM30: job 6479218 trains `20261113-2053-expricl` (10k steps, fp16, ~2 s/step), ends ~02:55 on 2026-11-14; quick
  evals flat since 4-6k, so no resume. Read-outs queued after it: sc 6479915, bulk 6479916 → health 6479917
  (200 shuffles), perturb 6479918.

## Results so far (details: `docs/experiments.md`, `data/output/summary/experiments.parquet`)
- **RM32 (2026-11-13 22:00)**: per-cell clock on the RM28 mask encoder: attention ties mean pooling (+1.81 vs +1.78 y
  behind pseudobulk), per-cell losses make it worse (+2.22 to +2.41).
- **RM31 (2026-11-13 21:58)**: five cell types per donor side by side (TabICLv2) beat the whole-sample clock in 3
  of 7 cohorts (bar 4), mean −0.31 y; wins in two brain cohorts and one blood cohort.
- **RM28 (2026-11-13 21:12)**: our own single-cell pretraining (mask, mask+perturb, universal; 30 epochs) gives
  donor-age clocks 1.3-2.0 y worse than pseudobulk in all 7 cohorts, frozen or fine-tuned; universal bulk MAE 18.7.
- **RM27 (2026-11-13 19:33)**: a Pasta-style age-order term on same study × tissue pairs leaves the health MLP's
  aging-linked gap at +2.69 y (3-seed mean; control 2.55; bar +2.9 not met), test MAE 0.27 y better; null p 0.005.
- **RM26 (2026-11-13 17:50)**: adding 3,289 recount3 bulk samples (2,046 training rows) lowers the 3-seed MLP test MAE
  from 7.90 to 7.73 y on the old test samples, −0.17 [−0.39, +0.02]: in the right direction but not beyond the CI; bar
  not met, no fine-tune retrain. New recount3 test studies read 11.2 y.
- **RM25 (2026-11-13 02:39)**: TabICLv2 as a bulk clock fails the lead's bar by far: test MAE 15.3 y, r 0.51
  (n 8,843; 2,000 HVG, 4,000 nearest-study context rows) vs the MLP's 8.1 and the headline's 7.46. TabICL stays a
  single-cell clock; no bulk fine-tuning step.
- **Sample sets (2026-11-11, RM12)**: meningioma and 169 plate single-cell series removed; eligible 100,312 →
  94,526. Figures before 06:58 (the 9.20 headline included) are on the old set and not comparable with figures on
  the new one; each figure below says which. On the new set, 20k-HVG MLP seeds: val 8.73, test 8.14 (old seeds 9.02 /
  8.09 on the same samples).
- **RM16 (2026-11-12 14:20)**: against the best of ridge, elastic net and boosted trees, TabICLv2 on 2,000 genes wins
  in 4 of 7 single-cell cohorts on the whole sample (0.8-2.0 y, CI below 0), loses in the largest brain cohort (+0.8 y);
  Terekhova whole blood 6.9 y vs 8.9 (elastic net).
- **RM14 (2026-11-12 02:40)**: TabICLv2 as a single-cell (per-donor pseudobulk) clock beats ridge in 4 of 7 cohorts on
  the whole sample (0.3-0.8 y, CI below 0), loses in none; best single classes gain up to 1.8 y (Terekhova CD4 T 6.2 y).
  D7's bar is met; the lead reads it.
- **RM1 done (2026-11-12 00:30, RM12 set)**: health-aware Atlas fine-tune, three seeds on the clean table. Test MAE
  11.47 / 11.12 / 10.20 (seeds 0 / 1 / 2); aging-linked case − control gap +3.09 / +2.88 / +2.85 y, each CI above 0.
  Mean of the three (`20261112-0022-ens-atlasft-rank1-3s`): test 10.62 (r 0.78), aging-linked **+2.94 y [1.24, 4.74]**
  (33 held-out studies), neurodegenerative +4.68 [1.46, 7.96], autoimmune +4.58; 0 of 200 label shuffles reach it.
  Health ensemble rebuilt with that mean (`20261112-0022-ens-health3t2-plus-atlasft-rank1-3s`): test 9.53 (r 0.81),
  aging-linked +2.79 [0.88, 4.88]; the earlier one (old-table fine-tune) test 9.21, +2.81. The gap is stable across
  seeds and ensembles; accuracy moves ~1 y between seeds.
- **Health-aware clock (afternoon)**: MLP on 20k HVGs with a within-study ranking loss (cases older than their
  study's controls) + disease head makes the case − control age gap on **held-out studies** significant:
  aging-linked diseases +2.35 [0.61, 4.17] y (seed 0), +2.62 [0.64, 4.56] (seed 1), +2.34 [0.49, 4.27] (seed 2),
  at control MAE 10.1–11.3; control-only / healthy-only models +0.8–1.1 y (n.s.). Neurodegenerative stays n.s.
- **Composition vs within-cell-type (blood pseudobulk, 4 cohorts, 196–981 donors)**: equalising the cell mix
  keeps the clock (r 0.85 → 0.78–0.85), composition alone carries little (0.20–0.43); CD4 T cells alone match
  whole PBMC. The aging signal in blood is within cell types, mostly T cells.
- **Health-aware Atlas fine-tune (RM1, 04:05)**: ranking loss on the fine-tuned encoder: aging-linked gap
  +2.93 y [1.17, 4.67] on held-out studies (MLP +2.35), neurodegenerative +3.84 [0.66, 7.47] (first significant);
  test MAE 10.52 (plain 10.39), validation and pseudobulk transfer worse. One seed; null check pending.
- **Composition-adjusted disease gaps (blood, 18 held-out studies)**: deconvolving PBMC classes and adjusting
  shrinks the case gap 15–30 % but it survives (health-aware MLP 2.46 → 2.07 y [1.11, 3.32]; headline 1.25 → 0.89
  [0.17, 2.04]); unchanged with a granulocyte reference added (health-aware 2.08 [0.96, 3.08]).
- **Composition outside blood**: brain (PsychENCODE, 783 and 299 donors) shows the blood pattern: fixed composition
  r 0.82 / 0.91 ≈ whole tissue, composition alone 0.41 / 0.46; oligodendrocytes or astrocytes alone ≈ whole tissue.
  Shuffled-age nulls give r ≈ 0 (brain and blood). Lung cohorts too small.
- **Tabular foundation models** (TabICLv2, GeneICL, so far): test MAE 15.6–16.3, far behind our MLP (10.4).
- **PRIMO (local, official scorer, 16 public tasks, nothing submitted)**: age fine-tunes match frozen Atlas
  (mean per-task difference −0.007 / −0.005, within fold noise): no forgetting, no gain.
- **Morning report**: `docs/reports/2026-11-10-overnight.md` (+ the report page linked at the top).
- **Headline (13:35)**: 7-MLP ensemble + mean of the two all-health-states Atlas fine-tunes (last 4 and last 6
  blocks): val 8.70, **test MAE 9.20**, r 0.817, transfer 13.5 / 0.67.
- Previous headline (10:25): 7-model MLP ensemble + all-health-states Atlas fine-tune (mean): val 8.72, **test MAE
  9.26**, r 0.816, transfer 13.6 / 0.67.
- Earlier headline: 7-model MLP ensemble on 20k log-CPM HVGs trained on all health states: val 9.00, **test MAE
  9.52**, r 0.793; + fine-tuned Atlas (last 4 blocks): test 9.45, r 0.817. Atlas fine-tune alone 10.85 (r 0.763),
  best pseudobulk transfer (13.8 / 0.66). Frozen Atlas and frozen Latest add nothing over expression.
- Clocks on identical samples (within-stratum r): tAge 0.33 / 0.32 (ARCHS4 / GTEx) vs ours 0.70 / 0.77;
  ClockBase 0.23 (chronological) vs ours 0.66.
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
- **Best now**: MLP on 20k HVG trained on all health states, 5-model ensemble: val 9.02, **test MAE 9.63**,
  r 0.786, GTEx bin hit 0.46; vs tAge on identical samples r 0.70 vs 0.33 (ARCHS4), 0.77 vs 0.32 (GTEx).
- Frozen Atlas CLS adds nothing on top of HVGs; gradient boosting ≈ ridge.
- Disease age gap (age-adjusted): +0.49 y overall; within-study +0.40 ± 0.61 y (69 studies), not significant.
- Terekhova PBMC transfers well from bulk-only models (ridge MAE 8.1, r 0.88); Census datasets do not.
- RM23 (2026-11-12): MaxToki-217M's in-silico knockdown in blood CD4 T cells does not agree with the CD4 CRISPRi
  screen (Spearman −0.03 / −0.03 / +0.01 Rest / Stim48hr / Stim8hr, all inside the shuffled-gene null; our clock
  0.25 on the same genes). Gated: the released 217M's time answer does not track true timelapses (Spearman −0.02 on
  600 prompts); its paper used a regression head that is not released. `maxtoki_cd4.parquet`.

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

## Next (order and owner: docs/ROADMAP.md)
1. RM1 and RM12 closed (D6, the lead): the reported health clock is the three-seed fine-tune mean
   (`20261112-0022-ens-atlasft-rank1-3s`); the MLP-plus-fine-tune ensemble is its accuracy-balanced variant; the earlier
   health ensemble is retired (pre-D4 fine-tune). Page (v14) and state switched.
2. RM8/RM9 canceled (D6): KronLSTM stopped, the default of the unanswered decision; its code stays in the tree.
3. Done 2026-11-11: RM4 (Terekhova composition), RM10 (biology: stage effects yes, interventions no; closed by the lead),
   RM11 (robustness), RM12 (clean table: meningioma and plate single-cell series out).
4. Blocked on Anton: PRIMO submission (desk); tabular foundation models paused.
