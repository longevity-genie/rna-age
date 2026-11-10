# Overnight report, 2026-11-10

**Report page (private artifact, republished as results change):** https://claude.ai/artifact/TfY8RYwiVDLfm6pEwj81ic


_Draft, updated through the night. Numbers come from the parquet summaries in `data/output/summary/` (see
`report_tables.md` there) and the run directories; the experiment log with every decision is
`docs/experiments.md`._

## 1. What we have: human RNA-seq samples with donor age

| source | what | eligible samples with age | age type |
|---|---|---|---|
| ARCHS4 (file of 2026-11-06, 64 GB) | GEO/SRA human bulk RNA-seq, ages parsed from `characteristics_ch1` | 74,293 exact (+795 binned, 81 censored), 1,676 studies | exact |
| GTEx v10 (public) | 68 tissues, 946 donors | 19,461 | 10-year bins |
| CELLxGENE Census LTS 2025-11-08 | donor × tissue pseudobulk, 12 datasets (OneK1K, AIDA, HLCA, …) | 2,958 exact (+71 binned/censored) | exact (dev. stage) |
| Terekhova 2023 (HCA copy) | PBMC pseudobulk per tube, 166 donors | 317 | exact |
| Bgee 15.2 | curated healthy libraries | 825 exact + 428 decade bins (753 already in ARCHS4) | exact / binned |

- **98,476 eligible, de-duplicated samples** with study-grouped splits (train 66,609 / val 18,413 / test
  13,454); the brief's target (≥ 20,000 exact-age samples from ≥ 2 independent sources + GTEx binned) is met.
- Label check: parsed ARCHS4 ages agree with the independent Shokhirev & Johnson 2021 curation within 1 year
  for 5,370 / 5,371 overlapping samples.
- Excluded from training (flagged, kept in tables): tumours, cell lines, iPSC/ESC/organoids, prenatal,
  single-cell runs inside ARCHS4, non-human; disease samples are a separate evaluation group.

## 2. Downloads (where, how big)

- `data/download/archs4/` 64.4 GB (S3 ETag verified) · `data/download/gtex/` 0.98 GB (GCS md5) ·
  `data/download/census_pseudobulk/h5ads/` ≈ 150 GB (14 h5ads, S3 ETags) · `data/download/terekhova2023/`
  16 GB (Azul sha256) · `data/download/bgee/` 1.3 GB · `data/download/geo_cohorts/` 0.2 GB (Mendeley sha256) ·
  `data/download/longevitybench/` 0.14 GB · `data/download/models/bulkformer/` Atlas + Latest weights.
- Parsed inputs: `data/input/<source>/{samples,genes}.parquet` + `counts/` (gene × ≤1000-sample parquet parts).

## 3. Models (exact-age MAE in years on held-out studies; full table in `data/output/summary/experiments.parquet`)

Protocol fixed before the first model (see `docs/experiments.md`): study-grouped splits, training on bulk
`healthy-or-unknown` samples unless stated, GTEx bins as midpoints / scored as interval hits, pseudobulk
(Census + Terekhova) predicted in full as a bulk→pseudobulk transfer test.

| model | val MAE | test MAE | test r | GTEx bin hit | pseudobulk MAE / r |
|---|---|---|---|---|---|
| constant (train median) | – | 18.17 | – | – | – |
| ridge, 20k log-CPM HVG + Horvath target | 11.57 | 12.64 | 0.653 | 0.430 | 14.2 / 0.54 |
| ridge on frozen Atlas CLS | 14.73 | 15.45 | 0.542 | 0.313 | 19.4 / 0.09 |
| ridge on Atlas's 4,096 input genes (no encoder) | 12.87 | 13.70 | 0.634 | 0.367 | 18.8 / 0.39 |
| MLP on frozen Atlas CLS | 12.63 | 13.69 | 0.634 | 0.346 | 17.0 / 0.23 |
| MLP, 20k log-CPM HVG (5-seed ensemble) | 10.00 | 10.44 | 0.767 | 0.440 | 14.6 / 0.62 |
| **MLP, 20k HVG, trained on all health states (5-model ensemble)** | **9.02** | **9.63** | **0.786** | **0.463** | 15.0 / 0.60 |
| MLP, Atlas CLS + 20k HVG, all health states | 9.22 | 9.80 | 0.783 | 0.445 | 18.6 / 0.54 |
| Atlas fine-tune, last 2 blocks + head | 13.27 (val subsample) | running | | | |
| Atlas fine-tune, last 4 blocks + head | 11.59 after epoch 1 (val subsample) | running | | | |

- Best model: test MAE 9.63 y (median 6.65), r 0.786, R² 0.61; bias slope −0.33 (regression to the mean).
- **tAge** on identical test samples (within dataset × tissue relative age, r): ARCHS4 0.33 vs our best 0.70;
  GTEx 0.32 vs 0.77; excluding tAge's training series changes nothing.
- **LongevityBench** GTEx decade-bin task, our held-out samples: 0.54 (val) / 0.57 (test) accuracy.
- **Transfer**: Terekhova PBMC pseudobulk is predicted well by bulk-only models (ridge MAE 8.1, r 0.88); Census
  datasets are not (dataset-level shift; snRNA brain cohorts sit 20–30 y low).
- **Disease age gap** (healthy-trained ensemble, age-adjusted): +0.49 y overall, within-study +0.40 ± 0.61 y
  (69 studies): not significant.
- Run-to-run noise ≈ 0.5 y test MAE (same data, same seed); conclusions use ensembles.

## 4. Problems and decisions

- NCBI's GEO download endpoint now answers this client with a reCAPTCHA page; NCBI-generated counts are parked
  (GEO metadata text still works). We do not work around bot protection.
- The Synapse token in `~/.synapseConfig` returns 401 from the cluster (also with `synapseclient`), so
  Terekhova raw counts came from the HCA copy, recovered from log1p-CP10k (100 % integral) as the brief allows.
- The CELLxGENE Census SOMA API stalls through the proxy; we use the same LTS release's h5ads over HTTPS.
  Four chosen datasets are not in that release and were skipped (release stays pinned).
- GPUs are RTX 2080 Ti (11 GB, no native bf16) → fp16 autocast; Atlas's all-true attention mask forced the
  full-matrix SDPA kernel, patched at runtime (identical maths).
- tAge has no human-only clock; its multispecies clocks are relative (centred per dataset × tissue), so the
  comparison uses age deltas within strata.

## 5. What exists now

- CLI: `rna-age fetch | parse | merge | summary | baseline | embed | embed-latest | probe | finetune | ensemble |
  calibrate | age-gap | tage | compare-tage | compare-clockbase | figures | report-tables | artifact`.
- Sources: GTEx v10, ARCHS4, Bgee, CELLxGENE Census pseudobulk, Terekhova 2023, plus labels (named GEO cohorts,
  Shokhirev 2021), LongevityBench GTEx tasks, ClockBase predictions; NCBI GEO counts module (parked).
- Every run under `data/output/runs/` (config, metrics, predictions), one row each in the tracked
  `data/output/summary/experiments.parquet`; decisions in `docs/experiments.md`; figures in
  `data/output/figures/`; the tabbed report page (link at the top) built by `rna-age artifact`.
- Atlas CLS embeddings for all 99.5k samples, Atlas panel inputs (fp16), BulkFormer-Latest embeddings for the
  evaluation samples + 8k training samples per source, Atlas fine-tune checkpoints (`data/interim/`).

## 6. Recommended next steps for BulkFormer training

1. **Finish the Atlas fine-tune properly**: unfreezing 4 blocks reached val ≈ 10.4 after 5 short epochs of 12k
   samples and was still improving; run more epochs over the full training set, then unfreeze 8 / all blocks with
   layer-wise LR decay and seeds ×3. The frozen encoder is not the route (CLS adds nothing over its own inputs).
2. **Fix the input normalisation for cross-assay use**: Atlas divides every sample by gene length; for 10x UMI
   pseudobulk this is wrong (our TPM experiment collapsed transfer). Prepare UMI data without the length step, or
   fine-tune with both assays and an assay token.
3. **Use all labelled data**: training on disease samples too lowered error by ~0.8 y; add GTEx through the
   interval loss in fine-tuning, and pseudobulk as a separate domain with its own head or adapter.
4. **Match the bar on the same split**: the MLP on 20k log-CPM HVGs (test MAE 9.63 as an all-health-states
   ensemble) is the number a fine-tuned BulkFormer has to beat; tAge (r 0.33) and ClockBase (r 0.23) are far
   below on identical samples.
5. **Evaluation gaps to close**: a pseudobulk test split (all pseudobulk studies happened to land in train/val),
   disease age acceleration with matched controls per study (currently +0.4 ± 0.6 y, not significant), and the
   regression to the mean (bias slope −0.33).
