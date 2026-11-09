# Experiments log

One dated entry per decision: hypothesis → what ran → result → next. Run directories:
`data/output/runs/<YYYYMMDD-HHMM>-<name>/` (config, metrics, predictions); one row per run in the tracked
`data/output/summary/experiments.parquet`.

## Evaluation protocol (fixed 2026-11-10 00:20, before the first model)

- **Splits** from `rna-age merge` (`data/interim/splits.parquet`): whole studies (GSE / Bgee experiment /
  Census dataset; GTEx donor) go to train / val / test by a stable hash (70/15/15). No series or donor spans
  two splits. Duplicates across sources are removed (ARCHS4 > NCBI GEO > Bgee).
- **Training set**: bulk, `split == train`, `is_healthy != False` (healthy or unknown; disease, tumour, cell
  line, stem/organoid, prenatal, single-cell runs excluded), exact ages; GTEx bins enter as midpoints only in
  variants that say so (scored as intervals regardless).
- **Evaluation**: val for model selection, test for reporting. External/transfer sets: pseudobulk sources
  (Census OneK1K/AIDA/…, Terekhova 2023) predicted in full as `split = transfer`; named cohorts (Fleischer
  2018, …) reported via the `cohort` grouping; LongevityBench GTEx samples via `in_longevitybench`.
- **Metrics** (`src/rna_age/train/evaluation.py`): MAE, median AE, Pearson r, R², bias slope, mean signed
  error; GTEx bins: interval hit rate and distance to the bin (+ midpoint metrics). Groupings: source,
  tissue (n ≥ 20), decade, healthy vs disease (age gap of disease samples should be positive), bulk vs
  pseudobulk, cohort, and "not in tAge training".
- **Controls**: every model vs the ridge baseline on the same split; tAge (relative; compared on age deltas
  within dataset × tissue strata) and ClockBase on their overlap, with and without their training series.

## Ladder

1. Baselines: ridge / ElasticNet on log-CPM HVGs (2k / 5k / 10k; chosen on train only).
2. BulkFormer weights (Atlas, Latest).
3. Frozen embeddings + probes (Atlas CLS; Latest pooled; + HVG).
4. Fine-tuning Atlas (head → last blocks → full), Gaussian/Huber + interval loss for bins.
5. Variants: healthy vs all, bulk vs bulk+pseudobulk, tissue-conditioned, study weighting, bias correction.

## Entries

### 2026-11-10 00:15 · tAge control smoke test
- **Ran**: `rna-age tage gtex --split test --max-samples 600`, EN Chronoage Multispecies Multi-Tissue (scaled,
  YuGene), strata = GTEx tissue (centred on all stratum samples).
- **Result**: n = 551; Pearson r of tAge vs (age − stratum mean age, bin midpoints) = 0.38 (scaled), 0.33
  (YuGene). No GTEx in tAge training. Relative output in years.
- **Next**: run on full ARCHS4/GTEx test sets once counts exist; same delta metric for our models.

### 2026-11-10 00:35 · GPU probe
- GPU jobs run on a requeue-capable partition (settings only in the git-ignored `slurm/env.local.sh`), so
  every GPU job must resume from its own outputs (embedding shards are skipped when present).
- Probe: **NVIDIA GeForce RTX 2080 Ti, 11 GB, compute capability 7.5**; torch 2.15 (CUDA 13.2 build) works.
  Matmul: fp32 3.2, fp16 10.2, bf16 6.8 TFLOP/s (bf16 is emulated) → **fp16 autocast** for Atlas, with an
  fp16-vs-fp32 CLS cosine check logged on the first chunk of every embedding job.
- Implication: Atlas (242 M params, 4,097 tokens) costs ~0.4 s/sample forward on this card → ~100k samples
  ≈ 11 GPU-h (≈ 3 h on 4 GPUs). Full fine-tuning on 11 GB cards is impractical tonight; rung 4 starts with
  head + last blocks on a training subsample.
- **Next**: Atlas panel (train only) → embedding jobs (val/test first) → probes; ridge baselines on CPU meanwhile.

### 2026-11-10 00:45 · Data merged; rung 1 submitted; Atlas embeddings running
- **Data**: 98,476 eligible, de-duplicated samples with splits (train 66,609 / val 18,413 / test 13,454):
  ARCHS4 74,293 exact (1,676 studies), GTEx 19,461 binned, Census pseudobulk 2,958 exact + 71 binned/censored,
  Terekhova 317 exact, Bgee 72 not already in ARCHS4. Independent label check: parsed ARCHS4 ages agree with
  Shokhirev & Johnson's curation within 1 year for 5,370 / 5,371 overlapping samples.
- **Rung 1**: ridge on 2k / 5k / 10k HVGs and ElasticNet on 5k HVGs submitted as CPU jobs (queued behind the
  shared CPU cap).
- **Atlas panel**: 4,096 genes chosen on 3,299 training samples (1,500 ARCHS4, 1,500 GTEx, 299 Bgee) →
  `data/interim/embeddings/atlas_panel.parquet`.
- **Embeddings**: 4 GPU jobs. Fix needed: Atlas passes an all-true boolean mask to SDPA, which forces the
  full-matrix kernel (OOM at batch 4 on 11 GB); `rna_age.train.atlas_patch` drops the mask when every position
  is valid (identical maths). fp16 vs fp32 CLS cosine = 1.000; throughput ≈ 0.1 s/sample/GPU.
- **Note**: jobs started at the same moment on one node all picked the same "free" GPU; shards are now pinned
  to different nodes.
