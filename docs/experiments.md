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
