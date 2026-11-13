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

### 2026-11-10 00:55 · Rung 1: ridge on log-CPM HVGs
- **Bug fixed first**: the "healthy or unknown" filter used `is_healthy != False`, which in polars drops nulls
  (training would have used only 5,208 explicitly healthy exact-age samples). Now `fill_null(True)`.
- **Reference**: constant (train median 36 y) → val MAE 19.0, test MAE 18.2 (exact ages; std ≈ 22 y; ~22 %
  children).
- **Ran**: ridge, HVGs chosen on 20k train samples, alpha by val exact MAE; GTEx midpoints in training.

| run | val MAE | test MAE | test r | bias slope | GTEx bin hit | GTEx MAE (mid) | transfer MAE / r |
|---|---|---|---|---|---|---|---|
| ridge 2k | 14.13 | 14.82 | 0.575 | −0.60 | 0.343 | 9.12 | 15.7 / 0.49 |
| ridge 5k | 12.85 | 13.72 | 0.614 | −0.51 | 0.376 | 8.10 | 16.7 / 0.51 |
| ridge 10k | 12.11 | 13.06 | 0.653 | −0.47 | 0.400 | 7.46 | 15.8 / 0.44 |

- **Reading**: more genes keep helping; strong regression to the mean (children +14 y, 80+ −20 y); whole ARCHS4
  studies are off together (study batch effects); bulk→pseudobulk transfer is poor, and one Census dataset
  (probably single-nucleus brain) sits 20–30 y below the rest.
- **Next**: ridge 20k genes; Horvath log-linear age target (for the young end); ElasticNet 5k running; Atlas
  probes when embeddings land.

### 2026-11-10 01:05 · Rung 1 continued + attribution
| run | val MAE | test MAE | test r | bias slope | GTEx bin hit | transfer MAE |
|---|---|---|---|---|---|---|
| ridge 20k HVG | 11.97 | **12.81** | 0.653 | −0.43 | 0.427 | **13.5** |
| ridge 10k HVG, Horvath target | **11.93** | 12.91 | 0.649 | −0.45 | 0.398 | 16.1 |
- The Horvath log-linear target helps a little on val; 20k genes helps test and transfer most.
- **Attribution (ridge 20k)** → `data/output/summary/ridge20k_top_genes.parquet`: strongest "older" genes
  include PTCHD4 (rank 5 overall), MIR34AHG and MIR29B2CHG (miR-34a / miR-29 hosts, both aging-linked), NEFL,
  IGFBP3; "younger" include LRRN3 (the classic blood aging gene), CD248 (rank 24), KLRB1, NEFH. Sensible
  biology; batch-driven features are not dominating the top of the list.
- **Next**: ridge 20k + Horvath target as the bar to beat; Atlas probes.

### 2026-11-10 01:30 · tAge control on GTEx test; ridge 20k + Horvath
- **ridge 20k HVG + Horvath target**: val 11.57, **test 12.64**, r 0.653, GTEx bin hit 0.430, transfer 14.2
  → current rung-1 bar.
- **tAge vs ridge on identical samples/strata** (GTEx test, 2,926 samples, 52 tissue strata; metric = Pearson r
  between the prediction's deviation from its stratum mean and the age deviation from the stratum mean, bin
  midpoints): tAge EN multispecies scaled 0.316, YuGene 0.284; ridge 20k 0.727 (Horvath 0.727).
  Caveat: ridge trained on other GTEx donors (in-distribution), tAge never saw GTEx; the ARCHS4 comparison
  (tAge training series flagged) is the fairer one, rerunning after a per-stratum fix (one stratum broke tAge's
  preprocessing).

### 2026-11-10 01:50 · Rung 3: probes on frozen Atlas CLS
- **Hypothesis**: Atlas CLS (1,024-d, the PRIMO-winning sample representation) carries age signal comparable
  to or better than log-CPM HVGs.
- **Ran**: ridge (alpha by val) and a 2-layer MLP (Huber, early stopping on val) on standardised CLS; same
  training set and splits as rung 1.

| run | val MAE | test MAE | test r | bias slope | GTEx bin hit | transfer MAE / r |
|---|---|---|---|---|---|---|
| Atlas CLS + ridge | 14.73 | 15.45 | 0.542 | −0.61 | 0.313 | 19.4 / 0.09 |
| Atlas CLS + MLP | 12.63 | 13.69 | 0.634 | −0.54 | 0.346 | 17.0 / 0.23 |
| (bar) ridge 20k HVG + Horvath | 11.57 | 12.64 | 0.653 | −0.41 | 0.430 | 14.2 / 0.54 |

- **Reading**: frozen CLS is non-linearly informative (MLP ≫ ridge) but below the HVG bar, and transfers
  worse to pseudobulk; the contrastive objective seems to encode sample identity / tissue more than age.
- **Next**: CLS + HVG probes (running); fine-tuning the last 2 / 4 Atlas blocks with a regression head
  (running, 12k samples per epoch, early stopping on val).

### 2026-11-10 01:45 · Rung 3 continued: what does the encoder add?
| run | val MAE | test MAE | test r | GTEx bin hit | transfer MAE / r |
|---|---|---|---|---|---|
| ElasticNet 5k HVG | 12.82 | 13.40 | 0.642 | 0.371 | (backfilled) |
| **panel only** (Atlas's 4,096 input genes, log2 TPM, ridge, no encoder) | 12.87 | 13.70 | 0.634 | 0.367 | 18.8 / 0.39 |
| Atlas CLS + 1,200 HVG, ridge | 13.62 | 14.61 | 0.587 | 0.321 | 17.7 / 0.28 |
| Atlas CLS, ridge | 14.73 | 15.45 | 0.542 | 0.313 | 19.4 / 0.09 |
| Atlas CLS, MLP | 12.63 | 13.69 | 0.634 | 0.346 | 17.0 / 0.23 |
- **Reading**: linearly, the frozen CLS is *worse* than the 4,096 panel genes it is computed from (14.7 vs
  12.9 val MAE); a non-linear readout recovers it (MLP 12.6). The encoder does not add age information over
  its own input when frozen; the CLS + HVG concatenation also loses to HVGs alone. Frozen Atlas is not the
  route to a better clock; fine-tuning (running) is the real test of the model.
- A Census probe crashed when the Census re-merge rewrote count parts mid-read; rerun (CLS + HVG MLP), plus a
  panel-only MLP control.
- **Next**: fine-tune results (u2, u4); then decide between more fine-tuning budget and the bulk→pseudobulk arm.

### 2026-11-10 02:10 · Non-linear readouts; pseudobulk in training; first fine-tune epoch
| run | val MAE | test MAE | test r | transfer MAE / r |
|---|---|---|---|---|
| **MLP on Atlas panel genes (4,096 log2 TPM, no encoder)** | **10.69** | **11.45** | **0.751** | 16.3 / 0.48 |
| MLP on Atlas CLS + 1,200 HVG | 11.60 | 12.46 | 0.682 | 19.9 / 0.14 |
| ridge 20k HVG + Horvath, trained with pseudobulk | 12.03 | 12.65 | 0.653 | (val Census 16.1 / 0.46) |
| Atlas fine-tune, last 2 blocks, after epoch 0 (val subsample of 3k) | 15.95 | – | – | – |
- **Reading**: a non-linear readout on the right genes is the biggest gain so far (−1.2 y test MAE vs the
  ridge bar). The encoder still adds nothing on top of its inputs. Pseudobulk: Terekhova PBMC transfers well
  from bulk-only ridge (MAE 8.1, r 0.88); the Census datasets do not, and training on 10 Census datasets does
  not help the 4 held-out ones (dataset-level shift: assay, nuclei vs cells, tissue). No pseudobulk study fell
  into the test split (few, large studies), so pseudobulk stays a transfer / val evaluation.
- **Next**: MLP on log-CPM HVG 4,096 and 20,000 (is it the panel or the MLP?); fine-tuning continues (u2, u4).

### 2026-11-10 02:15 · MLP on log-CPM HVGs (is it the panel or the MLP?)
| run | val MAE | test MAE | test r | bias slope | GTEx bin hit | transfer MAE / r |
|---|---|---|---|---|---|---|
| **MLP, log-CPM HVG 20k** | **10.13** | **10.43** | **0.773** | −0.38 | 0.406 | 14.8 / **0.65** |
| MLP, Atlas panel (4,096, log2 TPM) | 10.69 | 11.45 | 0.751 | −0.37 | 0.376 | 16.3 / 0.48 |
| MLP, log-CPM HVG 4,096 | 11.07 | 11.88 | 0.711 | −0.42 | 0.438 | 20.4 / 0.18 |
- **Reading**: the MLP is what matters (−2.2 y test MAE vs ridge on the same 20k genes); at equal size the
  Atlas panel with BulkFormer's gene-length TPM normalisation beats log-CPM HVGs (11.45 vs 11.88), so the
  normalisation is worth porting to the 20k feature set. Best transfer to pseudobulk so far as well.
- **Next**: 20k MLP with the Horvath target; seeds; TPM-normalised 20k features; fine-tuning keeps running.

### 2026-11-10 02:25 · tAge vs our models on identical test samples
Metric: Pearson r between a model's deviation from its stratum mean and the true age deviation from the
stratum mean (strata = dataset × tissue; tAge is a relative clock). Only samples tAge scored are used.

| test set | samples / strata | tAge EN scaled | tAge EN YuGene | ridge 20k + Horvath | MLP 20k |
|---|---|---|---|---|---|
| ARCHS4 | 8,643 / 185 | 0.331 | 0.336 | 0.560 | **0.662** |
| ARCHS4 without tAge training series | 8,561 / 184 | 0.328 | 0.334 | 0.557 | **0.659** |
| GTEx | 2,926 / 52 | 0.316 | 0.284 | 0.727 | **0.758** |
- tAge's training series barely overlap our test split (82 samples), so excluding them changes nothing.
- One ARCHS4 stratum (a blood series) failed inside tAge's preprocessing and was skipped.

### 2026-11-10 02:20 · MLP seeds, Horvath target, ensemble
| run | val MAE | test MAE | test r | bias slope | GTEx bin hit | transfer MAE / r |
|---|---|---|---|---|---|---|
| MLP 20k HVG, seed 0 | 10.13 | 10.43 | 0.773 | −0.38 | 0.406 | 14.8 / 0.65 |
| MLP 20k HVG, seed 1 | 10.04 | 10.64 | 0.759 | −0.38 | 0.447 | 14.0 / 0.66 |
| MLP 20k HVG, Horvath target | 10.44 | 10.78 | 0.765 | −0.34 | 0.451 | 13.9 / 0.69 |
| **ensemble of the three (mean)** | **10.00** | **10.36** | **0.775** | −0.36 | 0.440 | 14.0 / 0.68 |
- Seed spread ≈ ±0.1 val / ±0.2 test MAE. The Horvath target trades a little MAE for less regression to the
  mean and better transfer; the ensemble is the current best (`rna-age ensemble`).
- Fine-tuning after epoch 0: last 2 blocks 15.95, last 4 blocks 14.14 val MAE (3k val subsample).

### 2026-11-10 02:25 · Gene-length TPM features break UMI pseudobulk
| run | val MAE | test MAE | test r | GTEx bin hit | transfer MAE / r |
|---|---|---|---|---|---|
| MLP, log2 TPM HVG 20k (gene-length, all samples) | 10.02 | **10.24** | 0.773 | 0.436 | 22.1 / **0.09** |
| (for comparison) MLP, log-CPM HVG 20k | 10.13 | 10.43 | 0.773 | 0.406 | 14.8 / 0.65 |
- **Reading**: gene-length TPM is the best single-model input for bulk, but it destroys transfer to 10x
  pseudobulk: 3′/5′-tag UMI counts are not proportional to gene length, so dividing by length distorts them.
  This also plausibly explains part of Atlas's poor pseudobulk transfer (its preparation applies the same
  length normalisation to every sample).
- **Next**: assay-aware TPM (no length division for UMI pseudobulk rows; run `…-tpmhvg20000-mlp` at 02:26
  onwards uses it), interval-loss MLP running.

### 2026-11-10 02:27 · Interval loss for GTEx bins in the MLP
| run | val MAE | test MAE | test r | GTEx bin hit | transfer MAE / r |
|---|---|---|---|---|---|
| MLP 20k HVG, interval hinge for binned ages | 10.05 | 11.23 | 0.730 | 0.427 | 18.0 / 0.30 |
- No gain over midpoint targets (test and transfer worse, GTEx bin hits unchanged); midpoints stay the default.
- **Next**: study-weighted MLP (1/sqrt(study size)); assay-aware TPM running.

### 2026-11-10 02:40 · Disease age acceleration (best ensemble) + health-label fix
- **Label audit**: the health parser treated `ctrl` / `HC` / `non-demented` as diseases and study-design labels
  from `group:` fields (`discovery`, `validation`, …) as disease names. Fixed in `annotate.py` (+ tests);
  ARCHS4 metadata re-parsed (counts unchanged: eligibility does not depend on health): explicitly healthy
  9,051 → 14,303, disease 65,024 → 59,289. Runs before 02:35 trained with the old labels (training set =
  healthy-or-unknown, so ~5k more control samples are now eligible for training).
- **Age gap** (`rna-age age-gap`): residual trend on age fitted on healthy-or-unknown val+test samples and
  removed. Disease overall +0.49 y (SEM 0.12; 7,451 samples, 150 studies). **Within-study contrast
  (disease − control, 69 studies with ≥ 5 of each): +0.40 ± 0.61 y — not significant.** Per-label gaps rest
  on 2–5 studies each and are study-confounded (e.g. MS +17.7 from 2 studies). The clock does not yet show a
  reliable disease acceleration.

### 2026-11-10 02:45 · Assay-aware TPM; run-to-run noise
| run | val MAE | test MAE | test r | transfer MAE / r |
|---|---|---|---|---|
| MLP log2 TPM 20k, UMI pseudobulk as CPM | 10.04 | 10.75 | 0.742 | 31.2 / 0.21 |
| (02:20) MLP log2 TPM 20k, all samples length-normalised | 10.02 | 10.24 | 0.773 | 22.1 / 0.09 |
- Mixing scales (TPM for bulk, CPM for UMI) makes transfer worse: the model learns per-gene levels on the TPM
  scale. log-CPM stays the input for anything that must transfer to pseudobulk.
- **Noise floor**: both runs train on exactly the same samples and features (they differ only in how
  pseudobulk rows are scaled), same seed, yet test MAE differs by 0.5 y (GPU nondeterminism + early stopping).
  Single-run differences below ~0.5 y are not meaningful; final comparisons use seed ensembles.

### 2026-11-10 02:50 · Study weighting; fine-tune epoch 1
- MLP 20k HVG with per-sample weight 1/sqrt(study size): val 10.39, test 11.93, r 0.700, transfer 15.7 / 0.52
  → worse than equal weights (beyond the 0.5 y noise floor); equal weighting stays.
- Atlas fine-tune, last 2 blocks: val MAE 15.95 → 13.34 after epoch 1 (improving, still behind the MLP).
- Running: seeds 2–4 of the 20k MLP; MLP trained on all health states (disease included).

### 2026-11-10 03:00 · Seeds, training on all health states, ensembles, calibration
| run | val MAE | test MAE | test r | bias slope | GTEx bin hit | transfer MAE / r |
|---|---|---|---|---|---|---|
| MLP 20k, seeds 0–4 (range) | 10.04–10.31 | 10.43–11.14 | 0.73–0.77 | −0.36 to −0.41 | 0.41–0.45 | 14.0–15.9 |
| **5-seed ensemble** (healthy-or-unknown) | 10.00 | 10.44 | 0.767 | −0.39 | 0.440 | 14.6 / 0.62 |
| MLP 20k trained on all health states | **9.48** | 10.28 | 0.765 | −0.37 | 0.437 | 16.5 / 0.46 |
| **7-model ensemble** (5 seeds + train-all + Horvath) | 9.83 | **10.26** | **0.774** | −0.38 | 0.443 | 14.6 / 0.62 |
| 7-model ensemble, val-fitted linear de-compression | 13.39 | 14.02 | 0.774 | **0.00** | 0.265 | 14.3 / 0.62 |
- Training on disease samples too adds data and lowers val MAE (9.48, best single model); a train-all
  ensemble (seeds + Horvath) is running.
- De-compressing predictions with a line fitted on val removes the regression to the mean completely but
  raises MAE by ~4 y: the MAE-optimal model shrinks towards the mean. Both versions are reported; the
  uncalibrated ensemble is the headline.

### 2026-11-10 03:10 · Training on all health states (seeds + Horvath) → new best
| run | val MAE | test MAE | test r | GTEx bin hit | transfer MAE / r |
|---|---|---|---|---|---|
| MLP 20k, all health states, seeds 0–3 | 9.14–9.48 | 9.40–10.28 | 0.765–0.796 | 0.44–0.46 | 14.9–16.5 |
| MLP 20k, all health states, Horvath target | 9.22 | 10.14 | 0.767 | 0.456 | 13.6 / 0.71 |
| **ensemble of the five** | **9.02** | **9.63** | **0.786** | **0.463** | 15.0 / 0.60 |
- Test median AE 6.65 y, R² 0.61, bias slope −0.33; GTEx midpoint MAE 6.25 y. Disease samples add training
  signal without hurting healthy test samples. For disease *age acceleration* the healthy-trained ensemble is
  the right model (a model trained on disease learns to absorb it).
- **LongevityBench** GTEx multiclass task (6 decade bins), samples in our held-out splits: bin accuracy 0.54
  (val, n 114), 0.57 (test, n 74); chance ≈ 0.17, majority class ≈ 0.34. Published task baselines not verified.
- **Atlas fine-tuning**: last 4 blocks val 14.14 → 11.59 after epoch 1 (improving fast); last 2 blocks plateaued
  (15.95 → 13.34 → 13.27) and is being finalised from its best checkpoint. Final predictions now cover only
  val/test/transfer samples (predicting all 99k would take ~2.7 GPU-hours).

### 2026-11-10 03:20 · Does Atlas CLS add to a strong expression model? Wider MLP?
| run (all health states) | val MAE | test MAE | test r | transfer MAE / r |
|---|---|---|---|---|
| MLP Atlas CLS + 20k HVG | 9.22 | 9.80 | 0.783 | 18.6 / 0.54 |
| MLP 20k HVG, 1024×256, dropout 0.3 | 9.18 | 9.72 | 0.786 | 14.7 / 0.57 |
| (reference) MLP 20k HVG, seeds 0–3 | 9.14–9.48 | 9.40–10.28 | 0.77–0.80 | 14.9–16.5 |
- Frozen Atlas CLS adds nothing on top of 20k HVGs (within the ±0.5 y noise floor) and hurts transfer; a wider
  MLP is also within noise.
- **Next**: BulkFormer-Latest frozen embeddings (full transcriptome in; pooled 768-d) on test/val/transfer +
  8k training samples per source, to close rung 3 for both released models.

### 2026-11-10 03:30 · Atlas fine-tuning progress; BulkFormer-Latest embeddings started
- Last 4 blocks + head: val MAE (3k val subsample) 14.14 → 11.59 → **10.79** after epochs 0–2, still improving;
  resumed from its epoch-2 checkpoint under the patched code (evaluation-only predictions); 2 more epochs.
- Last 2 blocks + head plateaued at 13.27 (best checkpoint), now predicting evaluation samples.
- BulkFormer-Latest (full 19,973-position transcriptome in, fp16 vs fp32 cosine 1.00000, 0.4 s/sample): 2 GPU
  shards over test + val + transfer + 8k training samples per source; ETA ~05:00.

### 2026-11-10 03:35 · Gradient boosting baseline
- HistGradientBoosting (absolute-error loss, 5k HVG, early stopping): val 13.70, test 13.81, r 0.645, GTEx bin
  hit 0.358, transfer 16.5 / 0.58 → on par with ridge 5k, well behind the MLPs; not pursued further tonight.

### 2026-11-10 04:05 · Fine-tune (last 2 blocks) final
- Atlas fine-tune, last 2 blocks + head (best of 3 epochs): val 13.22, test 13.67, r 0.657, bias slope −0.49,
  GTEx bin hit 0.345, transfer 16.5 / 0.61 → about ridge level; the last-4-blocks run (val 10.79 at epoch 2)
  is the stronger one. Unfreezing more of the encoder is what helps.

### 2026-11-10 04:30 · ClockBase control
- ClockBase per-GSE API, our held-out series (530 queried): 36,616 GSMs with the two human RNA clocks
  (tAge-family, computed on ARCHS4 counts, control-centred, so relative).
- Same GSMs, val + test exact ages, series with ≥ 6 samples (19,099 samples, 280 series); Pearson r of
  within-series deviations with within-series age deviations:
  ClockBase chronological clock **0.228**, hazard clock 0.302; our healthy-trained 5-seed MLP ensemble 0.629;
  all-health-states ensemble **0.661**.
- Atlas fine-tune, last 4 blocks: epoch 3 val 12.22 (worse than 10.79 at epoch 2): early stopping will keep the
  epoch-2 checkpoint.

### 2026-11-10 04:55 · Atlas fine-tune, last 4 blocks: epoch 4
- val MAE (3k val subsample) by epoch: 14.14, 11.59, 10.79, 12.22, **10.35** → best checkpoint = epoch 4; the
  run ends here (5 epochs × 12k samples) and predicts the evaluation samples (test/val/transfer).
- Still improving at the end of its budget: a longer run over all training samples is the obvious next step.

### 2026-11-10 05:35 · Rung 4 result: Atlas fine-tuning works, and complements the MLP
| run | val MAE | test MAE | test r | bias slope | GTEx bin hit | transfer MAE / r |
|---|---|---|---|---|---|---|
| **Atlas fine-tune, last 4 blocks + head** (5 × 12k samples, healthy-or-unknown + GTEx intervals) | 10.35 | 10.85 | 0.763 | −0.40 | 0.307 | **13.8 / 0.66** |
| Atlas fine-tune, last 2 blocks + head | 13.22 | 13.67 | 0.657 | −0.49 | 0.345 | 16.5 / 0.61 |
| MLP 20k HVG, all health states, 5-model ensemble | **9.02** | 9.63 | 0.786 | −0.33 | 0.463 | 15.0 / 0.60 |
| **mean of the two above** (MLP ensemble + Atlas FT u4) | 9.10 | **9.50** | **0.814** | −0.37 | 0.378 | **13.6 / 0.67** |
| MLP 20k, all health states, seed 4 | 9.25 | 9.36 | 0.803 | −0.32 | 0.443 | 16.0 / 0.46 |
- Unfreezing more of Atlas is what makes it work (2 blocks ≈ ridge; 4 blocks ≈ the healthy MLP ensemble,
  with the best single-model transfer to pseudobulk). The fine-tuned encoder is complementary to the expression
  MLP: averaging them lifts test r from 0.786 to 0.814 and improves transfer. By the protocol (selection on
  val) the MLP ensemble remains the headline (val 9.02 vs 9.10); both are reported.
- tAge on identical samples, within-stratum relative-age r: Atlas FT 0.61 vs tAge 0.33 (ARCHS4), 0.50 vs 0.32
  (GTEx).
- Caveat: the fine-tune trained on the health labels as they were at 01:30 (before the ctrl/HC fix) and on
  60k samples in total; a longer run on all training samples is the next step.

### 2026-11-10 05:55 · Final ensembles for the morning report
| run | val MAE | test MAE | test r | GTEx bin hit | transfer MAE / r |
|---|---|---|---|---|---|
| **MLP 20k HVG, all health states, 7-model ensemble** (seeds 0–5 + Horvath) | **9.00** | 9.52 | 0.793 | 0.459 | 15.2 / 0.57 |
| same + Atlas fine-tune (last 4 blocks), mean of the two | 9.09 | **9.45** | **0.817** | 0.378 | 13.7 / 0.66 |
- Headline (selected on val): the 7-model MLP ensemble; test median AE 6.68 y, bias slope −0.33.
- Same samples, within-stratum relative-age r: ARCHS4 0.698 vs tAge 0.331; GTEx 0.770 vs tAge 0.316;
  ClockBase series (19,099 samples, 280 series) 0.661 vs ClockBase chronological 0.228 / hazard 0.302.
- BulkFormer-Latest embeddings done for 50,582 samples (all evaluation samples + 8k train per source);
  matched probes running (Latest pooled, Latest panel-pooled, 20k HVG MLP on the same samples).
- Longer Atlas fine-tune started (last 6 blocks, 20k samples/epoch, all health states), runs past 07:00.

### 2026-11-10 06:00 · Rung 3 for BulkFormer-Latest (matched samples)
Training restricted to the samples with Latest embeddings (8k ARCHS4 + 8k GTEx training samples; all
evaluation samples), so absolute numbers are worse than the full-data runs; compare within the table.

| run | val MAE | test MAE | test r | GTEx bin hit | transfer MAE / r |
|---|---|---|---|---|---|
| MLP, Latest pooled sample embedding (768-d) | 15.06 | 16.08 | 0.477 | 0.330 | 19.1 / 0.38 |
| MLP, Latest gene embeddings pooled over the panel | 15.19 | 16.08 | 0.481 | 0.306 | 21.5 / 0.34 |
| **MLP, 20k log-CPM HVG, same samples** | **12.78** | **14.02** | **0.580** | **0.352** | **18.8 / 0.49** |
- Frozen Latest, like frozen Atlas, carries less age information than the expression it is computed from.
  Across both released models, the value of BulkFormer for age shows up only when the encoder is fine-tuned.

### 2026-11-10 06:35 · Follow-up fine-tunes (after the morning report)
- Last 4 blocks, all health states (`atlas-ft-u4-all`): val MAE 12.80 → 10.81 → 10.45 → 10.60 → 9.73 → **9.56** after epochs 0–5 (final; evaluation predictions running) (healthy-only run: 14.14 → 11.59 → 10.79).
- Last 6 blocks, 20k samples/epoch, all health states (`atlas-ft-u6-all`): val MAE 12.09 → 10.24 → **9.97** after epochs 0–2 (~71 min/epoch).

### 2026-11-10 10:25 · Fine-tune on all health states (last 4 blocks) → part of the new headline
| run | val MAE | test MAE | test median AE | test r | GTEx bin hit | transfer MAE / r |
|---|---|---|---|---|---|---|
| Atlas fine-tune, last 4 blocks, all health states (6 × 12k samples) | 9.58 | 10.39 | 7.63 | 0.782 | 0.322 | **13.3 / 0.69** |
| (earlier) same, healthy-or-unknown only | 10.35 | 10.85 | 8.56 | 0.763 | 0.307 | 13.8 / 0.66 |
| MLP 20k HVG, all health states, 7-model ensemble | 9.00 | 9.52 | 6.68 | 0.793 | 0.459 | 15.2 / 0.57 |
| **mean of the two above (new headline)** | **8.72** | **9.26** | **6.56** | **0.816** | 0.384 | 13.6 / 0.67 |
- Disease samples help fine-tuning as they help the MLP. The fine-tuned Atlas has the best single-model transfer
  to pseudobulk, and combining it with the expression MLP is now the best model on validation, so a fine-tuned
  BulkFormer is part of the selected model.
- Same samples, within-stratum relative-age r: ARCHS4 0.711 vs tAge 0.331; GTEx 0.743 vs 0.316; ClockBase series
  0.692 vs 0.228 / 0.302. Fine-tune alone: ARCHS4 0.632, GTEx 0.532.
- Last 6 blocks + head (all health states) continues: val 12.09 → 10.24 → 9.97 → 9.62 → 9.86 → 9.68 (epochs 0–5; best = epoch 3; evaluation predictions running).

### 2026-11-10 11:00 · Validation-weighted blend; second fine-tune seed
- Blend of the 7-MLP ensemble and the all-health-states Atlas fine-tune with the weight chosen on val (0.65 MLP):
  val 8.68, test 9.19 (median 6.45), r 0.815, transfer 13.9 / 0.65: within noise of the plain mean (8.72 / 9.26),
  which stays the headline.
- Started a second seed of the last-4-blocks all-health-states fine-tune to measure fine-tune seed variance and to
  ensemble two fine-tunes.
- Fine-tune seed 1 (last 4 blocks, all health states): val 12.35 → 10.81 → 10.62 (epochs 0–2); seed 0 had
  12.80 → 10.81 → 10.45, so seed variance is small.

### 2026-11-10 13:30 · Last-6-blocks fine-tune; two fine-tunes + MLP = headline
| run | val MAE | test MAE | test median AE | test r | GTEx bin hit | transfer MAE / r |
|---|---|---|---|---|---|---|
| Atlas fine-tune, last 6 blocks, all health states (6 × 20k) | 9.78 | 10.28 | 7.31 | 0.784 | 0.405 | **13.0 / 0.70** |
| mean of the two all-health-states fine-tunes (u4, u6) | 9.44 | 10.10 | 7.24 | 0.794 | 0.366 | 13.1 / 0.70 |
| **7-MLP ensemble + mean of the two fine-tunes (headline)** | **8.70** | **9.20** | **6.45** | **0.817** | 0.415 | 13.5 / 0.67 |
| (previous) 7-MLP ensemble + u4 fine-tune | 8.72 | 9.26 | 6.56 | 0.816 | 0.384 | 13.6 / 0.67 |
- The last-6-blocks fine-tune is the best fine-tune on test and the best model of all on pseudobulk transfer.
  Headline: `20261110-1329-ens-mlp7-plus-atlasft2` (plain mean, no tuned weights).

### 2026-11-10 14:45 · Health-aware clocks (Anton: "can we push our model to learn health")
Evaluation (`rna-age health-eval`): held-out val + test studies, bulk, exact ages; disease labels normalised into
categories (generic labels such as "case" left out); per study with ≥ 5 cases and ≥ 5 controls, case − control
mean age gap (pred − age, minus the control residual trend); mean over studies with 95 % study-bootstrap CI.

| model (MLP, 20k log-CPM HVG) | control test MAE | aging-linked diseases (35 studies) | all diseases (44) | autoimmune/inflammatory (18) | infection (3, never in the ranking loss) |
|---|---|---|---|---|---|
| healthy-or-unknown 5-seed ensemble (existing) | 10.74 | +0.80 [−0.43, 1.97] | +1.04 [−0.15, 2.12] | +2.51 [0.83, 4.61] | +4.46 |
| age loss on controls only | 11.21 | +1.10 [−0.43, 2.50] | +1.04 | +3.03 [0.96, 5.55] | +4.12 |
| + within-study ranking loss (weight 1, margin 3 y) | 11.85 | **+2.43 [0.60, 4.24]** | +2.25 [0.68, 3.88] | +4.55 | +4.79 |
| **+ ranking loss + disease head** | **11.32** | **+2.35 [0.61, 4.17]** | **+2.34 [0.83, 3.86]** | +4.59 [1.82, 7.51] | **+5.98** |
| control-only age + disease head (no ranking) | 10.84 | +0.70 [−0.56, 1.90] | +0.83 | +2.51 | +4.17 |
- The within-study ranking loss turns the aging-linked disease gap significant on unseen studies (+2.4 y); the
  disease head keeps chronological accuracy close to the control-only model (11.32 vs 11.21). The gap also grows
  for infection, a category the ranking loss never saw. β for `bio_age = pred + β·logit` was chosen as 0 within
  the ≤ 1 y MAE budget, so the head acts as a regulariser rather than as an additive term.
- The all-health-states headline model trains on disease samples and shows smaller gaps (autoimmune +1.5).
- **Next**: ranking weight 3, seeds for the best variant; then the same loss in Atlas fine-tuning.
- Seeds / ranking weight (all with disease head; held-out studies): aging-linked gap rank 1 seed 1 **+2.62
  [0.64, 4.56]**, seed 2 +2.34 [0.49, 4.27]; rank 3 +2.64 [0.52, 5.00]; control test MAE 10.1–10.6. The effect is
  stable across seeds and costs no chronological accuracy relative to the control-only model.

### 2026-11-10 14:55 · Composition vs within-cell-type aging signal (blood, single-cell pseudobulk)
Per donor: broad cell classes from Census `cell_type`; clocks trained inside each cohort (ridge on log-CPM of 3k
HVGs chosen in-fold, 5-fold CV grouped by donor); Pearson r with age on held-out donors (`rna-age composition`).

| cohort (donors) | natural PBMC | fixed composition (own profiles, cohort-mean mix) | composition only (mean profiles, own mix) | proportions only | CD4 T alone | CD8 T alone | B | NK | monocyte |
|---|---|---|---|---|---|---|---|---|---|
| OneK1K (981) | 0.85 | 0.78 | 0.43 | 0.34 | 0.83 | 0.75 | 0.47 | 0.47 | 0.39 |
| AIDA (625) | 0.85 | **0.85** | 0.20 | 0.35 | **0.86** | 0.81 | 0.58 | 0.70 | 0.66 |
| SLE + controls (261) | 0.82 | 0.80 | 0.32 | 0.19 | 0.80 | 0.71 | 0.48 | 0.48 | 0.53 |
| COVID (196) | 0.64 | 0.67 | 0.23 | 0.26 | – | 0.72 | 0.52 | 0.48 | 0.55 |
- In blood the age signal is overwhelmingly within cell types: equalising the cell mix leaves the clock almost
  unchanged, while composition alone carries little. T cells (CD4, then CD8) carry most of it; a CD4 T pseudobulk
  alone matches whole PBMC.
- **Next**: tissues (lung, brain) and bulk: deconvolve bulk samples with these references and test whether the
  disease age gaps survive adjustment for estimated composition.

### 2026-11-10 14:55 · PRIMO preparation
- `rna-age fetch primo`: the 44 bulk datasets (6,390 samples) + public task labels at the leaderboard's data
  revision (`4b5aaaf9`), sha256-verified. `rna-age primo-embed`: frozen Atlas and our age fine-tunes (u4-all,
  u6-all) with the authors' PRIMO protocol → `data/output/primo/submission_*.parquet`.
- Local scoring with the official `primo evaluate` needs installing and running the external primo-eval code,
  which the session's safety check blocked pending Anton's approval; leaderboard submission needs Anton's HF token.

### 2026-11-10 15:20 · PRIMO, local scoring (official `primo evaluate`, primomics 0.1.7.dev0, nothing submitted)
16 public bulk tasks, all datasets validated; each task's metric (Pearson, AUROC, centred Spearman) from the
leaderboard probe; per-task scores in `data/output/summary/primo_local_scores.parquet`.

| category (tasks) | Atlas frozen | Atlas age fine-tune, last 4 blocks | last 6 blocks |
|---|---|---|---|
| clinical scores, Pearson (6) | 0.450 | 0.446 | 0.442 |
| endotype, AUROC (2) | 0.887 | 0.878 | 0.882 |
| perturbation, centred Spearman (2) | 0.336 | 0.331 | 0.323 |
| prognosis, AUROC (3) | 0.718 | 0.710 | 0.723 |
| treatment outcome, AUROC (3) | 0.589 | 0.579 | 0.587 |
- Age fine-tuning leaves the representation's usefulness for other clinical tasks unchanged: mean per-task
  difference vs frozen −0.007 (u4, better on 4/16 tasks) and −0.005 (u6, better on 8/16), all within the probe's
  fold sd. No forgetting, no transfer gain. A submission of our fine-tunes would sit next to frozen Atlas.

### 2026-11-10 15:35 · Fine-tune seed 1; tabular foundation model (TabICLv2)
| run | val MAE | test MAE | test r | transfer MAE / r |
|---|---|---|---|---|
| Atlas fine-tune, last 4 blocks, all health states, seed 0 (earlier) | 9.58 | 10.39 | – | 13.3 / 0.69 |
| same, **seed 1** | 9.97 | 11.34 | 0.745 | **11.5 / 0.76** |
| mean of three fine-tunes (u4 s0, u6, u4 s1) | 9.46 | 10.34 | 0.786 | – |
| 7-MLP ensemble + mean of three fine-tunes | 8.73 | 9.34 | 0.812 | – |
| (headline, unchanged) 7-MLP ensemble + mean of two fine-tunes | **8.70** | **9.20** | 0.817 | 13.5 / 0.67 |
- **Correction**: fine-tune seed variance is not small. Two seeds with near-equal validation MAE (9.58, 9.97)
  differ by ~1 y on test (10.39 vs 11.34); the early-epoch similarity noted at 13:30 did not hold. Seed 1 is the
  best model so far on pseudobulk transfer. Selected on validation, the two-fine-tune headline stays.

TabICLv2 (`rna-age tabfm`, regressor checkpoint 2026-02-12) as an in-context age regressor: prompt = 10k train
samples (≤ 200 per study, 1,353 studies), features = 200 PCs fitted on train rows; no tuning.

| model, features (200 PCs), 10k context | val MAE | test MAE | test r | bias slope |
|---|---|---|---|---|
| TabICLv2, log-CPM 20k HVGs | 15.18 | 15.73 | 0.512 | −0.70 |
| TabICLv2, Atlas CLS | 15.25 | 15.77 | 0.515 | −0.67 |
| ridge, same HVG PCs and same 10k rows (control) | 16.61 | 16.28 | 0.473 | −0.76 |
- TabICLv2 beats a linear readout of the same inputs, but the inputs are the limit: 200 PCs of 10k samples lose
  most of what the MLP on 20k genes and 60k+ samples uses (test 10.4).
- 30k context (1,583 studies): val 14.86, test 15.62, r 0.515: three times the context buys 0.1 y. The bottleneck
  is the 200 unsupervised PCs (ridge on them: 16.3; ridge on the 20k genes: 12.6), not the prompt size.
  Off-the-shelf tabular foundation models need a supervised or much wider gene input to compete here; GeneICL
  (20k genes, transcriptome-specific prior) is the one worth a run; the generic ones are deprioritised.

### 2026-11-10 17:45 · Do the disease age gaps survive adjustment for blood cell composition? (`rna-age deconv-gap`)
Reference: 8 broad classes from OneK1K + AIDA single-cell pseudobulk (pooled counts → CPM), 50 markers per class
(397 genes in our bulk axis); held-out bulk blood studies (val + test, ≥ 5 cases and ≥ 5 controls): fractions by
NNLS on the markers. Pooled OLS `pred − age ~ study + age + case [+ fractions]`, 95 % CI by resampling studies.

| model | cases | raw case gap (y) [95 % CI] | composition-adjusted [95 % CI] | studies / samples |
|---|---|---|---|---|
| healthy-only MLP ensemble (5 seeds) | all | 1.51 [0.60, 2.55] | 1.17 [0.23, 2.48] | 18 / 2,013 |
| headline (7 MLP + 2 Atlas fine-tunes) | all | 1.25 [0.51, 2.19] | 0.89 [0.17, 2.04] | 18 / 2,013 |
| health-aware MLP (ranking + disease head) | all | **2.46 [1.62, 3.54]** | **2.07 [1.11, 3.32]** | 18 / 2,013 |
| health-aware MLP | aging-linked | 1.95 [0.71, 2.74] | 1.36 [0.42, 2.95] | 14 / 1,064 |
- The gaps shrink by 15–30 % but stay positive with CIs above zero: the clocks' disease signal in blood is not a
  cell-mix artefact. Consistent with the single-cell result (the age signal sits within cell types).
- **Caveat**: the PBMC reference has no neutrophils, and many samples are whole blood; mean estimated fractions
  (monocyte 0.33, DC 0.21, B 0.16) are implausible, so neutrophil signal is absorbed by other classes. Next: add a
  granulocyte reference (whole-blood single-cell or sorted-cell bulk) and repeat.

GeneICL (normal inference, 4k context of raw log-CPM 20k HVGs): val 15.75, test 16.26 — no better than fast mode
(test 16.2) or TabICLv2 on PCs (15.7). Large FMs on 11 GB cards: TabFM 1.1 runs at a 2k context (4k and 10k ran out
of memory); Mitra-v2 ran out of memory even at a 2k fine-tuning cap; LimiX-2 needed huggingface_hub in its env.
- Mitra-v2 is not feasible on our 11 GB cards: with fine-tuning capped at 1,024 support rows it still asks for 9 GB at
  prediction (every `predict` re-runs the 8-fold bagged fine-tune, so the 34,784 evaluation rows cannot be chunked).
  Needs a larger GPU; not pursued (A100 nodes are reserved for other work).

### 2026-11-10 19:00 · Composition adjustment with a granulocyte reference (Tabula Sapiens blood added)
Reference now 9 classes (+ granulocyte from Tabula Sapiens blood, 149 donor × cell-type groups; a reference-only
dataset, never parsed into age samples), 441 markers. Mean estimated fractions over the 2,013 held-out blood samples
(PBMC and whole blood mixed): granulocyte 0.28, monocyte 0.16, CD4 T 0.10, NK 0.08, platelet 0.07, CD8 T 0.07, DC
0.06 (DC was 0.21 without the granulocyte class: the neutrophil signal had been absorbed).

| model | cases | raw gap [95 % CI] | composition-adjusted [95 % CI] |
|---|---|---|---|
| healthy-only MLP ensemble | all | 1.51 [0.67, 2.48] | 1.18 [0.11, 2.21] |
| headline | all | 1.25 [0.53, 2.18] | 0.91 [0.19, 1.86] |
| health-aware MLP | all | **2.46 [1.62, 3.46]** | **2.08 [0.96, 3.08]** |
| health-aware MLP | aging-linked | 1.95 [0.67, 2.78] | 1.31 [0.18, 2.86] |
- Same conclusion with neutrophils modelled: adjustment removes 15–33 % of the gap, every adjusted CI stays above
  zero. The disease age gap in blood is mostly not a cell-mix artefact.

### 2026-11-10 21:15 · Composition vs within-cell-type outside blood (brain, lung; `--mapping cell_type`)
Same design as the blood analysis, with each dataset's own cell-type labels (the 12 most abundant kept, the 5 most
abundant as the counterfactual classes); ridge clock, 5-fold CV by donor, Pearson r on held-out donors.

| cohort (donors with exact age) | natural | fixed composition | composition only | proportions | best single classes |
|---|---|---|---|---|---|
| PsychENCODE MSSM brain (783) | 0.82 | **0.82** | 0.41 | 0.42 | oligodendrocyte 0.78, astrocyte 0.76, microglia 0.73 |
| PsychENCODE brain, 0–85 y (299) | 0.92 | **0.91** | 0.46 | 0.64 | oligodendrocyte 0.91, astrocyte 0.89, microglia 0.89 |
| ILD lung atlas (95; 61 in the mixes) | 0.60 | 0.43 | 0.09 | 0.52 | ciliated 0.60 |
| CellRef lung (21) / RADC brain (100, ages 70–90+) | too few donors / too narrow an age range: no clock (r ≤ 0.53 / ≈ 0) | | | | |
- Brain shows the blood pattern: mixing each donor's own profiles at the cohort-mean composition keeps the clock,
  the composition alone carries about half of r. Glia alone (oligodendrocytes, astrocytes) predict age as well as
  whole tissue. In the 0–85 cohort composition alone is stronger (proportions r 0.64), as development shifts it.
- Lung: too few donors for the counterfactuals to settle anything.
- Null check (`--null-seed 0`, donors' ages shuffled), MSSM brain: natural r −0.01, proportions −0.01, fixed
  composition 0.01, composition only 0.00 (vs 0.82 / 0.42 / 0.82 / 0.41 on the real ages). OneK1K blood null: natural 0.01, proportions −0.02, fixed
  composition −0.08, composition only −0.04 (vs 0.85 / 0.34 / 0.78 / 0.43). The CV clock does not leak, in either tissue.

### 2026-11-11 01:40 · RM4: Terekhova 2023 PBMC by published cluster (`rna-age terekhova-celltype`)
317 tubes (visits) of 166 donors, 2,811 tube × cluster groups from the 1.9M-cell h5ad; each tube its own sample with
its age at that visit; composition analysis with the dataset's own clusters, **5-fold CV grouped by donor**.

| representation | r (CV by donor) | null (ages shuffled) |
|---|---|---|
| natural PBMC | 0.87 | −0.04 |
| fixed composition | **0.88** | −0.13 |
| composition only | 0.40 | −0.07 |
| proportions | 0.47 | −0.04 |
| CD4+ T cells alone | **0.91** | 0.01 |
| CD8+ T cells (TRAV1-2−) alone | 0.82 | 0.02 |
| myeloid / NK / B alone | 0.63 / 0.62 / 0.40 | ≤ 0 |
- Same as the four Census blood cohorts: within-cell-type signal, CD4 T cells alone at least as good as whole PBMC.
- **Gate caught a leak first**: the first run folded by tube, so one donor's visits sat on both sides (natural 0.93,
  fixed 0.95, CD4 T 0.95, null slightly negative); the code now folds by person when a cohort carries a `cv_group`
  (83958f2). Only the donor-grouped numbers above count.

RM1 progress: health-aware fine-tune val 12.59 → 11.72 → 11.04 → 10.65 → 10.82 (epoch 4); last epoch running.

### 2026-11-11 04:05 · RM1: health-aware Atlas fine-tune (within-study ranking loss)
`rna-age finetune --unfreeze 4 --epochs 6 --max-train 20000 --rank-weight 1` (healthy-or-unknown age loss; ranking
loss on 64,520 case/control pairs from 80 training studies, aging-linked categories, 2 pairs per step); jobs
6427392 + 6427754 (resumed at epoch 2 after the 4 h limit); val 12.59 → 11.72 → 11.04 → **10.65** → 10.82 → 10.84,
best checkpoint epoch 3. Run `20261110-1516-atlas-ft-u4-rank1`.

| model | val MAE | test MAE (r) | pseudobulk transfer MAE (r) |
|---|---|---|---|
| Atlas fine-tune, last 4 blocks, plain (all health states) | 9.58 | 10.39 (0.78) | 13.3 (0.69) |
| **same + ranking loss** | 10.74 | 10.52 (0.77) | 17.7 (0.67) |

Within-study case − control age gap, held-out studies (95 % study-bootstrap CI):

| category (studies) | health-aware fine-tune | health-aware MLP (rank 1 + head, s0) | headline (no health loss) |
|---|---|---|---|
| aging-linked (35) | **+2.93 [1.17, 4.67]** | +2.35 [0.64, 4.25] | +0.75 [−0.38, 1.96] |
| all categorised (44) | +2.92 [1.45, 4.37] | +2.34 [0.82, 3.92] | +0.98 [−0.01, 2.04] |
| neurodegenerative (9) | **+3.84 [0.66, 7.47]** | +0.99 [−1.39, 3.89] | +1.98 [0.09, 4.04] |
| autoimmune / inflammatory (18) | +4.39 [1.24, 7.70] | +4.59 [1.83, 7.40] | +1.52 [−0.39, 3.86] |
- The ranking loss gives the fine-tuned encoder a larger disease gap than the MLP, and the first significant
  neurodegenerative gap (9 studies: wide CI). Cost: ~1 y validation MAE and pseudobulk transfer (13.3 → 17.7); test
  MAE nearly unchanged. One seed only (fine-tune seed spread ~1 y on test).
- Gate: no null check yet on the gap itself (e.g. case labels permuted within study); needed before this is a
  headline. Conclusions and what trains next are the lead's.
- Null check (lead, 054db35: `health-eval --permutations 200`, case/control labels shuffled within each held-out
  study): null mean of the gap −0.02 to 0.02 in every category; the health-aware fine-tune's observed gap is above
  all 200 shuffles (p = 1/201) for aging-linked, all, autoimmune and neurodegenerative. Health-aware MLP (rank 1 +
  head, s0): p = 1/201 except neurodegenerative (p 0.20). Control-only MLP (rank 0 + head): aging-linked p 0.10,
  all 0.06, autoimmune 1/201, neurodegenerative 0.55. Summaries `health_*_pred_null.parquet`.

### 2026-11-11 04:30 · RM1 step 1: the health ensemble
`rna-age ensemble` (plain mean) of the three health-aware MLPs (rank 1 + head, seeds 0-2) and the health-aware Atlas
fine-tune → run `20261111-0406-ens-health3-plus-atlasft-rank1`; `health-eval --permutations 200` (job 6428619).
- Accuracy: val MAE 9.81, test MAE 10.10 (control test MAE 9.85).
- Within-study gap (held-out studies): aging-linked +2.56 y [0.71, 4.26], all +2.55 [1.05, 4.03], autoimmune +4.76
  [1.85, 7.77] (each above all 200 shuffles, p = 1/201); neurodegenerative +1.93 [−0.60, 4.86] (p 0.02).
- Against the fine-tune alone (+2.93 aging-linked, neurodegenerative +3.84 [0.66, 7.47]): the ensemble buys ~0.4 y of
  test MAE and loses the neurodegenerative signal, which only the fine-tune carries.

### 2026-11-11 07:00 · RM10 first biology battery and RM11 robustness preview (run by the lead, logged here)
**RM10** (registry 99f56bc committed before the run; code a407f4b; `data/output/summary/biology_tests.parquet`):
does the age gap (age regressed out within study) rise with disease severity? 12 directional tests + 1 negative
control, 2,000 within-study permutations, four models.

| test (study) | headline (no disease labels) | health ensemble | health fine-tune | control-only MLP |
|---|---|---|---|---|
| Braak stage, AD fusiform gyrus (GSE95587) | ρ 0.28, p 0.001 | **ρ 0.56, p 0.0005** | **ρ 0.56, p 0.0005** | ρ 0.24, p 0.007 |
| liver fibrosis stage, cohort A (GSE162694) | ρ 0.21, p 0.006 | ρ 0.38, p 0.0005 | – | ρ 0.30, p 0.0005 |
| liver fibrosis stage, cohort B (GSE174478) | ρ 0.22, p 0.013 | ρ 0.24, p 0.013 | – | n.s. |
| Lewy stage, PD cortex (GSE216281) | ρ 0.20, p 0.03 | – | ρ 0.27, p 0.008 | ρ 0.23, p 0.017 |
| COVID / acute viral severity, RA activity, MMSE, NYHA, LVAD unloading | n.s. or opposite | n.s. or opposite | n.s. or opposite | n.s. or opposite |
| negative control: one exercise bout (GSE202295) | nothing | nothing | nothing | nothing |
- Every model, the headline included (never trained on disease), puts higher fibrosis, Braak and Lewy stages older;
  the health-aware models most strongly. Blood-based acute severity, cognition and heart failure class: nothing or
  the opposite. Braak and fibrosis cohort A survive Bonferroni over the 12 tests. ("–" = row not shown in the
  lead's summary; all 52 rows in the parquet.)

**RM11 preview** (robustness of the 3-fine-tune headline `20261110-1528-ens-mlp7-plus-atlasft3`, before RM12's
re-merge; `robustness_*.parquet`): test MAE 9.34 over 232 studies; by sequencer, HiSeq 2000/2500 5.9-6.7 y,
NovaSeq 6000 10.2, NextSeq 500 15.5 (1,692 samples, 33 studies; flagged as an unsupported group), MiSeq 22.4
(46 samples). To be redone on the RM12 table.

### 2026-11-11 06:58 · RM12: re-parse ARCHS4 and re-merge (D4: meningioma as tumour; plate single-cell series out)
Job 6429048 (`rna-age parse archs4 && rna-age merge`, code 98f1c5e/2a5a05b, 2.5 min). Checked on the merged table:
GSE116672 (single-cell plates, 1,161 test "samples" from 10 donors) 0 eligible; the 77 samples labelled
meningioma are `is_tumor` and none eligible.

| split | ARCHS4 eligible before → after | all sources after |
|---|---|---|
| train | 51,450 → 47,493 | 63,735 |
| val | 13,312 → 12,793 | 17,894 |
| test | 10,407 → 9,097 | 12,144 |
| (no split: Bgee) | – | 753 |
- 100,312 → 94,526 eligible (−5,786, all ARCHS4; GTEx, Bgee, Census pseudobulk and Terekhova unchanged).
- Jobs started before 06:53 (the KronLSTM sweep, RM1 seed 1) trained on the old table; RM1 seed 2 and every later
  run on the new one. Evaluations read only currently eligible samples.

### 2026-11-11 07:15 · RM8 context sweep: 512 tokens done (`20261111-0539-kron-sweep-rank512-e8`)
KronLSTM, rank encoding, 512 tokens, 8 epochs, batch 32, seed 0 (job 6428862, code 937866e, started 05:35 on the
pre-RM12 table): val MAE per epoch 16.48, 14.98, 14.94, 14.74, 14.69, 14.45, 14.42, **14.41**; test 13.76 (r 0.59).
~11.6 min per epoch alone on a card (the 21.5 min of the smoke included Triton compilation and the shared-card time).
Still improving slowly at epoch 7; far from the 20k-HVG MLP (val ~9.2-10.0). 2,048 (micro 16) and 4,096 (micro 8)
running.

### 2026-11-11 07:25 · RM12 refresh: 20k-HVG MLP (all health states), seeds 0-2 on the new table (job 6429121)
Runs `20261111-0707/0717/0724-hvg20000-mlp-trainall[-seed1|-seed2]`. As each run reports: val 8.76 / 8.70 / 8.73,
test 8.03 / 8.24 / 8.14 (r 0.85-0.86), pseudobulk transfer 18.5-20.4. **Not comparable with the old runs' reported
numbers**: the test and val sets themselves changed (RM12 removed 1,161 plate "samples" of one series from test).
Old and new seeds scored on the same, currently eligible exact-age bulk samples (val n 12,864, test n 8,843):

| 20k-HVG MLP, seeds 0-2 | val MAE (mean) | test MAE (mean) |
|---|---|---|
| old (trained on the pre-RM12 table) | 9.21 / 8.90 / 8.94 (9.02) | 8.48 / 7.85 / 7.94 (8.09) |
| new (RM12 table) | 8.76 / 8.70 / 8.73 (**8.73**) | 8.03 / 8.24 / 8.14 (8.14) |
- Cleaner training gains ~0.3 y on validation, nothing on test; most of the drop from ~9.5 to ~8.1 test MAE is the
  test set losing hard plate-single-cell samples. Pseudobulk transfer worsens (old 15-16.5 → new 18.5-20.4): the
  removed plate series may have helped carry the model to single-cell pseudobulk. Every comparison from here reads
  currently eligible samples only.
- Lead's reading (2026-11-11 07:28): earlier headline figures (test 9.20, on the pre-RM12 sample set) are not
  comparable with figures on the RM12 set; every number is quoted with its sample set from here. The transfer drop is
  plausible (the plate series were the nearest thing to pseudobulk in training); pseudobulk stays a separate domain
  by the rules, so it is an open question, not a reason to put single cells back.

### 2026-11-11 07:48 · RM12 refresh: headline and health ensembles rebuilt from the new MLPs
Health-aware MLPs (rank 1 + head) seeds 0-2 on the RM12 table: job 6429122, runs `20261111-0718/0731/0743-health-mlp-rank1-head-s0..s2`.
Rebuilt (plain means; the fine-tunes are still the old-table ones until RM1's new seeds land):
`20261111-0747-ens-mlp3t2-plus-atlasft2` = mean(3 new 20k-HVG MLPs) + mean(2 Atlas fine-tunes, u4 and u6), and
`20261111-0747-ens-health3t2-plus-atlasft-rank1` = 3 new health MLPs + the health-aware fine-tune.
All scored on the **RM12 sample set** (currently eligible exact-age bulk: val n 12,864, test n 8,843):

| ensemble | val MAE (r) | test MAE (median, r) |
|---|---|---|
| old headline (7 old MLPs + 2 fine-tunes) | 8.44 (0.846) | 7.43 (5.41, 0.884) |
| **new headline (3 new MLPs + 2 fine-tunes)** | **8.36 (0.849)** | **7.46 (5.31, 0.882)** |
| old health ensemble | 9.57 (0.808) | 9.01 (6.71, 0.835) |
| new health ensemble | 9.62 (0.803) | 9.21 (6.85, 0.825) |
- On the RM12 set the headline is test MAE 7.46 (r 0.88); not comparable with the 9.20 of the pre-RM12 set. Three
  clean-table MLPs match seven old ones. New health ensemble's gap eval with the 200-shuffle null: job 6429191.

### 2026-11-11 07:50 · Clean-table (RM12) re-runs: RM10 biology, RM11 robustness, the new health ensemble's gap
Runs: headline `20261111-0747-ens-mlp3t2-plus-atlasft2`, health ensemble `20261111-0747-ens-health3t2-plus-atlasft-rank1`.
`biology_tests.parquet` and `robustness_*` are overwritten by these runs (earlier versions: the pre-RM12 runs in
the 07:00 entry). Lead's run, commits 18497d9, f0a79ec.

**RM10 biology** (age gap vs severity within study, 2,000 permutations; ρ, p):

| test | headline | health ensemble |
|---|---|---|
| liver fibrosis A / B | 0.25, 0.003 / 0.25, 0.007 | **0.42, 0.0005** / 0.29, 0.005 |
| Braak stage (AD) | 0.27, 0.003 | **0.52, 0.0005** |
| Lewy stage (PD) | 0.18, 0.049 | 0.25, 0.012 |
| NYHA class (heart failure) | −0.07 (opposite) | 0.19, 0.027 |
| COVID / viral severity, RA activity, MMSE, LVAD unloading, NAFLD activity | n.s. or opposite | n.s. or opposite |
| **negative control**: one exercise bout | 0.09, 0.36 (passes) | **0.21, 0.034 (fails)** |
- Stage effects replicate on the clean table for both ensembles; the health ensemble also fails the negative
  control, so part of its signal tracks acute physiological state, not only disease stage.

**RM11 robustness** of the headline on the RM12 set (test, 8,843 samples, 212 studies): MAE 7.46 (r 0.88); the
earlier NextSeq flag and blood bias were the single-cell series and are gone. Best tissues: gut 2.9, skeletal muscle
5.2, PBMC 6.4, blood 6.5, liver 6.9; worst: bone marrow 13.4, oral 13.0, plasma cfRNA 12.3 (r −0.23, 28 samples);
flagged unsupported: HiSeq X Ten (66 samples), bone marrow, "other".

**Health ensemble gap** (job 6429191, 200 shuffles, RM12 set): aging-linked +2.81 y [0.80, 4.88] (33 studies), all
+2.55 [0.85, 4.33], autoimmune +4.89 [1.90, 8.11], neurodegenerative +2.33 [−0.89, 5.71] (7 studies); each above all
200 shuffles.

### 2026-11-11 09:22 · RM10 step 2: the registered before/after tests (bba614a), run once
Job 6429223 (`slurm/rm10_paired.sh`, 29d886e): `rna-age external` of GSE65540, GSE274137, GSE192587, GSE66306,
GSE137631, GSE107894 (285 samples, never in the age table or training); retrained on the RM12 table with
`--external`: plain 20k-HVG MLPs `20261111-0821/0830/0837-hvg20000-mlp-trainall[-seed1|-seed2]` and health MLPs
`20261111-0854/0908/0918-health-mlp-rank1-head-s0..s2`; `biology-paired` (5,000 permutations, three seeds
averaged). The external predictions were not opened before these tests. Expected direction for every test: the
gap falls after the intervention (one-sided p).

| test (study, pairs) | plain: Δ gap y (p) | health: Δ gap y (p) |
|---|---|---|
| bariatric surgery, adipose (GSE65540, 22) | +0.73 (0.86) | +0.67 (0.81) |
| weight loss, abdominal SAT (GSE274137, 8) | −0.03 (0.49) | −0.56 (0.41) |
| weight loss, subcutaneous fat (GSE274137, 8) | +4.08 (0.98) | +4.82 (0.98) |
| calorie restriction, adipose (GSE192587, 10) | +1.03 (0.72) | +1.61 (0.75) |
| protein restriction, adipose (GSE192587, 9) | −1.12 (0.28) | −1.46 (0.26) |
| bariatric surgery, monocytes (GSE66306, 19) | −0.34 (0.35) | +0.88 (0.83) |
| gastric bypass, skeletal muscle (GSE137631, 24) | +3.66 (0.95) | +4.29 (0.99) |
| metformin, muscle (GSE107894, 13) | +0.82 (0.65) | +2.00 (0.82) |
| metformin, adipose (GSE107894, 14) | +2.06 (0.78) | +2.87 (0.83) |
- None of the nine registered tests passes for either model; seven of nine move the opposite way for the health
  models, six for the plain ones (bypass muscle and fat loss by ~+4 y). Results in
  `biology_paired_plain.parquet`, `biology_paired_health.parquet`. What it means is the lead's (step 3).

### 2026-11-11 12:29 · RM8 context sweep ends: 2,048 done, 4,096 stopped early (Anton's word)
All rank encoding, 8 epochs planned, batch 32, seed 0, pre-RM12 table; validation MAE by epoch:

| length | curve | result |
|---|---|---|
| 512 | 16.48, 14.98, 14.94, 14.74, 14.69, 14.45, 14.42, 14.41 | val 14.41, test 13.76 (r 0.59) |
| 2,048 (micro 16) | 15.71, 14.49, 14.87, 14.90, 14.12, 14.04, 13.95, 13.95 | val 13.95 (`20261111-0613-kron-sweep-rank2048-e8-m16`) |
| 4,096 (micro 8) | 15.86, 14.84, 14.82, 14.52 | **stopped at 12:29 after epoch 3** (job 6429024) |
- Stopped on Anton's instruction (#rna-age, "do we have early stop?", and earlier "if something is super bad in
  metrics and not promising just early stop"): 4,096 tracked 512 and 2,048 within ~0.5 y while all three sit
  ~5 y behind the 20k-HVG MLP (val ~8.7); ~6 GPU-hours left. Context length buys ~0.5 y at most.
- The lead's rule (stop when > 2 y worse than the rank-MLP at the same length for 2 epochs) is applied with the
  ablation (job 6429586, running); the best length for the full-length seeds is 2,048.
- 12:45: the rank-MLP ablation (RM8 step 4) cancelled on Anton's word ("do not resume ablation, we have more
  important stuff to do than numerous ablation experiments", relayed by the lead). Its first try (6429586) ran out
  of GPU memory at the MLP step (the dense rank matrix, 15.6 GB, on an 11 GB card); the CPU retry (6429643) was
  cancelled before a result. No new KronLSTM run until Anton and the lead settle the next step.

### 2026-11-11 18:52 · RM1 seed 2 on the RM12 table done
`20261111-0717-atlas-ft-u4-rank1-s2` (job 6428869, same flags as seed 0, `--seed 2`): val by epoch 11.95, 11.49, …,
10.94, 10.80, 9.84 (best, epoch 5). On the RM12 sample set: val MAE 10.07, test 10.20 (r 0.79).
Within-study gap, held-out studies: aging-linked **+2.85 y [1.20, 4.66]** (33 studies), all +2.67 [1.29, 4.26],
autoimmune +4.40 [1.40, 7.68], neurodegenerative +4.65 [1.55, 7.86] (7 studies). Seed 0 (old table) gave +2.93
[1.17, 4.67]. Seed 1 finishing; seed 0 on the RM12 table ends ~00:35; then the three-seed gap with the null.
- 18:57 · seed 1 on the RM12 table done: `20261111-0659-atlas-ft-u4-rank1-s1-t2` (job 6429117), best val 10.22 (epoch 5);
  on the RM12 set val MAE 10.38, test 11.12 (r 0.75). Gap: aging-linked **+2.88 y [1.10, 4.79]**, all +2.83 [1.36,
  4.45], autoimmune +4.36 [1.27, 7.69], neurodegenerative +4.82 [1.59, 8.04]. Accuracy differs by ~0.9 y between seeds
  1 and 2; the aging-linked gap does not (+2.88 vs +2.85).

### 2026-11-12 00:24 · RM1 done: seed 0 on the RM12 table, the three-seed gap with its null, the health ensemble rebuilt
`20261111-1233-atlas-ft-u4-rank1-s0-t2` (job 6429118, 11 h 47 min, same flags as seeds 1-2, `--seed 0`): val by
epoch 12.57, 11.12, 11.09, 10.54 (best, epoch 3), 10.79, 10.83. On the RM12 set val MAE 10.71, test 11.47 (r 0.75).
Gap: aging-linked **+3.09 y [1.26, 4.95]**, all +3.15 [1.60, 4.77], autoimmune +4.97 [1.91, 8.15], neurodegenerative
+4.56 [1.13, 8.02].
Built at 909fbda (plain means, `rna-age ensemble`; null by `health-eval --permutations 200`, labels shuffled within study):

| run (RM12 set) | val MAE | test MAE (median, r) | aging-linked gap [95 % CI] | null mean, shuffles ≥ gap |
|---|---|---|---|---|
| `20261112-0022-ens-atlasft-rank1-3s` (mean of seeds 0-2) | 10.13 | 10.62 (8.10, 0.78) | +2.94 [1.24, 4.74] | 0.02, 0 of 200 |
| `20261112-0022-ens-health3t2-plus-atlasft-rank1-3s` (3 health MLPs + that mean) | 9.63 | 9.53 (7.15, 0.81) | +2.79 [0.88, 4.88] | 0.02, 0 of 200 |
| `20261111-0747-ens-health3t2-plus-atlasft-rank1` (earlier: old-table fine-tune) | 9.62 | 9.21 (6.85, 0.83) | +2.81 [0.80, 4.88] | 0.01, 0 of 200 |

- Three-seed fine-tune mean, other categories: neurodegenerative +4.68 [1.46, 7.96] (7 studies), autoimmune +4.58
  [1.68, 7.67] (18), infection +4.77 (3), psychiatric +3.15 [0.26, 6.63] (7); cardiovascular −1.69 (4, n.s.). In the
  health ensemble neurodegenerative falls to +2.28 [−0.83, 5.43] and psychiatric to +0.77: the MLPs dilute what the
  fine-tune sees in brain.
- Seeds: test MAE 11.47 / 11.12 / 10.20, aging-linked gap +3.09 / +2.88 / +2.85. The gap is the stable part; the new
  health ensemble keeps it with a 0.3 y worse test MAE than the earlier one. The headline (plain) ensemble is
  unchanged: it uses the plain fine-tunes, not these.

### 2026-11-12 02:10 · RM14 (D7): tabular foundation models as single-cell aging clocks, runs submitted
`uv run --with tabicl rna-age sc-tabular <cohort>` at 2b73ff0 (the lead's code, d998476): per donor, log-CPM, 5-fold CV
grouped by person, the fold's 500 most variable genes; ridge vs MLP vs TabICLv2; whole pseudobulk and each cell class
alone. Two chains, one GPU each: blood (broad classes) terekhova2023 6430967 → 3faad104 6430968 → c838aec3 6430969 →
218acb0f 6430970; brain (`--mapping cell_type`) 37a17b78 6430972 → 5c97eeeb 6430973 → 5e57cd50 6430974. Bar (D7):
TabICLv2 beats ridge with a donor-bootstrap CI excluding 0 in at least two cohorts.

### 2026-11-12 02:30 · RM14 results: TabICLv2 vs ridge vs MLP, seven single-cell cohorts
All seven jobs completed (3-7 min each). Per-row tables (every representation × model, MAE, r, MAE minus ridge with a
donor-bootstrap 95 % CI): `data/output/summary/sc_tabular_compare.parquet`; out-of-fold predictions
`sc_tabular_<cohort>.parquet`. Whole-sample (natural) pseudobulk, MAE in years:

| cohort | donors | ridge | MLP | TabICLv2 | TabICLv2 − ridge [95 % CI] |
|---|---|---|---|---|---|
| 3faad104 (blood) | 981 | 8.23 | 11.18 | 7.91 | −0.32 [−0.56, −0.07] |
| c838aec3 (blood) | 625 | 6.97 | 8.40 | 6.45 | −0.53 [−0.84, −0.21] |
| 218acb0f (blood) | 261 | 8.01 | 12.05 | 7.55 | −0.46 [−0.91, −0.01] |
| terekhova2023 (blood, tubes) | 317 (166 people) | 9.25 | 11.24 | 8.63 | −0.62 [−1.23, 0.02]; by person [−1.35, 0.09] |
| 37a17b78 (brain) | 783 | 6.60 | 9.53 | 6.65 | +0.05 [−0.24, 0.33] |
| 5c97eeeb (brain) | 299 | 7.16 | 9.64 | 6.35 | −0.81 [−1.36, −0.27] |
| 5e57cd50 (brain) | 100 | 6.16 | 24.58 | 6.00 | −0.16 [−0.69, 0.35] |

- Whole sample: TabICLv2 beats ridge with the CI below 0 in 4 of 7 cohorts, never loses. Per cell class: it wins
  38 of 68 class/whole rows and loses 3, all in the smallest cohort (5e57cd50, 100 donors: VIP, lamp5 interneurons,
  oligodendrocytes, +0.7 y). Gains are 0.3-0.8 y on the whole sample, up to 1.7-1.8 y on single classes (Terekhova
  CD4 T 6.23 vs 7.93, by-person CI [−2.39, −1.02]; 5c97eeeb L2/3-6 IT neurons 5.36 vs 7.02, endothelial −1.82).
- The small MLP is worse than ridge everywhere (24.6 on 100 donors): it is no bar to beat.
- Gate notes: one TabICLv2 seed; the per-class rows are many comparisons (read the whole-sample row per cohort as the
  test); the row bootstrap resamples tubes in Terekhova (the by-person CI is given). D7's bar (beats ridge, CI
  excluding 0, ≥ 2 cohorts) is met on the whole sample; what it means is the lead's.

### 2026-11-12 12:10 · RM16: stronger baselines and a 2,000-gene TabICLv2 arm, submitted
Anton: ridge is a low bar. At 56541c1 (the lead's code, 0969bcf), per RM14 cohort, same `--mapping`: CPU
`rna-age sc-tabular <c> --models enet,gbm` (3 cores), then GPU `uv run --with tabicl rna-age sc-tabular <c> --models
tabicl --n-genes 2000` (after its CPU job; model saved as `tabicl-2000g`), new models merged into
`sc_tabular_<c>.parquet`. Jobs (cpu → gpu): terekhova 6432543 → 6432544, 3faad104 6432545 → 6432546, c838aec3
6432547 → 6432548, 218acb0f 6432549 → 6432550, 37a17b78 6432551 → 6432552, 5c97eeeb 6432553 → 6432554, 5e57cd50
6432555 → 6432556; GPU jobs in two chains (blood, brain). Verdict against the best of ridge, enet and gbm per cohort.

### 2026-11-12 14:15 · RM16 results: TabICLv2 against the best of ridge, elastic net and boosted trees
All 14 jobs completed (CPU baselines 2-54 min, GPU 2,000-gene arm). `sc_tabular_compare.parquet` now holds every
comparison against each of ridge, enet and gbm (column `ref`). Whole-sample pseudobulk, MAE in years:

| cohort | ridge | enet | gbm | TabICLv2 500 genes | TabICLv2 2,000 genes | 2,000 genes − best baseline [95 % CI] |
|---|---|---|---|---|---|---|
| 3faad104 (blood, 981) | 8.23 | 8.16 | 8.41 | 7.91 | 7.37 | −0.78 [−1.11, −0.46] vs enet |
| c838aec3 (blood, 625) | 6.97 | 7.04 | 7.51 | 6.45 | 5.77 | −1.21 [−1.55, −0.87] vs ridge |
| 218acb0f (blood, 261) | 8.01 | 7.44 | 8.83 | 7.55 | 6.88 | −0.56 [−1.14, 0.04] vs enet, n.s. |
| terekhova2023 (blood, 166 people) | 9.25 | 8.92 | 8.95 | 8.63 | 6.92 | −2.00 [−2.88, −1.09] vs enet, by person |
| 37a17b78 (brain, 783) | 6.60 | 6.70 | 7.88 | 6.65 | 7.40 | +0.79 [0.42, 1.16] vs ridge: worse |
| 5c97eeeb (brain, 299) | 7.16 | 7.12 | 8.23 | 6.35 | 5.81 | −1.31 [−1.87, −0.76] vs enet |
| 5e57cd50 (brain, 100) | 6.16 | 6.17 | 6.12 | 6.00 | 6.19 | +0.07 [−0.48, 0.59] vs gbm, n.s. |

- Against the best baseline per cohort, the 2,000-gene TabICLv2 wins in 4 of 7 cohorts (3 blood, 1 brain) by 0.8-2.0 y
  and loses in 1 (37a17b78, the largest brain cohort, where 500 genes ties ridge). The 500-gene arm wins in 3 of 7
  (3faad104 −0.25, c838aec3 −0.53, 5c97eeeb −0.77). Elastic net and boosted trees rarely beat ridge (best baseline:
  ridge in 35 of 68 rows, enet 24, gbm 9).
- Per row (whole sample and each class), against that row's best baseline: 500 genes wins 34, loses 3 of 68;
  2,000 genes wins 26, loses 1.
- One TabICLv2 seed per arm; two arms tried, so the per-cohort best of the two is optimistic. What it means: the lead.

### 2026-11-12 14:56 · RM17: CD4 T-cell Perturb-seq pseudobulk fetch submitted
`rna-age fetch cd4_perturbseq --connections 8` (the lead's fetcher, f617256): 44.6 GB pseudobulk from a public S3
bucket plus three small files, S3-ETag verified, into `data/download/cd4_perturbseq/`. CPU job 6434376, 2 cores. Scoring queued after it: `rna-age perturb-score` (9a158aa, ridge CD4 T clock from 3faad104, 2,000 genes), CPU job 6434582, 4 cores, 100 GB; the hit list is not read before the lead's D8 check.
- 15:12 · fetch done (8 min 26 s): GWCD4i.pseudobulk_merged.h5ad 44,566,657,140 B (S3 ETag verified) and three small files (md5 verified), sizes match their sidecars. Scoring 6434582 waited on the account's CPU cap (AssocGrpCpuLimit), then ran.
- 16:07 · perturb-score 6434582 done (4 min 51 s, no errors): perturb_cd4_profiles.parquet (202,110 rows) and perturb_cd4_genes.parquet (35,282 rows). Not opened beyond row counts: D8's check is the lead's first.
- 16:18 · `rna-age perturb-check` (D8, 807dac6) run on the login node in seconds: `perturb_cd4_check.parquet`; committed with the two scoring outputs.

### 2026-11-12 16:30 · RM18 step 2b: per-donor metadata of the lupus dataset (218acb0f, Perez et al. 2022)
New `rna-age census-donor-obs <dataset_id>` (`census_pseudobulk.donor_obs`): reads only the obs of the fetched h5ad
and keeps every column constant within a donor and varying between donors, one row per donor, to
`data/input/census_pseudobulk/donor_obs/<dataset_id>.parquet`. CPU job 6435226.
- 16:38 · job 6435226 done (17 s): 261 donors. Constant per donor: `donor_id`, `disease` (162 SLE, 99 normal), `ind_cov`,
  `self_reported_ethnicity` (149 European American, 107 Asian, 3 African American, 2 Hispanic), `sex` (244 female),
  and their ontology ids. `Processing_Cohort` (4 levels) varies within 57 donors, so it is not in the per-donor table:
  added `<id>.mixed.parquet` (donor × column × value × n_cells for columns of ≤ 20 levels that vary within a donor;
  also lists cell-type columns and `disease_state`), rerun on the login node (seconds).

### 2026-11-12 17:34 · RM20-RM23 started (Anton: "test more ideas"; the lead's numbering and bars, ca7429d)
Four arms built in parallel, each in its own worktree by an agent of the runner seat, one GPU job each at most, merged
to main after the lead's review: RM20 multitask (age + knockdown Δ, DEG-weighted, rank term, placebo), RM21 unseen
knockdowns from ESM-2 gene embeddings (TabICL, ridge, mean, shuffled-embedding ridge), RM22 frozen virtual-cell
embeddings as clock features (Tahoe-x1 / Geneformer / scGPT, the seven RM14 cohorts, D7 folds), RM23 MaxToki in-silico
screen of CD4 T cells against the Marson aging effects and our RM17 shifts. Bars as the lead wrote them in the roadmap.

### 2026-11-12 18:05 · RM21 result: a knockdown's age effect is barely predictable from the gene's protein embedding
Branch `worktree-agent-a06ba6dbd4c0fcc51` (3487491 build, ea8cdd9 result; tests pass), awaiting the lead's review.
Jobs 6435514 (ESM-2 650M embeddings of 11,993 target genes with a reviewed UniProt protein, 9 min) and 6435705 (5-fold
CV over genes, 11 min). Targets: our clock's per-gene shift and the study's aging coefficients, per condition. Bar
(beat the mean and a shuffled-embedding ridge, lower CIs above 0) met in 2 of 12 model × set tests, both for our
clock's Stim48hr shift, and tiny: Spearman 0.03 (ridge) and 0.05 (TabICLv2), skill over the mean ≤ 0.005. Not met for
Rest, Stim8hr, or any study coefficient; label-shuffle nulls |ρ| ≤ 0.04. Twelve tests, no correction: read as no
usable signal. Not done: Δ-profile principal components (target c). Result: `data/output/summary/unseen_knockdown.parquet`.

### 2026-11-12 18:20 · RM20 result: the multitask network learns no knockdown effects; a rank term gives a hint worth seeds
Branch `worktree-agent-afeeb4c691cef5fce` (fdf1310 code, 8050fc1 result), awaiting the lead's review. Job 6436115
(9 min). 981 donors (3faad104 CD4 T, 5-fold CV), 123,896 knockdown units, 2,386 of 11,929 genes and one context
(one donor's Stim48hr) held out; gene embedding from the target's coexpression with the 2,000-gene panel.
- Bar 1, Δ on held-out genes: **fail**. DEG recall@50 0.052 vs per-context mean 0.085 and donor+condition ridge
  0.091; Pearson-Δ ≈ 0 even for the baselines (single knockdown profiles are mostly noise).
- Bar 2, age MAE within 0.2 y of age-only: pass for age+Δ (10.64 vs 10.67). The rank arm is 8.77, but the placebo is
  8.68: the gain is regularisation, not aging information; ridge on the same panel 8.17 beats every network.
- Bar 3, held-out ρ > 0.26: **fail** for the predicted shift (−0.03). On measured profiles the rank arm's clock gives
  0.30 (0.31 / 0.32 / 0.28) vs 0.25 for the RM17 ridge clock and 0.21 for age-only, on the same genes.
- Bar 4, placebo fails: mixed. Measured-profile ρ placebo 0.23 vs rank 0.30 (+0.07, one seed, no CI).
One seed, one split, no CIs. The +0.07 needs repeat seeds before it counts.

### 2026-11-12 18:43 · RM20 merged (1ee5197, cli.py conflict with RM21 resolved by keeping both); RM24 and RM25 submitted
- RM24 (the lead's bar, fixed before the run): 5 seeds of RM20's rank arm and its placebo, same seed → same split and
  panel within a pair. Jobs 6436762, 6436763, 6436765, 6436766, 6436767 chained on one GPU,
  `rna-age rm20-multitask --arms age_delta_rank,placebo --seed S --out data/output/summary/rm24_seedS.parquet`.
- RM25: `uv run --with tabicl rna-age tabfm --model tabicl --features hvg --n-genes 2000 --context nearest-study
  --n-context 4000 --cap-per-study 200` (the lead's 46635e3), job 6436768. Bar: within 1 y of the 20k-HVG MLP (8.1
  test) earns fine-tuning; beating the headline 7.46 beyond the paired CI is a finding.
- 18:47 · RM24 seed 0 (6436762) failed in 1 s: it started at the moment the shared checkout was mid-rebase, and imported a cli.py with conflict markers (SyntaxError). The cause is gone (markers absent, rebase aborted); rerun as 6436814 at the end of the chain. Jobs import code from the shared working tree at start, so a rebase or merge in progress breaks any job starting then. RM25 started the same second and loaded fine.

### 2026-11-12 19:28 · RM24 result: the rank term passes both parts of the lead's bar, by small margins
Five seed pairs (jobs 6436763, 6436765, 6436766, 6436767, 6436814; seed 0 reproduces RM20's numbers exactly).
Measured-profile ρ with the study's aging effects on held-out genes (mean of the three conditions):

| seed | rank arm | placebo | RM17 ridge clock | age MAE rank / placebo / ridge |
|---|---|---|---|---|
| 0 | 0.303 | 0.233 | 0.250 | 8.78 / 8.68 / 8.17 |
| 1 | 0.291 | 0.216 | 0.259 | 8.73 / 8.78 / 8.17 |
| 2 | 0.266 | 0.216 | 0.265 | 8.66 / 8.63 / 8.17 |
| 3 | 0.335 | 0.269 | 0.281 | 8.77 / 8.65 / 8.17 |
| 4 | 0.254 | 0.237 | 0.250 | 8.69 / 8.61 / 8.17 |

- (a) rank − placebo: +0.056, paired bootstrap 95 % CI [0.035, 0.071], positive in 5 of 5: **passes**.
- (b) rank − RM17 ridge: +0.029, CI [0.009, 0.049], positive in 5 of 5: **passes** (seed 2 by 0.001).
- Age MAE: the rank arm 8.66-8.78 vs ridge 8.17, so no gain on age itself.
- Caveat: a bootstrap over 5 values is coarse; the five held-out gene sets overlap and two pairs share a held-out
  context. Summary: `data/output/summary/rm24_summary.parquet`; per seed `rm24_seed{0..4}.parquet`.

### 2026-11-12 17:30 · RM23: MaxToki-217M in-silico knockdown of blood CD4 T cells vs the CD4 CRISPRi screen and our clock
`uv run --with transformers rna-age maxtoki-cd4 --n-prompts 600 --batch 8` (0c93566's code; the jobs ran at ca7429d's
script with the module from the worktree). Weights: `theodoris-lab/MaxToki` @581b94c, **licence Apache-2.0** (model
card and Hub tag; research use permitted). The released temporal 217M is the BioNeMo checkpoint (vocabulary 23,277:
genes + time tokens −1500…1499 months + `<boq>`/`<eoq>`); the `-HF` 217M folder is stage 1 only (no time tokens).
Read without NeMo/Megatron (allow-list unpickler for the DCP metadata, `torch.load(weights_only=True)` per chunk),
mapped to `LlamaForCausalLM`. Cells: 992 naive + central-memory CD4 T cells of 3faad104 (one per donor, ≤ 40 donors
per type × sex × decade), MaxToki rank encoding (its medians), ≤ 1,300 genes. Prompts: the paper's task 1 within type
and sex, `[cell a1] [Δ months] [cell a2] <boq> [query a3] <eoq>`, answer = expected time token at `<eoq>`. Screen: each of
the screen's regulators expressed in the query removed from it (paper's in-silico inhibition), effect = Δ predicted
time (> 0 = older). GPU jobs 6435661 (validation, then OOM at batch 16 after 5 prompts) and 6435721 (resumed, 600
prompts, 2 h 20 min, ~13 s per prompt, ~310 deletions each).
- Gate failed. Validation (150 prompts): cell next-gene cross-entropy 3.24-3.30 (uniform ≈ 9.9: weights map right), but
  the timelapse answer does not follow truth: Spearman 0.02 / 0.02 / 0.10 at RoPE factor 1 / 4 / 8, MAE 276-286 months
  vs 234 for a constant, slope ≤ 0.005, only 0.1-7.6 % of probability on time tokens. On the 600 screen prompts (factor
  8): Spearman −0.024. The paper (Methods) trained the 217M's timelapse task with a separate MSE regression head; that
  head is not in the release, so the 217M's time answer cannot be read as published. The 1B is the headless one.
- Screen anyway, as specified: 3,556 genes scored, 3,063 with ≥ 5 expressing query cells (median 22); median |effect|
  0.21 months; effect vs the deleted gene's mean rank Spearman −0.03.

| condition | vs screen ρ [95 % CI] | null q97.5 | p_perm | clock ρ same genes | vs clock ρ [95 % CI] | p_perm |
|---|---|---|---|---|---|---|
| Rest | −0.032 [−0.077, 0.012] | 0.044 | 0.15 | 0.246 | −0.002 [−0.039, 0.035] | 0.91 |
| Stim8hr | 0.012 [−0.029, 0.054] | 0.045 | 0.58 | 0.238 | −0.044 [−0.079, −0.009] | 0.014 |
| Stim48hr | −0.030 [−0.072, 0.014] | 0.043 | 0.18 | 0.256 | −0.045 [−0.082, −0.009] | 0.014 |

- Top fifth of the screen by |coef| (397-432 genes): sign agreement 0.45-0.46 vs 0.50-0.51 under the shuffle (p ≥ 0.96);
  Spearman there −0.08 to −0.10. Bar (ρ ≈ 0.26, p < 0.05 above the placebo): not met in any condition. The two
  small negative clock rows are tiny and opposite in sign to agreement; with a time read-out that does not track age
  they are not read as biology.

### 2026-11-12 20:25 · RM23 1B: stopped before download, no released 1B checkpoint has time tokens
Lead approved one 1B download and one GPU job, behind a gate (Spearman ≥ 0.5 with the true timelapse and MAE below
the constant guess on the same 150 validation prompts). Licence first: `theodoris-lab/MaxToki` @581b94c, one model card
for the whole repository, **Apache-2.0** (no separate 1B card or licence file); research use permitted. Then the
checkpoints' own metadata, read without fetching weights (configs, `context/io.json`, DCP `.metadata` via the
allow-list unpickler):
- `MaxToki-1B-HF`: `LlamaForCausalLM`, vocabulary 20,275 (genes + 4 specials), embedding [20275, 2304]: stage 1 only.
- `MaxToki-1B-bionemo`: tokenizer 20,277 entries = genes + specials + `<boq>`/`<eoq>`, no numeric time tokens; embedding
  and output layer [20352, 2304] bf16 (padded to a multiple of 128); 68 weight files, 14.7 GB with optimizer state
  (lr 1.6e-4 in its config: a training checkpoint, not an import).
- With no time tokens the headless time answer (expected value over the time tokens at `<eoq>`) does not exist, so the
  gate cannot be computed and the screen cannot be run. Nothing downloaded beyond metadata; no GPU job. The temporal
  217M (vocabulary 23,277) is the only released checkpoint with time tokens, and its answer failed (RM23 above).
- 20:28 · RM25: 6436768 cancelled at 1 h 44 min. At the measured rate (about 1 query study a minute, 81 of 810 studies
  in 80 min) it needs about 13.5 h, while the repository's default limit is 4 h and `scontrol` may not raise it, so it
  would have been killed with nothing written. It is not resumable. Resubmitted as 6439020 with TIME=16:00:00, same
  command.
- 20:29 · RM25: 6439020 cancelled after 1.5 min. Resubmitted as 6439059 with the lead's `--splits test` (844a953: the bar reads
  test MAE only, 376 query studies instead of 810, about 6.5 h), TIME=08:00:00.

### 2026-11-12 22:30 · RM22 results: frozen Geneformer embeddings as donor-age clock features
`uv run --with tabicl --with transformers rna-age fm-embed-clock <cohort>` at 705332b, one GPU job at a time: 5e57cd50
6435664 (rate probe), 5c97eeeb 6436081, 218acb0f 6436082, c838aec3 6436083, terekhova2023 6436084, 3faad104 6436085,
37a17b78 6436086; all completed. Model: Geneformer V2-104M (Apache 2.0), fp16 autocast on the 11 GB cards. Tahoe-x1-70M
(first choice) was not used: its config needs FlashAttention and bf16 (Composer/llm-foundry), which these cards lack.
Per donor ≤ 200 cells, stratified by class, plus ≤ 200 CD4 T cells in blood; seed 0; Geneformer's V2 rank-value input
(counts / total × 10⁴ / gene median, 4,096 tokens) and its default layer (second to last); `cls` token (primary) and
gene-mean vectors averaged per donor. Ridge and TabICLv2 on RM14's folds and donors. Rates: brain 35-39 cells/s
(23-26 % of cells truncated at 4,096 genes), blood 70-221 cells/s. Tables: `fm_embedding_clock.parquet` (vs the
pseudobulk TabICLv2 2,000 genes and the row's best of ridge/enet/gbm, paired bootstrap over persons),
`fm_embedding_oof.parquet`. Whole sample, Geneformer `cls` + TabICLv2 against pseudobulk TabICLv2 2,000 genes, MAE:

| cohort | donors | embedding | pseudobulk | difference [95 % CI] |
|---|---|---|---|---|
| terekhova2023 (blood) | 317 | 8.73 | 6.92 | +1.82 [1.05, 2.54] |
| 3faad104 (blood) | 981 | 8.30 | 7.37 | +0.93 [0.62, 1.23] |
| c838aec3 (blood) | 625 | 6.20 | 5.77 | +0.43 [0.12, 0.72] |
| 218acb0f (blood; batch confounded with disease, age result only) | 261 | 7.97 | 6.88 | +1.10 [0.51, 1.75] |
| 37a17b78 (brain) | 783 | 7.49 | 7.40 | +0.09 [−0.18, 0.36] |
| 5c97eeeb (brain) | 299 | 7.88 | 5.81 | +2.08 [1.46, 2.67] |
| 5e57cd50 (brain) | 100 | 6.12 | 6.19 | −0.07 [−0.59, 0.45] |

- Bar (beat pseudobulk TabICL beyond the CI in ≥ 4 of 7): not met. 0 of 7; worse in 5, tie in 2. No embedding
  variant (cls/mean × ridge/TabICLv2) beats the 2,000-gene TabICLv2 in any of 11 cohort × representation rows; CD4 T
  rows are worse in all four blood cohorts.
- Against the best classical pseudobulk baseline the embeddings mostly tie; the one win is c838aec3 whole sample
  (cls + ridge 6.13 vs ridge 6.97, −0.84 [−1.24, −0.44]), where TabICLv2 on genes is still better.
- Limits: one model, one seed, ≤ 200 cells per donor (pseudobulk uses every cell), no fine-tuning.

### 2026-11-13 02:45 · RM25 result: TabICLv2 is no bulk clock (test MAE 15.3 y vs the bar of 9.1)
Job 6439059 (6 h 09 min on one 2080 Ti), run `20261113-0239-tfm-tabicl-hvg-ctx4k-near`: `uv run --with tabicl
rna-age tabfm --model tabicl --features hvg --n-genes 2000 --context nearest-study --n-context 4000 --cap-per-study 200
--splits test`; 376 test studies, each predicted from 4,000 context rows drawn from the 1,069 nearest training studies.

| set | n | MAE | median AE | r | R² | bias slope |
|---|---|---|---|---|---|---|
| test, all | 8,843 | 15.30 | 12.12 | 0.51 | 0.21 | −0.69 |
| test, ARCHS4 | — | 15.12 | — | 0.52 | — | — |

- Bar (the lead's, fixed before the run): within 1 y of the 20k-HVG MLP (8.1 test) earns the fine-tuning step;
  beating the headline 7.46 beyond the paired CI is a finding. Neither: 7.2 y worse than the MLP, so no paired CI
  was needed and no fine-tuning step follows.
- The bias slope of −0.69 says predictions shrink hard towards the context's mean age: with 4,000 rows and 2,000
  genes across hundreds of bulk studies the in-context model does not find the age signal that a trained network
  (or ridge) finds. On single-cell pseudobulk (RM14/RM16: one cohort, one technology, ≤ ~1,000 donors) it does.
- Limits: one seed, one context size and gene count; 4,000 rows is the most the 11 GB card held (RUNNER_NOTES).

### 2026-11-13 17:00 · RM26 building: recount3 source, 3,289 new adult bulk samples in 198 studies
`rna-age fetch recount3` then `rna-age parse recount3` (login node, nice, 4 threads; fetch 85 s, parse 1 min): for
the 248 projects holding the inventory's 4,430 `new` samples, `sra.gene_sums.<SRP>.G026.gz` and
`sra.recount_qc.<SRP>.MD.gz` (496 files, 1.30 GB, each md5-checked against its S3 ETag, sidecars written).
Output `data/input/recount3/` (samples, genes, 4 count parts, `README.json` with every count below,
`studies.parquet`, `read_length_check.parquet`, `audit.parquet`); merged to `data/interim/samples_all_recount3.parquet`
and `splits_recount3.parquet` (`rna-age merge --suffix _recount3`; the shared `samples_all.parquet` is untouched).

| step | samples | studies |
|---|---|---|
| inventory targets | 4,430 | 248 |
| dropped: already in Bgee (by SRX/SRR) | 104 | — |
| dropped: in ARCHS4 or GTEx by GSM/SRX/SRR | 0 | — |
| dropped: project spans several ARCHS4 series (SRP013565) | 19 | 1 |
| excluded: single cell (D4, whole study) | 574 | 18 |
| excluded: tumour / cancer patient | 172 / 257 | — |
| excluded: prenatal, cell line, stem/organoid | 15 | — |
| eligible, with counts | 3,289 | 198 |

- Ages: 3,236 exact, 53 decade bins. Health: 762 healthy stated (`is_healthy` true), 2,527 nothing stated (null, ARCHS4's
  convention), 0 disease stated. 21 projects partly in ARCHS4 take ARCHS4's GEO series as `study_id` (147 samples).
- Coverage → counts: sums divided by each run's `star.average_mapped_length`, rounded, runs summed per SRX. Check
  over 3,335 runs: library size / featureCounts assigned reads median 1.04 (5-95 %: 0.74-1.15), single-end 1.10,
  paired-end 0.98 (median mapped length 198 = both mates, so the counts are fragments); / STAR mapped reads 0.77.
- D4 rules: a study is single cell when > 0.4 of its runs are predicted `scrna-seq` by recount3 or its ARCHS4 series
  is single cell; tumour wording through `add_flags`, plus every sample of a project with a tumour sample is
  `cancer_patient` (101 normals of SRP114315's lung squamous carcinoma patients otherwise passed).
- Splits by the same study hash: no existing assignment moves (test `test_real_merge_keeps_old_assignments`). recount3:
  2,322 train, 457 val, and 510 samples of 36 studies in `test_new`, scored apart; `test` stays the old 12,144.
- Limits: GENCODE v26 lacks 1,600 of the 46,028 shared bulk genes (zeros for recount3 samples); `BULK_SOURCES` left
  unchanged so the headline's gene axis does not move. SRP223445 (240 plasma cell-free RNA samples) is eligible.

### 2026-11-13 17:11 · RM26 merged (7afe44c source, c624a5c plumbing; both reviewed by the lead); arms submitted
- Control: job 6470325, `rna-age probe --features hvg --model mlp --n-hvg 20000 --train-set all --genes-with recount3
  --seed S` for S in 0-2 (old table, gene axis 44,428 = shared bulk ∩ GENCODE v26).
- recount3 arm: job 6470326 (after 6470325), same plus `--samples-suffix _recount3 --exclude-train-studies SRP223445`
  (train rows 62,350 vs 60,304; recount3 val samples kept in early stopping, the lead's call).
- Bar (the lead, fixed before the run): `rna-age rm26-compare` arm − control on the old test (8,843 exact), 3-seed
  mean per arm, CI over test studies; pass if hi < 0. Noise floor of two identical 3-seed means: +0.11 [−0.09, 0.32].
  test_new and the ensemble vs 7.46 are secondary read-outs. Fine-tunes retrained only on a pass.

### 2026-11-13 17:50 · RM26 result: recount3 helps a little, not beyond the CI (bar not met)
Control runs `20261113-1716/1720/1724-hvg20000-mlp-trainall-genes-recount3[-seedN]` (job 6470325); recount3 arm
`20261113-1727/1731/1735-...-table-recount3-excl1[-seedN]` (job 6470326); `rna-age rm26-compare` →
`data/output/summary/rm26_compare.parquet`. 3-seed means, old test (8,843 exact, 212 studies), CI over 2,000 study
resamples:

| comparison | MAE a | MAE b | diff [95 % CI] |
|---|---|---|---|
| recount3 arm − control (all) | 7.73 | 7.90 | −0.17 [−0.39, +0.02] |
| recount3 arm − control (ARCHS4 test) | 7.75 | 7.92 | −0.16 [−0.38, +0.02] |
| recount3 arm − control (Bgee test, 4 studies) | 5.78 | 6.41 | −0.64 [−1.64, −0.20] |
| recount3 arm − headline 7.46 | 7.73 | 7.46 | +0.27 [−0.14, +0.69] |
| control − headline 7.46 | 7.90 | 7.46 | +0.44 [−0.01, +0.94] |

- Bar (hi < 0 on arm − control, all): not met; the upper bound is +0.02. The direction is consistent across seeds
  (control 7.92 / 8.32 / 7.99, arm 7.99 / 7.87 / 7.80 single-seed test MAE) and larger than the +0.11 noise floor
  in size, but not resolved. By the lead's rule no fine-tune retrain follows.
- test_new (494 recount3 test samples, 35 studies, arm only): MAE 11.24, i.e. new studies from a new pipeline are
  much harder than the old test; the control cannot score them on the same footing (it never saw recount3), so no
  paired read.
- Headline-style MLP-only means are worse than the 7.46 ensemble, as expected (no fine-tunes in either arm).

### 2026-11-13 18:32 · RM27 submitted: health MLP plus the age-order term (the lead's cefc4fc)
Job 6470599 (CPU, 8 cores): `rna-age train-health --rank-weight 1 --disease-head --age-pair-weight 1 --seed S
--external` for S in 0-2 (`--external` only scores the RM10 external series, as the control runs did). Control: the
RM12-table runs `20261111-0854/0908/0918-health-mlp-rank1-head-s0..s2` (same recipe without the term). Bar (the lead):
aging-linked within-study gap (`health-eval`, mean of 3 seeds) ≥ +2.9 y, and control test MAE no more than 0.5 y
worse than the control runs'. Then `health-eval --permutations 200` on the best seed.

### 2026-11-13 19:14 · RM28 submitted: own single-cell pretraining (the lead's ee227b2; bars in ROADMAP RM28)
- Pool: job 6472026 (CPU 8, 96 GB, 12 h limit; the lead's estimate 1-3 h, not measured), `rna-age sc-pretrain-pool`.
- Arms after it (afterok), one GPU each, 30 epochs: `rna-age sc-pretrain --arm mask` 6472027, `--arm mask+perturb`
  6472097, `--arm universal` 6472138 (96 GB). Estimates were for an A100; ours are 11 GB cards, so memory and time
  are unmeasured; the built-in epoch-1 stop rule (masked MSE must beat the mean-gene baseline) applies.
- Then per run: `sc-pretrain-embed` (first alone), `sc-pretrain-clock` (`--with tabicl`), `sc-pretrain-finetune`;
  universal also `sc-pretrain-universal-eval`.

### 2026-11-13 19:32 · RM28 fp32 fix picked up; RM30 prep submitted
- The lead's 1234b0b makes RM28's loop use fp32 on cards without bf16; the queued scpre jobs (still pending) pick it up.
- RM30 (TabICLv2 initialised from its released weights, trained on expression episodes; bars in ROADMAP RM30):
  `uv run --with tabicl rna-age expr-icl-prepare`, job 6472840 (CPU 8, 96 GB, 12 h limit, unmeasured). Training
  (`expr-icl-train --steps 20000`) waits for RM28's first arm's result, per the plan; first 500 steps get timed.

### 2026-11-13 19:33 · RM27 result: the age-order term does not lift the gap to +2.9 (bar not met)
Job 6470599 (58 min, CPU): runs `20261113-1848/1905/1931-health-mlp-rank1-head-agepair1-s0..s2`; control the RM12-table
runs `20261111-0854/0908/0918-health-mlp-rank1-head-s0..s2`. `rna-age health-eval`, aging-linked within-study gap
(33 held-out studies, 2,282 cases) and control test MAE:

| seed | arm gap [95 % CI] | control gap | arm control-MAE | control control-MAE |
|---|---|---|---|---|
| 0 | 2.71 [0.83, 4.76] | 2.30 [0.31, 4.41] | 9.07 | 8.88 |
| 1 | 2.95 [0.99, 5.00] | 2.44 [0.43, 4.42] | 9.30 | 9.30 |
| 2 | 2.40 [0.42, 4.55] | 2.92 [0.90, 5.14] | 8.99 | 9.98 |
| mean | **2.69** | 2.55 | 9.12 | 9.39 |

- Bar (the lead): mean gap ≥ +2.9 y: not met (2.69); MAE no more than 0.5 y worse: met (0.27 y better). The term
  moves the mean gap by +0.14 y, well inside seed spread (control seeds alone span 2.30-2.92).
- Best seed (s1), 200 shuffles within study: null mean +0.01, p 0.005 (1 of 200 bounds): the gap is real, as for the
  controls; the term just does not add to it.

### 2026-11-13 19:44 · RM31 submitted: several cell types per donor (the lead's fcdca9c; bar in ROADMAP RM31)
Job 6474667 (one GPU, 64 GB, 10 h limit, unmeasured): `uv run --with tabicl rna-age sc-multitype` (7 D7 cohorts,
ridge + TabICLv2, 5 classes, 5 folds × 7 representations). Writes sc_multitype.parquet and sc_multitype_compare.parquet.

### 2026-11-13 19:52 · RM28 arms mask and mask+perturb trained; read-outs and RM30 training submitted
- Trained in 5 and 6.5 min on one 2080 Ti each (30 epochs; the estimate was hours): runs
  `20261113-1942-scpre-mask-s0`, `20261113-1942-scpre-mask-perturb-s0`; both pass the epoch-1 stop rule. Universal
  (6472138) still training.
- Read-outs: embed mask 6475125 (CPU, builds the cohort cache) → embed mask+perturb 6475126; clocks (`--with tabicl`,
  CPU) 6475127, 6475128; fine-tunes (GPU, chained on one card) 6475129, 6475130.
- RM30: `expr-icl-train --steps 20000` job 6475131 (GPU, 24 h limit until the first 500 steps are timed).

### 2026-11-13 20:02 · RM28 universal trained; RM30 does not fit an 11 GB card as specified
- RM28 universal (6472138, 16 min, 30 epochs): run `20261113-1942-scpre-universal-s0`, best bulk val MAE 15.2 (its
  selection score), pool pseudobulk val MAE 6.9. Read-outs: embed 6475903, clock 6475904, fine-tune 6475905 (after the
  other two fine-tunes), universal-eval 6475906.
- RM30 `expr-icl-train`: 6475131 out of memory in the first backward pass (micro-batch 4); 6475493 with
  `--micro-batch 1 --accum 16` and expandable segments out of memory too (1.86 GiB short). Smaller episodes or
  gradient checkpointing needed; asked the lead.
