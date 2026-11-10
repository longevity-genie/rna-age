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
- Last 4 blocks, all health states (`atlas-ft-u4-all`): val MAE 12.80 → 10.81 → 10.45 → 10.60 → **9.73** after epochs 0–4 (healthy-only run: 14.14 → 11.59 → 10.79).
- Last 6 blocks, 20k samples/epoch, all health states (`atlas-ft-u6-all`): val MAE 12.09 → **10.24** after epochs 0–1 (~71 min/epoch); best fine-tune so far.
