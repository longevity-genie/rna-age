# Overnight report, 2026-11-10

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

## 3. Models (to be filled in as runs finish)

See section 6 of `docs/experiments.md` and `data/output/summary/experiments.parquet`.

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

## 5. Next steps

(to be written at the end of the night)
