# rna-age

Transcriptome-based age prediction with [BulkFormer](https://github.com/KangBoming/BulkFormer)
(Kang et al., Cell Systems 2026): BulkFormer-Atlas sample embeddings → age regressor.

This repository holds the data side: reproducible downloaders and parsers that turn public human RNA-seq
resources with donor age into one harmonised sample table plus raw count matrices that BulkFormer reads
(raw counts keyed by Ensembl gene ID).

## Quick start

```bash
uv sync
uv run rna-age fetch gtex      # → data/download/gtex/ (resumable, checksummed)
uv run rna-age parse gtex      # → data/input/gtex/{samples,genes}.parquet + counts/
uv run rna-age summary         # → data/output/summary/*.parquet
uv run pytest -q
```

Heavy parsers run as Slurm jobs through `slurm/submit.sh` (cluster settings go in the git-ignored
`slurm/env.local.sh`; copy `slurm/env.local.example.sh`).

## Sources

| source | what | age |
|---|---|---|
| `gtex` | GTEx v10 RNA-SeQC gene read counts, ~19.8k samples, 946 donors, 68 tissues (public GCS bucket) | 10-year bins (`binned`) |
| `archs4` | ARCHS4 human gene counts (all GEO/SRA human bulk RNA-seq, Maayan lab; file of 2026-11-06) | parsed from GEO `characteristics_ch1` |
| `ncbi_geo` | NCBI-generated GEO RNA-seq counts (GRCh38.p13), for age-annotated series ARCHS4 lacks and as a cross-check | GEO characteristics |
| `bgee` | Bgee curated healthy wild-type samples | developmental stage (`NN-year-old human stage`) |
| `geo_cohorts` | curated aging cohorts (e.g. Fleischer 2018 fibroblasts, GSE113957; series from transcriptomic-clock papers) | exact |
| `census_pseudobulk` | CZ CELLxGENE Census, donor × tissue pseudobulk of primary data with exact age (OneK1K, AIDA, ...) | development stage |
| `terekhova2023` | Terekhova et al. 2023 PBMC aging atlas (166 donors, 25–85 y), pseudobulk | exact |

Pseudobulk sources are kept separate from bulk (`is_pseudobulk`) to measure the domain shift.

**Not used:** cancer cohorts (TCGA, TARGET; out of scope for now; tumour samples inside mixed sources are
flagged and excluded from training and evaluation). Controlled access: GTEx exact age, TOPMed, UK Biobank,
most of PsychENCODE.

## Data layout and formats

Everything under `data/` (git-ignored except `data/output/summary/`); all tables are parquet written by polars.

- `data/download/<source>/`: raw files as fetched, each with `<file>.meta.json` (URL, size, checksum, date).
- `data/input/<source>/samples.parquet`: one row per sample, same columns in every source
  (`src/rna_age/schema.py`): ids, study and donor (grouping units for splits), `age_years`/`age_min`/`age_max`/
  `age_type` (exact | binned | censored) with the raw string in `age_raw`, sex, tissue, cell type, disease,
  `is_healthy`, `is_tumor`, `is_cell_line`, `exclude_reason`, platform, library, pseudobulk fields.
- `data/input/<source>/genes.parquet` + `counts/part-NNNNN.parquet`: raw integer counts as gene × sample blocks
  (≤ 1000 samples per part, `gene_id` = Ensembl ID without version); `counts/index.json` maps sample → part.
- `data/interim/`: merged tables, gene mapping, splits (grouped by study/donor), embeddings.
- `data/output/`: results.

Age strings are parsed in `src/rna_age/ages.py` (`age: 54`, `54 years`, `6 months`, `50-59`, `>89`, ...;
rejects gestational age, passage, time post infection, `adult`, `NA`); the raw string is kept for audit.

Status of the current build: see `STATUS.md`.
