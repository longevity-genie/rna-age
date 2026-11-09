# Run the tAge clocks (Tyshkovskiy et al., Nature 2026; R package Gladyshev-Lab/tAge) on one prepared input.
# Usage: Rscript scripts/tage_predict.R INPUT_DIR CLOCK_DIR
#   INPUT_DIR/counts.parquet  gene_id (Ensembl, no version) + one column of raw counts per sample
#   INPUT_DIR/samples.parquet sample_id, stratum (dataset x tissue; centring unit), age_years, ...
# Writes INPUT_DIR/predictions.parquet: sample_id, stratum, one column per clock (relative tAge), units.
# Called by `rna-age tage`; run inside the isolated R env (scripts/setup_tage_env.sh).
suppressPackageStartupMessages({
  library(tAge)
  library(arrow)
  library(Biobase)
})
args <- commandArgs(trailingOnly = TRUE)
input_dir <- args[[1]]
clock_dir <- args[[2]]

counts <- as.data.frame(read_parquet(file.path(input_dir, "counts.parquet")))
rownames(counts) <- counts$gene_id
counts$gene_id <- NULL
meta <- as.data.frame(read_parquet(file.path(input_dir, "samples.parquet")))
rownames(meta) <- meta$sample_id
meta <- meta[colnames(counts), , drop = FALSE]

# Human-applicable clocks: multispecies multi-tissue Elastic Net chronological age (scaled + YuGene).
clocks <- list_clocks(type = "EN", outcome = "Chronoage", species = "Multispecies", tissue = "Multi-Tissue")
clocks <- download_clocks(clocks, dest_dir = clock_dir)

eset <- make_ExpressionSet(counts, meta)
# No dedicated control group: every stratum is filtered, normalised and centred on its own samples
# (tAge's default of centring on the overall per-gene median, per stratum), so outputs are relative.
tage_eset <- tAge_preprocessing(eset, species = "human", split_by = "stratum")
res <- predict_tAge(tage_eset, clocks)
res <- as.data.frame(res)
res$sample_id <- rownames(res)
units <- attr(res, "tage_units")
write_parquet(res, file.path(input_dir, "predictions.parquet"))
cat("tAge predictions:", nrow(res), "samples;", ncol(res) - 1, "columns\n")
if (!is.null(units)) print(units)
