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
pdf(NULL)  # tAge draws diagnostics; do not write Rplots.pdf into the repository
input_dir <- args[[1]]
clock_dir <- args[[2]]

counts <- as.data.frame(read_parquet(file.path(input_dir, "counts.parquet")))
rownames(counts) <- counts$gene_id
counts$gene_id <- NULL
meta <- as.data.frame(read_parquet(file.path(input_dir, "samples.parquet")))
rownames(meta) <- meta$sample_id
meta <- meta[colnames(counts), , drop = FALSE]

# Human-applicable clocks: multispecies multi-tissue Elastic Net chronological age (scaled + YuGene).
clocks <- list_clocks(type = "EN", outcome = "Chronological", species = "Multispecies", tissue = "Multi-Tissue")
clocks <- download_clocks(clocks, dest_dir = clock_dir)

# Strata (dataset x tissue) are processed one at a time, as tAge centres each on its own samples; a stratum that
# tAge's preprocessing cannot handle (e.g. too few expressed genes) is skipped and listed, not fatal.
results <- list()
skipped <- character(0)
for (st in unique(meta$stratum)) {
  ids <- rownames(meta)[meta$stratum == st]
  lib <- colSums(counts[, ids, drop = FALSE])
  ids <- ids[lib > 0]
  if (length(ids) < 6) { skipped <- c(skipped, st); next }
  res <- tryCatch({
    eset <- make_ExpressionSet(counts[, ids, drop = FALSE], meta[ids, , drop = FALSE])
    pre <- suppressMessages(tAge_preprocessing(eset, species = "human", verbose = FALSE))
    as.data.frame(predict_tAge(pre, clocks))
  }, error = function(e) { message("skip stratum ", st, ": ", conditionMessage(e)); NULL })
  if (is.null(res)) { skipped <- c(skipped, st); next }
  res$sample_id <- rownames(res)
  results[[st]] <- res
}
res <- do.call(rbind, results)
write_parquet(res, file.path(input_dir, "predictions.parquet"))
cat("tAge predictions:", nrow(res), "samples from", length(results), "strata; skipped", length(skipped), "\n")
if (length(skipped)) writeLines(skipped, file.path(input_dir, "skipped_strata.txt"))
