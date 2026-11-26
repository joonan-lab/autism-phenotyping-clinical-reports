#!/usr/bin/env Rscript
# Table S4. Bayesian information criterion for Gaussian mixtures and unimodal fits
rm(list = ls())
source("_common.R")

observed <- read_tsv(P_BIC_OBS)
bic_columns <- c(paste0("gmm_bic_k", 1:8), "gauss_bic", "t_bic", "skewt_bic", "dbic_min")
for (column in bic_columns) observed[[column]] <- round(observed[[column]], 1)
design <- c(T = "Type", TL = "Type + length", TLx = "Type × length")
best_bic <- vapply(seq_len(nrow(observed)), function(i) {
  observed[[paste0("gmm_bic_k", observed$gmm_best_k[i])]][i]
}, numeric(1))
x <- data.frame(observed$pseudocount, unname(design[observed$design]),
                observed$gmm_best_k, best_bic, observed$gauss_bic,
                observed$t_bic, observed$skewt_bic, observed$dbic_min)
save_table(4, x,
  c("Pseudocount", "Adjustment", "Best k", "Best GMM BIC", "Gaussian BIC",
    "t BIC", "Skew-t family BIC", "Observed ΔBIC"))
