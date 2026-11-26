#!/usr/bin/env Rscript
# Table S9. Association of domain contributions with logit changes after domain deletion
rm(list = ls())
source("_common.R")

x <- read_tsv(P_DEL_ASSOC)
x <- x[x$feature %in% c("E", "L_times_P"), ]
scopes <- c(all = "All reports", setNames(paste("Fold", 1:5), paste0("fold", 1:5)), ASD = "ASD", nonASD = "Non-ASD")
x$scope <- unname(scopes[x$scope])
x$feature <- ifelse(x$feature == "E", "E", "L × P")
save_table(9,
  select_fields(x, "scope feature n spearman_signed_delta spearman_signed_excess spearman_abs_delta spearman_absolute_excess sign_agreement"),
  c("Scope", "Predictor", "Report–domain pairs", "Signed change ρ", "Signed excess ρ", "Absolute change ρ",
    "Absolute excess ρ", "Sign agreement"))
