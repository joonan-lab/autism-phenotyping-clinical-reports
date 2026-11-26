#!/usr/bin/env Rscript
# Table S3. Sentence-selection sensitivity of agreement with full-report gold profiles
rm(list = ls())
source("_common.R")

results <- read_tsv(P_SELECTION_GOLD)
methods <- c(full = "All eligible sentences", attention_top10 = "Attention top-10 sentences")
format_metric <- function(method, metric) {
  r <- results[results$method == method & results$metric == metric, ]
  stopifnot(nrow(r) == 1L)
  ci <- fromJSON(r$percentile_interval)
  sprintf("%.4f [%.4f, %.4f]", r$mean, ci[1], ci[2])
}
x <- do.call(rbind, lapply(names(methods), function(method) {
  r <- results[results$method == method & results$metric == "cosine_full_gold", ]
  stopifnot(nrow(r) == 1L)
  data.frame(input = unname(methods[method]), reports = r$n_reports,
             cosine = format_metric(method, "cosine_full_gold"),
             tv = format_metric(method, "TV_full_gold"),
             spearman = format_metric(method, "rank_correlation_full_gold"))
}))

save_table(3, x,
  c("Sentence input", "Reports", "Mean cosine [95% CI]", "Mean TV [95% CI]",
    "Mean within-report domain-rank Spearman ρ [95% CI]"))
