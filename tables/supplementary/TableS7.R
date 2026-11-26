#!/usr/bin/env Rscript
# Table S7. Performance of the ASD classifier and matched TF-IDF baselines
rm(list = ls())
source("_common.R")

models <- classifier_names
metrics <- c(auroc = "AUROC", auprc = "Average precision", accuracy = "Accuracy", precision = "Precision",
             recall = "Sensitivity", specificity = "Specificity", f1 = "F1", brier = "Brier score",
             calibration_slope = "Calibration slope")
x <- read_tsv(P_CLF_METRICS)
x <- x[x$group == "All" & x$model %in% names(models) & x$metric %in% names(metrics), ]
x <- x[order(match(x$model, names(models)), match(x$metric, names(metrics))), ]
delta <- read_tsv(P_CLF_PAIRED)
key <- function(d) paste(d$group, d$model, d$metric)
delta <- delta[match(key(x), key(delta)), ]
x <- data.frame(unname(models[x$model]), unname(metrics[x$metric]),
                select_fields(x, "pooled_estimate ci95_low ci95_high fold_mean fold_sd"),
                select_fields(delta, "difference_baseline_minus_v4_1 ci95_low ci95_high"))
save_table(7, x,
  c("Model", "Metric", "Pooled estimate", "95% CI low", "95% CI high", "Fold mean",
    "Fold SD", "Paired Δ", "Δ CI low", "Δ CI high"))
