#!/usr/bin/env Rscript
# Table S1. Agreement between silver and gold report profiles
rm(list = ls())
source("_common.R")

results <- fromJSON(P_PROFILE_BASE)
rho <- fromJSON(P_PROFILE_RHO)
groups <- c(All = "All", A = "A-type", P = "P-type")
profiles <- c(silver_report = "Report silver",
              cohort_mean_loo_report = "Cohort mean baseline",
              type_mean_loo_report = "Report-type mean baseline")
one <- function(x) {
  stopifnot(nrow(x) == 1L)
  x
}
format_ci <- function(value, low, high) sprintf("%.4f [%.4f, %.4f]", value, low, high)
x <- do.call(rbind, lapply(names(groups), function(group) {
  do.call(rbind, lapply(names(profiles), function(model) {
    m <- one(results$metrics[results$metrics$group == group &
               results$metrics$model == model & results$metrics$metric == "mean_cosine", ])
    r <- one(rho[rho$group == group & rho$model == model, ])
    difference <- "-"
    if (model != "silver_report") {
      d <- one(results$paired_differences[results$paired_differences$group == group &
                 results$paired_differences$baseline == model, ])
      difference <- format_ci(d$mean_cosine_difference_silver_minus_baseline, d$ci95_low, d$ci95_high)
    }
    data.frame(subset = paste0(unname(groups[group]), ", n=", m$n_reports), profile = unname(profiles[model]),
               cosine = format_ci(m$estimate, m$ci95_low, m$ci95_high),
               spearman = format_ci(r$new_estimate, r$new_ci[[1]][1], r$new_ci[[1]][2]),
               difference = difference)
  }))
}))

save_table(1,
  x,
  c("Subset", "Compared profile", "Mean cosine [95% CI]", "Pooled Spearman ρ [95% CI]", "Cosine difference [95% CI]"))
