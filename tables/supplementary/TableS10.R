#!/usr/bin/env Rscript
# Table S10. Sensitivity of the phenotype space to domain exclusion and weighting
rm(list = ls())
source("_common.R")

drop <- read_json(P_DROP)$variants
down <- read_json(P_DOWNWEIGHT)$variants
tail <- read_json(P_HEAVYTAIL)$variants
fp <- read_json(P_FINGERPRINT)$variants
variants <- c(baseline_full19 = "All domains", drop_sentence_f1_lt_0.3 = "Exclude sentence F1 < 0.3",
              drop_report_rho_lt_0.3 = "Exclude report ρ < 0.3", drop_report_rho_lt_0 = "Exclude report ρ < 0",
              downweight_sentence_f1 = "Weight by sentence F1", downweight_report_rho = "Weight by report ρ")
x <- do.call(rbind, lapply(names(variants), function(v) {
  s <- if (v %in% names(drop)) drop[[v]] else down[[v]]
  data.frame(setting = variants[[v]], domains = num(s$n_domains), dim = num(s$eff_dim), k = num(s$best_gmm_k),
             seed = num(s$bestk_stability$cross_seed_ari), boot = num(s$bestk_stability$bootstrap_ari),
             kurtosis = num(tail[[v]]$resid_z), mean_distance = num(fp[[v]]$layer1_mahalanobis$mean),
             rank_rho = num(fp[[v]]$layer1_vs_baseline$rank_spearman))
}))
save_table(10, x,
  c("Domain setting", "Domains", "Effective dimension", "BIC-selected k", "Seed ARI", "Bootstrap ARI",
    "Residual kurtosis z", "Mean prototype distance", "Distance rank ρ vs full"))
