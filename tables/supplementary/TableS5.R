#!/usr/bin/env Rscript
# Table S5. Stability of Gaussian mixture partitions
rm(list = ls())
source("_common.R")

s6 <- read_json(P_STEP6)$per_k
bp <- read_json(P_BOOT)$per_k
k_all <- as.character(sort(as.integer(names(s6))))
x <- do.call(rbind, lapply(k_all, function(k) {
  s <- s6[[k]]; b <- bp[[k]]
  data.frame(k = as.integer(k), silhouette = num(s$silhouette), seed_ari_mean = num(s$seed_ari_mean),
             seed_ari_sd = num(s$seed_ari_sd), bootstrap_ari_mean = num(b$bootstrap$mean),
             bootstrap_lo = num(b$bootstrap$ci95_lo), bootstrap_hi = num(b$bootstrap$ci95_hi),
             null_lo = num(b$null$ci95_lo), null_hi = num(b$null$ci95_hi))
}))
save_table(5, x,
  c("k", "Silhouette", "Seed ARI mean", "Seed ARI SD", "Bootstrap ARI mean", "Bootstrap 2.5th pct.",
    "Bootstrap 97.5th pct.", "Null 2.5th pct.", "Null 97.5th pct."))
