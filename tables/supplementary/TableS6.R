#!/usr/bin/env Rscript
# Table S6. Sensitivity to the compositional pseudocount
rm(list = ls())
source("_common.R")

p <- fromJSON(P_PSEUDOCOUNT)$results
save_table(6,
  select_fields(p, "pseudocount pc13_var pc4plus_var pc1_3_z pc4plus_z full_z tail_frac_beyond_chi99"),
  c("Pseudocount", "Leading variance", "Residual variance", "Leading kurtosis z",
    "Residual kurtosis z", "Full-space kurtosis z", "Above Gaussian 99th percentile"))
