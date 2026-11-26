#!/usr/bin/env Rscript
# Figure S3. Report-length matching.
rm(list = ls())
source("_common.R")

len_rep <- read.delim(LENGTH_REP_TSV)
len_rep <- len_rep[len_rep$status == "ok", ]
len_obs <- read.delim(LENGTH_SUMMARY_TSV)
obs_of <- function(tab, metric) tab$observed[tab$metric == metric]

hist_panel <- function(values, observed, xlab, ylab, tag = NULL, breaks = NULL, bins = 40,
                       ref = NULL, ref_label = NULL, legend_keys = NULL) {
  q <- quantile(values, c(0.025, 0.975), names = FALSE)
  lines <- data.frame(x = c(observed, q), what = c("observed", rep("replicate 2.5-97.5th percentile", 2)))
  if (!is.null(ref)) lines <- rbind(lines, data.frame(x = ref, what = ref_label))
  hist <- if (is.null(breaks)) geom_histogram(bins = bins, fill = PAL$navy, alpha = 0.80, colour = "white", linewidth = 0.2)
          else geom_histogram(breaks = breaks, fill = PAL$navy, alpha = 0.80, colour = "white", linewidth = 0.2)
  cols <- c(observed = PAL$red, `replicate 2.5-97.5th percentile` = PAL$blue)
  ltys <- c(observed = "solid", `replicate 2.5-97.5th percentile` = "dashed")
  if (!is.null(ref)) { cols[ref_label] <- PAL$navy; ltys[ref_label] <- "dotted" }
  if (is.null(legend_keys)) legend_keys <- names(cols)
  ggplot(data.frame(v = values), aes(v)) +
    hist +
    geom_vline(data = lines, aes(xintercept = x, colour = what, linetype = what), linewidth = 0.5) +
    scale_colour_manual(values = cols, breaks = legend_keys, name = NULL) +
    scale_linetype_manual(values = ltys, breaks = legend_keys, name = NULL) +
    scale_y_continuous(expand = expansion(mult = c(0, 0.05))) +
    labs(x = xlab, y = ylab, tag = tag) +
    theme_all(SI_TEXT_PT) + theme(legend.position = "top", legend.key.width = unit(0.6, "cm"))
}

beta_lab <- "Residual Mardia β₂ (PC4–18)"
fraction <- len_rep$tail_A_fraction * 100
observed_A <- obs_of(len_obs, "tail_A_fraction") * 100
observed_P <- obs_of(len_obs, "tail_P_fraction") * 100

# (a) Residual kurtosis after shortening A-type reports to P-type lengths.
pa <- hist_panel(len_rep$residual_beta2, obs_of(len_obs, "residual_beta2"),
                 beta_lab, "Length-matched repetitions", tag = "a", bins = 30) +
  guides(colour = guide_legend(nrow = 2), linetype = guide_legend(nrow = 2))

# (b) Percentage of A-type reports in the full-space distance tail.
pb <- hist_panel(fraction, observed_A, "A-type reports in the distance tail (%)",
                 "Length-matched repetitions", tag = "b",
                 breaks = seq(-0.25, max(fraction, observed_P) + 1, by = 0.52),
                 ref = observed_P, ref_label = "observed P-type",
                 legend_keys = "observed P-type")

fig <- pa + pb
save_fig(fig, "FigureS3", width = 7.0866, height = 3.25)
