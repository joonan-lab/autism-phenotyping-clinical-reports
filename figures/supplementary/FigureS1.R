#!/usr/bin/env Rscript
# Figure S1. Silver-label confidence and agreement with gold labels.
rm(list = ls())
source("_common.R")

rb <- read.delim(CAL_BINS_TSV)
rb <- rb[rb$source == "adopted" & rb$group == "All" & rb$strategy == "fixed", ]
sm <- fromJSON(CAL_SUMMARY_JSON)$metrics
gap <- sm[sm$source == "adopted" & sm$group == "All" & sm$metric == "confidence_minus_agreement", ]
p <- ggplot(rb, aes(mean_confidence, expert_agreement)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", colour = PAL$grey, linewidth = 0.4) +
  geom_line(colour = PAL$blue, linewidth = 0.6) +
  geom_point(aes(size = n_pairs), colour = PAL$blue) +
  annotate("text", x = 0.52, y = 0.98, hjust = 0, vjust = 1, size = 8 / .pt,
           label = sprintf("confidence minus agreement %.3f\n(95%% CI %.3f to %.3f)",
                           gap$estimate, gap$ci95_low, gap$ci95_high), lineheight = 0.9) +
  scale_size_continuous(range = c(1, 3.5), name = "label pairs") +
  coord_cartesian(xlim = c(0.5, 1), ylim = c(0, 1)) +
  labs(x = "Mean silver-label confidence", y = "Agreement with gold labels") +
  theme_all() +
  theme(legend.position = "inside",
        legend.position.inside = c(0.98, 0.05),
        legend.justification.inside = c(1, 0),
        legend.key.size = unit(0.30, "cm"),
        plot.margin = margin(3, 4, 3, 4))
save_fig(p, "FigureS1", width = 169 / 25.4, height = 90 / 25.4)
