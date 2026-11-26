#!/usr/bin/env Rscript
# Figure S5. Domain logit contributions E.
rm(list = ls())
source("_common.R")

# (a) Domain contributions versus proportional allocation by domain share.
assoc <- read.delim(EV_ASSOC_TSV)
assoc <- assoc[assoc$feature %in% c("E", "L_times_P") & assoc$scope %in% c("all", paste0("fold", 1:5)), ]
assoc$scope <- factor(assoc$scope, levels = c("all", paste0("fold", 1:5)),
                      labels = c("All", "Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5"))
assoc$feature <- factor(assoc$feature, levels = c("E", "L_times_P"), labels = c("E", "L × P"))
pa <- ggplot(assoc, aes(scope, spearman_signed_delta, fill = feature)) +
  geom_col(position = position_dodge(width = 0.75), width = 0.7) +
  geom_hline(yintercept = 0, linewidth = 0.36) +
  scale_fill_manual(values = c("E" = PAL$red, "L × P" = PAL$grey), name = NULL) +
  labs(x = NULL, y = "Spearman ρ with deletion change", tag = "a") +
  theme_all(SI_TEXT_PT) + theme(legend.position = c(0.99, 0.99), legend.justification = c(1, 1),
                     legend.key.size = unit(0.30, "cm"))

# (b) Increase in held-out R² from adding E to L × P and the covariates.
cmp <- read.delim(EV_HELDOUT_TSV)
cmp <- cmp[cmp$scope == "all" & cmp$method == "nuisance_LP_E", ]
ci <- do.call(rbind, lapply(regmatches(cmp$delta_R2_ci95, gregexpr("-?[0-9.]+(e-?[0-9]+)?", cmp$delta_R2_ci95)),
                            as.numeric))
cmp$lo <- ci[, 1]; cmp$hi <- ci[, 2]
tlab <- c(delta_logit = "signed change", signed_excess = "signed excess\nover random",
          absolute_delta = "absolute change", absolute_excess = "absolute excess\nover random")
cmp$target <- factor(tlab[cmp$target], levels = rev(unname(tlab)))
pb <- ggplot(cmp, aes(delta_R2, target)) +
  geom_errorbar(aes(xmin = lo, xmax = hi), orientation = "y", width = 0.2,
                linewidth = 0.45, colour = PAL$blue) +
  geom_point(size = 1.8, colour = PAL$blue) +
  geom_text(aes(x = hi + 0.03, label = sprintf("%.3f", delta_R2)), hjust = 0, size = SI_TEXT_PT / .pt) +
  expand_limits(x = c(0, 1)) +
  scale_x_continuous(breaks = seq(0, 1, by = 0.25), expand = expansion(mult = c(0, 0.06))) +
  labs(x = "Held-out ΔR² adding E after L × P", y = NULL, tag = "b") +
  theme_all(SI_TEXT_PT)

# (c) Agreement in domain ranks after reallocating E with gold labels.
gold <- read.delim(EV_GOLD_TSV)
mean_rank <- mean(gold$rank_spearman)
pc <- ggplot(gold, aes(rank_spearman)) +
  geom_histogram(binwidth = 0.1, boundary = 0, fill = PAL$navy, alpha = 0.80, colour = "white", linewidth = 0.36) +
  geom_vline(xintercept = mean_rank, colour = PAL$red, linewidth = 0.6) +
  annotate("text", x = mean_rank, y = Inf, vjust = -0.45, size = SI_TEXT_PT / .pt,
           label = sprintf("mean %.2f", mean_rank), colour = "#9E473C") +
  scale_y_continuous(expand = expansion(mult = c(0, 0.05))) +
  coord_cartesian(clip = "off") +
  labs(x = "Per-report Spearman ρ between gold and silver E", y = "Reports", tag = "c") +
  theme_all(SI_TEXT_PT) + theme(plot.margin = margin(t = 13, r = 5.5, b = 5.5, l = 5.5))

fig <- wrap_plots(A = pa, B = pb, C = pc, design = "AB\nCC", heights = c(1, 0.72))
save_fig(fig, "FigureS5", width = 7.0866, height = 5.9)
