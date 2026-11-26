#!/usr/bin/env Rscript
# Figure S2. PC1 scores before and after adjustment for report type.
rm(list = ls())
source("_common.R")

FS <- 8
pc1 <- jsonlite::fromJSON(CONFOUND_JSON)$panel_b

scores <- rbind(
  data.frame(stage = "unadjusted", type = pc1$report_type,
             pc1 = as.numeric(pc1$pc1_raw)),
  data.frame(stage = "type-adjusted", type = pc1$report_type,
             pc1 = as.numeric(pc1$pc1_adjusted))
)
scores$stage <- factor(scores$stage, levels = c("unadjusted", "type-adjusted"))
scores$type  <- factor(scores$type, levels = c("A", "P"),
                     labels = c("A-type", "P-type"))

r_raw <- pc1$corr_pc1_type_raw
r_adj <- pc1$corr_pc1_type_adjusted
ann <- data.frame(
  stage = factor(c("unadjusted", "type-adjusted"),
                 levels = c("unadjusted", "type-adjusted")),
  lab = c(sprintf("r = %.2f", r_raw), sprintf("r = %.2f", r_adj)),   # signed; PC1 sign is arbitrary
  y = max(scores$pc1) * 1.06
)

fig <- ggplot(scores, aes(type, pc1, fill = type)) +
  geom_boxplot(width = 0.56, outlier.size = 0.35, outlier.colour = "#777777",
               linewidth = 0.36, colour = "#333333", alpha = 0.9) +
  geom_text(data = ann, aes(x = 1.5, y = y, label = lab), inherit.aes = FALSE,
            family = FONT, size = FS / .pt, colour = "#333333") +
  facet_wrap(~ stage, nrow = 1) +
  scale_fill_manual(values = c(`A-type` = "#6BA6A0", `P-type` = "#C6A85C"), guide = "none") +
  scale_y_continuous(expand = expansion(c(0.05, 0.14))) +
  labs(x = NULL, y = "PC1 score") +
  theme_all(FS) +
  theme(strip.background = element_blank(),
        strip.text = element_text(size = FS, family = FONT, colour = "black"),
        plot.margin = margin(6, 4, 6, 8))

save_fig(fig, "FigureS2", width = 5.85, height = 3.5)
