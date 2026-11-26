#!/usr/bin/env Rscript
# Figure S6. Cross-seed stability of Gaussian mixtures in A-type reports only.
rm(list = ls())
source("_common.R")

d   <- jsonlite::fromJSON(ATYPE_JSON, simplifyVector = FALSE)
per_k <- d$per_k
ks  <- sort(as.integer(names(per_k)))
ari <- vapply(ks, function(k) as.numeric(per_k[[as.character(k)]]$seed_ari), numeric(1))

df <- data.frame(k = factor(ks), ari = ari)

fig <- ggplot(df, aes(k, ari)) +
  geom_col(width = 0.78, fill = PAL$blue, colour = "white", alpha = 0.9) +
  geom_hline(yintercept = 0.8, linetype = "dotted", colour = "#444444", linewidth = 0.36) +
  annotate("text", x = length(ks) - 0.3, y = 0.82, label = "stable >= 0.8",
           hjust = 0, vjust = 0, size = SI_TEXT_PT / .pt, family = FONT, colour = "#444444") +
  scale_y_continuous(limits = c(0, 1), expand = expansion(c(0, 0))) +
  labs(x = expression("mixture components " * italic(k)), y = "cross-seed ARI") +
  theme_all(SI_TEXT_PT) +
  theme(plot.margin = margin(8, 8, 6, 6))

save_fig(fig, "FigureS6", width = 7.0866, height = 3.5)
