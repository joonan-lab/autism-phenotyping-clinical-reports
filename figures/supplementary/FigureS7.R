#!/usr/bin/env Rscript
# Figure S7. Full-space and subspace Mahalanobis distances.
rm(list = ls())
source("_common.R")

K_LEAD <- 3                                   # leading subspace = PC1..PC3 (same split as Figure 4d)

dom  <- domain_columns()
df   <- read.delim(TSV_PROP, sep = "\t", check.names = FALSE)
asd  <- df[df$asd_label == 1, ]
Pmat <- as.matrix(asd[, dom]); storage.mode(Pmat) <- "double"
rid  <- asd$report_id

# type-residual ILR -> PCA (same as Figure 4)
ILR    <- ilr_transform(Pmat)
RES    <- residualize_on_ptype(ILR, rid)
pc     <- prcomp(RES, center = TRUE, scale. = FALSE)
scores <- pc$x
p      <- ncol(scores)
lead   <- scores[, 1:K_LEAD, drop = FALSE]
resid  <- scores[, (K_LEAD + 1):p, drop = FALSE]

# per-subspace Mahalanobis (PC scores are uncorrelated -> blocks separate)
d_lead <- sqrt(mahalanobis(lead,  colMeans(lead),  cov(lead)))
d_res  <- sqrt(mahalanobis(resid, colMeans(resid), cov(resid)))
d_full <- sqrt(mahalanobis(scores, colMeans(scores), cov(scores)))

# Gaussian reference: under normality each subspace distance^2 ~ chi-square(df)
ref_res  <- sqrt(qchisq(0.99, df = ncol(resid)))
ref_lead <- sqrt(qchisq(0.99, df = K_LEAD))

dd <- data.frame(d_lead = d_lead, d_res = d_res, d_full = d_full, rid = rid)

# Same illustrative report as Figure 5c and d.
EX <- fromJSON(EXEMPLAR_JSON)$chosen_report
ex <- dd[dd$rid == EX, ][1, ]

# (a) chi-square QQ of full Mahalanobis distance
maha2 <- d_full^2
ndim  <- p
nn    <- length(maha2)
qq    <- data.frame(q = qchisq((seq_len(nn) - 0.5) / nn, df = ndim), d2 = sort(maha2))
ks    <- suppressWarnings(ks.test(maha2, "pchisq", df = ndim))
lim   <- max(qq$q, qq$d2)
pp_qq <- ggplot(qq, aes(q, d2)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", colour = PAL$grey, linewidth = 0.4) +
  geom_point(size = 1.4, alpha = 0.85, stroke = 0, colour = PAL$blue) +
  annotate("text", x = lim, y = 0,
           label = sprintf("KS D = %.3f", ks$statistic),
           family = FONT, size = SI_TEXT_PT / .pt, hjust = 1, vjust = 0, lineheight = 0.95, colour = "#333") +
  coord_equal(xlim = c(0, lim), ylim = c(0, lim)) +
  labs(x = sprintf("theoretical chi-square quantile (df = %d)", ndim),
       y = "observed squared Mahalanobis distance", tag = "a") +
  theme_all(SI_TEXT_PT)

# (b) leading versus residual distance
pp_dd <- ggplot(dd, aes(d_lead, d_res)) +
  geom_hline(yintercept = ref_res, linetype = "dashed", colour = PAL$red, linewidth = 0.4) +
  geom_vline(xintercept = ref_lead, linetype = "dotted", colour = PAL$grey, linewidth = 0.36) +
  geom_point(aes(colour = d_full), size = 1.5, alpha = 0.85, stroke = 0) +
  geom_point(data = ex, shape = 21, size = 2.6, stroke = 0.6,
             colour = "black", fill = NA) +
  annotate("text", x = min(dd$d_lead) - 0.2, y = max(dd$d_res) * 0.82, label = "example report (Figure 5c, d)",
           family = FONT, size = SI_TEXT_PT / .pt, colour = "black", hjust = 0) +
  annotate("text", x = ref_lead - 0.15, y = max(dd$d_res) + 1.70,
           label = "leading\nGaussian 99%", family = FONT, size = SI_TEXT_PT / .pt,
           colour = "#535D69", hjust = 1, vjust = 1, lineheight = 0.9) +
  annotate("text", x = max(dd$d_lead) * 1.45, y = ref_res + 0.2,
           label = "residual\nGaussian 99%", family = FONT,
           size = SI_TEXT_PT / .pt, colour = "#9E473C", hjust = 1, vjust = 0, lineheight = 0.9) +
  scale_colour_gradientn(colours = c(PAL$gold, PAL$coral, PAL$navy),
                         name = "Layer 1\ndistance") +
  labs(x = "leading-subspace distance (PC1-3)",
       y = "residual-subspace distance (PC4-18)", tag = "b") +
  coord_cartesian(xlim = c(0, max(dd$d_lead) * 1.45),
                  ylim = c(0, max(dd$d_res) * 1.13), clip = "on") +
  theme_all(SI_TEXT_PT) +
  theme(aspect.ratio = 1,
        legend.position = "right", legend.direction = "vertical",
        legend.key.height = unit(0.34, "cm"), legend.key.width = unit(0.30, "cm"))

fig <- pp_qq + pp_dd + plot_layout(widths = c(1, 1))
save_fig(fig, "FigureS7", width = 7.0866, height = 3.2)
