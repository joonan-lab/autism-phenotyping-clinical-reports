#!/usr/bin/env Rscript
# Figure 3. Dimensionality of the type-adjusted ASD phenotype space.
#   (a) per-component variance against the uniform reference
#   (b) cumulative variance and effective dimensionality
#   (c) share of phenotype variance by domain
rm(list = ls())
source("_common.R")

dom <- domain_columns()
df  <- read.delim(TSV_PROP, sep = "\t", check.names = FALSE)
sub <- df[df$asd_label == 1, ]
P   <- as.matrix(sub[, dom]); storage.mode(P) <- "double"
rid <- sub$report_id

ILR <- ilr_transform(P)
RES <- residualize_on_ptype(ILR, rid)
pca_evr <- function(M) { p <- prcomp(M, center = TRUE, scale. = FALSE); ev <- (p$sdev^2)[seq_len(ncol(M))]; ev / sum(ev) }
evr <- pca_evr(RES)
ed  <- eff_dim(evr)
ND  <- length(evr)
uniform <- 1 / ND
x <- seq_along(evr)

# (a) scree
pa <- ggplot(data.frame(x = x, v = evr * 100), aes(x, v)) +
  geom_col(fill = PAL$blue, alpha = 0.85, colour = "white") +
  geom_hline(yintercept = uniform * 100, linetype = "dashed", colour = "#888", linewidth = 0.5) +
  annotate("text", x = 12, y = max(evr * 100) * 0.9,
           label = sprintf("PC1 = %.1f%%", evr[1] * 100), colour = TEXT_RED, fontface = "bold", size = FIG_ANNOTATION_SIZE) +
  annotate("text", x = 13.5, y = uniform * 100 + 1.2,
           label = sprintf("uniform (1/%d = %.1f%%)", ND, uniform * 100), colour = TEXT_GREY, size = FIG_ANNOTATION_SIZE) +
  scale_x_continuous(breaks = seq(2, max(x), by = 2)) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.05))) +   # bars flush to the x-axis
  labs(x = "principal component", y = "variance explained (%)", tag = "a") +
  theme_all()

# (b) cumulative
cum <- data.frame(x = x, ctrl = cumsum(evr) * 100, unif = x * uniform * 100)
pb <- ggplot(cum, aes(x)) +
  geom_line(aes(y = unif), colour = "#888", linetype = "dashed", linewidth = 0.5) +
  geom_hline(yintercept = 80, linetype = "dotted", colour = PAL$red, linewidth = 0.5) +
  geom_line(aes(y = ctrl), colour = PAL$blue, linewidth = 0.7) +
  geom_point(aes(y = ctrl), colour = PAL$blue, size = 1.6) +
  annotate("text", x = 18, y = 28, label = sprintf("effective dimensionality\n= %.1f / %d", ed, ND),
           colour = TEXT_RED, fontface = "bold", size = FIG_ANNOTATION_SIZE, hjust = 1, lineheight = 1) +
  scale_x_continuous(breaks = seq(2, max(x), by = 2)) +
  scale_y_continuous(breaks = seq(0, 100, 20)) +
  labs(x = "number of PCs", y = "cumulative variance (%)", tag = "b") +
  theme_all()

# (c) per-domain variance share, coloured by domain group
J   <- fromJSON(STEP11_JSON)
pcd <- J$per_domain_clr
pcd <- pcd[order(pcd$contribution_pct), ]
pcd$label <- factor(dlabel(pcd$domain), levels = dlabel(pcd$domain))
grp  <- unname(DOMAIN_GROUP[pcd$domain])   # map groups by domain id
stopifnot(all(grp %in% c("CO", "AS", "RE")), length(grp) == 19)
gt   <- tapply(pcd$contribution_pct, grp, sum)            # per-group variance total
glab <- sprintf("%s (%d%%)", GROUP_LABEL, round(gt[c("CO", "AS", "RE")]))
# wrap the long group-AS legend label so it does not run off the panel
glab[2] <- sub("co-occurring ", "co-occurring\n", glab[2])
pcd$grp <- factor(grp, levels = c("CO", "AS", "RE"), labels = glab)
gcols <- setNames(unname(GROUP_PAL), glab)
pc <- ggplot(pcd, aes(contribution_pct, label, fill = grp)) +
  geom_col(width = 0.72, colour = "white", linewidth = 0.36) +
  scale_fill_manual(values = gcols, name = NULL) +
  scale_x_continuous(expand = expansion(mult = c(0, 0.08))) +
  labs(x = "share of ASD phenotype variance (%)", y = NULL, tag = "c") +
  theme_all() + theme(legend.position = c(0.99, 0.22), legend.justification = c(1, 0.5),
                     legend.text = element_text(size = FIG_TEXT_PT),
                     legend.key.size = unit(0.30, "cm"), axis.text.y = element_text(size = FIG_TEXT_PT))

fig <- (pa + pb) / patchwork::free(pc, type = "panel", side = "l") + patchwork::plot_layout(heights = c(1, 1.5))
save_fig(fig, "Figure3", width = 7.0866, height = 6.3)
