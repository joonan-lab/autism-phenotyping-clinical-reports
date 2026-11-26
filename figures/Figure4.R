#!/usr/bin/env Rscript
# Figure 4. No reproducible subtype partition and a descriptive distance tail.
#   (a) BIC of Gaussian mixtures minus the best unimodal fit, after adjustment for type or type and length
#   (b) stability of the k-way split: cross-seed ARI, bootstrap ARI and permutation null
#   (c) Mahalanobis distance to the prototype against the Gaussian chi(df) reference
#   (d) tail by subspace: Mardia z and 99th-percentile squared-distance ratio
#   (e) domains carrying individual deviation (|z| > 2)
rm(list = ls())
source("_common.R")

seed <- fromJSON(SEED_JSON)
s4 <- fromJSON(STEP4_JSON)
bp <- fromJSON(BOOT_JSON)

# (a) BIC difference between GMM (k = 1..8, 10 seeds) and the best unimodal fit
# (Gaussian, multivariate t, skew-t family) in PC1-3 at pseudocount 1e-6.
# T = type-adjusted; TL = type + log sentence count.
obs  <- read.delim(BIC_OBS_TSV)
ka   <- 1:8
dcol <- paste0("dbic_k", ka)
obs  <- obs[abs(obs$pseudocount - 1e-6) < 1e-12 & obs$design %in% c("T", "TL"), ]
la <- do.call(rbind, lapply(c("T", "TL"), function(d) {
  r <- obs[obs$design == d, ]
  data.frame(k = ka, dbic = as.numeric(r[1, dcol]),
             design = if (d == "T") "type adjusted" else "type + length adjusted")
}))
la$design <- factor(la$design, levels = c("type adjusted", "type + length adjusted"))
tl <- la[la$design == "type + length adjusted", ]
best <- tl[which.min(tl$dbic), ]
pa <- ggplot() +
  geom_hline(yintercept = 0, linewidth = 0.36) +
  geom_line(data = la, aes(k, dbic, colour = design), linewidth = 0.6) +
  geom_point(data = la, aes(k, dbic, colour = design, shape = design), size = 1.6) +
  annotate("text", x = best$k, y = best$dbic + 60, label = sprintf("k = %d, %.0f", best$k, best$dbic),
           colour = TEXT_RED, size = FIG_ANNOTATION_SIZE, vjust = 0) +
  scale_colour_manual(values = c("type adjusted" = PAL$grey, "type + length adjusted" = PAL$red), name = NULL) +
  scale_shape_manual(values = c("type adjusted" = 15, "type + length adjusted" = 16), name = NULL) +
  scale_x_continuous(breaks = ka) +
  scale_y_continuous(breaks = seq(-1000, 0, 250)) +
  labs(x = "mixture components k", y = "BIC(GMM) −\nBIC(best unimodal)", tag = "a") +
  expand_limits(y = 260) +
  theme_all() + theme(plot.margin = margin(t = 11, r = 4.6, b = 4.6, l = 4.6),
                     legend.position = c(0.99, 1.0), legend.justification = c(1, 1),
                     legend.key.size = unit(0.30, "cm"), legend.background = element_blank())
ks <- 2:8   # panel b k range

# (b) stability: cross-seed ARI + bootstrap ARI + permutation null
seed_m <- vapply(ks, function(k) seed$per_k[[as.character(k)]]$seed_ari_mean, numeric(1))
seed_s <- vapply(ks, function(k) seed$per_k[[as.character(k)]]$seed_ari_sd,   numeric(1))
boot_m <- vapply(as.character(ks), function(k) bp$per_k[[k]]$bootstrap$mean,    numeric(1))
boot_lo<- vapply(as.character(ks), function(k) bp$per_k[[k]]$bootstrap$ci95_lo, numeric(1))
boot_hi<- vapply(as.character(ks), function(k) bp$per_k[[k]]$bootstrap$ci95_hi, numeric(1))
null_lo<- vapply(as.character(ks), function(k) bp$per_k[[k]]$null$ci95_lo,      numeric(1))
null_hi<- vapply(as.character(ks), function(k) bp$per_k[[k]]$null$ci95_hi,      numeric(1))

off <- 0.16
db_seed <- data.frame(k = ks - off, m = seed_m, lo = pmax(0, seed_m - seed_s), hi = pmin(1, seed_m + seed_s), grp = "cross-seed ARI")
db_boot <- data.frame(k = ks + off, m = boot_m, lo = boot_lo, hi = boot_hi, grp = "bootstrap ARI")
db_pts  <- rbind(db_seed, db_boot)
db_pts$grp <- factor(db_pts$grp, levels = c("cross-seed ARI", "bootstrap ARI"))
db_null <- data.frame(k = ks, lo = null_lo, hi = null_hi)

pb <- ggplot() +
  geom_rect(data = db_null, aes(xmin = k - 0.07, xmax = k + 0.07, ymin = lo, ymax = hi), fill = PAL$grey, alpha = 0.6) +
  annotate("text", x = max(ks), y = 0.06, label = "permutation null (95%)",
           colour = TEXT_GREY, size = FIG_ANNOTATION_SIZE, hjust = 1, vjust = 0) +
  geom_hline(yintercept = 0.8, linetype = "dotted", colour = "#444", linewidth = 0.4) +
  annotate("text", x = max(ks) + 0.25, y = 1.06, label = "stable ≥ 0.8", size = FIG_ANNOTATION_SIZE, hjust = 1,
           colour = "#444") +
  geom_errorbar(data = db_pts, aes(x = k, ymin = lo, ymax = hi, colour = grp), width = 0.12, linewidth = 0.45) +
  geom_point(data = db_pts, aes(x = k, y = m, colour = grp), size = 1.7) +
  scale_colour_manual(values = c("cross-seed ARI" = PAL$blue, "bootstrap ARI" = PAL$red), name = NULL) +
  scale_x_continuous(breaks = ks) +
  scale_y_continuous(breaks = seq(0, 1, 0.25)) +
  coord_cartesian(ylim = c(-0.05, 1.20)) +
  labs(x = "mixture components k", y = "ARI vs reference", tag = "b") +
  theme_all() +
  theme(legend.position = c(0.99, 1.02), legend.justification = c(1, 1), legend.direction = "horizontal",
        legend.text = element_text(size = FIG_TEXT_PT), legend.key.size = unit(0.30, "cm"),
        legend.background = element_blank())

# distribution recompute (type-residual ILR -> PCA -> Mahalanobis)
dom <- domain_columns()
df  <- read.delim(TSV_PROP, sep = "\t", check.names = FALSE)
asd <- df[df$asd_label == 1, ]
Pmat <- as.matrix(asd[, dom]); storage.mode(Pmat) <- "double"
rid  <- asd$report_id
ILR  <- ilr_transform(Pmat)
RES  <- residualize_on_ptype(ILR, rid)
pc   <- prcomp(RES, center = TRUE, scale. = FALSE)
scores <- pc$x
ndim <- ncol(scores)
maha2 <- mahalanobis(scores, colMeans(scores), cov(scores))
maha  <- sqrt(maha2)
chi99 <- sqrt(qchisq(0.99, df = ndim))

# (c) Mahalanobis-to-prototype histogram + chi(df) reference
dchi <- function(x, dfree) 2 * x * dchisq(x^2, df = dfree)   # density of chi(df): X=sqrt(chisq)
xg <- seq(0, max(maha) * 1.05, length.out = 400)
ref <- data.frame(x = xg, y = dchi(xg, ndim))
pc_hist <- ggplot(data.frame(m = maha), aes(m)) +
  geom_histogram(aes(y = after_stat(density)), bins = 26, fill = PAL$navy, alpha = 0.80,
                 colour = "white", linewidth = 0.36) +
  geom_line(data = ref, aes(x, y), colour = PAL$red, linewidth = 0.7) +
  geom_vline(xintercept = chi99, colour = PAL$grey, linetype = "dashed", linewidth = 0.4) +
  annotate("text", x = chi99, y = Inf, label = "Gaussian 99%", colour = TEXT_GREY,
           size = FIG_ANNOTATION_SIZE, hjust = -0.05, vjust = 1.4) +
  annotate("text", x = min(ref$x[ref$y >= max(ref$y) * 0.85]) - 0.15, y = max(ref$y) * 0.95,
           label = sprintf("Gaussian ref\nχ(df=%d)", ndim), colour = TEXT_RED,
           size = FIG_ANNOTATION_SIZE, hjust = 1, vjust = 1, lineheight = 0.9) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.05))) +
  labs(x = "Mahalanobis distance to prototype", y = "density", tag = "c") +
  theme_all()

# (d) tail by subspace (twin axis)
MARDIA_COL <- PAL$blue
EXCESS_COL <- PAL$red
keys <- c("pc1_3", "pc4plus"); labk <- c("PC1-3", "PC4+")
z_vals <- vapply(keys, function(k) s4[[k]]$z, numeric(1))
excess <- vapply(keys, function(k) s4[[k]]$excess99, numeric(1))
zlo <- -8; zhi <- max(z_vals) * 1.18
e_scale <- zhi / (max(excess) * 1.1)
e_to_z <- function(e) e * e_scale
z_to_e <- function(z) z / e_scale
dd <- data.frame(grp = rep(labk, 2), metric = rep(c("z", "excess"), each = 2),
                 yval = c(z_vals, e_to_z(excess)),
                 xpos = c(1:2 - 0.18, 1:2 + 0.18),
                 col = rep(c(MARDIA_COL, EXCESS_COL), each = 2))
pd <- ggplot(dd) +
  geom_col(aes(xpos, yval, fill = col), width = 0.34, alpha = 0.85, colour = "white") +
  geom_hline(yintercept = 0, linewidth = 0.36) +
  geom_text(data = data.frame(x = 1:2 - 0.18, z = z_vals),
            aes(x, pmax(z, 0) + 5, label = sprintf("%.1f", z)),
            colour = MARDIA_COL, fontface = "bold", size = FIG_ANNOTATION_SIZE) +
  scale_fill_identity() +
  scale_x_continuous(breaks = 1:2, labels = labk) +
  scale_y_continuous(name = "Mardia z", sec.axis = sec_axis(~ z_to_e(.), name = "99th-percentile\nsquared-distance ratio", breaks = seq(0, 3, by = 0.5))) +
  coord_cartesian(ylim = c(zlo, zhi)) +
  labs(x = NULL, tag = "d") +
  theme_all() +
  theme(axis.title.y.left = element_text(colour = MARDIA_COL),
        axis.text.y.left = element_text(colour = MARDIA_COL),
        axis.title.y.right = element_text(colour = TEXT_RED),
        axis.text.y.right = element_text(colour = TEXT_RED))

# (e) domains carrying individual deviation (|z| > 2)
prof <- read.delim(PROFILE_TSV, sep = "\t", check.names = FALSE)
zcols <- grep("^z_", names(prof), value = TRUE)
frac <- vapply(zcols, function(cc) mean(abs(prof[[cc]]) > 2) * 100, numeric(1))
names(frac) <- sub("^z_", "", zcols)
od  <- order(frac)
ids <- names(frac)[od]
ddv <- data.frame(name = factor(dlabel(ids), levels = dlabel(ids)),
                  val = frac[od], grp = dgroup(ids))
p_dev <- ggplot(ddv, aes(val, name, fill = grp)) +
  geom_col(colour = "white", width = 0.78) +
  geom_vline(xintercept = 4.55, linetype = "dotted", colour = "#333", linewidth = 0.4) +
  annotate("text", x = 4.62, y = 1.5, label = "Gaussian 4.6%", size = FIG_ANNOTATION_SIZE, hjust = 0, colour = "#333") +
  scale_fill_manual(values = GROUP_FILL, name = NULL) +
  scale_x_continuous(expand = expansion(mult = c(0, 0.02))) +
  labs(x = "% of reports with |deviation z| > 2", y = NULL, tag = "e") +
  theme_all() + theme(legend.position = c(0.81, 0.22), legend.text = element_text(size = FIG_TEXT_PT),
                     legend.key.size = unit(0.30, "cm"), axis.text.y = element_text(size = FIG_TEXT_PT),
                     plot.margin = margin(t = 4, r = 4, b = 2, l = 2))

r1 <- wrap_elements(full = pa) + pb + plot_layout(widths = c(1, 1))
r2 <- pc_hist + pd + plot_layout(widths = c(1, 1))
fig <- r1 / r2 / free(p_dev, type = "panel", side = "l") + plot_layout(heights = c(1, 1, 1.32))

save_fig(fig, "Figure4", width = 7.0866, height = 7.85)   # 180 mm wide
