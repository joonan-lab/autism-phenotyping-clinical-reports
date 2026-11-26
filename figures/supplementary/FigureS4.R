#!/usr/bin/env Rscript
# Figure S4. Classifier discrimination and calibration (out-of-fold).
rm(list = ls())
source("_common.R")

oof <- read.delim(OOF_TSV)
audit <- fromJSON(AUDIT_JSON)
y <- oof$asd_label
p <- oof$saved_probability
stopifnot(!anyNA(y), !anyNA(p), all(y %in% c(0, 1)))

# (a) ROC: one point per distinct threshold, ties handled together
thr <- sort(unique(p), decreasing = TRUE)
tpr <- c(0, vapply(thr, function(t) mean(p[y == 1] >= t), numeric(1)))
fpr <- c(0, vapply(thr, function(t) mean(p[y == 0] >= t), numeric(1)))
pa <- ggplot(data.frame(fpr, tpr), aes(fpr, tpr)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", colour = PAL$grey, linewidth = 0.36) +
  geom_path(colour = PAL$blue, linewidth = 0.6) +
  annotate("text", x = 0.98, y = 0.04, hjust = 1, vjust = 0, size = SI_TEXT_PT / .pt,
           label = sprintf("AUROC %.3f", audit$pooled$AUROC_saved_probability)) +
  coord_equal(xlim = c(0, 1), ylim = c(0, 1)) +
  labs(x = "False-positive rate", y = "True-positive rate", tag = "a") +
  theme_all(SI_TEXT_PT)

# (b) calibration by decile of the out-of-fold probability
br <- unique(quantile(p, seq(0, 1, 0.1), names = FALSE))
bin <- cut(p, br, include.lowest = TRUE)
cal <- data.frame(p = tapply(p, bin, mean), y = tapply(y, bin, mean))
pb <- ggplot(cal, aes(p, y)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", colour = PAL$grey, linewidth = 0.36) +
  geom_line(colour = PAL$blue, linewidth = 0.6) +
  geom_point(colour = PAL$blue, size = 1.8) +
  annotate("text", x = 0.98, y = 0.03, hjust = 1, vjust = 0, size = SI_TEXT_PT / .pt, lineheight = 0.9,
           label = sprintf("Brier %.3f\nCalibration slope %.3f", audit$pooled$Brier, audit$pooled$calibration_slope)) +
  coord_equal(xlim = c(0, 1), ylim = c(0, 1)) +
  labs(x = "Mean out-of-fold probability", y = "Observed ASD fraction", tag = "b") +
  theme_all(SI_TEXT_PT)

fig <- pa + pb
save_fig(fig, "FigureS4", width = 7.0866, height = 3.35)
