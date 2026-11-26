#!/usr/bin/env Rscript
# Figure 5. Two-layer fingerprint with classifier-derived directional annotation.
#   (a) mean domain composition of the 346 ASD reports
#   (b) forced two-way split variance decomposition with a single-Gaussian split reference
#   (c) exemplar: Layer 1 distance on the cohort scale and Layer 2 deviation z
#   (d) same exemplar, Layer 2 emphasis versus signed domain logit contribution E
rm(list = ls())
source("_common.R")

dom <- domain_columns()
df  <- read.delim(TSV_PROP, sep = "\t", check.names = FALSE)
asd <- df[df$asd_label == 1, ]
Pmat <- as.matrix(asd[, dom]); storage.mode(Pmat) <- "double"
ex_id <- fromJSON(EXEMPLAR_JSON)$chosen_report

# (a) mean composition
proto <- colMeans(Pmat)
oa <- order(proto)
da <- data.frame(name = factor(dlabel(dom[oa]), levels = dlabel(dom[oa])),
                 val = proto[oa] * 100, grp = dgroup(dom[oa]))
pa <- ggplot(da, aes(val, name, fill = grp)) +
  geom_col(colour = "white", width = 0.78) +
  geom_text(aes(label = sprintf("%.1f", val)), hjust = -0.15, size = FIG_ANNOTATION_SIZE) +
  scale_fill_manual(values = GROUP_FILL,
                    labels = function(x) vapply(x, function(s)
                      paste(strwrap(s, width = 15), collapse = "\n"), character(1)),
                    name = NULL) +
  scale_x_continuous(expand = expansion(mult = c(0, 0.22))) +
  labs(x = "Mean domain composition (%)", y = NULL, tag = "a") +
  theme_all() + theme(legend.position = c(0.99, 0.23), legend.justification = c(1, 0.5),
                     legend.key.size = unit(0.30, "cm"), legend.spacing.y = unit(0.05, "cm"))

# (b) forced two-way variance decomposition
k2 <- fromJSON(VAR_JSON)$residual_GMM_k2_core_vs_periphery
between <- k2$full_eta2_between
RES_b  <- residualize_on_ptype(ilr_transform(Pmat), asd$report_id)
ev_b   <- prcomp(RES_b, center = TRUE, scale. = FALSE)$sdev^2
pc1_b  <- ev_b[1] / sum(ev_b)
gauss_ref <- (2 / pi) * pc1_b
xlab_b <- "2-way split"

db <- data.frame(grp = xlab_b, part = factor(c("within", "between"), levels = c("within", "between")),
                 val = c(1 - between, between))
pb <- ggplot(db, aes(grp, val, fill = part)) +
  geom_col(width = 1.02) +
  annotate("text", x = 1, y = between / 2, label = sprintf("%.1f%%", between * 100),
           fontface = "bold", size = FIG_ANNOTATION_SIZE, colour = "black") +
  annotate("text", x = 1, y = between + (1 - between) / 2, label = sprintf("%.1f%%", (1 - between) * 100),
           fontface = "bold", size = FIG_ANNOTATION_SIZE) +
  geom_hline(yintercept = gauss_ref, linetype = "dashed", colour = "#444", linewidth = 0.4) +
  annotate("text", x = 1.67, y = gauss_ref + 0.03, label = sprintf("one Gaussian\nsplit at PC1\nmean %.1f%%", gauss_ref * 100),
           size = FIG_ANNOTATION_SIZE, hjust = 0, vjust = 0, lineheight = 0.9, colour = "#444") +
  scale_fill_manual(values = c(between = PAL$red, within = PAL$panel), breaks = c("between", "within"),
                    labels = c(between = "between groups", within = "within groups"), name = NULL) +
  scale_y_continuous(limits = c(0, 1), expand = expansion(mult = c(0, 0.05))) +
  coord_cartesian(xlim = c(0.6, 2.05), clip = "off") +
  labs(x = NULL, y = "ASD variance share", tag = "b") +
  theme_all() + theme(legend.position = "right", legend.key.size = unit(0.30, "cm"))

fp <- read.delim(FP_TSV, sep = "\t", check.names = FALSE)

# (c, d) exemplar domains, ordered by Layer 2 z
ex <- read.delim(EXEMPLAR_TSV, sep = "\t", check.names = FALSE)
ex$reliable <- as.logical(ex$reliable)
stopifnot(nrow(ex) == length(dom), all(ex$report_id == ex_id),
          setequal(ex$domain, dom), !anyNA(ex$reliable), all(is.finite(ex$E)))
ex <- ex[order(ex$layer2_z), ]
lab <- paste0(dlabel(ex$domain), ifelse(ex$reliable, "", " †"))
ex$name <- factor(lab, levels = lab)
ex$grp  <- dgroup(ex$domain)
ex$y    <- seq_len(nrow(ex))
ybreaks <- scale_y_continuous(breaks = ex$y, labels = lab, expand = expansion(add = c(0.6, 0.6)))
d_ex  <- fp$distance[fp$report_id == ex_id]
pct   <- mean(fp$distance <= d_ex) * 100

# (c) top: Layer 1 distance of the exemplar on the cohort scale
d_mean <- mean(fp$distance); d_q90 <- quantile(fp$distance, 0.9, names = FALSE); d_max <- max(fp$distance)
ticks <- data.frame(x = c(0, d_mean, d_q90, d_max),
                    lab = c("0 prototype", sprintf("mean %.2f", d_mean), sprintf("top-10%% tail \u2265 %.2f", d_q90),
                            sprintf("max %.2f", d_max)),
                    hj = c(0, 0.5, 0, 1), y = c(-0.31, 0.24, -0.31, 0.24), vj = c(1, 0, 1, 0))
pr <- ggplot() +
  annotate("rect", xmin = d_q90, xmax = d_max, ymin = -0.12, ymax = 0.12, fill = PAL$red, alpha = 0.15) +
  annotate("segment", x = 0, xend = d_max, y = 0, yend = 0, colour = "grey45", linewidth = 0.5) +
  geom_segment(data = ticks, aes(x = x, xend = x, y = -0.12, yend = 0.12), colour = "grey45", linewidth = 0.5) +
  geom_text(data = ticks, aes(x = x, y = y, label = lab, hjust = hj, vjust = vj), size = FIG_ANNOTATION_SIZE, colour = "grey30") +
  annotate("point", x = d_ex, y = 0.32, shape = 25, size = 2.6, fill = PAL$red, colour = PAL$red) +
  annotate("text", x = d_max / 2, y = 0.87, label = sprintf("This report: d = %.2f (%.0fth percentile)", d_ex, pct),
           fontface = "bold", size = FIG_ANNOTATION_SIZE, vjust = 0) +
  annotate("text", x = d_max / 2, y = -0.99, label = "Mahalanobis distance from prototype", size = FIG_ANNOTATION_SIZE,
           colour = "grey40", vjust = 1) +
  scale_x_continuous(limits = c(0, d_max)) +
  scale_y_continuous(limits = c(-1.33, 1.16)) +
  theme_void(base_family = FONT) +
  theme(plot.tag = element_text(size = FIG_TEXT_PT, face = "bold", family = FONT), plot.margin = margin(2, 6, 0, 5.5))

# (c) bottom: Layer 2 deviation z, coloured by domain group;
# dagger marks domains with low agreement between silver and gold labels. Values are printed where |z| > 2,
# outside positive bars and inside the left end of negative bars.
ex$big  <- abs(ex$layer2_z) > 2
pd <- ggplot(ex, aes(layer2_z, y, fill = grp)) +
  geom_col(colour = "white", width = 0.78, orientation = "y") +
  geom_vline(xintercept = 0, linewidth = 0.36) +
  geom_vline(xintercept = c(-2, 2), linetype = "dotted", colour = "#666", linewidth = 0.36) +
  geom_text(data = ex[ex$big, ], aes(x = layer2_z + 0.1, y = y, label = sprintf("%+.2f", layer2_z), hjust = 0),
            inherit.aes = FALSE, size = FIG_ANNOTATION_SIZE, fontface = "bold") +
  scale_fill_manual(values = GROUP_FILL, name = NULL, guide = "none") +
  scale_x_continuous(expand = expansion(mult = c(0.04, 0.22))) +
  ybreaks +
  coord_cartesian(clip = "off") +
  labs(x = "Layer 2 deviation z", y = NULL) +
  theme_all()

# (d) recorded domains: Layer 2 deviation versus additive logit contribution.
de <- ex[ex$n_sentences > 0 & is.finite(ex$E), ]
n_nr <- sum(ex$n_sentences == 0)
short <- c(social_emotional_reciprocity = "Social-emotional\nreciprocity",
           nonverbal_communication = "Nonverbal\ncommunication",
           relationship_play = "Social\nrelationships",
           stereotyped_behavior = "Stereotyped\nbehavior",
           restricted_interests = "Restricted\ninterests",
           sensory_processing = "Sensory\nreactivity",
           language_skills = "Language\nskills",
           physiological_function = "Physiological\nfunction",
           adaptive_behavior = "Adaptive\nbehavior",
           motor_skills = "Motor skills",
           other_general = "Other/general")
stopifnot(setequal(de$domain, names(short)))
de$short <- paste0(unname(short[de$domain]), ifelse(de$reliable, "", "\u2020"),
                   ifelse(de$n_sentences < 3, "\u2021", ""))
label_values <- c("social_emotional_reciprocity", "motor_skills")
de$label <- paste0(de$short, ifelse(de$domain %in% label_values, sprintf("\nE %+.2f", de$E), ""))
label_positions <- data.frame(
  domain = names(short),
  lx = c(2.65, 1.40, 0.55, 2.75, 2.65, 0.55, 2.65, 2.65, 0.55, 0.55, 1.55),
  ly = c(1.18, 1.18, 0.40, 0.62, -0.16, 0.76, -0.47, 0.88, 0.06, -0.54, -0.17))
de <- merge(de, label_positions, by = "domain")
pe <- ggplot(de, aes(layer2_z, E)) +
  annotate("rect", xmin = -Inf, xmax = Inf, ymin = 0, ymax = Inf,
           fill = PAL$red, alpha = 0.06) +
  annotate("rect", xmin = -Inf, xmax = Inf, ymin = -Inf, ymax = 0,
           fill = PAL$blue, alpha = 0.06) +
  geom_hline(yintercept = 0, linewidth = 0.36) +
  geom_vline(xintercept = 2, linetype = "dotted", colour = "#666", linewidth = 0.36) +
  geom_segment(aes(xend = lx, yend = ly), linewidth = 0.3, colour = "grey45") +
  geom_point(aes(fill = grp), size = 2.8, shape = 21, colour = "grey20", stroke = 0.4) +
  geom_label(aes(lx, ly, label = label), size = 8.5 / .pt, lineheight = 0.9,
             linewidth = 0, label.padding = unit(0.04, "lines"), fill = "white") +
  annotate("text", x = 3.28, y = 1.60, label = "Positive logit contribution", hjust = 1,
           size = FIG_ANNOTATION_SIZE, colour = TEXT_RED) +
  annotate("text", x = 3.28, y = -0.92, label = "Negative logit contribution", hjust = 1,
           size = FIG_ANNOTATION_SIZE, colour = PAL$blue) +
  scale_fill_manual(values = GROUP_FILL, guide = "none") +
  scale_x_continuous(limits = c(0, 3.35), breaks = c(0, 1, 2, 3),
                     expand = expansion(add = c(0.04, 0.04))) +
  scale_y_continuous(limits = c(-1.00, 1.72), breaks = c(-0.5, 0, 0.5, 1, 1.5)) +
  labs(x = "Layer 2 deviation z (emphasis)", y = "Domain logit contribution E", tag = "d",
       caption = sprintf("‡ 1–2 assigned sentences
%d unrecorded domains: E = 0 (not shown)", n_nr)) +
  theme_all() + theme(plot.caption = element_text(size = FIG_TEXT_PT, hjust = 0, colour = "#555"))

row1 <- pa + pb + plot_layout(widths = c(1.35, 1))

pc <- free(pr, type = "panel", side = "l") / pd + plot_layout(heights = c(0.29, 1))
no_margin <- theme(plot.margin = margin(0, 0, 0, 0))
row2 <- wrap_elements(full = pc) + labs(tag = "c") + no_margin +
  theme(plot.tag = element_text(size = FIG_TEXT_PT, face = "bold", family = FONT)) + pe + plot_layout(widths = c(1.35, 1))

fig <- row1 / (wrap_elements(full = row2) + no_margin) + plot_layout(heights = c(0.8, 1))
save_fig(fig, "Figure5", width = 7.0866, height = 8.3)
