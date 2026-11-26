#!/usr/bin/env Rscript
# Figure 2. Agreement between silver and gold domain proportions across 26 reports.
rm(list = ls())
source("_common.R")

src <- read.delim(GOLD_DOMAIN_TSV,
                  check.names = FALSE)
dom <- domain_columns()
stopifnot(setequal(src$domain, dom))
codes <- c(paste0("CO", 1:7), paste0("AS", 1:9), paste0("RE", seq_len(length(dom) - 16)))
rho <- setNames(src$spearman, src$domain)[dom]
rho[as.logical(src$constant_column[match(dom, src$domain)])] <- NA

map <- data.frame(code = codes, name = dlabel(dom), r = as.numeric(rho), stringsAsFactors = FALSE)
map$group <- factor(substr(map$code, 1, 2), levels = c("CO", "AS", "RE"),
                    labels = c("Core ASD\ndomains", "Associated and\nco-occurring\nfeatures",
                               "Report\nelements\nand other"))
map$label <- factor(paste(map$code, map$name), levels = rev(paste(map$code, map$name)))
map$rv  <- ifelse(is.na(map$r), 0, map$r)
map$txt <- ifelse(is.na(map$r), "n/a", sprintf("%.2f", map$r))

BLUE <- "#35618F"; MIDW <- "#F7F7F7"; RED <- "#C15B4E"
p <- ggplot(map, aes(x = rv, y = label)) +
  geom_vline(xintercept = 0, colour = "#c2c8d2", linewidth = 0.4) +
  geom_segment(aes(x = 0, xend = rv, yend = label, colour = r), linewidth = 1.1, lineend = "round") +
  geom_point(aes(fill = r), shape = 21, size = 3.0, stroke = 0.4, colour = "grey35") +
  geom_text(aes(label = txt, hjust = ifelse(rv < 0, 1.3, -0.3)), size = FIG_ANNOTATION_SIZE, colour = "#2b3a52",
            fontface = "bold") +
  facet_grid(rows = vars(group), scales = "free_y", space = "free_y", switch = "y") +
  scale_fill_gradient2(name = "Spearman ρ", low = BLUE, mid = MIDW, high = RED, midpoint = 0,
                       limits = c(-0.5, 1.0), breaks = c(-0.5, 0, 0.5, 1.0), na.value = "#c9cdd4") +
  scale_colour_gradient2(low = BLUE, mid = MIDW, high = RED, midpoint = 0,
                         limits = c(-0.5, 1.0), na.value = "#c9cdd4", guide = "none") +
  scale_x_continuous(limits = c(-0.75, 1.15), breaks = c(-0.5, 0, 0.5, 1.0)) +
  labs(x = "Spearman ρ between silver and gold label proportions (26 reports)", y = NULL) +
  guides(fill = guide_colourbar(display = "rectangles", nbin = 256,
                                barwidth = unit(30, "mm"), barheight = unit(3, "mm"),
                                title.position = "left", title.vjust = 1)) +
  theme_all() +
  theme(legend.position = "top", legend.justification = "center",
        strip.placement = "outside",
        strip.text.y.left = element_text(angle = 90, face = "bold", size = FIG_TEXT_PT, colour = "#16233d"),
        axis.text.y = element_text(size = FIG_TEXT_PT, colour = "#1d2b40"),
        axis.title.x = element_text(size = FIG_TEXT_PT, margin = margin(t = 5)),
        axis.line.y = element_blank(), axis.ticks.y = element_blank(),
        plot.margin = margin(4, 8, 4, 4))

save_fig(p, "Figure2", width = 7.0866, height = 7.6)
