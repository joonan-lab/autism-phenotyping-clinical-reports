#!/usr/bin/env Rscript
# Table S2. Silver–gold agreement by domain at the sentence and report levels
rm(list = ls())
source("_common.R")

gold <- Filter(Negate(is.null), read_jsonl(P_GOLD))
reps <- sort(unique(vapply(gold, function(r) r$report_id, character(1))))
silver_lines <- readLines(P_SILVER, warn = FALSE)
keep <- Reduce(`|`, lapply(reps, function(id) grepl(id, silver_lines, fixed = TRUE)))
silver <- Filter(function(r) !is.null(r) && r$report_id %in% reps,
                 lapply(silver_lines[keep], fromJSON, simplifyVector = FALSE))
rm(silver_lines)

get_domains <- function(labels) {
  if (is.null(labels) || length(labels) == 0) return(character(0))
  vapply(labels, function(x) if (is.list(x)) as.character(x$domain_id) else as.character(x), character(1))
}
ndom <- length(doms)
tp <- fp <- fn <- support <- setNames(numeric(ndom), doms)
for (rid in reps) {
  sr <- Filter(function(r) r$report_id == rid, silver)
  gr <- Filter(function(r) r$report_id == rid, gold)
  sr <- sr[order(vapply(sr, function(r) r$sentence_idx, numeric(1)))]
  gr <- gr[order(vapply(gr, function(r) r$sentence_idx, numeric(1)))]
  stopifnot(length(sr) == length(gr),
            identical(vapply(sr, function(r) trimws(r$sentence), character(1)),
                      vapply(gr, function(r) trimws(r$sentence), character(1))))
  sb <- do.call(rbind, lapply(sr, function(r) doms %in% get_domains(r$labels)))
  gb <- do.call(rbind, lapply(gr, function(r) doms %in% get_domains(r$labels)))
  tp <- tp + colSums(sb & gb)
  fp <- fp + colSums(sb & !gb)
  fn <- fn + colSums(!sb & gb)
  support <- support + colSums(gb)
}

report <- read_tsv(P_GOLD_DOMAIN)
x <- data.frame(code = codes, domain = unname(display_of[doms]),
                sentence_support = unname(support),
                sentence_precision = round(unname(tp / pmax(tp + fp, 1)), 3),
                sentence_recall = round(unname(tp / pmax(tp + fn, 1)), 3),
                sentence_f1 = round(unname(2 * tp / pmax(2 * tp + fp + fn, 1)), 3),
                report_spearman = round(report$spearman[match(doms, report$domain)], 3))
stopifnot(!anyNA(x$report_spearman))
x$domain[x$domain == "Intellectual functioning and learning skills"] <- "Intellectual functioning/learning skills"

group <- substr(x$code, 1, 2)
median_rows <- data.frame(code = NA_character_, domain = paste(group_of, "median"),
                          sentence_support = NA_real_, sentence_precision = NA_real_, sentence_recall = NA_real_,
                          sentence_f1 = vapply(names(group_of), function(g) median(x$sentence_f1[group == g]), numeric(1)),
                          report_spearman = vapply(names(group_of), function(g) median(x$report_spearman[group == g]), numeric(1)))
save_table(2,
  rbind(x, median_rows),
  c("Code", "Domain or group summary", "Gold sentence support", "Sentence precision",
    "Sentence recall", "Sentence F1", "Report Spearman ρ"))
