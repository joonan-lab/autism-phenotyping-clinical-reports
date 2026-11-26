#!/usr/bin/env Rscript
# Shared setup for the supplementary-table scripts.
# Requires jsonlite, openxlsx, xml2 and zip.
suppressMessages(library(jsonlite))

.script_dir <- function() {
  a <- commandArgs(trailingOnly = FALSE)
  f <- gsub("~+~", " ", sub("^--file=", "", a[grep("^--file=", a)]), fixed = TRUE)
  if (length(f)) dirname(normalizePath(f[1])) else getwd()
}
SCRIPT_DIR <- .script_dir()

TS_DIR <- normalizePath(Sys.getenv(
  "REPORT_LLM_TABLE_SOURCE_DIR",
  unset = "~/Dropbox/ReportLLM_analysis/table_source"
), mustWork = TRUE)
ts <- function(f) file.path(TS_DIR, f)
args <- commandArgs(TRUE)
if (length(args) > 1L) stop("Usage: Rscript <script>.R [output]")
OUT <- if (length(args)) args[1] else
  Sys.getenv("REPORT_LLM_SUPPLEMENTARY_TABLE_OUT_DIR", unset = dirname(SCRIPT_DIR))

P_PSEUDOCOUNT <- ts("step2_pseudocount_sensitivity_results.json")
P_BIC_OBS     <- ts("observed_bic.tsv")
P_STEP6       <- ts("step5_seed_stability_results.json")
P_BOOT        <- ts("bootstrap_permutation_results.json")
P_CLF_METRICS <- ts("baseline_metrics.tsv")
P_CLF_FOLDS   <- ts("per_fold_metrics.tsv")
P_CLF_PAIRED  <- ts("paired_model_differences.tsv")
P_FOLD_QUALITY <- ts("fold_quality.tsv")
P_DEL_ASSOC   <- ts("domain_associations.tsv")
P_HELDOUT     <- ts("heldout_model_comparisons.tsv")
P_DROP        <- ts("sensitivity_drop_domains_results.json")
P_DOWNWEIGHT  <- ts("sensitivity_downweight_results.json")
P_HEAVYTAIL   <- ts("sensitivity_heavytail_results.json")
P_FINGERPRINT <- ts("sensitivity_fingerprint_results.json")
P_PROFILE_BASE <- ts("profile_baselines.json")
P_PROFILE_RHO <- ts("profile_spearman_round12.json")
P_SILVER      <- ts("silver_labels_489reports.jsonl")
P_GOLD        <- ts("gold_labels_26reports.jsonl")
P_GOLD_DOMAIN <- ts("per_domain_all_sentences.tsv")
P_SELECTION_GOLD <- Sys.getenv(
  "REPORT_LLM_GOLD_SELECTION_SUMMARY",
  unset = ts("gold_sentence_selection_summary.tsv"))
P_PROFILES    <- ts("controlled_individual_profiles.tsv")
P_FINGERPRINTS <- ts("two_channel_fingerprints.tsv")
P_META        <- ts("domain_vectors_meta_latest.json")

# 19-domain order, codes and reader-facing names
doms <- fromJSON(P_META)$domain_columns
codes <- c(paste0("CO", 1:7), paste0("AS", 1:9), paste0("RE", seq_len(length(doms) - 16)))
display_of <- c(
  social_emotional_reciprocity = "Social-emotional reciprocity",
  nonverbal_communication      = "Nonverbal communication",
  relationship_play            = "Social relationships",
  stereotyped_behavior         = "Stereotyped behavior",
  insistence_on_sameness       = "Insistence on sameness",
  restricted_interests         = "Restricted interests",
  sensory_processing           = "Sensory reactivity",
  externalizing                = "Externalizing behaviors",
  internalizing                = "Internalizing behaviors",
  language_skills              = "Language skills",
  physiological_function       = "Physiological function",
  adaptive_behavior            = "Adaptive behavior",
  intelligence_learning        = "Intellectual functioning and learning skills",
  executive_function           = "Executive function",
  motor_skills                 = "Motor skills",
  family_environment           = "Family environment",
  test_scores                  = "Test scores",
  other_general                = "Other/general",
  recommendations              = "Recommendations")
stopifnot(identical(doms, names(display_of)))
group_of <- c(CO = "Core ASD domains",
              AS = "Associated and co-occurring features",
              RE = "Report elements and other")
classifier_names <- c(v4_1 = "ASD classifier", char_tfidf_lr = "Character TF-IDF",
                      word_tfidf_lr = "Word TF-IDF")

read_tsv <- function(p) read.delim(p, check.names = FALSE, stringsAsFactors = FALSE)
read_json <- function(p) fromJSON(p, simplifyVector = FALSE)
read_jsonl <- function(p) lapply(readLines(p, warn = FALSE),
  function(l) if (nzchar(trimws(l))) fromJSON(l, simplifyVector = FALSE) else NULL)
select_fields <- function(data, fields) data[strsplit(fields, " ", fixed = TRUE)[[1]]]
num <- function(x) if (is.null(x)) NA_real_ else as.numeric(x)

save_submission_workbook <- function(wb, file, data_range) {
  stage <- tempfile("xlsx-")
  dir.create(stage)
  on.exit(unlink(stage, recursive = TRUE), add = TRUE)
  raw <- file.path(stage, "raw.xlsx")
  parts <- file.path(stage, "parts")
  dir.create(parts)
  openxlsx::saveWorkbook(wb, raw, overwrite = TRUE)
  utils::unzip(raw, exdir = parts)

  sheet_path <- file.path(parts, "xl", "worksheets", "sheet1.xml")
  sheet <- xml2::read_xml(sheet_path)
  dimension <- xml2::xml_find_first(sheet, "//*[local-name()='dimension']")
  xml2::xml_set_attr(dimension, "ref", data_range)
  xml2::write_xml(sheet, sheet_path)

  rel_path <- file.path(parts, "xl", "worksheets", "_rels", "sheet1.xml.rels")
  if (file.exists(rel_path)) {
    rel <- xml2::read_xml(rel_path)
    nodes <- xml2::xml_children(rel)
    target <- xml2::xml_attr(nodes, "Target")
    unused <- startsWith(target, "../drawings/") &
      !file.exists(file.path(parts, "xl", "worksheets", target))
    xml2::xml_remove(nodes[unused])
    xml2::write_xml(rel, rel_path)
  }

  types_path <- file.path(parts, "[Content_Types].xml")
  types <- xml2::read_xml(types_path)
  nodes <- xml2::xml_children(types)
  part <- xml2::xml_attr(nodes, "PartName", default = "")
  unused <- startsWith(part, "/xl/drawings/") &
    !file.exists(file.path(parts, substring(part, 2)))
  xml2::xml_remove(nodes[unused])
  xml2::write_xml(types, types_path)

  xlsx <- file.path(stage, "workbook.xlsx")
  zip::zipr(xlsx, list.files(parts, all.files = TRUE, recursive = TRUE),
            root = parts, include_directories = FALSE, mode = "mirror")
  dir.create(dirname(file), recursive = TRUE, showWarnings = FALSE)
  if (!file.copy(xlsx, file, overwrite = TRUE)) stop("Could not save ", file)
}

save_table <- function(n, x, headers) {
  stopifnot(ncol(x) == length(headers), !anyDuplicated(headers), nrow(x) > 0L)
  names(x) <- headers
  wb <- openxlsx::buildWorkbook(as.data.frame(x), asTable = FALSE, gridLines = TRUE,
                                creator = NULL)
  openxlsx::setColWidths(wb, "Sheet 1", cols = seq_len(ncol(x)), widths = "auto")
  file <- file.path(OUT, sprintf("TableS%d.xlsx", n))
  data_range <- paste0("A1:", openxlsx::int2col(ncol(x)), nrow(x) + 1L)
  save_submission_workbook(wb, file, data_range)
}
