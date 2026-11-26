#!/usr/bin/env Rscript
# Shared setup for the main table scripts. Each TableN.R sources this, then writes one XLSX.
rm(list = ls())
suppressMessages(library(jsonlite))
suppressMessages(library(openxlsx))

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
OUT <- Sys.getenv("REPORT_LLM_TABLE_OUT_DIR", unset = dirname(SCRIPT_DIR))

META <- file.path(TS_DIR, "metadata_489reports.csv")   # Table 1 cohort

.xlsx_compat <- function(path) {
  work <- tempfile("xlsx-")
  packed <- tempfile(fileext = ".xlsx")
  dir.create(work)
  on.exit(unlink(c(work, packed), recursive = TRUE), add = TRUE)
  unzip(path, exdir = work)
  ns <- c(r = "http://schemas.openxmlformats.org/officeDocument/2006/relationships")
  sheets <- list.files(file.path(work, "xl/worksheets"), "^sheet[0-9]+\\.xml$",
                       full.names = TRUE)
  for (sheet in sheets) {
    doc <- xml2::read_xml(sheet)
    relfile <- file.path(dirname(sheet), "_rels", paste0(basename(sheet), ".rels"))
    if (file.exists(relfile)) {
      rel <- xml2::read_xml(relfile)
      nodes <- xml2::xml_find_all(rel, "//*[local-name()='Relationship']")
      used <- xml2::xml_text(xml2::xml_find_all(doc, "//@r:id", ns))
      for (node in nodes) {
        target <- xml2::xml_attr(node, "Target")
        type <- xml2::xml_attr(node, "Type")
        if (grepl("/(drawing|vmlDrawing)$", type) &&
            !identical(xml2::xml_attr(node, "TargetMode"), "External") &&
            !file.exists(file.path(dirname(sheet), target))) {
          if (xml2::xml_attr(node, "Id") %in% used)
            stop("Missing drawing referenced by worksheet: ", target)
          xml2::xml_remove(node)
        }
      }
      xml2::write_xml(rel, relfile)
    }
    refs <- xml2::xml_attr(xml2::xml_find_all(doc, "//*[local-name()='sheetData']//*[local-name()='c']"), "r")
    dim <- xml2::xml_find_first(doc, "//*[local-name()='dimension']")
    if (length(refs) && !inherits(dim, "xml_missing")) {
      last_col <- openxlsx::int2col(max(openxlsx::convertFromExcelRef(refs)))
      last_row <- max(as.integer(sub("^[A-Z]+", "", refs)))
      xml2::xml_set_attr(dim, "ref", paste0("A1:", last_col, last_row))
      xml2::write_xml(doc, sheet)
    }
  }
  typesfile <- file.path(work, "[Content_Types].xml")
  types <- xml2::read_xml(typesfile)
  for (node in xml2::xml_find_all(types, "//*[local-name()='Override']")) {
    part <- sub("^/", "", xml2::xml_attr(node, "PartName"))
    if (startsWith(part, "xl/drawings/") && !file.exists(file.path(work, part)))
      xml2::xml_remove(node)
  }
  xml2::write_xml(types, typesfile)
  zip::zipr(packed, list.files(work, recursive = TRUE, all.files = TRUE),
            root = work, include_directories = FALSE, mode = "mirror")
  if (!file.copy(packed, path, overwrite = TRUE)) stop("Could not save: ", path)
  invisible(path)
}

write_table <- function(df, base) {
  dir.create(OUT, recursive = TRUE, showWarnings = FALSE)
  p <- file.path(OUT, paste0(base, ".xlsx"))
  draft <- tempfile(fileext = ".xlsx")
  on.exit(unlink(draft), add = TRUE)
  openxlsx::write.xlsx(df, draft, overwrite = TRUE)
  .xlsx_compat(draft)
  if (!file.copy(draft, p, overwrite = TRUE)) stop("Could not save: ", p)
  cat("  ->", p, "\n")
  invisible(p)
}
