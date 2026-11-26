#!/usr/bin/env Rscript
# ESM_2. Report-level phenotype fingerprints and classifier annotations.
# The first 24 columns hold the phenotype fingerprint (distance, largest deviation and Layer 2 z);
# the remaining columns hold the leading-space distance and the out-of-fold classifier results.
rm(list = ls())
source("_common.R")

out <- if (length(args)) OUT else file.path(OUT, "ESM_2.xlsx")
profiles <- read.delim(P_PROFILES, check.names = FALSE)
classifier <- read.delim(P_FINGERPRINTS, check.names = FALSE)
fold_quality <- read.delim(P_FOLD_QUALITY, check.names = FALSE)

domains <- doms
domain_labels <- setNames(paste(codes, unname(display_of)), domains)

stopifnot(nrow(profiles) == 346L, !anyNA(profiles$report_id), !anyDuplicated(profiles$report_id),
          !anyNA(classifier$report_id), !anyDuplicated(classifier$report_id))
index <- match(profiles$report_id, classifier$report_id)
stopifnot(!anyNA(index))
classifier <- classifier[index, ]
stopifnot(identical(profiles$report_type, classifier$report_type),
          !anyNA(classifier$fold), !anyNA(fold_quality$fold),
          !anyDuplicated(fold_quality$fold),
          all(fold_quality$near_constant_logit %in% c("True", "False")))
fold_index <- match(classifier$fold, fold_quality$fold)
stopifnot(!anyNA(fold_index))

z_columns <- paste0("z_", domains)
data <- profiles[c("report_id", "report_type", "mahalanobis_full", "max_dev_domain", "max_dev_z", z_columns)]
names(data)[3] <- "prototypicality_mahalanobis_full"
data$max_dev_domain <- unname(display_of[data$max_dev_domain])
stopifnot(!anyNA(data$max_dev_domain))
names(data)[match(z_columns, names(data))] <- paste0(unname(domain_labels), " (z)")

data$mahalanobis_lead3 <- profiles$mahalanobis_lead3
data$fold <- classifier$fold
data$raw_logit <- classifier$raw_logit
data$model_probability <- classifier$model_probability
data$near_constant_logit_fold <- fold_quality$near_constant_logit[fold_index] == "True"

for (domain in domains) {
  label <- domain_labels[[domain]]
  data[[paste0(label, " (E)")]] <- classifier[[paste0("E_", domain)]]
}
stopifnot(ncol(data) == 48L, !anyDuplicated(names(data)),
          all(is.finite(as.matrix(data[paste0(unname(domain_labels), " (E)")]))))

# Keep the internal ID through the join and replace it only for export.
map_path <- file.path(TS_DIR, "internal", "ESM_2_report_id_mapping.tsv")
id_map <- read.delim(map_path, colClasses = "character", check.names = FALSE)
expected_ids <- sprintf("RS-%03d", seq_len(nrow(data)))
if (!identical(names(id_map), c("report_id", "submission_id")) ||
    nrow(id_map) != nrow(data) ||
    anyNA(id_map) || anyDuplicated(id_map$report_id) || anyDuplicated(id_map$submission_id) ||
    !setequal(id_map$report_id, data$report_id) ||
    !setequal(id_map$submission_id, expected_ids)) {
  stop("The ESM_2 ID mapping must pair each report with one unique submission ID.")
}
data$report_id <- id_map$submission_id[match(data$report_id, id_map$report_id)]
data <- data[order(data$report_id), , drop = FALSE]

wb <- openxlsx::buildWorkbook(data, asTable = FALSE, creator = NULL)
openxlsx::addStyle(wb, "Sheet 1", openxlsx::createStyle(numFmt = "0.000"),
                    rows = seq_len(nrow(data)) + 1L, cols = c(3L, 5:24), gridExpand = TRUE)
data_range <- paste0("A1:", openxlsx::int2col(ncol(data)), nrow(data) + 1L)
save_submission_workbook(wb, out, data_range)
