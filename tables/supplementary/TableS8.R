#!/usr/bin/env Rscript
# Table S8. Sample sizes and classifier performance by outer fold
rm(list = ls())
source("_common.R")

splits <- read_json(ts("internal/classifier_split_manifest.json"))
labels <- read_tsv(ts("internal/classifier_oof_labels.tsv"))
stopifnot(!anyDuplicated(labels$report_id), all(labels$asd_label %in% 0:1))
label_of <- setNames(labels$asd_label, labels$report_id)

partition_counts <- function(split) {
  members <- data.frame(
    index = as.integer(c(unlist(split$train_indices_cv), unlist(split$valid_indices_cv))),
    report = c(unlist(split$train_report_ids), unlist(split$valid_report_ids)),
    individual = c(unlist(split$train_patient_ids), unlist(split$valid_patient_ids)),
    stringsAsFactors = FALSE)
  stopifnot(!anyDuplicated(members$index), !anyDuplicated(members$report))
  members$asd <- unname(label_of[members$report])
  stopifnot(!anyNA(members$asd))
  partitions <- list(train = "inner_train_indices_cv", validation = "inner_valid_indices_cv",
                     test = "valid_indices_cv")
  selected <- lapply(partitions, function(key) {
    indices <- match(as.integer(unlist(split[[key]])), members$index)
    stopifnot(!anyNA(indices))
    members[indices, , drop = FALSE]
  })
  stopifnot(!length(intersect(selected$train$individual, selected$validation$individual)),
            !length(intersect(selected$train$individual, selected$test$individual)),
            !length(intersect(selected$validation$individual, selected$test$individual)),
            sum(vapply(selected, nrow, integer(1))) == nrow(members))
  result <- list(fold = as.integer(split$fold))
  for (name in names(selected)) {
    a <- selected[[name]]
    result[[paste0(name, "_reports")]] <- nrow(a)
    result[[paste0(name, "_individuals")]] <- length(unique(a$individual))
    result[[paste0(name, "_asd")]] <- sum(a$asd == 1L)
    result[[paste0(name, "_non_asd")]] <- sum(a$asd == 0L)
  }
  as.data.frame(result)
}
counts <- do.call(rbind, lapply(splits, partition_counts))
stopifnot(!anyDuplicated(counts$fold))

models <- classifier_names
x <- read_tsv(P_CLF_FOLDS)
x <- x[x$group == "All" & x$model %in% names(models), ]
x <- x[order(match(x$model, names(models)), x$fold), ]
a <- counts[match(x$fold, counts$fold), ]
stopifnot(!anyNA(a$fold), all(x$n_reports == a$test_reports),
          all(x$n_patients == a$test_individuals), all(x$n_asd == a$test_asd))
for (partition in c("train", "validation", "test")) {
  x[[paste0(partition, "_n")]] <- sprintf("%d (%d)", a[[paste0(partition, "_reports")]],
                                         a[[paste0(partition, "_individuals")]])
  x[[paste0(partition, "_classes")]] <- sprintf("%d/%d", a[[paste0(partition, "_asd")]],
                                               a[[paste0(partition, "_non_asd")]])
}
quality <- read_tsv(P_FOLD_QUALITY)
x$logit_sd <- ifelse(x$model == "v4_1", quality$logit_sd[match(x$fold, quality$fold)], NA_real_)
x$model <- unname(models[x$model])
save_table(8,
  select_fields(x, paste("model fold train_n train_classes validation_n validation_classes",
                        "test_n test_classes auroc auprc recall specificity brier calibration_slope logit_sd")),
  c("Model", "Fold", "Train n (N)", "Train ASD/non", "Inner val. n (N)", "Inner val. ASD/non",
    "Test n (N)", "Test ASD/non", "AUROC", "AP", "Sens.", "Spec.", "Brier score",
    "Cal. slope", "Logit SD"))
