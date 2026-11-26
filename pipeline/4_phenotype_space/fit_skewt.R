#!/usr/bin/env Rscript
# Multivariate skew-t (sn::selm, family "ST") and skew-normal ("SN", the nu -> infinity limit) fits for every dataset
# in a long CSV. selm often stops with "false convergence" when nu runs off to very large values, and the default SN
# start can land in a local optimum. Each family is therefore fitted from two starts and the higher log-likelihood is
# kept: ST from the default start and from the SN solution (nu = 30); SN from the default start and from the ST
# solution's (xi, Omega, alpha). Downstream, the skew-t family BIC is min(ST BIC with 13 parameters, SN BIC with 12).
# Input CSV columns: dataset, PC1, PC2, PC3.
# Output TSV: dataset, n, st_loglik, st_bic, st_nu, st_converged, st_start, st_message,
#             sn_loglik, sn_bic, sn_converged, sn_start, sn_message.
# selm uses a deterministic optimiser (no random numbers); set.seed is kept for the code rule.
set.seed(42)
suppressPackageStartupMessages(library(sn))
args <- commandArgs(trailingOnly = TRUE)
inp <- args[1]; out <- args[2]
d <- read.csv(inp)
ids <- unique(d$dataset)
clean <- function(s) gsub("[\t\n]", " ", as.character(s))
none <- list(fit = NULL, loglik = NA, bic = NA, nu = NA, converged = FALSE, start = "none", message = "")

fit_one <- function(X, family, start = NULL, start_name = "default") {
  n <- nrow(X); p <- ncol(X)
  n_par <- p + p * (p + 1) / 2 + p + (family == "ST")   # xi, Omega, alpha (+ nu)
  tryCatch({
    f <- if (is.null(start)) selm(X ~ 1, family = family) else selm(X ~ 1, family = family, start = start)
    ll <- as.numeric(logLik(f))
    nu <- if (family == "ST") as.numeric(coef(f, "DP", vector = FALSE)$nu) else Inf
    conv <- f@opt.method$convergence
    list(fit = f, loglik = ll, bic = -2 * ll + n_par * log(n), nu = nu,
         converged = if (is.null(conv)) NA else conv == 0, start = start_name,
         message = clean(f@opt.method$message))
  }, error = function(e) modifyList(none, list(start = start_name, message = clean(conditionMessage(e)))))
}
better <- function(a, b) if (is.na(b$loglik) || (!is.na(a$loglik) && a$loglik >= b$loglik)) a else b

rows <- vector("list", length(ids))
for (i in seq_along(ids)) {
  X <- as.matrix(d[d$dataset == ids[i], c("PC1", "PC2", "PC3")])
  st0 <- fit_one(X, "ST"); sn0 <- fit_one(X, "SN")
  sn1 <- if (!is.null(st0$fit)) {
    dp <- coef(st0$fit, "DP", vector = FALSE)
    fit_one(X, "SN", list(beta = matrix(dp$beta, 1), Omega = dp$Omega, alpha = dp$alpha), "from_ST")
  } else none
  st1 <- if (!is.null(sn0$fit)) {
    dp <- coef(sn0$fit, "DP", vector = FALSE)
    fit_one(X, "ST", list(beta = matrix(dp$beta, 1), Omega = dp$Omega, alpha = dp$alpha, nu = 30), "from_SN")
  } else none
  st <- better(st0, st1); sn_ <- better(sn0, sn1)
  rows[[i]] <- data.frame(dataset = ids[i], n = nrow(X),
                          st_loglik = st$loglik, st_bic = st$bic, st_nu = st$nu, st_converged = st$converged,
                          st_start = st$start, st_message = st$message,
                          sn_loglik = sn_$loglik, sn_bic = sn_$bic, sn_converged = sn_$converged,
                          sn_start = sn_$start, sn_message = sn_$message)
}
write.table(do.call(rbind, rows), out, sep = "\t", row.names = FALSE, quote = FALSE)
