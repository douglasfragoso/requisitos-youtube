#!/usr/bin/env Rscript
# Factorization bridge for the official NMFregress 1.0.1 package.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) %% 2 != 0) stop("Arguments must be --name value pairs")
keys <- sub("^--", "", args[seq(1, length(args), by = 2)])
opts <- as.list(args[seq(2, length(args), by = 2)])
names(opts) <- keys
required <- c("mode", "tdm", "vocab", "output_dir", "seed", "topics")
missing <- setdiff(required, names(opts))
if (length(missing)) stop(paste("Missing arguments:", paste(missing, collapse = ", ")))
if (!opts$mode %in% c("fit", "fit_and_regress")) stop("Unknown mode")
if (!requireNamespace("NMFregress", quietly = TRUE)) stop("NMFregress is not installed")
if (!requireNamespace("jsonlite", quietly = TRUE)) stop("jsonlite is not installed")

set.seed(as.integer(opts$seed))
tdm <- as.matrix(read.csv(opts$tdm, header = FALSE, check.names = FALSE))
storage.mode(tdm) <- "double"
vocab <- as.character(read.csv(opts$vocab, header = FALSE)[[1]])
if (length(vocab) != nrow(tdm)) stop("Vocabulary and TDM rows are not aligned")
if (anyNA(tdm) || any(tdm < 0)) stop("TDM must contain nonnegative finite counts")
covariates <- NULL
if (!is.null(opts$covariates)) {
  covariates <- as.matrix(read.csv(opts$covariates, header = TRUE, check.names = FALSE))
  storage.mode(covariates) <- "double"
  if (nrow(covariates) != ncol(tdm)) stop("Covariates and documents are not aligned")
}

input <- NMFregress::create_input(tdm, vocab = vocab,
                                  topics = as.integer(opts$topics),
                                  covariates = covariates)
fit <- NMFregress::solve_nmf(input)
# NMFregress 1.0.1 stores input$covariates but solve_nmf copies
# input$covariate (singular). Restore the supplied design for inference.
fit$covariates <- covariates
dir.create(opts$output_dir, recursive = TRUE, showWarnings = FALSE)
write.csv(fit$theta, file.path(opts$output_dir, "theta.csv"), row.names = FALSE)
phi <- data.frame(word = fit$vocab, fit$phi, check.names = FALSE)
write.csv(phi, file.path(opts$output_dir, "phi.csv"), row.names = FALSE)
saveRDS(fit, file.path(opts$output_dir, "model.rds"))

result <- list(mode = opts$mode, topics = as.integer(opts$topics),
               n_docs = ncol(tdm), vocab_size = nrow(tdm),
               package_version = as.character(utils::packageVersion("NMFregress")),
               anchors = unname(as.character(fit$anchors)),
               theta_csv = "theta.csv", phi_csv = "phi.csv",
               empty_fraction = mean(colSums(fit$theta) == 0),
               seed = as.integer(opts$seed))
if (opts$mode == "fit_and_regress") {
  if (is.null(covariates)) stop("Regression mode requires covariates")
  beta <- NMFregress::get_regression_coefs(fit, model = "BETA",
                                          return_just_coefs = TRUE)
  write.csv(beta, file.path(opts$output_dir, "beta_coefs.csv"), row.names = TRUE)
  result$beta_csv <- "beta_coefs.csv"
  ols <- NMFregress::boot_reg(fit, samples = as.integer(opts$bootstrap_n),
                              model = "OLS")
  saveRDS(ols, file.path(opts$output_dir, "ols_boot.rds"))
  result$ols_boot_rds <- "ols_boot.rds"
}
jsonlite::write_json(result, file.path(opts$output_dir, "brett_final.json"),
                     auto_unbox = TRUE, pretty = TRUE)
