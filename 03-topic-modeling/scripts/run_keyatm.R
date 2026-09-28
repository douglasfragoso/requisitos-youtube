#!/usr/bin/env Rscript
# keyATM training engine for the frozen H6 aspect lexicon.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) %% 2 != 0) stop("Arguments must be --name value pairs")
opts <- as.list(args[seq(2, length(args), by = 2)])
names(opts) <- sub("^--", "", args[seq(1, length(args), by = 2)])
required <- c("mode", "input", "keywords", "output_dir", "seed", "no_keyword_topics")
missing <- setdiff(required, names(opts))
if (length(missing)) stop(paste("Missing arguments:", paste(missing, collapse = ", ")))
if (!opts$mode %in% c("base", "covariates")) stop("mode must be base or covariates")

for (package in c("quanteda", "keyATM", "jsonlite")) {
  if (!requireNamespace(package, quietly = TRUE)) {
    stop(paste("R package not installed:", package))
  }
}

texts <- read.csv(opts$input, stringsAsFactors = FALSE, fileEncoding = "UTF-8")
if (!"text" %in% names(texts)) stop("Input CSV must have a text column")
if (anyNA(texts$text) || any(!nzchar(trimws(texts$text)))) {
  stop("Input contains empty documents; filter them with their metadata before fitting")
}
keywords <- jsonlite::fromJSON(opts$keywords, simplifyVector = TRUE)
if (!is.list(keywords) || !length(keywords)) stop("No effective keyword aspects")
keywords <- lapply(keywords, as.character)
dfm <- quanteda::dfm(quanteda::tokens(texts$text))
vocab <- quanteda::featnames(dfm)
absent <- setdiff(unique(unlist(keywords)), vocab)
if (length(absent)) stop(paste("Keywords absent from model vocabulary:", paste(absent, collapse = ", ")))
docs <- keyATM::keyATM_read(texts = dfm)

model_settings <- list()
if (opts$mode == "covariates") {
  if (is.null(opts$covariates) || is.null(opts$covariates_formula)) {
    stop("Covariate mode requires --covariates and --covariates_formula")
  }
  covariates <- read.csv(opts$covariates, stringsAsFactors = FALSE, fileEncoding = "UTF-8")
  if (nrow(covariates) != nrow(texts)) stop("Covariates and texts are not aligned")
  covariates$product_category <- as.factor(covariates$product_category)
  model_settings <- list(covariates_data = covariates,
                         covariates_formula = as.formula(opts$covariates_formula))
}

fit <- keyATM::keyATM(
  docs = docs, model = opts$mode,
  no_keyword_topics = as.integer(opts$no_keyword_topics),
  keywords = keywords, model_settings = model_settings,
  options = list(seed = as.integer(opts$seed),
                 iterations = as.integer(if (is.null(opts$iterations)) 1500 else opts$iterations))
)

dir.create(opts$output_dir, recursive = TRUE, showWarnings = FALSE)
topic_names <- c(names(keywords), paste0("livre_", seq_len(as.integer(opts$no_keyword_topics))))
theta <- as.matrix(fit$theta)
phi <- as.matrix(fit$phi)
if (nrow(theta) != nrow(texts) || ncol(theta) != length(topic_names)) {
  stop("Unexpected keyATM theta dimensions")
}
if (nrow(phi) != length(topic_names)) stop("Unexpected keyATM phi dimensions")
colnames(theta) <- topic_names
rownames(phi) <- topic_names
write.csv(theta, file.path(opts$output_dir, "keyatm_theta.csv"), row.names = FALSE)
write.csv(phi, file.path(opts$output_dir, "keyatm_phi.csv"), row.names = TRUE)
top_words <- as.data.frame(keyATM::top_words(fit, n = 20))
colnames(top_words) <- topic_names
utils::write.csv(top_words, file.path(opts$output_dir, "topics.csv"), row.names = FALSE)
saveRDS(fit, file.path(opts$output_dir, "keyatm_model.rds"))
jsonlite::write_json(
  list(theta_csv = "keyatm_theta.csv", phi_csv = "keyatm_phi.csv",
       topics_csv = "topics.csv", topic_names = topic_names,
       seed = as.integer(opts$seed), no_keyword_topics = as.integer(opts$no_keyword_topics)),
  file.path(opts$output_dir, "keyatm_final.json"), auto_unbox = TRUE, pretty = TRUE
)
