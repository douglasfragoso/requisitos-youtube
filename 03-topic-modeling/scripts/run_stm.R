#!/usr/bin/env Rscript
# ============================================================================
# run_stm.R — motor STM (Roberts et al.) do modulo 03-topic-modeling.
#
# Chamado por _helpers.py (grid_search_k_stm / grid_search_stm_hparams /
# train_stm) via subprocess — nao usar interativamente. O texto de entrada JA
# vem lematizado do Python (lemmatize_corpus): aqui NAO ha stem, stopwords
# nem lowercase — apenas tokenizacao por espaco e filtro de frequencia
# alinhado ao gensim (no_below/no_above).
#
# Modos:
#   --mode grid_k       1 STM por K (split heldout, como searchK) ->
#                       top words + diagnosticos nativos por K
#   --mode grid_hparams K fixo, varre sigma.prior x gamma.prior
#   --mode train        treino final + estimateEffect; theta/beta em CSV
#
# Saidas (em --output_dir): stm_grid_k.json | stm_hparams.json |
#   stm_final.json + stm_theta.csv + stm_beta.csv
# ============================================================================

suppressPackageStartupMessages({
  library(stm)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
opt <- list()
i <- 1
while (i <= length(args)) {
  if (!startsWith(args[[i]], "--")) stop(sprintf("argumento inesperado: %s", args[[i]]))
  opt[[sub("^--", "", args[[i]])]] <- args[[i + 1]]
  i <- i + 2
}
req <- function(name) {
  v <- opt[[name]]
  if (is.null(v)) stop(sprintf("argumento obrigatorio ausente: --%s", name))
  v
}
opt_or <- function(name, default) if (is.null(opt[[name]])) default else opt[[name]]

mode       <- req("mode")
input_csv  <- req("input")
output_dir <- req("output_dir")
seed       <- as.integer(opt_or("seed", "42"))
top_n      <- as.integer(opt_or("top_n", "10"))
no_below   <- as.integer(opt_or("no_below", "5"))
no_above   <- as.numeric(opt_or("no_above", "0.5"))
max_em_its <- as.integer(opt_or("max_em_its", "150"))
prevalence <- opt_or("prevalence", "")

set.seed(seed)
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

meta <- read.csv(input_csv, fileEncoding = "UTF-8", stringsAsFactors = FALSE,
                 colClasses = "character")
if (!"text" %in% names(meta)) stop("stm_input.csv sem coluna 'text'")
# covariaveis categoricas viram factor; 'date' fica character (a formula converte)
for (cn in setdiff(names(meta), c("text", "post_id", "date"))) {
  meta[[cn]] <- factor(meta[[cn]])
}

# Texto ja lematizado -> tokenizacao por espaco, sem qualquer reprocessamento
proc <- textProcessor(
  documents = meta$text, metadata = meta,
  lowercase = FALSE, removestopwords = FALSE, removenumbers = FALSE,
  removepunctuation = FALSE, stem = FALSE, wordLengths = c(1, Inf),
  verbose = FALSE
)
# lower.thresh: palavras em <= lower.thresh docs sao dropadas ->
#   no_below - 1 mantem palavras em >= no_below docs (= gensim filter_extremes)
# upper.thresh: palavras em > upper.thresh docs sao dropadas (= no_above)
prep <- prepDocuments(
  proc$documents, proc$vocab, proc$meta,
  lower.thresh = no_below - 1,
  upper.thresh = floor(no_above * length(proc$documents)),
  verbose = FALSE
)

# Indices 0-based (p/ o Python) dos docs removidos por textProcessor/prepDocuments
kept <- seq_len(nrow(meta))
if (length(proc$docs.removed) > 0) kept <- kept[-proc$docs.removed]
if (length(prep$docs.removed) > 0) kept <- kept[-prep$docs.removed]
# I() forca write_json a serializar como array mesmo com length 1 — sem isso,
# auto_unbox=TRUE colapsa um unico doc removido num escalar JSON solto, e o
# Python (que sempre itera docs_removed como lista) quebra com TypeError.
docs_removed0 <- I(setdiff(seq_len(nrow(meta)), kept) - 1L)

prev_formula <- if (nzchar(prevalence)) as.formula(prevalence) else NULL

fit_stm <- function(K, documents, vocab, data, sigma_prior = NULL, gamma_prior = "Pooled") {
  build_args <- function(init) {
    a <- list(documents = documents, vocab = vocab, K = K, data = data,
              init.type = init, seed = seed, max.em.its = max_em_its,
              gamma.prior = gamma_prior, verbose = FALSE)
    if (!is.null(prev_formula)) a$prevalence <- prev_formula
    if (!is.null(sigma_prior)) a$sigma.prior <- as.numeric(sigma_prior)
    a
  }
  # init.type="Spectral" e deterministico, mas a decomposicao espectral (anchor
  # words) pode falhar com chol() em matrizes perturbadas — tipicamente o train
  # do make.heldout no grid_k, onde ~50% dos tokens de alguns docs sao removidos
  # e a co-ocorrencia de palavras vira singular. Nesse caso cai para
  # init.type="LDA" (Gibbs com seed fixo -> reproduzivel run-a-run). O treino
  # final (train, docs completos) segue Spectral, que funciona.
  tryCatch(
    do.call(stm, build_args("Spectral")),
    error = function(e) {
      if (grepl("chol|decomposition|singular", conditionMessage(e), ignore.case = TRUE)) {
        cat(sprintf("[fit_stm] Spectral falhou (%s) -> fallback init.type='LDA' (seed=%d)\n",
                    conditionMessage(e), seed))
        do.call(stm, build_args("LDA"))
      } else {
        stop(e)
      }
    }
  )
}

as_topic_list <- function(mat) {
  out <- list()
  for (t in seq_len(nrow(mat))) out[[as.character(t - 1L)]] <- as.character(mat[t, ])
  out
}
top_words <- function(model) {
  lbl <- labelTopics(model, n = top_n)
  list(prob = as_topic_list(lbl$prob), frex = as_topic_list(lbl$frex))
}

cat(sprintf("[run_stm] mode=%s docs=%d vocab=%d removed=%d prevalence='%s'\n",
            mode, length(prep$documents), length(prep$vocab),
            length(docs_removed0), prevalence))

if (mode == "grid_k") {
  k_values <- as.integer(strsplit(req("k_grid"), ",")[[1]])
  heldout <- make.heldout(prep$documents, prep$vocab, seed = seed)
  results <- list()
  # Reseta o checkpoint de uma tentativa anterior: este script NAO le o
  # checkpoint para retomar de onde parou, cada chamada recomputa a grade
  # inteira. Sem o reset, um checkpoint de um processo morto no meio (o
  # motivo de existir, ver comentario abaixo) fica misturado com as linhas
  # da tentativa nova, porque write.table(append=...) so acrescenta.
  chk_path <- file.path(output_dir, "stm_grid_k_checkpoint.csv")
  if (file.exists(chk_path)) file.remove(chk_path)
  for (K in k_values) {
    t0 <- Sys.time()
    m <- fit_stm(K, heldout$documents, heldout$vocab, prep$meta)
    tw <- top_words(m)
    el <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
    res <- list(
      k = K,
      topics = tw$prob,
      topics_frex = tw$frex,
      heldout_likelihood = eval.heldout(m, heldout$missing)$expected.heldout,
      residual_dispersion = checkResiduals(m, heldout$documents)$dispersion,
      semantic_coherence = mean(semanticCoherence(m, heldout$documents)),
      exclusivity_stm = mean(exclusivity(m)),
      iterations = m$convergence$its,
      elapsed_sec = el
    )
    results[[length(results) + 1]] <- res
    cat(sprintf("[grid_k] K=%d heldout=%.4f disp=%.3f semcoh=%.2f excl=%.2f its=%d (%.0fs)\n",
                K, res$heldout_likelihood, res$residual_dispersion,
                res$semantic_coherence, res$exclusivity_stm, res$iterations, el))
    # Checkpoint: grava CADA K assim que termina, para sobreviver a kill/crash
    # no meio do grid — sem isso, so o write_json() no final do loop persiste
    # qualquer coisa (2026-08-22, mesma lacuna que o sweep_bertopic_grid tinha).
    chk_row <- data.frame(
      k = K, heldout_likelihood = res$heldout_likelihood,
      residual_dispersion = res$residual_dispersion,
      semantic_coherence = res$semantic_coherence,
      exclusivity_stm = res$exclusivity_stm,
      iterations = res$iterations, elapsed_sec = el
    )
    write.table(chk_row, chk_path, append = file.exists(chk_path), sep = ",",
                row.names = FALSE, col.names = !file.exists(chk_path))
  }
  write_json(
    list(mode = "grid_k", seed = seed, prevalence = prevalence,
         n_docs = length(prep$documents), vocab_size = length(prep$vocab),
         docs_removed = docs_removed0, results = results),
    file.path(output_dir, "stm_grid_k.json"),
    auto_unbox = TRUE, digits = 8
  )
} else if (mode == "grid_hparams") {
  K <- as.integer(req("k"))
  sigma_grid <- as.numeric(strsplit(req("sigma_grid"), ",")[[1]])
  gamma_grid <- strsplit(req("gamma_grid"), ",")[[1]]
  results <- list()
  # Mesmo reset do modo grid_k acima — cada chamada recomputa a grade
  # inteira, entao um checkpoint da tentativa anterior nunca deve sobreviver.
  chk_path <- file.path(output_dir, "stm_hparams_checkpoint.csv")
  if (file.exists(chk_path)) file.remove(chk_path)
  for (sg in sigma_grid) {
    for (gm in gamma_grid) {
      t0 <- Sys.time()
      m <- fit_stm(K, prep$documents, prep$vocab, prep$meta,
                   sigma_prior = sg, gamma_prior = gm)
      tw <- top_words(m)
      el <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
      results[[length(results) + 1]] <- list(
        sigma_prior = sg, gamma_prior = gm,
        topics = tw$prob, topics_frex = tw$frex,
        bound = tail(m$convergence$bound, 1),
        iterations = m$convergence$its, elapsed_sec = el
      )
      cat(sprintf("[grid_hparams] sigma=%.2f gamma=%s its=%d (%.0fs)\n",
                  sg, gm, m$convergence$its, el))
      # Checkpoint por combo — mesmo motivo do grid_k acima.
      last <- results[[length(results)]]
      chk_row <- data.frame(
        sigma_prior = last$sigma_prior, gamma_prior = last$gamma_prior,
        bound = last$bound, iterations = last$iterations, elapsed_sec = last$elapsed_sec
      )
      write.table(chk_row, chk_path, append = file.exists(chk_path), sep = ",",
                  row.names = FALSE, col.names = !file.exists(chk_path))
    }
  }
  write_json(
    list(mode = "grid_hparams", k = K, seed = seed, prevalence = prevalence,
         n_docs = length(prep$documents), vocab_size = length(prep$vocab),
         docs_removed = docs_removed0, results = results),
    file.path(output_dir, "stm_hparams.json"),
    auto_unbox = TRUE, digits = 8
  )
} else if (mode == "train") {
  K <- as.integer(req("k"))
  sigma_prior <- as.numeric(opt_or("sigma_prior", "0"))
  gamma_prior <- opt_or("gamma_prior", "Pooled")
  m <- fit_stm(K, prep$documents, prep$vocab, prep$meta,
               sigma_prior = sigma_prior, gamma_prior = gamma_prior)
  tw <- top_words(m)

  theta <- m$theta
  colnames(theta) <- paste0("t", seq_len(K) - 1L)
  write.csv(theta, file.path(output_dir, "stm_theta.csv"),
            row.names = FALSE, fileEncoding = "UTF-8")

  beta <- exp(m$beta$logbeta[[1]])
  colnames(beta) <- m$vocab
  write.csv(beta, file.path(output_dir, "stm_beta.csv"),
            row.names = FALSE, fileEncoding = "UTF-8")

  effects <- list()
  if (!is.null(prev_formula)) {
    ee <- estimateEffect(
      as.formula(paste0("1:", K, " ", prevalence)), m,
      metadata = prep$meta, uncertainty = "Global"
    )
    es <- summary(ee)
    for (t in seq_len(K)) {
      tab <- es$tables[[t]]
      effects[[as.character(t - 1L)]] <- data.frame(
        term = rownames(tab),
        estimate = unname(tab[, "Estimate"]),
        std_error = unname(tab[, "Std. Error"]),
        p_value = unname(tab[, "Pr(>|t|)"]),
        stringsAsFactors = FALSE
      )
    }
  }
  write_json(
    list(mode = "train", k = K, seed = seed, prevalence = prevalence,
         sigma_prior = sigma_prior, gamma_prior = gamma_prior,
         topics = tw$prob, topics_frex = tw$frex,
         prevalence_effects = effects,
         iterations = m$convergence$its,
         n_docs = length(prep$documents), vocab_size = length(prep$vocab),
         docs_removed = docs_removed0,
         theta_csv = "stm_theta.csv", beta_csv = "stm_beta.csv"),
    file.path(output_dir, "stm_final.json"),
    auto_unbox = TRUE, digits = 8
  )
} else {
  stop(sprintf("modo desconhecido: %s", mode))
}
cat("[run_stm] done\n")
