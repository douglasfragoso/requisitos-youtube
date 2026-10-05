"""Inlined helpers for the 03-topic-modeling notebooks.

This module consolidates what used to live under ``03-topic-modeling/src/``
(deleted in commit 002f06ac) into a single file co-located with the notebooks
so that ``from _helpers import ...`` works without any sys.path tweaks.

Originating modules (concatenated in order, imports merged + de-duplicated):

1. config.py        — params loading, multi-corpus resolution.
2. embeddings.py    — Ollama + SentenceTransformer embedding helpers with cache.
3. lemmatize.py     — language-aware spaCy lemmatization (PT/EN).
4. naming.py        — topic naming via Ollama LLM (retry + temperature escalation).
5. topic_utils.py   — coherence / exclusivity / FREX / diversity / NPMI etc.
6. lda_pipeline.py  — gensim LDA grid search, train, extract, doc distributions.
7. stm_pipeline.py  — prepare STM input + R subprocess orchestration.
8. nmf_pipeline.py  — gensim NMF grid search (K, then kappa x minimum_probability), train.

Original function bodies are preserved verbatim. Only the import blocks were
merged at the top of the file.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import time
from itertools import combinations
from pathlib import Path
from typing import Iterable

# ---------------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------------
import httpx
import numpy as np
import ollama
import pandas as pd
import yaml
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel, LdaMulticore, Nmf
from sentence_transformers import SentenceTransformer
from sklearn.metrics import cohen_kappa_score


# ===========================================================================
# config.py
# ===========================================================================
"""Configuration loading from params.yaml.

Multi-corpus aware. Backward-compatible with notebooks that pass the full
``params`` dict to ``get_column_names`` — in that case the default corpus
(``params['default_corpus']`` or first key) is used.
"""

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_params(path: str = None) -> dict:
    """Load parameters from a YAML file."""
    if path is None:
        path = PROJECT_ROOT / "configs" / "params.yaml"
    with open(path, "r", encoding="utf-8") as f:
        result = yaml.safe_load(f)
    if not isinstance(result, dict):
        raise ValueError(f"Expected dict from {path}, got {type(result)}")
    return result


def get_corpus_config(params: dict, corpus_id: str = None) -> tuple[str, dict]:
    """Resolve which corpus to use and return its configuration dict.

    Resolution order: explicit arg → params['default_corpus'] → first key.
    Returns (corpus_id, corpus_cfg). Falls back to ('default', params) for
    legacy flat layouts (no 'corpora' key).
    """
    corpora = params.get("corpora")
    if corpora is None:
        return ("default", params)

    if corpus_id is None:
        corpus_id = params.get("default_corpus")
    if corpus_id is None:
        corpus_id = next(iter(corpora))

    if corpus_id not in corpora:
        available = ", ".join(corpora.keys())
        raise KeyError(
            f"Corpus '{corpus_id}' not found in params['corpora']. "
            f"Available: {available}"
        )
    return (corpus_id, corpora[corpus_id])


def get_column_names(params_or_cfg: dict, corpus_id: str = None) -> dict:
    """Extract column names. Accepts full ``params`` dict or a ``corpus_cfg``.

    - If a full params dict is passed (has ``corpora`` key), the active corpus
      is resolved via ``get_corpus_config(params, corpus_id)``.
    - If a corpus_cfg dict is passed (has ``text_column`` at top), uses it.
    - Falls back to the legacy nested layout (``data.columns.text``) for
      compatibility with old configs.

    Returns dict with keys: 'text', 'date', 'post_id', 'covariates'.
    """
    # New layout — full params with corpora dict
    if "corpora" in params_or_cfg:
        _, cfg = get_corpus_config(params_or_cfg, corpus_id)
    else:
        cfg = params_or_cfg

    # Per-corpus flat layout
    if "text_column" in cfg:
        return {
            "text": cfg.get("text_column", "message"),
            "date": cfg.get("date_column", "data"),
            "post_id": cfg.get("post_id_column", "post_id"),
            "covariates": cfg.get("covariates", []) or [],
        }

    # Legacy nested layout (data.columns.*)
    columns = cfg.get("data", {}).get("columns", {})
    return {
        "text": columns.get("text", "message"),
        "date": columns.get("date", "data"),
        "post_id": columns.get("post_id", "post_id"),
        "covariates": columns.get("covariates", []) or [],
    }


def get_seed(params: dict) -> int:
    """Extract the global random seed."""
    return params.get("seed", 42)


def make_run_output_dir(base_dir, corpus_id, *, create: bool = True):
    """Diretorio de saida por execucao: ``base_dir/<corpus>_<YYYYmmdd_HHMMSS>``.

    Garante que rodar o mesmo notebook varias vezes (ex.: sweeps com configs
    diferentes, ou re-execucoes) NAO sobrescreva as saidas anteriores — cada
    run grava num subdiretorio carimbado com nome do dataset + data + hora.

    Parameters
    ----------
    base_dir : str | Path
        Diretorio-base do corpus (ex.: ``../data/output/<corpus>``). Caches que
        devem persistir entre runs (ex.: ``lda_metrics.csv`` do grid search)
        continuam vivendo aqui, no PAI do diretorio de run.
    corpus_id : str
        Nome do corpus, usado como prefixo legivel do subdiretorio.
    create : bool
        Cria o diretorio (``mkdir -p``) quando True (default).

    Returns
    -------
    Path
        ``base_dir/<corpus_id>_<timestamp>``.
    """
    from datetime import datetime

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_dir) / f"{corpus_id}_{stamp}"
    if create:
        run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


# Carimbo que make_run_output_dir grava: <nome>_AAAAMMDD_HHMMSS.
# resolve_latest_dir usa isto para nao deixar diretorio manual (backup, copia)
# vencer a ordenacao lexicografica dos carimbos.
_STAMP_RE = re.compile(r"_\d{8}_\d{6}$")


def resolve_latest_dir(base_output_dir, *, contains, fallback_dir=None, verbose: bool = True):
    """Resolve o diretório com a versão MAIS RECENTE de ``contains`` (latest-wins).

    Implementa o fluxo "input de um módulo = output do anterior, sem cópia": cada
    módulo grava num subdiretório carimbado ``<nome>_<AAAAMMDD_HHMMSS>`` no seu
    ``data/output/<corpus>/`` e o downstream lê a versão mais nova direto de lá.

    Ordem de resolução:
      1. subdiretórios **carimbados** de ``base_output_dir`` que contenham
         ``contains`` → retorna o de nome (= timestamp) MAIOR. "Carimbado" é
         validado por regex (``_AAAAMMDD_HHMMSS`` no fim do nome): sem isso a
         comparação lexicográfica deixa um diretório manual vencer o carimbo
         (``folha_backup/`` > ``folha_20260801_120000/``, porque 'b' > '2');
      2. layout plano legado (``contains`` direto em ``base_output_dir``);
      3. ``fallback_dir`` legado (ex.: ``data/input/<corpus>``).

    Diretório não-carimbado com o arquivo é ignorado no passo 1 e avisado — é
    quase sempre cópia manual ou backup, não uma versão do pipeline.

    Parameters
    ----------
    base_output_dir : str | Path
        ``data/output/<corpus>`` do módulo produtor (ex.: ``../../01-preprocessing/data/output/folha``).
    contains : str
        Nome do arquivo que identifica uma versão válida (ex.: ``corpus_limpo.csv``).
    fallback_dir : str | Path | None
        Diretório legado a usar se nada versionado/plano for encontrado.
    verbose : bool
        Imprime a versão resolvida.

    Returns
    -------
    Path
        Diretório que contém ``contains``. Use ``Path.name`` como string de versão
        para rastreabilidade (provenance) nos metrics.
    """
    base = Path(base_output_dir)
    com_arquivo = (
        [d for d in base.iterdir() if d.is_dir() and (d / contains).exists()]
        if base.exists()
        else []
    )
    stamped = [d for d in com_arquivo if _STAMP_RE.search(d.name)]
    sem_carimbo = [d for d in com_arquivo if d not in stamped]
    if stamped:
        chosen = max(stamped, key=lambda d: d.name)
        if verbose:
            print(f"  versão resolvida: {chosen.name}/ (latest)")
            for d in sem_carimbo:
                print(f"  (ignorado: '{d.name}/' tem {contains} mas não é "
                      f"carimbado <nome>_AAAAMMDD_HHMMSS)")
        return chosen
    if sem_carimbo:
        # Nenhum carimbado: cai no mais novo por nome, como antes — mas avisa,
        # porque aqui a ordenação lexicográfica não significa cronologia.
        chosen = max(sem_carimbo, key=lambda d: d.name)
        if verbose:
            print(f"  versão resolvida: {chosen.name}/ (SEM carimbo — ordem por "
                  f"nome, não por data; confira se é o diretório esperado)")
        return chosen
    if base.exists() and (base / contains).exists():
        if verbose:
            print(f"  layout plano legado: {base}")
        return base
    if fallback_dir is not None and (Path(fallback_dir) / contains).exists():
        if verbose:
            print(f"  fallback legado: {fallback_dir}")
        return Path(fallback_dir)
    raise FileNotFoundError(
        f"'{contains}' não encontrado em {base} (subdirs carimbados ou plano) "
        f"nem no fallback {fallback_dir}"
    )


def load_corpus(input_dir, encoding: str = "utf-8", verbose: bool = True) -> pd.DataFrame:
    """Carrega o corpus do diretório de input com auto-detecção de sentimento.

    Prefere ``corpus_com_sentimento.csv`` (gerado pelo módulo 03-sentiment —
    inclui colunas ``sentiment``, ``confidence``, ``sim_positive``,
    ``sim_negative`` além das colunas originais). Cai para ``corpus_limpo.csv``
    (gerado pelo 01-preprocessing) se a versão enriquecida não estiver presente.

    Isso permite que os outputs do 03-topic-modeling (``bertopic_results.csv``,
    ``lda_results.csv``, ``stm_results.csv``) herdem automaticamente a coluna
    ``sentiment`` quando o 03-sentiment tiver rodado antes — sem precisar de
    merge manual posterior.

    Parameters
    ----------
    input_dir : Path or str
        Diretório que contém o(s) CSV(s) (ex.: ``data/input/<corpus>/``).
    encoding : str
        Encoding do CSV (default: utf-8).
    verbose : bool
        Imprime qual arquivo foi carregado e se a coluna ``sentiment`` existe.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError
        Se nem ``corpus_com_sentimento.csv`` nem ``corpus_limpo.csv``
        estiverem em ``input_dir``.
    """
    input_dir = Path(input_dir)
    enriched = input_dir / "corpus_com_sentimento.csv"
    plain = input_dir / "corpus_limpo.csv"
    if enriched.exists():
        df = pd.read_csv(enriched, encoding=encoding)
        if verbose:
            print(f"Carregado: {enriched.name} ({len(df)} docs, com coluna 'sentiment')")
        return df
    if plain.exists():
        df = pd.read_csv(plain, encoding=encoding)
        if verbose:
            print(
                f"Carregado: {plain.name} ({len(df)} docs, SEM coluna 'sentiment' — "
                "rode 03-sentiment antes para ter cruzamento topic x sentiment automatico)"
            )
        return df
    raise FileNotFoundError(
        f"Nenhum corpus encontrado em '{input_dir}'. "
        "Esperado: 'corpus_com_sentimento.csv' (do 03-sentiment) ou "
        "'corpus_limpo.csv' (do 01-preprocessing)."
    )


# ===========================================================================
# embeddings.py
# ===========================================================================
"""Sentence-transformer embeddings with disk cache."""

logger = logging.getLogger(__name__)


async def _get_ollama_embedding(client, text, model, dimension, timeout=120.0, _retries=5):
    """Get embedding for a single text via Ollama API.

    If ``dimension`` is provided and smaller than the model's native size, the
    returned vector is **truncated to the first ``dimension`` components and
    re-normalized to unit length** — this is the Matryoshka-style truncation
    used by Qwen3-Embedding (and any model trained with MRL). When the native
    size is already smaller, returns as-is and logs a warning (the request
    cannot be honoured without retraining).

    Retries up to ``_retries`` times on 5xx errors **and on transient transport
    failures** (ConnectError/timeout) with exponential backoff — needed when
    Ollama runs in CPU/hybrid mode and the server momentarily drops the
    connection or restarts (queues saturated, model reload under memory
    pressure). Only 4xx (client) errors propagate immediately.
    """
    import asyncio as _asyncio
    last_exc = None
    for attempt in range(_retries):
        try:
            resp = await client.post(
                "http://localhost:11434/api/embeddings",
                json={"model": model, "prompt": text},
                timeout=timeout,
            )
        except (httpx.TransportError, httpx.TimeoutException) as exc:
            # Servidor caiu/reiniciou ou timeout — retenta com backoff.
            last_exc = exc
            wait = 2 ** attempt
            logger.warning(
                "Ollama conexão falhou (%s) (attempt %d/%d), retrying in %ds…",
                type(exc).__name__, attempt + 1, _retries, wait,
            )
            await _asyncio.sleep(wait)
            continue
        if resp.status_code < 500:
            resp.raise_for_status()
            break
        last_exc = resp
        wait = 2 ** attempt
        logger.warning("Ollama 5xx (attempt %d/%d), retrying in %ds…", attempt + 1, _retries, wait)
        await _asyncio.sleep(wait)
    else:
        # Retries esgotados: relança 5xx (Response) ou a última exceção de transporte.
        if isinstance(last_exc, httpx.Response):
            last_exc.raise_for_status()
        elif last_exc is not None:
            raise last_exc
    emb = resp.json()["embedding"]

    if dimension and len(emb) != dimension:
        if len(emb) > dimension:
            # Matryoshka truncation: take first D components, renormalize.
            arr = np.asarray(emb[:dimension], dtype=np.float32)
            norm = float(np.linalg.norm(arr))
            if norm > 0:
                arr = arr / norm
            return arr.tolist()
        else:
            logger.warning(
                "Ollama returned %dd embedding, requested %dd > native; "
                "returning native size.", len(emb), dimension,
            )
    return emb


async def _get_ollama_embeddings_chunk(client, texts, model, dimension, timeout=120.0, _retries=5):
    """Get embeddings for a chunk of texts in ONE request, via Ollama's batch
    endpoint (``/api/embed``, ``input: [...]``) — as opposed to the legacy
    ``/api/embeddings`` endpoint used by ``_get_ollama_embedding``, which
    takes one ``prompt`` per request.

    Same retry-on-5xx/transport-failure semantics as ``_get_ollama_embedding``.
    A 404 (server predates ``/api/embed``) is NOT retried — it propagates
    immediately so the caller can fall back to the per-text endpoint.
    """
    import asyncio as _asyncio
    last_exc = None
    for attempt in range(_retries):
        try:
            resp = await client.post(
                "http://localhost:11434/api/embed",
                json={"model": model, "input": texts},
                timeout=timeout,
            )
        except (httpx.TransportError, httpx.TimeoutException) as exc:
            last_exc = exc
            wait = 2 ** attempt
            logger.warning(
                "Ollama conexão falhou (%s) (attempt %d/%d), retrying in %ds…",
                type(exc).__name__, attempt + 1, _retries, wait,
            )
            await _asyncio.sleep(wait)
            continue
        if resp.status_code < 500:
            resp.raise_for_status()
            break
        last_exc = resp
        wait = 2 ** attempt
        logger.warning("Ollama 5xx (attempt %d/%d), retrying in %ds…", attempt + 1, _retries, wait)
        await _asyncio.sleep(wait)
    else:
        if isinstance(last_exc, httpx.Response):
            last_exc.raise_for_status()
        elif last_exc is not None:
            raise last_exc
    embs = resp.json()["embeddings"]

    if not dimension:
        return embs
    out = []
    for emb in embs:
        if len(emb) > dimension:
            arr = np.asarray(emb[:dimension], dtype=np.float32)
            norm = float(np.linalg.norm(arr))
            out.append((arr / norm).tolist() if norm > 0 else arr.tolist())
        elif len(emb) < dimension:
            logger.warning(
                "Ollama returned %dd embedding, requested %dd > native; "
                "returning native size.", len(emb), dimension,
            )
            out.append(emb)
        else:
            out.append(emb)
    return out


async def _get_ollama_embeddings_batch(texts, model, dimension, max_concurrent=1,
                                        timeout=120.0, chunk_size=32):
    """Get embeddings for multiple texts.

    Uses Ollama's batch endpoint (one HTTP request per ``chunk_size`` texts)
    — far fewer round-trips than one request per text. This is what made
    ``compute_embedding_coherence`` effectively hang on the BERTopic-folha
    run of 2026-08-21: ~20 keywords/tópico × 24 tópicos meant ~480 sequential
    requests to the old per-text endpoint at 3-16s each.

    Falls back to the per-text endpoint (``_get_ollama_embedding``) if the
    server predates ``/api/embed`` (404 on the first chunk).
    """
    if not texts:
        return []
    chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]
    semaphore = asyncio.Semaphore(max_concurrent)
    async with httpx.AsyncClient() as client:
        async def bounded_chunk(chunk):
            async with semaphore:
                return await _get_ollama_embeddings_chunk(client, chunk, model, dimension, timeout=timeout)
        try:
            chunk_results = await asyncio.gather(*(bounded_chunk(c) for c in chunks))
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 404:
                raise
            logger.warning(
                "Ollama sem /api/embed (servidor anterior a essa versão) — "
                "caindo para /api/embeddings, uma requisição por texto."
            )
            async def bounded_single(text):
                async with semaphore:
                    return await _get_ollama_embedding(client, text, model, dimension, timeout=timeout)
            return list(await asyncio.gather(*(bounded_single(t) for t in texts)))
    return [emb for chunk in chunk_results for emb in chunk]


def get_ollama_embeddings(
    texts: list[str],
    model: str,
    dimension: int = None,
    max_concurrent: int = 5,
    timeout: float = 120.0,
) -> np.ndarray:
    """Get embeddings from Ollama API. Returns numpy array.

    Handles Jupyter event loop via nest_asyncio if available.
    """
    try:
        import nest_asyncio
        nest_asyncio.apply()
    except ImportError:
        pass

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                results = pool.submit(
                    asyncio.run,
                    _get_ollama_embeddings_batch(texts, model, dimension, max_concurrent, timeout=timeout)
                ).result()
        else:
            results = loop.run_until_complete(
                _get_ollama_embeddings_batch(texts, model, dimension, max_concurrent, timeout=timeout)
            )
    except RuntimeError:
        results = asyncio.run(
            _get_ollama_embeddings_batch(texts, model, dimension, max_concurrent, timeout=timeout)
        )
    return np.array(results, dtype=np.float32)


# ===========================================================================
# lemmatize.py
# ===========================================================================
"""Lemmatization with language-aware spaCy model loading.

Supports PT-BR (`pt_core_news_lg`) and EN (`en_core_web_sm`).
Models are lazy-loaded and cached at module level.
"""

_NLP_CACHE: dict = {}

_LANG_TO_MODEL = {
    "pt": "pt_core_news_lg",
    "en": "en_core_web_sm",
}


def _get_nlp(lang: str):
    if lang not in _LANG_TO_MODEL:
        raise ValueError(
            f"Unsupported lang '{lang}'. Supported: {list(_LANG_TO_MODEL)}"
        )
    if lang not in _NLP_CACHE:
        import spacy
        try:
            _NLP_CACHE[lang] = spacy.load(_LANG_TO_MODEL[lang])
        except OSError as e:
            raise OSError(
                f"spaCy model '{_LANG_TO_MODEL[lang]}' not installed. "
                f"Run: python -m spacy download {_LANG_TO_MODEL[lang]}"
            ) from e
    return _NLP_CACHE[lang]


def lemmatize_corpus(
    docs: list[str],
    lang: str,
    params: dict,
    batch_size: int = 256,
    n_process: int = 1,
    model_key: str = "lda",
    extra_stopwords: list[str] | None = None,
) -> tuple[list[list[str]], Dictionary]:
    """Lemmatize a list of documents and build a filtered gensim Dictionary.

    Args:
        docs: raw document strings.
        lang: 'pt' or 'en'.
        params: dict with `<model_key>.no_below` and `<model_key>.no_above` for filter_extremes.
        batch_size: spaCy nlp.pipe batch size.
        n_process: spaCy nlp.pipe parallel workers (1 on Windows recommended).
        model_key: which params block to read `no_below`/`no_above` from
            ('lda' or 'nmf' — both share the same gensim Dictionary filtering).
        extra_stopwords: optional extra lemmas to drop (case-insensitive), on top of
            spaCy's built-in stop list — e.g. corpus_cfg['stopwords_emojis'], which
            already carries non-emoji noise tokens (dateline artifacts, HTML residue).
            None/default keeps prior behavior unchanged (opt-in, doesn't affect
            existing callers that don't pass it).

    Returns:
        (tokenized_docs, dictionary) where dictionary already had
        filter_extremes applied per params[model_key].
    """
    nlp = _get_nlp(lang)
    _extra_stop = {w.lower() for w in (extra_stopwords or [])}

    tokenized: list[list[str]] = []
    # Process via nlp.pipe for speed on large corpora
    for spacy_doc in nlp.pipe(docs, batch_size=batch_size, n_process=n_process):
        tokens = [
            tok.lemma_.lower()
            for tok in spacy_doc
            if (not tok.is_stop) and tok.is_alpha and len(tok.lemma_) > 2
            and tok.lemma_.lower() not in _extra_stop
        ]
        tokenized.append(tokens)

    dictionary = Dictionary(tokenized)
    model_cfg = params.get(model_key, {})
    dictionary.filter_extremes(
        no_below=model_cfg.get("no_below", 5),
        no_above=model_cfg.get("no_above", 0.5),
    )
    return tokenized, dictionary


# ===========================================================================
# naming.py
# ===========================================================================
"""Topic naming via Ollama LLM (PT-BR, with retry/temperature escalation)."""


# ---------------------------------------------------------------------------
# Prompt + response cleaning
# ---------------------------------------------------------------------------


def _strip_thinking(text: str) -> str:
    """Remove Qwen3 <think>...</think> blocks and surrounding quotes.

    Qwen3 may emit <think>...</think> followed by the answer. If the model
    returns *only* a thinking block (sem resposta após), extract a fallback
    candidate from inside the thinking block (last short line that looks like
    a label) — this prevents 100% fallback rate observed em prod.
    """
    original = text or ""
    cleaned = re.sub(r"<think>.*?</think>", "", original, flags=re.DOTALL).strip()
    cleaned = cleaned.strip('"').strip("'").strip()

    if cleaned:
        return cleaned

    # Modelo emitiu apenas <think>...</think> sem resposta. Tentar extrair
    # uma frase candidata de dentro do thinking (ultima linha curta).
    inner = re.findall(r"<think>(.*?)</think>", original, flags=re.DOTALL)
    if not inner:
        return ""
    last_block = inner[-1]
    # Pegar linhas curtas (3-50 chars) que pareçam título/rótulo
    candidates = [
        ln.strip(' "\'.,:;-')
        for ln in last_block.split("\n")
        if 3 <= len(ln.strip()) <= 50
    ]
    # Ultima candidata costuma ser a conclusao do raciocinio
    return candidates[-1] if candidates else ""


def _clean_label(raw: str) -> str:
    """Extract first useful line and strip prefixes/punctuation."""
    if not raw:
        return ""
    line = next((l.strip() for l in _strip_thinking(raw).split("\n") if l.strip()), "")
    line = line.strip('"\'`').rstrip(".!?;:")
    for pref in ("Rótulo:", "Rotulo:", "Label:", "Topic:", "Tópico:", "Nome:"):
        if line.lower().startswith(pref.lower()):
            line = line[len(pref):].strip()
    # Cap at 7 words — enough for descriptive labels without truncating mid-phrase.
    parts = line.split()
    if len(parts) > 7:
        line = " ".join(parts[:7])
    return line


_HTML_ENTITY_TOKENS = frozenset({
    "atilde", "ccedil", "iacute", "eacute", "aacute", "otilde", "uacute",
    "ocirc", "acirc", "ecirc", "iuml", "agrave", "egrave", "ograve",
    "amp", "nbsp", "quot", "apos", "lt", "gt",
    "ccedil atilde", "novamente representando",
})


def _filter_keywords(keywords: list[str]) -> list[str]:
    """Remove HTML entity tokens and other known-spurious terms from keyword list."""
    return [k for k in keywords if k.lower() not in _HTML_ENTITY_TOKENS]


def _build_prompt_pt(
    keywords: list[str],
    example_docs: list[str] | None,
    top_n: int,
    doc_max_chars: int,
    max_docs: int,
) -> str:
    """PT-BR prompt body (rótulos de tópico, jornalístico/notícias)."""
    clean_kws = _filter_keywords(keywords)
    keywords_str = ", ".join(clean_kws[:top_n])
    docs_block = ""
    if example_docs:
        snippets = []
        for i, d in enumerate(example_docs[:max_docs]):
            text = d if isinstance(d, str) else str(d)
            snippets.append(f"Documento {i + 1}: {text[:doc_max_chars].strip()}...")
        if snippets:
            docs_block = "\n\n" + "\n\n".join(snippets)

    return (
        "Voce e um especialista em analise de topicos de noticias jornalisticas brasileiras.\n\n"
        f"Palavras-chave: {keywords_str}"
        f"{docs_block}\n\n"
        "Tarefa: criar um rotulo descritivo em portugues brasileiro (4-7 palavras).\n"
        "O rotulo deve nomear o TEMA central, nao descrever uma acao.\n"
        "Responda APENAS com o rotulo, sem explicacoes, sem aspas, sem prefixos.\n"
        "Use substantivos concretos. Capitalize apenas a primeira palavra.\n\n"
        "/no_think\n\n"
        "Rotulo:"
    )


def _build_prompt_en(
    keywords: list[str],
    example_docs: list[str] | None,
    top_n: int,
    doc_max_chars: int,
    max_docs: int,
) -> str:
    """English prompt body (topic labels, generic news/benchmark domain)."""
    keywords_str = ", ".join(keywords[:top_n])
    docs_block = ""
    if example_docs:
        snippets = []
        for i, d in enumerate(example_docs[:max_docs]):
            text = d if isinstance(d, str) else str(d)
            snippets.append(f"Document {i + 1}: {text[:doc_max_chars].strip()}...")
        if snippets:
            docs_block = "\n\n" + "\n\n".join(snippets)

    return (
        "You are an expert in topic analysis for English-language news corpora.\n\n"
        f"Keywords: {keywords_str}"
        f"{docs_block}\n\n"
        "Task: create a short label in English (3-5 words).\n"
        "Respond ONLY with the label, no explanations, no quotes, no prefixes.\n"
        "Use concrete nouns. Capitalize only the first word.\n\n"
        "/no_think\n\n"
        "Label:"
    )


def build_prompt(
    keywords: list[str],
    example_docs: list[str] | None = None,
    top_n: int = 15,
    doc_max_chars: int = 200,
    max_docs: int = 1,
    lang: str = "pt",
) -> str:
    """Build a topic-naming prompt, language-aware via ``lang``.

    F2 — calibração para Qwen3:4b em CPU:
    - top_n=15 (vs 25 anterior) — reduz ruído da cauda longa
    - doc_max_chars=200 (vs 400) — encurta prompt
    - max_docs=1 (vs 2-3) — apenas o doc mais representativo
    - sufixo /no_think — instrui Qwen3 a desabilitar thinking mode

    Parameters
    ----------
    lang : str
        ``"pt"`` (default, preserva comportamento histórico — rotulos em
        portugues brasileiro) ou ``"en"`` (rotulos em ingles, para corpora
        EN). Qualquer outro valor cai em ``"pt"``.
    """
    builder = _build_prompt_en if lang == "en" else _build_prompt_pt
    return builder(keywords, example_docs, top_n, doc_max_chars, max_docs)


# ---------------------------------------------------------------------------
# Backend call (single attempt + retry wrapper)
# ---------------------------------------------------------------------------


_DEFAULT_SYSTEM_PROMPT_PT = (
    "Voce e um especialista em topicos. Responda APENAS com "
    "um rotulo curto (3-5 palavras), sem explicacoes."
)

_DEFAULT_SYSTEM_PROMPT_EN = (
    "You are a topic-labeling expert. Respond ONLY with "
    "a short label (3-5 words), no explanations."
)


def _call_ollama_once(
    keywords: list[str],
    example_docs: list[str] | None,
    model: str,
    base_url: str,
    temperature: float,
    prompt_builder=None,
    system_prompt: str | None = None,
    lang: str = "pt",
) -> str:
    """One Ollama call. Raises on transport/HTTP error.

    Parameters
    ----------
    prompt_builder : callable, optional
        Funcao customizada `(keywords, example_docs) -> str` para gerar o user
        prompt. Quando None, usa ``build_prompt`` padrao (language-aware via
        ``lang``). Permite que o notebook edite o prompt direto na cell sem
        mexer no naming.py.
    system_prompt : str, optional
        System prompt customizado. Default: enxuto otimizado para topic
        naming, selecionado por ``lang`` ("pt" ou "en").
    lang : str
        Idioma do rotulo quando ``prompt_builder``/``system_prompt`` nao sao
        passados explicitamente. Default ``"pt"`` preserva comportamento
        historico.
    """
    if prompt_builder is None:
        prompt_builder = lambda kws, docs: build_prompt(kws, docs, lang=lang)  # noqa: E731
    if system_prompt is None:
        system_prompt = (
            _DEFAULT_SYSTEM_PROMPT_EN if lang == "en" else _DEFAULT_SYSTEM_PROMPT_PT
        )

    client = ollama.Client(host=base_url)
    response = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_builder(keywords, example_docs)},
        ],
        options={
            "temperature": temperature,
            "top_p": 0.95,
            "num_predict": 256,        # Maior para acomodar thinking + resposta
            "repeat_penalty": 1.1,
            "think": False,            # Ollama recente: desabilita thinking no Qwen3
        },
    )
    return _clean_label(response["message"]["content"])


def _smart_fallback(keywords: list[str], lang: str = "pt") -> str:
    """Fallback inteligente: gera frase nominal em vez de lista de keywords.

    Em vez de "saude, hospital, leitos" (lista, parece MMR keywords), gera
    "Saude, hospital e leitos" — pelo menos é uma frase nominal natural.

    ``lang="en"`` gera o conector em ingles ("and") em vez de "e", para
    corpora EN. Default ``"pt"`` preserva comportamento historico.
    """
    kws = [k for k in keywords[:3] if k and len(k) >= 2]
    no_label_msg = "Topic without label" if lang == "en" else "Topico sem rotulo"
    connector = "and" if lang == "en" else "e"
    if not kws:
        return no_label_msg
    if len(kws) == 1:
        return kws[0].capitalize()
    if len(kws) == 2:
        return f"{kws[0].capitalize()} {connector} {kws[1]}"
    return f"{kws[0].capitalize()}, {kws[1]} {connector} {kws[2]}"


def _looks_like_keyword_list(label: str, keywords: list[str], threshold: float = 0.5) -> bool:
    """Detecta se o LLM devolveu lista de keywords disfarçada de rótulo.

    Um rótulo legítimo é uma frase nominal coerente. Se >threshold das partes
    separadas por vírgula são keywords do input, é lista (ruim).
    """
    if not label or "," not in label:
        return False
    parts = [p.strip().lower().rstrip(".") for p in label.split(",")]
    if len(parts) < 2:
        return False
    kws_lower = {k.lower() for k in keywords[:15] if k}
    matches = sum(1 for p in parts if p in kws_lower)
    return matches / len(parts) > threshold


def name_topic(
    keywords: list[str],
    model: str = "qwen3:4b",
    base_url: str = "http://localhost:11434",
    example_docs: list[str] | None = None,
    max_attempts: int = 3,
    prompt_builder=None,
    system_prompt: str | None = None,
    lang: str = "pt",
) -> str:
    """Name a single topic with retry + temperature escalation.

    Strategy
    --------
    Up to ``max_attempts`` calls; temperature escalates 0.2 → 0.5 → 0.8 to
    encourage variety on retries when a previous attempt returned an empty or
    too-short label. After all attempts fail, returns the top-3 keyword
    fallback so the pipeline never crashes on a single bad topic.

    Parameters
    ----------
    keywords : list[str]
        Representative keywords (top-10 recommended).
    model : str
        Ollama model identifier.
    base_url : str
        Ollama server base URL.
    example_docs : list[str] | None
        Optional representative documents to ground the prompt.
    max_attempts : int
        Max LLM calls before falling back.
    prompt_builder : callable, optional
        Override do prompt builder. Quando None, usa ``build_prompt`` com
        ``lang`` aplicado.
    system_prompt : str, optional
        Override do system prompt. Quando None, usa o default de ``lang``.
    lang : str
        Idioma do rotulo ("pt" ou "en") quando ``prompt_builder`` /
        ``system_prompt`` nao sao passados explicitamente. Tambem usado para
        selecionar o idioma do fallback (``_smart_fallback``). Default
        ``"pt"`` preserva comportamento historico.
    """
    fallback = _smart_fallback(keywords, lang=lang)
    temperatures = [0.2, 0.5, 0.8]
    last_error: str | None = None

    for attempt in range(max_attempts):
        temp = temperatures[min(attempt, len(temperatures) - 1)]
        try:
            label = _call_ollama_once(
                keywords, example_docs, model, base_url, temp,
                prompt_builder=prompt_builder, system_prompt=system_prompt,
                lang=lang,
            )
            if not label or len(label) < 3:
                last_error = (
                    f"resposta vazia/curta (tentativa {attempt + 1}, "
                    f"temp={temp})"
                )
            elif _looks_like_keyword_list(label, keywords):
                # LLM devolveu lista de keywords disfarcada de rotulo. Rejeita.
                last_error = (
                    f"resposta parece lista de keywords (tentativa "
                    f"{attempt + 1}, temp={temp}): {label!r}"
                )
            else:
                # Rotulo valido
                return label
        except Exception as e:
            last_error = (
                f"erro na chamada (tentativa {attempt + 1}): "
                f"{type(e).__name__}: {e}"
            )

        time.sleep(1)

    print(f"  [naming] fallback acionado: {last_error} -> '{fallback}'")
    return fallback


def name_all_topics(
    topics_keywords: dict[int, list[str]],
    model: str = "qwen3:4b",
    base_url: str = "http://localhost:11434",
    example_docs_map: dict[int, list[str]] | None = None,
    max_attempts: int = 3,
    prompt_builder=None,
    system_prompt: str | None = None,
    lang: str = "pt",
) -> dict[int, str]:
    """Name every topic. ``example_docs_map`` is optional per-topic context.

    Pass ``prompt_builder`` and/or ``system_prompt`` to override the defaults
    direto do notebook (sem editar src/naming.py).

    Parameters
    ----------
    lang : str
        Idioma do rotulo ("pt" ou "en"), usado para selecionar o prompt
        builder/system prompt/fallback padrao quando ``prompt_builder`` /
        ``system_prompt`` nao sao passados. Tipicamente vem de
        ``corpus_cfg["language"]`` (ver ``params.yaml``). Default ``"pt"``
        preserva comportamento historico para notebooks que nao passam este
        argumento.
    """
    return {
        tid: name_topic(
            kws,
            model=model,
            base_url=base_url,
            example_docs=(example_docs_map or {}).get(tid),
            max_attempts=max_attempts,
            prompt_builder=prompt_builder,
            system_prompt=system_prompt,
            lang=lang,
        )
        for tid, kws in topics_keywords.items()
    }


# ===========================================================================
# topic_utils.py
# ===========================================================================
"""Topic evaluation utilities: coherence, stability, Likert, Kappa, export."""


def _compute_coherence(
    topics_keywords: dict[int, list[str]],
    texts: list[list[str]],
    dictionary,
    measure: str,
    return_counts: bool = False,
    top_k: int | None = None,
):
    """Chamada compartilhada do ``CoherenceModel``. Ver os wrappers publicos.

    Filtra palavras fora do ``dictionary`` e descarta topicos que sobrem com
    menos de 2 palavras conhecidas — mesmo tratamento que ``_stm_cv`` ja fazia
    para o STM. O gensim TOLERA palavra OOV solta dentro de um topico, mas um
    topico 100% OOV levanta ValueError que derruba o calculo do conjunto
    INTEIRO: era o C_v NaN do BERTopic-tweets em top-10/15 (o topico de emojis
    nao tem nenhuma keyword no vocabulario de lemas — bigramas do vectorizer,
    tokens de emoji e fragmentos de URL). Verificado em 2026-08-01: onde o
    calculo ja funcionava, o filtro reproduz o mesmo valor ate 1e-6.

    ATENCAO ao comparar: descartar topico sem keyword conhecida remove
    justamente o pior topico, o que enviesa a coerencia para CIMA. Use
    ``return_counts=True`` para obter o diagnostico e reporte-o sempre que
    ``n_topicos_descartados > 0`` ou ``taxa_oov > 0``.
    """
    from gensim.models import CoherenceModel

    known = getattr(dictionary, "token2id", {})
    n_kw_total = 0
    n_kw_oov = 0
    topics_list: list[list[str]] = []
    for kws in topics_keywords.values():
        limpo = [w for w in kws if isinstance(w, str) and w]
        conhecidas = [w for w in limpo if w in known]
        n_kw_total += len(limpo)
        n_kw_oov += len(limpo) - len(conhecidas)
        topics_list.append(conhecidas)

    # Truncate topics BEFORE filtering (protocol §1.8, condition 1)
    if top_k is not None:
        topics_list = [kws[:top_k] for kws in topics_list]

    n_topicos = len(topics_list)
    topics_list = [kws for kws in topics_list if len(kws) >= 2]
    diag = {
        "n_topicos": n_topicos,
        "n_topicos_avaliados": len(topics_list),
        "n_topicos_descartados": n_topicos - len(topics_list),
        "n_keywords": n_kw_total,
        "n_keywords_oov": n_kw_oov,
        "taxa_oov": (n_kw_oov / n_kw_total) if n_kw_total else 0.0,
        "measure": measure,
        "top_k_declarado": top_k,
    }

    if not topics_list:
        return (0.0, diag) if return_counts else 0.0

    # ``topn`` do CoherenceModel tem default 20 e NAO acompanha o tamanho da lista:
    # pedir top-30 sem passa-lo devolve, silenciosamente, o valor de top-20. Era o
    # backlog P0c/C10 — e a tabela de robustez por top-N, que este projeto reporta
    # como contribuicao, repetia o mesmo numero de top-25 em diante para C_v e NPMI
    # enquanto Diversity/Exclusividade/FREX (que nao passam por aqui) variavam.
    # Verificado em 2026-08-16: sem esta linha, C_v(top-30) == C_v(top-20) bit a bit.
    profundidade = max(len(kws) for kws in topics_list)
    diag["topn_efetivo"] = profundidade

    cm = CoherenceModel(
        topics=topics_list,
        texts=texts,
        dictionary=dictionary,
        coherence=measure,
        topn=profundidade,
        processes=1,  # avoid multiprocessing issues on Windows
    )
    score = cm.get_coherence()
    return (score, diag) if return_counts else score


def compute_coherence_cv(
    topics_keywords: dict[int, list[str]],
    texts: list[list[str]],
    dictionary,
    return_counts: bool = False,
    top_k: int | None = None,
):
    """Coerencia C_v (Roder et al. 2015) via gensim.

    ATENCAO ao regime de janela. O C_v e
    ``P_sw(110) + S_one-set + m_cos(nlr,1) + sigma_a``: a janela deslizante de
    110 palavras e parte da definicao, e Roder et al. so validam as variantes
    com janela ``s >= 50``. Quando o documento e mais curto que a janela, ela
    degenera numa unica janela igual ao texto inteiro e o C_v deixa de ser o
    C_v medido em 0,731 — vira outra combinacao. Medido nos corpora deste
    projeto sobre os lemas que alimentam esta funcao (2026-08-16):

        folha           mediana 335 lemas —   2,2% dos docs abaixo de 110
        tweets_bre2022  mediana   9 lemas — 100,0% dos docs abaixo de 110

    Ou seja: na folha o C_v e legitimo; nos tweets o numero NAO deve ser
    publicado sob o rotulo "C_v". Use ``compute_coherence_npmi``, cuja janela
    de 10 palavras degenera para a definicao canonica de Bouma (2009) em vez
    de para uma medida nao validada. Detalhe em
    ``docs/protocolo-selecao-final-2026-08-16.md`` §3.0.

    Com ``return_counts=True`` devolve ``(score, diagnostico)`` — ver
    ``_compute_coherence``.

    ``top_k`` (protocolo secao 1.8, condicao 1): profundidade DECLARADA da
    lista. None mantem o comportamento historico — a profundidade e derivada
    da maior lista recebida, o que torna dois bracos incomparaveis sem aviso.
    A camada 3 SEMPRE passa top_k explicito.
    """
    return _compute_coherence(topics_keywords, texts, dictionary, "c_v", return_counts, top_k)


def compute_coherence_npmi(
    topics_keywords: dict[int, list[str]],
    texts: list[list[str]],
    dictionary,
    return_counts: bool = False,
    top_k: int | None = None,
):
    """Coerencia NPMI (gensim ``c_npmi``), a medida de Bouma (2009).

    Para um par de palavras,

        NPMI(w_i, w_j) = PMI(w_i, w_j) / -log P(w_i, w_j)
        PMI(w_i, w_j)  = log P(w_i, w_j) / (P(w_i) P(w_j))

    com imagem em [-1, 1]: +1 sempre coocorrem, 0 independentes, -1 nunca.
    Score do topico = media sobre os pares do top-N; do modelo = media sobre
    topicos.

    POR QUE ESTA E A COERENCIA DO PROTOCOLO, e nao o C_v:

    1. **Comparabilidade externa.** E a medida da avaliacao do proprio
       BERTopic (Grootendorst 2022, via OCTIS, citando Bouma 2009), ao lado da
       Topic Diversity de Dieng et al. (2020). Sem ela nenhum numero deste
       projeto confronta benchmark publicado do modelo.
    2. **Estabilidade de regime.** O ``c_npmi`` usa janela de 10 palavras. Nos
       tweets (mediana 9 lemas) ela degenera — mas degenera para coocorrencia
       por documento, que E a definicao canonica de Bouma. A degeneracao do
       C_v e maligna (troca a medida); a do NPMI e benigna (converge para a
       forma original). E a unica coerencia cujo estimador nao muda de regime
       entre os dois corpora deste projeto.
    3. O C_v usa NPMI internamente, mas acrescenta a etapa de cosseno entre
       vetores de contexto — justamente a criticada na literatura pos-Roder
       (Doogan & Buntine 2021; Hoyle et al. 2021). Este e o calculo direto.

    ESCALA DIFERENTE do C_v: e coluna adicional, nunca substituta na mesma
    coluna. Valores negativos sao normais e nao indicam erro — no benchmark do
    proprio Grootendorst a faixa vai de -0,213 a +0,167.

    NOTA de estimacao: as coocorrencias vem do corpus de modelagem, nao de uma
    colecao de referencia externa (a Wikipedia, no desenho original). Declarar
    isso junto com o numero.

    Com ``return_counts=True`` devolve ``(score, diagnostico)`` — ver
    ``_compute_coherence``.

    ``top_k`` (protocolo secao 1.8, condicao 1): profundidade DECLARADA da
    lista. None mantem o comportamento historico — a profundidade e derivada
    da maior lista recebida, o que torna dois bracos incomparaveis sem aviso.
    A camada 3 SEMPRE passa top_k explicito.
    """
    return _compute_coherence(topics_keywords, texts, dictionary, "c_npmi", return_counts, top_k)


def compute_jaccard(set_a: set, set_b: set) -> float:
    """Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union


def compute_stability(
    results_per_seed: dict[int, dict[int, list[str]]],
) -> tuple[float, float]:
    """Compute topic stability across seeds via greedy Jaccard matching."""
    seeds = list(results_per_seed.keys())
    if len(seeds) < 2:
        return float("nan"), float("nan")

    pair_scores = []
    for seed_a, seed_b in combinations(seeds, 2):
        topics_a = results_per_seed[seed_a]
        topics_b = results_per_seed[seed_b]

        matched_scores = []
        used_b = set()
        for tid_a, kws_a in topics_a.items():
            best_score = 0.0
            best_tid = None
            for tid_b, kws_b in topics_b.items():
                if tid_b in used_b:
                    continue
                score = compute_jaccard(set(kws_a), set(kws_b))
                if score > best_score:
                    best_score = score
                    best_tid = tid_b
            if best_tid is not None:
                used_b.add(best_tid)
            matched_scores.append(best_score)

        if matched_scores:
            pair_scores.append(np.mean(matched_scores))

    if not pair_scores:
        return 0.0, 0.0
    return float(np.mean(pair_scores)), float(np.std(pair_scores))


# ---------------------------------------------------------------------------
# Protocolo 2026-08-16 — regime de janela e tabela unica de metricas
# ---------------------------------------------------------------------------

CV_WINDOW = 110       # janela deslizante do C_v (Roder et al. 2015)
NPMI_WINDOW = 10      # janela do c_npmi do gensim (Bouma 2009)
CV_DEGENERACAO_MAX = 0.50   # acima disso o rotulo "C_v" deixa de ser honesto


def diagnose_cv_window(tokenized: list[list[str]], window: int = CV_WINDOW) -> dict:
    """Mede a degeneracao da janela deslizante NESTE corpus.

    O C_v e ``P_sw(110) + S_one-set + m_cos(nlr,1) + sigma_a``. Quando o
    documento e mais curto que a janela, ela produz UMA unica janela igual ao
    texto inteiro e o ``P_sw(110)`` degenera em ``P_bd`` (boolean document) —
    outra combinacao, que Roder et al. mediram e que NAO e a validada em 0,731.
    Os autores so sustentam as variantes com janela ``s >= 50``.

    Isto e propriedade do TEXTO, nao do modelo: LDA e BERTopic sobre o mesmo
    corpus degeneram igual. Por isso a funcao recebe ``tokenized`` e nao um
    modelo, e por isso a regra resultante e por corpus.

    Medido nos corpora deste projeto em 2026-08-16 (lemas, model_key="lda"):
    folha mediana 335 -> 2,2% degenerado; tweets_bre2022 mediana 9 -> 100%.

    Devolve tambem ``frac_abaixo_npmi``: a degeneracao do NPMI e BENIGNA (ele
    converge para coocorrencia por documento, que E a definicao de Bouma),
    enquanto a do C_v e maligna. O campo existe para poder declarar as duas.
    """
    tamanhos = np.array([len(t) for t in tokenized], dtype=float)
    if tamanhos.size == 0:
        return {"n_docs": 0, "mediana_tokens": 0.0, "frac_abaixo_janela": 0.0,
                "frac_abaixo_npmi": 0.0, "cv_degenerado": False, "janela": window}
    frac = float((tamanhos < window).mean())
    return {
        "n_docs": int(tamanhos.size),
        "mediana_tokens": float(np.median(tamanhos)),
        "frac_abaixo_janela": frac,
        "frac_abaixo_npmi": float((tamanhos < NPMI_WINDOW).mean()),
        "cv_degenerado": bool(frac > CV_DEGENERACAO_MAX),
        "janela": window,
    }


def export_results(
    topics: list[int],
    probs: list[list[float]],
    names: dict[int, str],
    texts: list[str],
    output_path: str,
    topic_type: str = "probabilistic",
    granularity: str = "unit",
    post_ids: list[str] = None,  # NEW: original post_id values for cross-axis joins
) -> None:
    """Export model results to standardized CSV.

    Parameters
    ----------
    topics: dominant topic index per document
    probs: full probability distribution per document
    names: mapping from topic_id to human-readable name
    texts: raw text per document
    output_path: destination CSV path
    topic_type: 'semantic' or 'probabilistic'
    granularity: 'unit' or 'monthly'
    post_ids: optional list of original post_id strings.  When provided a
        ``post_id`` column is inserted as the first column so outputs can be
        joined doc-a-doc com `03-sentiment/data/output/<corpus>/sentiment_results.csv`.
        ``doc_id`` (sequential integer) is kept for backward compatibility.
    """
    def _to_native(p):
        """Convert a probability row to native Python floats for JSON serialization."""
        return [float(v) for v in p]

    df = pd.DataFrame({
        "doc_id": range(len(topics)),
        "text": texts,
        "topic_id": topics,
        "topic_name": [names.get(t, f"Topic_{t}") for t in topics],
        "topic_prob_distribution": [json.dumps(_to_native(p)) for p in probs],
    })
    df["topic_type"] = topic_type
    df["granularity"] = granularity

    if post_ids is not None:
        df.insert(0, "post_id", post_ids)

    df.to_csv(output_path, index=False, encoding="utf-8")


def export_topics_for_eval(
    topics_keywords: dict[int, list[str]],
    topics_names: dict[int, str],
    model_name: str,
    output_path: str,
) -> None:
    """Export topics in standard format for comparative evaluation."""
    rows = []
    for tid, kws in topics_keywords.items():
        rows.append({
            "topic_id": tid,
            "topic_name": topics_names.get(tid, f"Topic_{tid}"),
            "keywords": ", ".join(kws),
            "model": model_name,
        })
    pd.DataFrame(rows).to_csv(output_path, index=False)


def compute_exclusivity(topics_keywords: dict[int, list[str]]) -> float:
    """Proportion of keywords that appear in only one topic.

    Coarse binary version (presence/absence). Use ``compute_exclusivity_ctfidf``
    for the continuous version that follows the BERTopic literature.
    """
    all_words = []
    for kws in topics_keywords.values():
        all_words.extend(kws)
    if not all_words:
        return 0.0
    unique_count = sum(1 for w in set(all_words) if all_words.count(w) == 1)
    return unique_count / len(set(all_words))


def compute_exclusivity_ctfidf(
    topics_keywords: dict[int, list[str]],
    topic_word_scores: dict[int, dict[str, float]],
    top_n: int = 10,
) -> tuple[float, dict[int, float]]:
    """Continuous exclusivity via c-TF-IDF weight distribution.

    For each top-N keyword w of topic t, exclusivity is the share of total
    c-TF-IDF mass of w concentrated in t:

        excl(w, t) = score(w, t) / sum_{t'} score(w, t')

    Topic exclusivity is the mean over its top-N keywords. The function returns
    both the corpus-level mean and the per-topic dict.

    Parameters
    ----------
    topics_keywords : dict[int, list[str]]
        topic_id -> ordered keyword list (top-N already truncated or fuller).
    topic_word_scores : dict[int, dict[str, float]]
        topic_id -> {word: c-TF-IDF score}. Required.
    top_n : int
        Number of keywords per topic to score.
    """
    per_topic: dict[int, float] = {}
    for tid, kws in topics_keywords.items():
        if not kws:
            per_topic[tid] = 0.0
            continue
        scores = []
        for w in kws[:top_n]:
            num = topic_word_scores.get(tid, {}).get(w, 0.0)
            denom = sum(
                topic_word_scores.get(other_tid, {}).get(w, 0.0)
                for other_tid in topics_keywords
            )
            if denom > 0:
                scores.append(num / denom)
        per_topic[tid] = float(np.mean(scores)) if scores else 0.0

    mean_excl = float(np.mean(list(per_topic.values()))) if per_topic else 0.0
    return mean_excl, per_topic


def compute_ctfidf_scores(docs_by_topic: dict[int, list[str]]) -> dict[int, dict[str, float]]:
    """Class-based c-TF-IDF (estilo BERTopic ClassTfidfTransformer).

    Para modelos SEM c-TF-IDF nativo (LDA/NMF/STM): agrupa os documentos por
    tópico dominante e devolve topic_word_scores = {tid: {word: peso c-TF-IDF}},
    comparável ao `topic_model.c_tf_idf_` que o BERTopic expõe.

    Fórmula: w(x,c) = tf(x,c) * log(1 + A / f_x), onde tf é a contagem de x na
    classe c normalizada pelo total de palavras da classe (L1), A = média de
    palavras por classe e f_x = frequência total de x entre as classes.

    Parameters
    ----------
    docs_by_topic : dict[int, list[str]]
        topic_id -> lista de documentos (cada doc = string de tokens já
        lematizados separados por espaço), agrupados pelo tópico dominante.
    """
    from sklearn.feature_extraction.text import CountVectorizer

    tids = sorted(docs_by_topic)
    class_docs = [" ".join(docs_by_topic[t]) for t in tids]  # 1 doc concatenado/classe
    if not any(d.strip() for d in class_docs):
        return {t: {} for t in tids}
    cv = CountVectorizer(token_pattern=r"(?u)\b\w+\b")  # tokens já lematizados; mantém curtos
    X = cv.fit_transform(class_docs).toarray().astype(float)  # n_classes x vocab
    vocab = cv.get_feature_names_out()
    words_per_class = X.sum(axis=1)
    tf = X / np.maximum(words_per_class[:, None], 1e-12)  # L1 por classe
    f_x = X.sum(axis=0)                                    # freq total por palavra
    A = X.sum() / max(len(tids), 1)                        # média de palavras/classe
    idf = np.log(1 + A / np.maximum(f_x, 1e-12))
    ctfidf = tf * idf
    scores: dict[int, dict[str, float]] = {}
    for i, t in enumerate(tids):
        row = ctfidf[i]
        scores[t] = {vocab[j]: float(row[j]) for j in range(len(vocab)) if row[j] > 0}
    return scores


def compute_topic_diversity(
    topics_keywords: dict[int, list[str]],
    top_k: int = 10,
) -> float:
    """Topic Diversity (Dieng et al., 2020).

    TD = |unique(top-k keywords across all topics)| / (k * |valid_topics|)

    Range: [0, 1]. Higher = topics share less vocabulary.

    Skips topics with fewer than 2 valid keywords (typical of -1/outlier or
    degenerate clusters when MMR returns empty strings).
    """
    valid_word_lists: list[list[str]] = []
    for tid, kws in topics_keywords.items():
        if tid == -1:
            continue
        clean = [w for w in kws[:top_k] if w]
        if len(clean) < 2:
            continue
        valid_word_lists.append(clean)

    if not valid_word_lists:
        return 0.0

    all_words = [w for kws in valid_word_lists for w in kws]
    unique_count = len(set(all_words))
    return unique_count / (top_k * len(valid_word_lists))


def compute_embedding_coherence(
    topics_keywords: dict[int, list[str]],
    embedding_func,
) -> float:
    """Mean cosine similarity between embeddings of top-N terms per topic.

    Renomeada em 2026-08-19 (protocolo, C5): chamava-se ``compute_semantic_coherence``,
    o que a confundia com a *semantic coherence* nativa do STM (Mimno et al. 2011,
    familia UMass — ``semantic_coherence`` em ``stm_grid_k_diagnostics.csv``), uma
    quantidade diferente. Esta e media de cosseno entre embeddings dos termos, nao
    tem relacao com a formula de Mimno. Ver ``docs/protocolo-selecao-final-2026-08-16.md``
    §4.2.3, "armadilha de rotulo 2".

    Parameters
    ----------
    topics_keywords: dict mapping topic_id to list of keywords
    embedding_func: callable that takes list[str] and returns np.ndarray of shape (n, dim)
    """
    from sklearn.metrics.pairwise import cosine_similarity
    scores = []
    for tid, kws in topics_keywords.items():
        if len(kws) < 2:
            continue
        try:
            embs = embedding_func(kws)
        except Exception as exc:  # backend de embeddings indisponível (ex.: Ollama fora)
            import warnings
            warnings.warn(
                "compute_embedding_coherence: backend de embeddings indisponível "
                f"({type(exc).__name__}: {exc}). Retornando NaN. "
                "Verifique se o Ollama está no ar (ollama serve) na porta 11434.",
                RuntimeWarning,
                stacklevel=2,
            )
            return float("nan")
        sim = cosine_similarity(embs)
        n = len(kws)
        total = sum(float(sim[i][j]) for i in range(n) for j in range(i + 1, n))
        pairs = n * (n - 1) / 2
        scores.append(total / pairs if pairs > 0 else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def compute_frex_score(
    topics_keywords: dict[int, list[str]],
    topic_word_matrix: np.ndarray,
    vocab_index: dict[str, int],
    topic_index: dict[int, int],
    top_n: int = 10,
    w_freq: float = 0.5,
) -> tuple[float, dict[int, float]]:
    """FREX score (Airoldi & Bischof, 2016) — harmonic mean of percentile ranks.

    For each topic t and each top-N keyword w:

        F(w, t) = percentile rank of c-TF-IDF(w, t) in topic t's row
        E(w, t) = percentile rank of exclusivity(w, t) in topic t's row
        FREX(w, t) = 1 / (w_freq / F + (1 - w_freq) / E)      (harmonic mean)

    Topic FREX is the mean over its top-N keywords; the corpus-level FREX is
    the mean across topics. Returns (mean_frex, per_topic_dict).

    Parameters
    ----------
    topics_keywords : dict[int, list[str]]
        topic_id -> ordered keyword list.
    topic_word_matrix : np.ndarray
        Matrix of shape (n_topics, vocab_size) with c-TF-IDF (or equivalent
        topic-word weights). Rows correspond to topic_index keys.
    vocab_index : dict[str, int]
        Maps vocabulary token -> column index in ``topic_word_matrix``.
    topic_index : dict[int, int]
        Maps topic_id -> row index in ``topic_word_matrix``. Must include
        every topic appearing in ``topics_keywords``; -1 (outliers) is ignored
        if absent.
    top_n : int
        Keywords per topic to score.
    w_freq : float
        Weight on frequency in the harmonic mean (0.5 is the canonical value).
    """
    from scipy.stats import rankdata

    if topic_word_matrix.size == 0:
        return 0.0, {}

    # Total word mass across topics for exclusivity computation.
    valid_rows = [topic_index[t] for t in topics_keywords if t in topic_index and t != -1]
    if not valid_rows:
        return 0.0, {}
    word_total = topic_word_matrix[valid_rows].sum(axis=0) + 1e-12

    per_topic: dict[int, float] = {}
    for tid, kws in topics_keywords.items():
        if tid == -1 or tid not in topic_index:
            continue
        row = topic_word_matrix[topic_index[tid]]
        excl = row / word_total

        # Percentile ranks within this topic
        f_rank = rankdata(row) / len(row)
        e_rank = rankdata(excl) / len(excl)

        idxs = [vocab_index[w] for w in kws[:top_n] if w in vocab_index]
        if not idxs:
            per_topic[tid] = 0.0
            continue

        scores = []
        for i in idxs:
            f = max(f_rank[i], 1e-12)
            e = max(e_rank[i], 1e-12)
            frex = 1.0 / (w_freq / f + (1 - w_freq) / e)
            scores.append(frex)
        per_topic[tid] = float(np.mean(scores))

    mean_frex = float(np.mean(list(per_topic.values()))) if per_topic else 0.0
    return mean_frex, per_topic


def _build_cooccurrence_matrix(texts: list[list[str]], vocab: set[str]) -> tuple[dict, dict]:
    """Pre-compute co-occurrence counts for all word pairs in vocab."""
    from collections import defaultdict
    cooccur = defaultdict(int)
    doc_freq = defaultdict(int)

    for doc in texts:
        words_in_doc = set(doc) & vocab
        for w in words_in_doc:
            doc_freq[w] += 1
        words_list = sorted(words_in_doc)
        for i, wa in enumerate(words_list):
            for wb in words_list[i + 1:]:
                cooccur[(wa, wb)] += 1

    return dict(cooccur), dict(doc_freq)


def compute_diversity(topic_distributions: list[list[float]]) -> float:
    """Compute topic diversity via entropy of document-topic distribution.

    Higher entropy = topics distribute more evenly across documents.
    """
    if not topic_distributions:
        return 0.0
    arr = np.array(topic_distributions)
    avg_dist = arr.mean(axis=0)
    avg_dist = avg_dist[avg_dist > 0]
    if len(avg_dist) == 0:
        return 0.0
    entropy = -np.sum(avg_dist * np.log2(avg_dist))
    max_entropy = np.log2(len(avg_dist)) if len(avg_dist) > 1 else 1.0
    return float(entropy / max_entropy)


# ===========================================================================
# bertopic_sweep.py
# ===========================================================================
"""BERTopic outlier-strategy/threshold sweep — multi-seed scoring of
reduce_outliers variants via coherence, diversity, exclusivity, FREX and
Jaccard stability.

Corpus-agnostic, same calling convention as ``grid_search_k`` below: pure
functions, no hardcoded corpus/paths/seeds, no file I/O — the notebook
supplies ``build_model`` + the corpus's docs/embeddings/tokenized/dictionary
and owns any caching of the returned DataFrames. Fills the gap the BERTopic
docs leave open: there's no built-in way to compare
off/c-tf-idf/embeddings/probabilities/distributions (or a threshold sweep
within one strategy) empirically across seeds.
"""


def _bertopic_postprocess(
    model,
    docs: list[str],
    embeddings: np.ndarray,
    strategy: str,
    threshold: float = 0.0,
    reduce_nr: int | None = None,
) -> tuple[float, float, int]:
    """Apply reduce_outliers(strategy, threshold) + conditional reduce_topics,
    in the canonical 4-step post-processing order (reduce_outliers ->
    update_topics -> reduce_topics -> update_topics). ``strategy="off"``
    skips step 1 entirely.

    Returns (outlier_pre, outlier_post, n_raw_topics).
    """
    topics0 = model.topics_
    outlier_pre = sum(1 for t in topics0 if t == -1) / len(topics0)
    n_raw = len([t for t in set(topics0) if t != -1])

    if strategy != "off":
        kw = dict(documents=docs, topics=topics0, strategy=strategy, threshold=threshold)
        if strategy == "embeddings":
            kw["embeddings"] = embeddings
        elif strategy == "probabilities":
            kw["probabilities"] = model.probabilities_
        new_topics = model.reduce_outliers(**kw)
        model.update_topics(
            docs, topics=new_topics, vectorizer_model=model.vectorizer_model,
            ctfidf_model=model.ctfidf_model, representation_model=model.representation_model,
        )

    n_now = len([t for t in model.get_topic_info()["Topic"] if t != -1])
    if reduce_nr and n_now > reduce_nr:
        model.reduce_topics(docs, nr_topics=reduce_nr)
        model.update_topics(
            docs, topics=model.topics_, vectorizer_model=model.vectorizer_model,
            ctfidf_model=model.ctfidf_model, representation_model=model.representation_model,
        )

    outlier_post = sum(1 for t in model.topics_ if t == -1) / len(model.topics_)
    return outlier_pre, outlier_post, n_raw


def _bertopic_sweep_metrics(
    model,
    tokenized: list[list[str]],
    dictionary,
    top_n_metrics: int = 20,
) -> tuple[dict, dict[int, list[str]]]:
    """Score a post-processed BERTopic model: K, c_v, diversity, exclusivity, FREX.

    Returns (metrics_dict, {topic_id: keywords}); the keyword map is what
    callers accumulate across seeds to feed ``compute_stability``.
    """
    topics = model.topics_
    valid_ids = sorted(t for t in set(topics) if t != -1)
    topics_keywords = {
        tid: [w for w, _ in (model.get_topic(tid) or []) if isinstance(w, str)]
        for tid in valid_ids
    }
    topic_index = {tid: i for i, tid in enumerate(sorted(model.get_topics().keys()))}
    ctfidf_matrix = model.c_tf_idf_.toarray() if model.c_tf_idf_ is not None else np.zeros((1, 1))
    vocab = model.vectorizer_model.get_feature_names_out()
    vocab_index = {w: i for i, w in enumerate(vocab)}

    # Exclusividade (F1): topic_word_scores DEVE vir da MATRIZ COMPLETA c-TF-IDF,
    # nao do top-N de model.get_topic() (cache). Se usar so o cache, o denominador
    # da exclusividade ignora o peso real de uma palavra nos OUTROS topicos (tratado
    # como 0 quando ela nao esta no top-N deles), inflando a exclusividade. Espelha
    # o fix F1 ja presente no notebook da folha — mantem a coluna Exclus do
    # sweep comparavel com a metrica do pipeline principal.
    topic_word_scores: dict[int, dict[str, float]] = {}
    if model.c_tf_idf_ is not None:
        for tid in valid_ids:
            if tid in topic_index:
                row = ctfidf_matrix[topic_index[tid]]
                topic_word_scores[tid] = {
                    vocab[j]: float(row[j]) for j in range(len(vocab)) if row[j] > 0
                }
            else:
                topic_word_scores[tid] = {}
    else:
        topic_word_scores = {tid: {w: float(s) for w, s in model.get_topic(tid)} for tid in valid_ids}

    tk_metrics = {tid: kws[:top_n_metrics] for tid, kws in topics_keywords.items()}
    # NPMI — eixo de ORDENACAO do protocolo 2026-08-16. O C_v continua sendo
    # calculado (comparabilidade com quem so reporta C_v) mas nao decide.
    try:
        npmi = compute_coherence_npmi(tk_metrics, tokenized, dictionary)
    except ValueError:
        npmi = float("nan")
    try:
        cv = compute_coherence_cv(tk_metrics, tokenized, dictionary)
    except ValueError:
        # gensim CoherenceModel exige que cada topico tenha >=1 token no
        # dictionary (unigramas); com ngram_range=(1,2) um topico pode cair
        # 100% em bigramas fora do vocabulario unigrama — mesmo modo de falha
        # ja tratado no pipeline principal (notebook, celula "Metricas
        # quantitativas completas"), aqui so precisa nao derrubar o sweep.
        cv = float("nan")
    div = compute_topic_diversity(topics_keywords, top_k=top_n_metrics)
    excl, _ = compute_exclusivity_ctfidf(topics_keywords, topic_word_scores, top_n=top_n_metrics)
    frex, _ = compute_frex_score(
        topics_keywords, topic_word_matrix=ctfidf_matrix,
        vocab_index=vocab_index, topic_index=topic_index, top_n=top_n_metrics, w_freq=0.5,
    )
    seed_kws = {tid: [w for w in kws if w] for tid, kws in topics_keywords.items()}
    seed_kws = {tid: kws for tid, kws in seed_kws.items() if len(kws) >= 2}
    return dict(K=len(valid_ids), npmi=npmi, cv=cv, div=div, excl=excl, frex=frex), seed_kws


def _checkpoint_append_row(path, row: dict) -> None:
    """Append ONE row to a checkpoint CSV, header written only on first call.

    Best-effort survival for long grids/sweeps: o sweep_bertopic_grid da
    Folha rodou 6h+ em 2026-08-22 e foi morto sem deixar nenhum resultado
    parcial (sem print, sem CSV incremental — so devolvia tudo no final).
    ``path=None`` desliga (comportamento antigo, sem custo).

    Grava tambem uma linha de texto plano num ``.log`` irmao do CSV, com
    flush + fsync imediatos — ver ``_progress_log_line``. Existe porque
    ``jupyter nbconvert --execute`` NAO repassa os ``print()`` de dentro das
    celulas para o stdout do proprio processo em tempo real: o canal IOPub do
    kernel so e escrito de volta no ``.ipynb`` quando a execucao inteira
    termina. Numa grade de horas isso deixa quem acompanha de fora (log do
    nbconvert) sem nenhum sinal de progresso ate o fim — so o CSV/log escrito
    direto em disco por esta funcao e visivel ao vivo (2026-08-23).
    """
    if not path:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)
    _progress_log_line(path.with_suffix(".log"), row)


def _progress_log_line(log_path, row: dict) -> None:
    """Acrescenta uma linha de progresso legivel a ``log_path``, com flush+fsync
    imediatos — ver a nota em ``_checkpoint_append_row`` sobre por que isso
    existe (nbconvert nao da visibilidade em tempo real de outra forma).
    """
    linha = " ".join(f"{k}={v}" for k, v in row.items())
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] {linha}\n")
        f.flush()
        os.fsync(f.fileno())


def _checkpoint_reset(path) -> None:
    """Remove a stale checkpoint from a previous attempt before a fresh grid starts.

    Nenhuma das funcoes que usam ``_checkpoint_append_row`` LE o checkpoint para
    retomar de onde parou — cada chamada recomputa a grade inteira do zero. Sem
    este reset, um checkpoint deixado por uma tentativa anterior (ex.: processo
    morto no meio, como o sweep da Folha em 2026-08-22) fica misturado com as
    linhas da tentativa nova em ``_checkpoint_append_row`` (que so faz append),
    e quem inspeciona o CSV pos-crash para medir progresso ve linhas duplicadas
    ou obsoletas de duas execucoes diferentes. ``path=None`` desliga, igual
    ``_checkpoint_append_row``.
    """
    if not path:
        return
    path = Path(path)
    if path.exists():
        path.unlink()


def sweep_outlier_strategies(
    build_model,
    docs: list[str],
    embeddings: np.ndarray,
    tokenized: list[list[str]],
    dictionary,
    seeds: Iterable[int],
    strategies: Iterable[str] = ("off", "c-tf-idf", "embeddings", "probabilities", "distributions"),
    reduce_nr: int | None = None,
    top_n_metrics: int = 20,
    checkpoint_path=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Multi-seed sweep over ``reduce_outliers`` strategies.

    ``checkpoint_path`` (opcional): grava cada linha em CSV assim que e
    computada — sobrevive a kill/crash no meio do sweep (ver
    ``_checkpoint_append_row``).

    For each seed, fits ONE base model via ``build_model(seed)`` then
    deep-copies it once per strategy so every variant starts from the same
    fit — only post-processing differs. Scores each variant with
    c_v/diversity/exclusivity/FREX plus, per strategy, multi-seed Jaccard
    stability (``compute_stability``).

    Parameters
    ----------
    build_model : Callable[[int], BERTopic]
        Factory returning a fresh, *unfit* BERTopic instance configured for
        the given seed (UMAP/HDBSCAN ``random_state``). The caller owns the
        embedding model, vectorizer, ctfidf and representation model choices
        — this function only fits/copies/post-processes/scores.
    docs, embeddings : corpus texts and pre-computed embeddings (aligned).
    tokenized, dictionary : gensim inputs for c_v coherence.
    seeds : seeds to fit the base model with (stability needs >= 2).
    strategies : reduce_outliers strategies to compare; ``"off"`` skips
        reduce_outliers entirely (the 4-step post-processing minus step 1).
    reduce_nr : if set and the post-RO topic count exceeds it, applies
        ``reduce_topics(nr_topics=reduce_nr)`` — step 3 of the post-processing.
    top_n_metrics : keywords per topic fed to the metric functions.

    Returns
    -------
    (raw, agg) : ``raw`` has one row per (seed, strategy); ``agg`` has one
        row per strategy with means + Jaccard stability across seeds.
    """
    _checkpoint_reset(checkpoint_path)
    import copy

    rows = []
    kws_by_strategy: dict[str, dict[int, dict[int, list[str]]]] = {s: {} for s in strategies}

    for seed in seeds:
        base = build_model(seed)
        base.fit_transform(docs, embeddings=embeddings)
        for strat in strategies:
            m = copy.deepcopy(base)
            o_pre, o_post, n_raw = _bertopic_postprocess(
                m, docs, embeddings, strat, reduce_nr=reduce_nr,
            )
            met, seed_kws = _bertopic_sweep_metrics(m, tokenized, dictionary, top_n_metrics)
            kws_by_strategy[strat][seed] = seed_kws
            row = dict(strategy=strat, seed=seed, n_raw=n_raw,
                       outlier_pre=o_pre, outlier_post=o_post, **met)
            rows.append(row)
            _checkpoint_append_row(checkpoint_path, row)

    raw = pd.DataFrame(rows)

    agg_rows = []
    for strat in strategies:
        sub = raw[raw.strategy == strat]
        if sub.empty:
            continue
        stab_m, stab_s = (
            compute_stability(kws_by_strategy[strat])
            if len(kws_by_strategy[strat]) >= 2 else (float("nan"), float("nan"))
        )
        agg_rows.append(dict(
            strategy=strat, n_seeds=len(sub),
            K=round(sub["K"].mean(), 1),
            outlier_pre=round(sub["outlier_pre"].mean(), 4),
            outlier_post=round(sub["outlier_post"].mean(), 4),
            C_v=round(sub["cv"].mean(), 4), Diversity=round(sub["div"].mean(), 4),
            Exclus=round(sub["excl"].mean(), 4), FREX=round(sub["frex"].mean(), 4),
            Stability=round(stab_m, 4), Stab_std=round(stab_s, 4),
        ))

    agg = pd.DataFrame(agg_rows)
    return raw, agg


def sweep_bertopic_grid(
    build_model,
    docs: list[str],
    embeddings: np.ndarray,
    tokenized: list[list[str]],
    dictionary,
    seeds: Iterable[int],
    n_neighbors_grid: Iterable[int],
    min_cluster_size_grid: Iterable[int],
    min_samples_grid: Iterable[int | None] = (None,),
    reduce_nr_grid: Iterable[int | None] = (None,),
    cluster_selection_methods_grid: Iterable[str] = ("leaf",),
    outlier_strategy: str = "off",
    outlier_threshold: float = 0.0,
    top_n_metrics: int = 20,
    checkpoint_path=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Multi-seed grid search over UMAP ``n_neighbors`` x HDBSCAN
    ``min_cluster_size`` x HDBSCAN ``min_samples`` x ``reduce_topics(nr_topics)``
    target — the structural-hyperparameter analogue of ``grid_search_k`` for LDA.

    UMAP ``n_neighbors`` and the two HDBSCAN knobs (``min_cluster_size`` +
    ``min_samples``) each force a fresh UMAP+HDBSCAN fit, so this refits once
    per ``(seed, n_neighbors, min_cluster_size, min_samples)`` and reuses that
    fit across ``reduce_nr_grid`` (post-processing only, same fit-once/copy-many
    pattern as the other sweeps here). ``min_samples`` controls how conservative
    HDBSCAN is (higher -> more points pushed to -1/outlier): the knob that most
    affects outlier rate on short/noisy text (e.g. tweets), which is exactly why
    it belongs in the grid alongside ``min_cluster_size``.

    A single-seed version of this grid search is cheap but unreliable: ARJ's
    own validation (see ``relatorio_arj.md`` sec 7.1) found a single-seed
    n_neighbors/min_cluster_size grid picked the wrong n_neighbors, only
    caught by multi-seed stability scoring. Always pass >= 2 seeds and trust
    ``Stability``/``Stab_std`` over a single run's coherence number.

    Parameters
    ----------
    build_model : Callable[[int, int, int, int | None, str], BERTopic]
        Factory ``(seed, n_neighbors, min_cluster_size, min_samples,
        cluster_selection_method) -> unfit BERTopic``. The caller owns
        embedding/vectorizer/ctfidf/representation choices.
    min_samples_grid : Iterable[int | None]
        HDBSCAN ``min_samples`` values to try. Default ``(None,)`` runs a single
        point using the factory's own default.
    cluster_selection_methods_grid : Iterable[str]
        HDBSCAN ``cluster_selection_method`` values to try. Default ``("leaf",)``.
        Pass ``("leaf", "eom")`` to compare both methods. ``"eom"`` (Excess of
        Mass) tends to produce fewer, larger clusters; ``"leaf"`` tends to produce
        more, smaller clusters.
    outlier_strategy, outlier_threshold : applied identically to every grid
        point via ``_bertopic_postprocess`` (canonical 4-step order). Should always be
        ``"off"`` here to isolate the structural-hyperparameter effect — compare
        outlier strategies separately with ``sweep_outlier_strategies``.
    checkpoint_path : opcional, grava cada linha em CSV assim que e computada
        — ver ``_checkpoint_append_row``. Sobrevive a kill/crash no meio do
        sweep (o motivo de existir: a Folha rodou 6h+ e foi morta sem deixar
        nenhum resultado parcial, 2026-08-22).

    Returns
    -------
    (raw, agg) : ``raw`` has one row per (seed, n_neighbors, min_cluster_size,
        min_samples, cluster_selection_method, reduce_nr); ``agg`` has one row
        per grid point with means + Jaccard stability across seeds.
    """
    _checkpoint_reset(checkpoint_path)
    import copy
    import itertools
    import time

    rows = []
    kws_by_combo: dict[tuple, dict[int, dict[int, list[str]]]] = {}

    combos = list(itertools.product(
        seeds, n_neighbors_grid, min_cluster_size_grid, min_samples_grid, cluster_selection_methods_grid,
    ))
    n_combos = len(combos)
    t0 = time.time()

    for i, (seed, nn, mcs, ms, csm) in enumerate(combos, start=1):
        base = build_model(seed, nn, mcs, ms, csm)
        base.fit_transform(docs, embeddings=embeddings)
        print(f"  [{i}/{n_combos}] seed={seed} nn={nn} mcs={mcs} ms={ms} csm={csm} "
              f"— fit ok (+{time.time() - t0:.0f}s total)", flush=True)
        for nr in reduce_nr_grid:
            m = copy.deepcopy(base)
            o_pre, o_post, n_raw = _bertopic_postprocess(
                m, docs, embeddings, outlier_strategy,
                threshold=outlier_threshold, reduce_nr=nr,
            )
            met, seed_kws = _bertopic_sweep_metrics(m, tokenized, dictionary, top_n_metrics)
            key = (nn, mcs, ms, csm, nr)
            kws_by_combo.setdefault(key, {})[seed] = seed_kws
            row = dict(
                n_neighbors=nn, min_cluster_size=mcs, min_samples=ms,
                cluster_selection_method=csm, reduce_nr=nr, seed=seed,
                n_raw=n_raw, outlier_pre=o_pre, outlier_post=o_post, **met,
            )
            rows.append(row)
            _checkpoint_append_row(checkpoint_path, row)

    raw = pd.DataFrame(rows)

    def _nan_safe_mask(col: pd.Series, value) -> pd.Series:
        # A None grid value (min_samples=None / reduce_nr=None) becomes NaN once
        # mixed with ints in the DataFrame; `col == None` is always False (not
        # NaN-aware), so it must be special-cased rather than compared directly.
        return col.isna() if value is None else col == value

    agg_rows = []
    for (nn, mcs, ms, csm, nr), kmap in kws_by_combo.items():
        sub = raw[
            (raw.n_neighbors == nn) & (raw.min_cluster_size == mcs)
            & _nan_safe_mask(raw.min_samples, ms)
            & (raw.cluster_selection_method == csm)
            & _nan_safe_mask(raw.reduce_nr, nr)
        ]
        stab_m, stab_s = compute_stability(kmap) if len(kmap) >= 2 else (float("nan"), float("nan"))
        agg_rows.append(dict(
            n_neighbors=nn, min_cluster_size=mcs, min_samples=ms,
            cluster_selection_method=csm, reduce_nr=nr,
            n_seeds=len(sub),
            K=round(sub["K"].mean(), 1),
            # A1 do protocolo 2026-08-16: K identico em todas as seeds e restricao
            # DURA de admissibilidade, e a media entre seeds ESCONDE colapso (o caso
            # do "K=11" que era 15/6/12). Por isso min/max vao para o agregado — sem
            # eles o filtro so seria aplicavel relendo o _raw.csv.
            K_min=int(sub["K"].min()), K_max=int(sub["K"].max()),
            # A4: n_raw <= 2K. Ja existia no raw; sem estar no agregado a restricao
            # era inavaliavel a partir do CSV que o seletor le.
            n_raw=round(sub["n_raw"].mean(), 1),
            razao_nraw_k=round(sub["n_raw"].mean() / max(sub["K"].mean(), 1e-9), 2),
            outlier_pre=round(sub["outlier_pre"].mean(), 4),
            outlier_post=round(sub["outlier_post"].mean(), 4),
            cobertura=round(1 - sub["outlier_pre"].mean(), 4),
            NPMI=round(sub["npmi"].mean(), 4),
            C_v=round(sub["cv"].mean(), 4), Diversity=round(sub["div"].mean(), 4),
            Exclus=round(sub["excl"].mean(), 4), FREX=round(sub["frex"].mean(), 4),
            Stability=round(stab_m, 4), Stab_std=round(stab_s, 4),
        ))

    agg = (
        pd.DataFrame(agg_rows)
        .sort_values(["n_neighbors", "min_cluster_size", "min_samples", "cluster_selection_method", "reduce_nr"])
        .reset_index(drop=True)
        if agg_rows else pd.DataFrame(agg_rows)
    )
    return raw, agg


# ===========================================================================
# lda_pipeline.py
# ===========================================================================
"""LDA pipeline: grid search K, train, extract topics, qualitative report.

Corpus-agnostic. Importado pelos notebooks de 03-topic-modeling.
"""


def cache_protocolo_apto(df_ou_linha, colunas: list[str]) -> list[str]:
    """Colunas do protocolo ausentes ou NaN num cache de grid — vazio = apto.

    Generaliza o check ad hoc que os notebooks de grid (LDA, NMF, STM) reinventavam
    cada um a sua maneira para decidir se um ``*_metrics.csv`` em disco e anterior
    ao protocolo de 2026-08-16 e precisa ser recomputado (C9). Uso tipico:

        faltando = cache_protocolo_apto(m, ["k_npmi_scores"])
        if faltando:
            print(f"Cache sem {faltando} (pre-protocolo) — recomputando.")
            <roda o grid de novo>
        else:
            <le do cache>

    Aceita tanto uma ``pd.Series`` (uma linha ja selecionada) quanto um
    ``pd.DataFrame`` inteiro (usa a ULTIMA linha, o run mais recente — mesma
    convencao ja usada pelos notebooks de grid).
    """
    linha = df_ou_linha.iloc[-1] if isinstance(df_ou_linha, pd.DataFrame) else df_ou_linha
    return [c for c in colunas if c not in linha.index or pd.isna(linha.get(c))]


def grid_search_k(
    corpus_bow: list[list[tuple[int, int]]],
    dictionary: Dictionary,
    tokenized: list[list[str]],
    k_range: Iterable[int],
    seed: int = 42,
    passes: int = 10,
    workers: int | None = None,
    return_npmi: bool = False,
    checkpoint_path=None,
    top_k: int = 10,
    return_diversity: bool = False,
):
    """Train LDA for each K and return (cv_scores, perplexity_scores).

    ``checkpoint_path`` (opcional): grava cada K em CSV assim que e computado
    — ver ``_checkpoint_append_row``.

    Fits each model once and computes both C_v coherence and perplexity
    (exp(-log_perplexity_per_word)) to avoid refitting.

    ``return_npmi=True`` devolve ``(cv_scores, perplexity_scores, npmi_scores)``
    — o mesmo fit, pontuado tambem por NPMI. **O NPMI e o eixo que ORDENA o K no
    protocolo de 2026-08-16** (secao 4.2.1); o C_v segue calculado por
    comparabilidade com quem so reporta C_v, e a perplexidade fica num eixo
    SEPARADO, de desempenho preditivo, nunca de qualidade de topico
    (Chang et al. 2009: verossimilhanca retida anticorrelaciona com julgamento
    humano de interpretabilidade). O default preserva a assinatura de 2 valores,
    de que os notebooks antigos dependem.

    ``return_diversity=True`` devolve tambem ``diversity_scores`` (mesmo fit,
    sem refit): **e o SEGUNDO eixo da fronteira de Pareto do LDA no protocolo**
    (secao 1.5 — NPMI e Diversity, nao NPMI sozinho). Combinavel com
    ``return_npmi``; a diversidade sai SEMPRE por ultimo na tupla, seguindo o
    mesmo padrao de flag independente e default ``False`` de ``return_npmi``,
    para preservar as assinaturas de que os notebooks de producao dependem:

    - nenhuma flag: ``(cv_scores, perplexity_scores)``
    - so ``return_npmi``: ``(cv_scores, perplexity_scores, npmi_scores)``
    - so ``return_diversity``: ``(cv_scores, perplexity_scores, diversity_scores)``
    - as duas: ``(cv_scores, perplexity_scores, npmi_scores, diversity_scores)``

    Reporte a CURVA de NPMI sobre a faixa de K, nao so o argmax: a auditoria de
    reprodutibilidade (mapeamento secao 2.11) mostrou que o argmax nao e estavel,
    e a curva e o objeto honesto a publicar — custa zero, ja que os valores saem
    todos deste mesmo grid.

    .. warning::
       **O resultado depende de ``workers``, e ``None`` o deriva da maquina**
       (``cpu_count()-1``). Medido em 2026-08-15, K=20, dois corpora:

       ===============  ==========  =========  ==========
       corpus           15 workers  1 worker   delta
       ===============  ==========  =========  ==========
       folha            0,5369      0,5635     2,7e-2
       tweets_bre2022   0,4323      0,3835     4,9e-2
       ===============  ==========  =========  ==========

       Nos tweets a mudanca inverte a curva inteira de C_v x K: com 15 workers
       o pico e K=20; com 1 worker, K=3.

       **Nao e nao-determinismo** — para uma contagem FIXA de workers o
       resultado reproduz bit a bit (verificado em 2 fits, nas duas
       configuracoes). E ``batch=True`` **nao** resolve: o desacoplamento
       falha igual (delta 9,4e-3 na folha, 5,2e-2 nos tweets), embora com 15
       workers ``batch=True`` e ``batch=False`` sejam identicos (1e-16) — ou
       seja, os runs publicados ja eram batch de fato.

       Mecanismo (``gensim/models/ldamulticore.py`` L236-239):
       ``updateafter = chunksize * workers`` (2000 x 15 = 30000 > os dois
       corpora, logo uma atualizacao por passe) contra 2000 com 1 worker
       (2 a 4 atualizacoes por passe). ``workers`` controla o regime
       online-vs-batch da inferencia variacional, nao so paralelismo.

       **Enquanto nao houver decisao**, o default segue ``None`` para nao
       mudar silenciosamente os numeros publicados. Quem precisar de resultado
       portavel entre maquinas deve passar ``workers`` explicito e registra-lo
       junto do resultado. Ver o registro da auditoria em
       ``docs/mapeamento-artigo-base-2026-08-12.md``.

    ``top_k`` (protocolo secao 1.8, condicao 1 e secao 3.4): profundidade passada
    ao gensim. SEM ele o CoherenceModel usa topn=20 e trunca listas maiores em
    silencio — a curva de K sairia medida numa base diferente da declarada.
    """
    _checkpoint_reset(checkpoint_path)
    cv_scores: dict[int, float] = {}
    perplexity_scores: dict[int, float] = {}
    npmi_scores: dict[int, float] = {}
    div_scores: dict[int, float] = {}
    for k in k_range:
        model = LdaMulticore(
            corpus=corpus_bow,
            id2word=dictionary,
            num_topics=k,
            random_state=seed,
            passes=passes,
            workers=workers,
        )
        cm = CoherenceModel(
            model=model,
            texts=tokenized,
            dictionary=dictionary,
            coherence="c_v",
            topn=top_k,
        )
        cv_scores[k] = float(cm.get_coherence())
        perplexity_scores[k] = float(np.exp(-model.log_perplexity(corpus_bow)))
        if return_npmi:
            cm_n = CoherenceModel(
                model=model,
                texts=tokenized,
                dictionary=dictionary,
                coherence="c_npmi",
                topn=top_k,
            )
            npmi_scores[k] = float(cm_n.get_coherence())
        if return_diversity:
            topicos_k = {i: [w for w, _ in model.show_topic(i, topn=top_k)]
                         for i in range(k)}
            div_scores[k] = compute_topic_diversity(topicos_k, top_k=top_k)
        checkpoint_row = {"k": k, "cv": cv_scores[k], "perplexity": perplexity_scores[k]}
        if return_npmi:
            checkpoint_row["npmi"] = npmi_scores[k]
        if return_diversity:
            checkpoint_row["diversity"] = div_scores[k]
        _checkpoint_append_row(checkpoint_path, checkpoint_row)
    if return_npmi and return_diversity:
        return cv_scores, perplexity_scores, npmi_scores, div_scores
    if return_npmi:
        return cv_scores, perplexity_scores, npmi_scores
    if return_diversity:
        return cv_scores, perplexity_scores, div_scores
    return cv_scores, perplexity_scores


def grid_search_alpha_eta(
    corpus_bow: list[list[tuple[int, int]]],
    dictionary: Dictionary,
    tokenized: list[list[str]],
    k: int,
    alpha_grid: list | None = None,
    eta_grid: list | None = None,
    seed: int = 42,
    passes: int = 10,
    workers: int | None = None,
    score_by: str = "cv",
    checkpoint_path=None,
) -> tuple[pd.DataFrame, object, object]:
    """Varre combinações alpha × eta com K fixo. Retorna (df_results, best_alpha, best_eta).

    ``checkpoint_path`` (opcional): grava cada combo em CSV assim que e
    computado — ver ``_checkpoint_append_row``.

    ``score_by="npmi"`` ordena (e escolhe) pelo NPMI em vez do C_v — é o que o
    protocolo de 2026-08-16 exige, para que os DOIS estágios do grid usem o mesmo
    eixo. O default ``"cv"`` preserva o comportamento antigo. Em ambos os casos
    as duas colunas são calculadas e persistidas: a divergência entre eixos é
    informação a reportar, não um detalhe a esconder.

    .. warning::
       O resultado depende de ``workers`` — ver a nota em ``grid_search_k``.

    alpha — prior Dirichlet doc→tópico:
      'symmetric': uniforme (default gensim); 'asymmetric': 1/k (favorece temas dominantes);
      float baixo (0.01): docs concentrados em poucos tópicos; alto (0.5): difusos.

    eta — prior Dirichlet tópico→palavra:
      None: gensim default (1/K); float baixo (0.01): tópicos com vocab concentrado
      (mais distintos); alto (0.1+): maior sobreposição de vocabulário entre tópicos.

    df_results ordenado por cv decrescente — primeira linha é o vencedor.
    """
    if score_by not in ("cv", "npmi"):
        raise ValueError(f"score_by deve ser 'cv' ou 'npmi', recebido {score_by!r}")
    _checkpoint_reset(checkpoint_path)
    if alpha_grid is None:
        alpha_grid = ["symmetric", "asymmetric", 0.1, 0.5]
    if eta_grid is None:
        eta_grid = [None, 0.01, 0.1]
    rows: list[dict] = []
    for alpha in alpha_grid:
        for eta in eta_grid:
            model = LdaMulticore(
                corpus=corpus_bow,
                id2word=dictionary,
                num_topics=k,
                random_state=seed,
                passes=passes,
                workers=workers,
                alpha=alpha,
                eta=eta,
            )
            cm = CoherenceModel(
                model=model, texts=tokenized, dictionary=dictionary, coherence="c_v"
            )
            cm_n = CoherenceModel(
                model=model, texts=tokenized, dictionary=dictionary, coherence="c_npmi"
            )
            row = {
                "alpha": alpha,
                "eta": eta,
                "cv": float(cm.get_coherence()),
                "npmi": float(cm_n.get_coherence()),
                "perplexity": float(np.exp(-model.log_perplexity(corpus_bow))),
            }
            rows.append(row)
            _checkpoint_append_row(checkpoint_path, row)
    df = (
        pd.DataFrame(rows)
        .sort_values(score_by, ascending=False)
        .reset_index(drop=True)
    )
    return df, df.iloc[0]["alpha"], df.iloc[0]["eta"]


# ===========================================================================
# nmf_pipeline.py
# ===========================================================================
"""NMF pipeline: grid search K, then kappa x minimum_probability (K fixed),
train. Mirrors lda_pipeline.py's two-stage protocol; extract_topics_keywords,
compute_doc_distributions and export_results are shared (model-agnostic).
"""


def _h_sparsity(topic_word, threshold: float = 1e-4) -> float:
    """Fracao de entradas ~zero na matriz topico x palavra (o "H" do NMF).

    Usa limiar, nao igualdade exata: fatoracao numerica quase nunca zera de
    verdade, e 1e-12 e zero para qualquer efeito pratico. H densa = topicos que
    nao se especializaram, um dos dois sinais do colapso de K.
    """
    m = np.asarray(topic_word, dtype=float)
    if m.size == 0:
        return 0.0
    return float((np.abs(m) < threshold).sum() / m.size)


def _doc_topic_concentration(dominant: Iterable, k: int) -> tuple[float, int]:
    """(fracao de docs no maior topico, nº de topicos que receberam doc).

    `dominant` e a lista do topico dominante de cada documento; -1 (ou None)
    significa "nenhum topico acima do piso" e **conta no denominador sem virar
    topico** — um K que deixa metade do corpus sem atribuicao nao deve parecer
    concentrado nem diverso. E o segundo sinal do colapso: poucos topicos
    absorvendo quase tudo.
    """
    ids = list(dominant)
    if not ids:
        return 0.0, 0
    from collections import Counter

    counts = Counter(int(i) for i in ids if i is not None and int(i) >= 0)
    if not counts:
        return 0.0, 0
    return float(max(counts.values()) / len(ids)), len(counts)


def grid_search_k_nmf(
    corpus_bow: list[list[tuple[int, int]]],
    dictionary: Dictionary,
    tokenized: list[list[str]],
    k_range: Iterable[int],
    seed: int = 42,
    passes: int = 10,
    return_diagnostics: bool = False,
    return_npmi: bool = False,
    checkpoint_path=None,
    top_k: int = 10,
    return_diversity: bool = False,
):
    """Train NMF for each K and return {K: c_v}.

    ``checkpoint_path`` (opcional): grava cada K em CSV assim que e computado
    — ver ``_checkpoint_append_row``.

    Analogous to grid_search_k (LDA), but gensim.models.Nmf has no
    log_perplexity — only coherence is reported here.

    ``return_npmi=True`` devolve tambem ``npmi_scores``, no mesmo fit (sem
    refit): **e o eixo que ORDENA o K do NMF no protocolo de 2026-08-16**
    (secao 4.2.2, mesma regra do LDA — secao 4.2.1). O C_v segue calculado por
    comparabilidade com quem so reporta C_v.

    Com ``return_diagnostics=True`` devolve tambem ``diag_df`` — mesmo
    contrato de ``grid_search_k_stm``. As duas flags sao independentes e
    combinaveis; o default preserva a assinatura antiga (dict puro), de que
    os 3 notebooks NMF dependem:

    - nenhuma flag: ``cv_scores``
    - so ``return_npmi``: ``(cv_scores, npmi_scores)``
    - so ``return_diagnostics``: ``(cv_scores, diag_df)``
    - as duas: ``(cv_scores, npmi_scores, diag_df)``

    ``return_diversity=True`` devolve tambem ``diversity_scores``, no mesmo
    fit: **e o SEGUNDO eixo da fronteira de Pareto do NMF no protocolo**
    (secao 1.6 — NPMI e Diversity; o NMF nao tem perplexidade). Independente
    das outras duas flags, sempre por ultimo na tupla:

    - so ``return_diversity``: ``(cv_scores, diversity_scores)``
    - ``return_npmi`` + ``return_diversity``: ``(cv_scores, npmi_scores, diversity_scores)``
    - ``return_diagnostics`` + ``return_diversity``: ``(cv_scores, diag_df, diversity_scores)``
    - as tres: ``(cv_scores, npmi_scores, diag_df, diversity_scores)``

    Os diagnosticos existem para explicar o colapso do NMF-tweets para K=8
    (backlog P3), que ate 2026-08-14 era so reportado:

    - ``h_sparsity`` — fracao de entradas ~zero na matriz topico x palavra;
    - ``top_topic_doc_share`` — fracao de docs no maior topico;
    - ``n_topics_with_docs`` — quantos dos K topicos receberam algum documento.

    ``top_k`` (protocolo secao 1.8, condicao 1 e secao 3.4): profundidade passada
    ao gensim. SEM ele o CoherenceModel usa topn=20 e trunca listas maiores em
    silencio — a curva de K sairia medida numa base diferente da declarada.
    """
    _checkpoint_reset(checkpoint_path)
    cv_scores: dict[int, float] = {}
    npmi_scores: dict[int, float] = {}
    div_scores: dict[int, float] = {}
    rows: list[dict] = []
    for k in k_range:
        t0 = time.time()
        model = Nmf(
            corpus=corpus_bow,
            id2word=dictionary,
            num_topics=k,
            random_state=seed,
            passes=passes,
        )
        cm = CoherenceModel(
            model=model,
            texts=tokenized,
            dictionary=dictionary,
            coherence="c_v",
            topn=top_k,
        )
        cv = float(cm.get_coherence())
        cv_scores[k] = cv
        npmi = None
        if return_npmi:
            cm_n = CoherenceModel(
                model=model,
                texts=tokenized,
                dictionary=dictionary,
                coherence="c_npmi",
                topn=top_k,
            )
            npmi = float(cm_n.get_coherence())
            npmi_scores[k] = npmi
        if return_diversity:
            topicos_k = {i: [w for w, _ in model.show_topic(i, topn=top_k)]
                         for i in range(k)}
            div_scores[k] = compute_topic_diversity(topicos_k, top_k=top_k)
        if not return_diagnostics:
            checkpoint_row = {"k": k, "cv": cv}
            if return_npmi:
                checkpoint_row["npmi"] = npmi
            if return_diversity:
                checkpoint_row["diversity"] = div_scores[k]
            _checkpoint_append_row(checkpoint_path, checkpoint_row)
            continue

        elapsed = time.time() - t0
        dominant = []
        for bow in corpus_bow:
            try:
                dist = model.get_document_topics(bow)
            except Exception:
                dist = []
            dominant.append(max(dist, key=lambda p: p[1])[0] if dist else -1)
        top_share, n_with_docs = _doc_topic_concentration(dominant, k)
        row = {
            "k": k,
            "cv": cv,
            "h_sparsity": _h_sparsity(model.get_topics()),
            "top_topic_doc_share": top_share,
            "n_topics_with_docs": n_with_docs,
            "docs_sem_topico": sum(1 for d in dominant if d < 0),
            "elapsed_sec": elapsed,
        }
        if return_npmi:
            row["npmi"] = npmi
        if return_diversity:
            row["diversity"] = div_scores[k]
        rows.append(row)
        _checkpoint_append_row(checkpoint_path, row)

    if return_npmi and return_diagnostics and return_diversity:
        return cv_scores, npmi_scores, pd.DataFrame(rows), div_scores
    if return_npmi and return_diagnostics:
        return cv_scores, npmi_scores, pd.DataFrame(rows)
    if return_npmi and return_diversity:
        return cv_scores, npmi_scores, div_scores
    if return_npmi:
        return cv_scores, npmi_scores
    if return_diagnostics and return_diversity:
        return cv_scores, pd.DataFrame(rows), div_scores
    if return_diagnostics:
        return cv_scores, pd.DataFrame(rows)
    if return_diversity:
        return cv_scores, div_scores
    return cv_scores


def _cobertura_doc_topico(
    doc_topics: list[list[tuple[int, float]]],
    bow_nao_vazio: list[bool] | None = None,
) -> float:
    """Fracao de documentos com ao menos um topico atribuido.

    Protocolo secao 1.6. No LDA a cobertura e trivialmente 1.0 (todo documento
    tem massa em todo topico); no NMF o ``minimum_probability`` trunca theta e
    um documento pode terminar com lista VAZIA. Como NPMI, Diversity,
    Exclusividade e FREX melhoram quando os documentos ambiguos somem da conta,
    esta e a coluna que impede o grid de premiar o descarte.

    ``bow_nao_vazio`` (opcional): mascara paralela a ``doc_topics``, True onde
    o BOW do documento tem >=1 token no vocabulario. Um documento com BOW
    vazio (ex.: tweet so com emoji/URL/hashtag, sem nada que sobreviva ao
    ``filter_extremes``) nao tem sinal nenhum para NENHUM modelo atribuir
    topico — nao e o ``kappa``/``minimum_probability`` descartando, e o
    documento chegando sem conteudo. Sem a mascara, esses casos contam contra
    a cobertura do mesmo jeito que um descarte real do hiperparametro, o que
    confunde as duas causas (achado 2026-08-24: 27 dos 8811 tweets do corpus
    tweets_bre2022 tem BOW vazio e SEMPRE ficam sem topico, mesmo com
    ``minimum_probability=0.0`` — cross-check 27/27, zero divergencia).
    Passando a mascara, o denominador conta so os documentos com conteudo,
    isolando o descarte que de fato varia com o hiperparametro.
    """
    if not doc_topics:
        return 0.0
    if bow_nao_vazio is not None:
        indices = [i for i, ok in enumerate(bow_nao_vazio) if ok]
        if not indices:
            return 0.0
        cobertos = sum(1 for i in indices if doc_topics[i])
        return cobertos / len(indices)
    cobertos = sum(1 for d in doc_topics if d)
    return cobertos / len(doc_topics)


def grid_search_nmf_hparams(
    corpus_bow: list[list[tuple[int, int]]],
    dictionary: Dictionary,
    tokenized: list[list[str]],
    k: int,
    kappa_grid: list | None = None,
    min_prob_grid: list | None = None,
    seed: int = 42,
    passes: int = 10,
    score_by: str = "cv",
    checkpoint_path=None,
    top_k: int = 10,
) -> tuple[pd.DataFrame, float, float]:
    """Varre combinações kappa x minimum_probability com K fixo. Retorna
    (df_results, best_kappa, best_min_prob) — análogo ao
    grid_search_alpha_eta do LDA.

    ``score_by="npmi"`` ordena (e escolhe) ``best_kappa``/``best_min_prob``
    pelo NPMI em vez do C_v — mesmo parâmetro, mesmos dois valores aceitos
    (``"cv"``/``"npmi"``) e mesmo ``ValueError`` na entrada inválida que
    ``grid_search_alpha_eta`` (LDA); é o que o protocolo (§1.6) exige, para
    que o segundo estágio do NMF use o mesmo eixo do braço LDA. O default
    ``"cv"`` preserva o comportamento histórico desta função para quem a
    chama sem o argumento — mas ``score_by="cv"`` é **sempre** fora do
    protocolo aqui: o §1.3 tira C_v da decisão nos três braços sem exceção,
    não é uma alternativa válida como poderia parecer no LDA. Por isso,
    diferente de ``grid_search_alpha_eta``, esta função AVISA em runtime
    quando ``score_by="cv"`` (print, não só docstring) — ver o corpo abaixo.

    AVISO — mesmo com ``score_by="npmi"``, ``best_kappa``/``best_min_prob``
    continuam sendo um atalho de conveniência, não a decisão do protocolo:
    o §1.6 manda Pareto NPMI x Diversity com a cobertura como PORTA de
    admissibilidade antes de qualquer ordenação — algo que um sort escalar
    não expressa. Quem decide sob o protocolo é ``_selecao.py >
    selecionar_nmf`` (Pareto NPMI x Diversity sobre o ``df_results``
    completo, após a porta de cobertura), não estes dois valores escalares.
    Os notebooks NMF (``03_nmf_folha.ipynb``, ``03_nmf_tweets_bre2022.ipynb``)
    chamam ``_selecao.selecionar_nmf`` para a decisão publicada e usam
    ``score_by="npmi"`` aqui só para que o CSV do grid já saia ordenado no
    eixo correto — os notebooks desempacotam ``best_kappa``/``best_min_prob``
    posicionalmente, e mudar o que eles significam por default alteraria
    números publicados sem que o lado do notebook tivesse sido revisto.

    ``checkpoint_path`` (opcional): grava cada combo em CSV assim que e
    computado — ver ``_checkpoint_append_row``.

    kappa — passo de gradiente da atualização de W: valores maiores aceleram
      a convergência mas podem instabilizar; valores menores são mais
      conservadores.
    minimum_probability — piso de esparsidade da distribuição doc-tópico,
      aplicado só na hora de *ler* o modelo treinado via
      ``get_document_topics`` — não afeta o treino (W/H). Por isso o modelo é
      treinado **uma vez por kappa**: ``get_document_topics`` é chamado **por
      combo** (com ``minimum_probability=min_prob`` explícito), porque é
      exatamente essa leitura que varia com o limiar — mas ``show_topic``
      (usada por ``cv``/``NPMI``/``Diversity``, todas leituras de
      tópico-palavra, não de doc-tópico) **não aceita** ``minimum_probability``
      (``inspect.signature(gensim.models.Nmf.show_topic)`` →
      ``(self, topicid, topn=10, normalize=None)``, verificado nesta
      revisão). A leitura de tópico-palavra é, portanto, matematicamente
      invariante ao eixo ``minimum_probability`` — diferente de ``cobertura``
      (via ``get_document_topics``, que TEM o parâmetro e onde ele importa de
      verdade). Documentado aqui porque uma versão anterior deste docstring
      afirmava, incorretamente, que ``show_topic`` também respeitava o piso.

    ``cv``, ``NPMI`` e ``Diversity`` são calculados **uma vez por kappa**
    (mesmo fit, sem refit) e replicados para cada ``minimum_probability`` do
    grid — pelo motivo acima, não por atalho: recomputá-los dentro do laço de
    ``min_prob`` devolveria, combo a combo, o mesmo valor bit-a-bit (mesma
    chamada a ``show_topic``), só |min_prob_grid| vezes mais caro.
    ``minimum_probability`` NÃO É um shortcut para pular o cálculo — é medido
    onde de fato pode variar (``cobertura``) e reaproveitado onde não pode
    (``cv``/``NPMI``/``Diversity``). ``top_k`` (protocolo secao 1.8/3.4):
    mesma profundidade da curva de K (``grid_search_k_nmf``), para o segundo
    estágio não medir numa base diferente da primeira.

    ``NPMI``/``Diversity`` em maiúsculas (colunas), ``cv``/``cobertura`` em
    minúsculas — deliberado: casa com o consumidor (``_selecao.py``, mesmo
    ``EIXOS``/``_pareto`` do braço BERTopic), à custa de inconsistência
    cosmética no CSV.

    ``cobertura`` (protocolo secao 1.6): fracao de documentos COM BOW NAO
    VAZIO que tem ao menos um topico acima de ``minimum_probability`` — ver
    ``_cobertura_doc_topico``. Documento com BOW vazio (0 tokens no
    vocabulario pos filter_extremes) fica de fora do denominador: ele nunca
    teria topico em nenhum modelo, entao nao e o descarte que esta coluna
    existe para vigiar. NAO e decorativa — enquanto ela nao for medida, nenhum
    numero do braco NMF pode ser comparado com os do LDA, que tem cobertura
    1.0 por construcao. ``n_docs_bow_vazio`` (coluna, constante na grade)
    reporta a contagem excluida, para transparencia.

    df_results ordenado por (``score_by`` desc, minimum_probability asc) —
    primeira linha é a vencedora; o tie-break por minimum_probability é
    determinístico (evita depender da ordem instável do quicksort quando
    ``score_by`` empata, o que acontece sempre nesta dimensão). Isto NÃO
    decide o vencedor do protocolo, que é responsabilidade de
    ``_selecao.selecionar_nmf`` sobre o CSV completo (Pareto NPMI x
    Diversity, cobertura como porta — §1.6).
    """
    if score_by not in ("cv", "npmi"):
        raise ValueError(f"score_by deve ser 'cv' ou 'npmi', recebido {score_by!r}")
    if score_by == "cv":
        print(
            "AVISO: grid_search_nmf_hparams ordenando best_kappa/best_min_prob por "
            "'cv' — FORA do protocolo (secao 1.3: C_v nunca decide em nenhum braco). "
            "Passe score_by='npmi' para o eixo que o protocolo (secao 1.6) exige; a "
            "decisao publicada, de qualquer forma, e sempre _selecao.selecionar_nmf "
            "sobre o CSV completo, nao estes dois valores escalares."
        )
    _checkpoint_reset(checkpoint_path)
    if kappa_grid is None:
        kappa_grid = [0.5, 1.0, 2.0]
    if min_prob_grid is None:
        min_prob_grid = [0.0, 0.01, 0.05]
    # BOW vazio (0 tokens no vocab pos filter_extremes) e constante para o
    # corpus inteiro, independente de kappa/minimum_probability -- medido uma
    # vez, fora do laco, e excluido do denominador de `cobertura` (ver
    # docstring de `_cobertura_doc_topico`).
    bow_nao_vazio = [len(b) > 0 for b in corpus_bow]
    n_bow_vazio = bow_nao_vazio.count(False)
    if n_bow_vazio:
        print(
            f"AVISO: {n_bow_vazio} de {len(corpus_bow)} documentos com BOW vazio "
            "(sem nenhum token no vocabulario pos filter_extremes) -- excluidos "
            "do denominador de 'cobertura' (nao ha sinal para nenhum modelo "
            "atribuir topico; nao e descarte de minimum_probability)."
        )
    rows: list[dict] = []
    for kappa in kappa_grid:
        model = Nmf(
            corpus=corpus_bow,
            id2word=dictionary,
            num_topics=k,
            random_state=seed,
            passes=passes,
            kappa=kappa,
        )
        cm = CoherenceModel(
            model=model, texts=tokenized, dictionary=dictionary,
            coherence="c_v", topn=top_k,
        )
        cv = float(cm.get_coherence())
        cm_n = CoherenceModel(
            model=model, texts=tokenized, dictionary=dictionary,
            coherence="c_npmi", topn=top_k,
        )
        npmi = float(cm_n.get_coherence())
        topicos_k = {i: [w for w, _ in model.show_topic(i, topn=top_k)]
                     for i in range(k)}
        diversity = compute_topic_diversity(topicos_k, top_k=top_k)
        for min_prob in min_prob_grid:
            doc_topics = [
                model.get_document_topics(doc, minimum_probability=min_prob)
                for doc in corpus_bow
            ]
            cobertura = _cobertura_doc_topico(doc_topics, bow_nao_vazio=bow_nao_vazio)
            row = {
                "kappa": kappa,
                "minimum_probability": min_prob,
                "cv": cv,
                "NPMI": npmi,
                "Diversity": diversity,
                "cobertura": cobertura,
                "n_docs_bow_vazio": n_bow_vazio,
            }
            rows.append(row)
            _checkpoint_append_row(checkpoint_path, row)
    # coluna interna: score_by="npmi" ordena pela coluna "NPMI" (maiuscula,
    # ver nota acima sobre casing deliberada); score_by="cv" pela "cv".
    coluna_sort = "NPMI" if score_by == "npmi" else "cv"
    df = (
        pd.DataFrame(rows)
        .sort_values([coluna_sort, "minimum_probability"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return df, float(df.iloc[0]["kappa"]), float(df.iloc[0]["minimum_probability"])


def train_nmf(
    corpus_bow: list[list[tuple[int, int]]],
    dictionary: Dictionary,
    k: int,
    seed: int = 42,
    passes: int = 20,
    kappa: float = 1.0,
    minimum_probability: float = 0.01,
    normalize: bool = True,
) -> Nmf:
    """Train final NMF model with chosen K, kappa and minimum_probability."""
    return Nmf(
        corpus=corpus_bow,
        id2word=dictionary,
        num_topics=k,
        random_state=seed,
        passes=passes,
        kappa=kappa,
        minimum_probability=minimum_probability,
        normalize=normalize,
    )


def extract_topics_keywords(
    model: LdaMulticore,
    k: int,
    top_n: int = 10,
) -> dict[int, list[str]]:
    """Return {topic_id: [top_n keywords]} ordered by topic-word probability."""
    out: dict[int, list[str]] = {}
    for tid in range(k):
        terms = model.show_topic(tid, topn=top_n)
        out[tid] = [w for w, _ in terms]
    return out


def compute_doc_distributions(
    model: LdaMulticore,
    corpus_bow: list[list[tuple[int, int]]],
    k: int,
) -> tuple[list[int], list[list[float]]]:
    """Return (dominant_topic_per_doc, full_distribution_per_doc)."""
    dominant: list[int] = []
    full: list[list[float]] = []
    for bow in corpus_bow:
        dist = dict(model.get_document_topics(bow, minimum_probability=0.0))
        row = [float(dist.get(t, 0.0)) for t in range(k)]
        full.append(row)
        dominant.append(int(np.argmax(row)))
    return dominant, full


def qualitative_report(
    topics_keywords: dict[int, list[str]],
    names: dict[int, str] | None = None,
    coherent_threshold: float = 0.0,
) -> str:
    """Markdown structured report. Heuristic categorization placeholders.

    The bulk of qualitative analysis is annotated manually after this report
    in the notebook (or in docs/baselines/). This function produces the
    template + auto-grouped sections by simple heuristics.
    """
    names = names or {tid: ", ".join(kws[:3]) for tid, kws in topics_keywords.items()}
    lines: list[str] = []
    lines.append("# Qualitative report\n")
    lines.append(f"**Total topics:** {len(topics_keywords)}\n")
    lines.append("## Tópicos coesos\n_(preencher manualmente: tópicos com keywords semanticamente próximas e tema único)_\n")
    lines.append("## Tópicos genéricos / discurso\n_(preencher manualmente: keywords genéricas, sem tema concreto)_\n")
    lines.append("## Tópicos lixo / boilerplate\n_(preencher manualmente: keywords são fórmulas, hashtags, headers, etc.)_\n")
    lines.append("## Redundâncias\n_(preencher manualmente: tópicos diferentes que cobrem o mesmo tema)_\n")
    lines.append("## Candidatos a stop word\n_(preencher manualmente: termos repetidos em 3+ tópicos sem agregar diferenciação)_\n")
    lines.append("## Tabela de tópicos\n")
    lines.append("| ID | Nome | Top keywords |\n|---|---|---|\n")
    for tid in sorted(topics_keywords):
        kws = topics_keywords[tid]
        kws_s = ", ".join(kws[:10])
        nm = names.get(tid, "—")
        lines.append(f"| T{tid} | {nm} | {kws_s} |\n")
    return "".join(lines)


# ===========================================================================
# stm_pipeline.py
# ===========================================================================
"""STM pipeline: Python e dono do protocolo (C_v decide), R e motor de treino.

O R (`scripts/run_stm.R`) recebe texto JA lematizado (mesmo vocabulario do
LDA/NMF) e devolve top words / theta / beta / estimateEffect. O C_v e
calculado aqui, sobre o mesmo `tokenized`/`dictionary` dos outros modelos.
"""

_STM_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "run_stm.R"
# funcoes/identificadores permitidos em formulas R alem das colunas do input
_STM_FORMULA_FUNCS = {"s", "log", "I", "factor", "poly"}


def prepare_stm_input(
    df: pd.DataFrame,
    tokenized: list[list[str]],
    covariates: list[str],
    date_col: str = "data",
) -> pd.DataFrame:
    """Monta o stm_input: texto lematizado (join dos tokens) + covariaveis.

    O texto enviado ao R e o MESMO `tokenized` usado no C_v — o vocabulario
    do STM fica alinhado ao do LDA/NMF por construcao. `date_col` (ex. 'data')
    e renomeada para 'date' para casar com a stm_prevalence_formula.

    Levanta ValueError se covariavel estiver ausente ou tiver NA (o STM nao
    aceita NA na matriz de desenho — trate com df.dropna ANTES de lematizar,
    senao df e tokenized desalinham).
    """
    if len(df) != len(tokenized):
        raise ValueError(f"df ({len(df)}) e tokenized ({len(tokenized)}) desalinhados")
    d = df.copy()
    if date_col in d.columns and date_col != "date":
        d = d.rename(columns={date_col: "date"})
    missing = [c for c in covariates if c not in d.columns]
    if missing:
        raise ValueError(
            f"Covariavel(s) ausente(s): {missing}. Cols: {d.columns.tolist()}"
        )
    out = pd.DataFrame({"text": [" ".join(t) for t in tokenized]})
    if "post_id" in d.columns:
        out["post_id"] = d["post_id"].astype(str).values
    for c in covariates:
        if d[c].isna().any():
            raise ValueError(
                f"Covariavel '{c}' tem {int(d[c].isna().sum())} NA(s) — aplique "
                f"df.dropna(subset=[...]) ANTES de lemmatize_corpus para manter alinhamento"
            )
        out[c] = d[c].values
    return out


def _validate_stm_formula(formula: str, columns: list[str]) -> None:
    """Confere que a formula R so referencia colunas existentes no input."""
    idents = set(re.findall(r"[A-Za-z_][A-Za-z0-9_.]*", formula))
    unknown = {
        t for t in idents
        if "." not in t and t not in _STM_FORMULA_FUNCS and t not in columns
    }
    if unknown:
        raise ValueError(
            f"Formula STM referencia colunas inexistentes: {sorted(unknown)} — "
            f"colunas disponiveis: {columns}"
        )


def _run_stm_r(
    mode: str,
    input_csv,
    output_dir,
    stm_cfg: dict,
    seed: int,
    extra: dict | None = None,
) -> dict:
    """Executa run_stm.R e retorna o JSON do modo. RuntimeError com stdout/stderr em falha."""
    rscript = stm_cfg.get("rscript_path", "Rscript")
    if shutil.which(rscript) is None and not Path(rscript).exists():
        raise RuntimeError(
            f"Rscript nao encontrado ({rscript!r}). Instale o R (https://cran.r-project.org) "
            f"ou aponte stm.rscript_path no params.yaml para o binario, ex.: "
            f'"C:/Program Files/R/R-4.5.3/bin/Rscript.exe"'
        )
    if not _STM_SCRIPT.exists():
        raise RuntimeError(f"Script R nao encontrado: {_STM_SCRIPT}")
    cmd = [
        rscript, str(_STM_SCRIPT),
        "--mode", mode,
        "--input", str(input_csv),
        "--output_dir", str(output_dir),
        "--seed", str(seed),
        "--no_below", str(stm_cfg.get("no_below", 5)),
        "--no_above", str(stm_cfg.get("no_above", 0.5)),
        "--max_em_its", str(stm_cfg.get("max_em_its", 150)),
    ]
    for key, val in (extra or {}).items():
        cmd += [f"--{key}", str(val)]
    timeout_sec = int(stm_cfg.get("timeout_sec", 21600))
    print(f"[stm] cmd: {' '.join(cmd)}")
    # errors="replace": o R no Windows pode emitir cp1252 em stderr (mensagens
    # localizadas). Com decodificacao estrita, um byte invalido levantaria
    # UnicodeDecodeError AQUI e mascararia a falha real do R.
    r = subprocess.run(cmd, capture_output=True, text=True,
                       timeout=timeout_sec, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise RuntimeError(
            f"run_stm.R exit {r.returncode}\n=== stdout ===\n{r.stdout}\n=== stderr ===\n{r.stderr}"
        )
    print(r.stdout[-2000:])
    json_name = {
        "grid_k": "stm_grid_k.json",
        "grid_hparams": "stm_hparams.json",
        "train": "stm_final.json",
    }[mode]
    p = Path(output_dir) / json_name
    if not p.exists():
        raise RuntimeError(f"R nao gravou {p}. stdout tail:\n{r.stdout[-500:]}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _stm_topicos_conhecidos(topics: dict, dictionary, min_palavras: int = 3) -> dict[int, list[str]]:
    """``{topic_id: [palavras]}`` do JSON do R, filtrado ao vocab do Dictionary gensim.

    Palavras fora do ``dictionary`` sao descartadas; topicos que sobrem com
    menos de ``min_palavras`` palavras conhecidas sao removidos (CoherenceModel
    quebra com topico 100% OOV — mesmo tratamento de ``_compute_coherence``).

    Extraida como funcao propria (Task 11, protocolo secao 1.7) para que a
    coerencia (C_v) e a Diversity de um mesmo K sejam medidas sobre EXATAMENTE
    a mesma lista de palavras por topico — ``grid_search_k_stm`` monta este
    dicionario uma unica vez por K e reusa nas duas chamadas, em vez de
    reconstruir o filtro duas vezes.

    ASSIMETRIA DECLARADA entre bracos (revisao Task 11, fix round 1): o filtro
    de OOV e o piso ``min_palavras=3`` existem por exigencia do
    ``CoherenceModel`` (C_v), NAO da Diversity — a propria porta de
    ``compute_topic_diversity`` e ``>= 2`` palavras validas, sem nocao de OOV
    nenhuma. Como este dicionario alimenta as DUAS metricas para preservar
    "mesma lista para os dois" (o proprio pedido do brief da Task 11), a
    Diversity do braco STM sai medida sobre uma lista mais filtrada — sem
    palavras fora do vocabulario gensim e sem topicos com <3 palavras
    conhecidas — do que a Diversity dos outros tres bracos (LDA/NMF/BERTopic),
    que nao passam por filtro de OOV algum antes de entrar em
    ``compute_topic_diversity``. Na pratica a diferenca so aparece quando o
    vocab do R diverge do ``Dictionary`` gensim (raro — thresholds quase
    identicos) ou quando um topico do STM fica com <3 palavras conhecidas
    (nesse caso ele simplesmente NAO entra na Diversity do STM, enquanto o
    mesmo grau de degeneracao nos outros bracos so cairia com <2). Registrado
    tambem em ``docs/protocolo_selecao.md`` secao 1.7 — quem comparar Diversity
    entre bracos na camada 3 precisa saber que a base de calculo do STM nao e
    identica a dos demais.
    """
    known = dictionary.token2id
    filt = {int(t): [w for w in ws if w in known] for t, ws in topics.items()}
    return {t: ws for t, ws in filt.items() if len(ws) >= min_palavras}


def _stm_cv(
    topics: dict, tokenized: list[list[str]], dictionary, top_k: int | None = None,
) -> float:
    """C_v sobre top words do R, filtrando palavras fora do Dictionary gensim.

    O vocab do R (prepDocuments) e quase identico ao do gensim (mesmos
    thresholds), mas casos de borda existem — palavras desconhecidas quebram
    o CoherenceModel, dai o filtro.

    ``top_k`` (opcional): profundidade declarada passada ao
    ``compute_coherence_cv``/``CoherenceModel`` (protocolo secao 1.8, condicao
    1). ``None`` preserva o comportamento historico (profundidade derivada da
    lista recebida) — quem precisa da comparacao com o eixo Diversity de
    ``grid_search_k_stm`` deve passar o mesmo ``top_k`` explicito.
    """
    filt = _stm_topicos_conhecidos(topics, dictionary)
    if not filt:
        return float("nan")
    return compute_coherence_cv(filt, tokenized, dictionary, top_k=top_k)


def grid_search_k_stm(
    stm_df: pd.DataFrame,
    tokenized: list[list[str]],
    dictionary: Dictionary,
    k_range: Iterable[int],
    prevalence_formula: str,
    stm_cfg: dict,
    work_dir,
    seed: int = 42,
    top_n: int = 10,
    top_k: int = 10,
) -> tuple[dict[int, float], pd.DataFrame]:
    """Grid de K do STM: 1 fit por K no R (split heldout), C_v decidido aqui.

    Analogo ao grid_search_k (LDA): retorna (cv_scores, diag_df), onde diag_df
    traz os diagnosticos nativos do STM (heldout likelihood, dispersao dos
    residuos, semantic coherence, exclusivity) como apoio — o K e escolhido
    por C_v, comparavel entre os 4 modelos.

    ``diag_df`` tambem traz ``diversity`` (Topic Diversity, Dieng et al.) —
    protocolo secao 1.7: a fronteira do STM e semantic_coherence x
    exclusivity_stm (Roberts et al.) MAIS Diversity, para ordenar o STM pelo
    mesmo eixo de repeticao de vocabulario que os outros tres bracos. As top
    words de cada K sao extraidas UMA vez do JSON do R
    (``_stm_topicos_conhecidos``) e reusadas para C_v e Diversity — a mesma
    lista alimenta as duas metricas, em vez de duas reconstrucoes que podem
    divergir silenciosamente.

    ``top_k`` (protocolo secao 1.8, condicao 1): profundidade declarada
    passada ao ``CoherenceModel`` (via ``_stm_cv``) e a ``compute_topic_diversity``
    — mesma razao do ``top_k`` em ``grid_search_k``/``grid_search_k_nmf``.
    Independente de ``top_n``, que controla quantas palavras o R devolve por
    topico (`extra={"top_n": ...}`); ``top_k`` so afeta a profundidade lida
    dessas listas para as metricas de qualidade.
    """
    _validate_stm_formula(prevalence_formula, stm_df.columns.tolist())
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    input_csv = work_dir / "stm_input.csv"
    stm_df.to_csv(input_csv, index=False, encoding="utf-8")
    d = _run_stm_r(
        "grid_k", input_csv, work_dir, stm_cfg, seed,
        extra={
            "prevalence": prevalence_formula,
            "k_grid": ",".join(str(int(k)) for k in k_range),
            "top_n": top_n,
        },
    )
    cv_scores: dict[int, float] = {}
    rows: list[dict] = []
    for res in d["results"]:
        k = int(res["k"])
        topicos_k = _stm_topicos_conhecidos(res["topics"], dictionary)
        cv = (compute_coherence_cv(topicos_k, tokenized, dictionary, top_k=top_k)
              if topicos_k else float("nan"))
        diversity = compute_topic_diversity(topicos_k, top_k=top_k)
        cv_scores[k] = cv
        rows.append({
            "k": k,
            "cv": cv,
            "heldout_likelihood": res.get("heldout_likelihood"),
            "residual_dispersion": res.get("residual_dispersion"),
            "semantic_coherence": res.get("semantic_coherence"),
            "exclusivity_stm": res.get("exclusivity_stm"),
            "diversity": diversity,
            "iterations": res.get("iterations"),
            "elapsed_sec": res.get("elapsed_sec"),
        })
        print(f"  K={k}: C_v={cv:.4f}")
    diag_df = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)
    return cv_scores, diag_df


def grid_search_stm_hparams(
    stm_df: pd.DataFrame,
    tokenized: list[list[str]],
    dictionary: Dictionary,
    k: int,
    prevalence_formula: str,
    stm_cfg: dict,
    work_dir,
    sigma_grid: list | None = None,
    gamma_grid: list | None = None,
    seed: int = 42,
    top_n: int = 10,
) -> tuple[pd.DataFrame, float, str]:
    """Varre sigma.prior x gamma.prior com K fixo — analogo ao alpha/eta do LDA.

    sigma.prior — regularizacao [0,1] da covariancia entre topicos induzida
      pelas covariaveis de prevalencia (0 = sem regularizacao).
    gamma.prior — prior dos coeficientes de prevalencia: 'Pooled' (normal
      hierarquica) ou 'L1' (esparso, via glmnet).

    df_results ordenado por cv decrescente — primeira linha e a vencedora.
    """
    if sigma_grid is None:
        sigma_grid = [0, 0.3, 0.5, 0.7]
    if gamma_grid is None:
        gamma_grid = ["Pooled", "L1"]
    _validate_stm_formula(prevalence_formula, stm_df.columns.tolist())
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    input_csv = work_dir / "stm_input.csv"
    stm_df.to_csv(input_csv, index=False, encoding="utf-8")
    d = _run_stm_r(
        "grid_hparams", input_csv, work_dir, stm_cfg, seed,
        extra={
            "prevalence": prevalence_formula,
            "k": int(k),
            "sigma_grid": ",".join(str(s) for s in sigma_grid),
            "gamma_grid": ",".join(gamma_grid),
            "top_n": top_n,
        },
    )
    rows: list[dict] = []
    for res in d["results"]:
        rows.append({
            "sigma_prior": float(res["sigma_prior"]),
            "gamma_prior": str(res["gamma_prior"]),
            "cv": _stm_cv(res["topics"], tokenized, dictionary),
            "bound": res.get("bound"),
            "iterations": res.get("iterations"),
            "elapsed_sec": res.get("elapsed_sec"),
        })
    df_results = (
        pd.DataFrame(rows).sort_values("cv", ascending=False).reset_index(drop=True)
    )
    return df_results, float(df_results.iloc[0]["sigma_prior"]), str(df_results.iloc[0]["gamma_prior"])


def train_stm(
    stm_df: pd.DataFrame,
    k: int,
    prevalence_formula: str,
    stm_cfg: dict,
    work_dir,
    sigma_prior: float = 0.0,
    gamma_prior: str = "Pooled",
    seed: int = 42,
    top_n: int = 10,
) -> tuple[dict, dict, np.ndarray, pd.DataFrame, dict, dict]:
    """Treino final do STM + estimateEffect.

    Returns:
        keywords: {tid: [top words por probabilidade]}
        keywords_frex: {tid: [top words por FREX]} (nativo do STM)
        theta: ndarray (n_docs_kept, k) — docs em meta['docs_removed'] fora
        beta_df: DataFrame K x vocab (colunas = termos)
        effects: {tid: DataFrame(term, estimate, std_error, p_value)} do estimateEffect
        meta: k, sigma_prior, gamma_prior, iterations, vocab_size, n_docs,
              docs_removed (indices 0-based do stm_df a descartar no notebook)
    """
    _validate_stm_formula(prevalence_formula, stm_df.columns.tolist())
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    input_csv = work_dir / "stm_input.csv"
    stm_df.to_csv(input_csv, index=False, encoding="utf-8")
    d = _run_stm_r(
        "train", input_csv, work_dir, stm_cfg, seed,
        extra={
            "prevalence": prevalence_formula,
            "k": int(k),
            "sigma_prior": sigma_prior,
            "gamma_prior": gamma_prior,
            "top_n": top_n,
        },
    )
    keywords = {int(t): list(ws) for t, ws in d["topics"].items()}
    keywords_frex = {int(t): list(ws) for t, ws in d["topics_frex"].items()}
    theta = pd.read_csv(work_dir / d["theta_csv"]).values
    beta_df = pd.read_csv(work_dir / d["beta_csv"], encoding="utf-8")
    effects = {
        int(t): pd.DataFrame(rows)
        for t, rows in (d.get("prevalence_effects") or {}).items()
    }
    meta = {
        "k": int(d["k"]),
        "sigma_prior": d.get("sigma_prior"),
        "gamma_prior": d.get("gamma_prior"),
        "iterations": d.get("iterations"),
        "vocab_size": d.get("vocab_size"),
        "n_docs": d.get("n_docs"),
        "docs_removed": [int(i) for i in (d.get("docs_removed") or [])],
    }
    return keywords, keywords_frex, theta, beta_df, effects, meta
