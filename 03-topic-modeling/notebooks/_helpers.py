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
import json
import logging
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
import numpy as np
import ollama
import pandas as pd
import yaml
from gensim.corpora import Dictionary
from gensim.models import LdaMulticore, Nmf


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
# lda_pipeline.py
# ===========================================================================
"""LDA pipeline: grid search K, train, extract topics, qualitative report.

Corpus-agnostic. Importado pelos notebooks de 03-topic-modeling.
"""


# ===========================================================================
# nmf_pipeline.py
# ===========================================================================
"""NMF pipeline: grid search K, then kappa x minimum_probability (K fixed),
train. Mirrors lda_pipeline.py's two-stage protocol; extract_topics_keywords,
compute_doc_distributions and export_results are shared (model-agnostic).
"""


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
