"""Helpers for the local (per-topic) NMF: lemmatized BoW, training, keywords, diversity."""

from __future__ import annotations

import numpy as np
from gensim.corpora import Dictionary
from gensim.models import Nmf

SPACY_MODEL = "en_core_web_sm"
_NLP = None


def _get_nlp():
    global _NLP
    if _NLP is None:
        import spacy
        try:
            _NLP = spacy.load(SPACY_MODEL)
        except OSError as error:
            raise OSError(f"spaCy model '{SPACY_MODEL}' not installed. "
                          f"Run: python -m spacy download {SPACY_MODEL}") from error
    return _NLP


def lemmatize_corpus(docs: list[str], no_below: int = 5, no_above: float = 0.5,
                     extra_stopwords: list[str] | None = None,
                     batch_size: int = 256) -> tuple[list[list[str]], Dictionary]:
    """Lemmatize English documents and build a gensim Dictionary with filter_extremes.

    Keeps alphabetic, non-stop lemmas longer than two characters and drops
    ``extra_stopwords`` (case-insensitive).
    """
    nlp = _get_nlp()
    extra = {word.lower() for word in (extra_stopwords or [])}
    tokenized = [
        [token.lemma_.lower() for token in doc
         if not token.is_stop and token.is_alpha and len(token.lemma_) > 2
         and token.lemma_.lower() not in extra]
        for doc in nlp.pipe(docs, batch_size=batch_size, n_process=1)
    ]
    dictionary = Dictionary(tokenized)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above)
    return tokenized, dictionary


def train_nmf(corpus_bow: list[list[tuple[int, int]]], dictionary: Dictionary, k: int,
              seed: int = 42, passes: int = 20, kappa: float = 1.0,
              minimum_probability: float = 0.01, normalize: bool = True) -> Nmf:
    """Train a gensim NMF with ``k`` topics on raw BoW counts."""
    return Nmf(corpus=corpus_bow, id2word=dictionary, num_topics=k, random_state=seed,
               passes=passes, kappa=kappa, minimum_probability=minimum_probability,
               normalize=normalize)


def extract_topics_keywords(model: Nmf, k: int, top_n: int = 10) -> dict[int, list[str]]:
    """Return {topic_id: [top_n keywords]} ordered by topic-word weight."""
    return {topic: [word for word, _ in model.show_topic(topic, topn=top_n)]
            for topic in range(k)}


def compute_doc_distributions(model: Nmf, corpus_bow: list[list[tuple[int, int]]],
                              k: int) -> tuple[list[int], list[list[float]]]:
    """Return (dominant topic per document, full topic distribution per document)."""
    dominant: list[int] = []
    full: list[list[float]] = []
    for bow in corpus_bow:
        weights = dict(model.get_document_topics(bow, minimum_probability=0.0))
        row = [float(weights.get(topic, 0.0)) for topic in range(k)]
        full.append(row)
        dominant.append(int(np.argmax(row)))
    return dominant, full


def compute_topic_diversity(topics_keywords: dict[int, list[str]], top_k: int = 10) -> float:
    """Topic diversity (Dieng et al., 2020): unique top-k words / (k * valid topics).

    Skips the -1 topic and topics with fewer than two non-empty keywords.
    """
    valid = []
    for topic, words in topics_keywords.items():
        if topic == -1:
            continue
        clean = [word for word in words[:top_k] if word]
        if len(clean) >= 2:
            valid.append(clean)
    if not valid:
        return 0.0
    unique = len({word for words in valid for word in words})
    return unique / (top_k * len(valid))
