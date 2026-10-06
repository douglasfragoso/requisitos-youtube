import sys
from pathlib import Path

import numpy as np
import pytest
from gensim.corpora import Dictionary

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nmf_utils import (  # noqa: E402
    compute_doc_distributions,
    compute_topic_diversity,
    extract_topics_keywords,
    lemmatize_corpus,
    train_nmf,
)

TEXTOS = [["a", "b", "c"], ["a", "b", "d"], ["c", "d", "e"], ["e", "f", "a"]] * 8


@pytest.fixture
def bow():
    dic = Dictionary(TEXTOS)
    return dic, [dic.doc2bow(t) for t in TEXTOS]


def test_topic_diversity_extremos():
    assert compute_topic_diversity({0: ["a", "b"], 1: ["c", "d"]}, top_k=2) == 1.0
    assert compute_topic_diversity({0: ["a", "b"], 1: ["a", "b"]}, top_k=2) == 0.5


def test_topic_diversity_ignora_outlier_e_topico_degenerado():
    assert compute_topic_diversity({-1: ["x", "y"], 0: ["a", "b"], 1: [""]}, top_k=2) == 1.0


def test_topic_diversity_com_top_k_maior_que_as_keywords_subestima():
    assert compute_topic_diversity({0: ["a", "b"]}, top_k=10) == pytest.approx(0.2)


def test_topic_diversity_sem_topico_valido_e_zero():
    assert compute_topic_diversity({}, top_k=5) == 0.0


def test_nmf_get_topics_normaliza_linhas_por_padrao(bow):
    """O projeto usa normalize=True (default do gensim); sem isso as linhas nao somam 1."""
    dic, corpus = bow
    normal = train_nmf(corpus, dic, k=2, passes=5)
    assert np.allclose(normal.get_topics().sum(axis=1), 1.0, atol=1e-6)
    cru = train_nmf(corpus, dic, k=2, passes=5, normalize=False)
    assert not np.allclose(cru.get_topics().sum(axis=1), 1.0, atol=1e-6)


def test_keywords_e_distribuicoes_tem_o_formato_esperado(bow):
    dic, corpus = bow
    model = train_nmf(corpus, dic, k=2, passes=5)
    keywords = extract_topics_keywords(model, k=2, top_n=3)
    assert set(keywords) == {0, 1} and all(len(v) == 3 for v in keywords.values())
    dominant, dist = compute_doc_distributions(model, corpus, k=2)
    assert len(dominant) == len(dist) == len(corpus)
    assert all(len(row) == 2 for row in dist) and set(dominant) <= {0, 1}


def test_lemmatize_corpus_remove_stopwords_e_aplica_filtro_de_frequencia():
    pytest.importorskip("spacy")
    docs = ["The batteries last hours and the battery charges fast."] * 6 + ["The screen is bright."]
    tokenized, dic = lemmatize_corpus(docs, no_below=3, no_above=1.0,
                                      extra_stopwords=["fast"])
    assert "battery" in dic.token2id and "fast" not in dic.token2id
    assert "screen" not in dic.token2id          # aparece em 1 documento (< no_below)
    assert all("the" not in doc for doc in tokenized)
