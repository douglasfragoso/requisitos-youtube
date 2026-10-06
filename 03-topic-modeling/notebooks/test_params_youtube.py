"""Contrato dos params.yaml: STM de documento e NMF global/local de sentencas."""
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _helpers import get_corpus_config, load_params  # noqa: E402

RAIZ = Path(__file__).resolve().parents[2]


def test_preprocessing_params_so_tem_corpus_youtube():
    p = yaml.safe_load(open(RAIZ / "01-preprocessing/configs/params.yaml", encoding="utf-8"))
    assert list(p["corpora"]) == ["youtube"]
    c = p["corpora"]["youtube"]
    assert c["language"] == "en"
    assert c["text_column"] == "message"
    assert c["id_column"] == "post_id"
    assert c["emoji_handling"] == "none"
    assert p["default_corpus"] == "youtube"


def test_topic_modeling_params_tem_youtube_doc_e_youtube_sent():
    p = load_params()
    assert list(p["corpora"]) == ["youtube_doc", "youtube_sent"]
    assert p["default_corpus"] == "youtube_doc"
    assert "stm" in p and "evaluation" in p
    for k in ("advisor", "guided", "brett", "nmf_restrito"):
        assert k not in p, f"bloco {k} saiu na limpeza"


def test_youtube_sent_tem_so_o_que_o_nmf_usa():
    _, c = get_corpus_config(load_params(), "youtube_sent")
    assert c["text_column"] == "sentence"
    assert c["language"] == "en"
    assert c["token_pattern"] and c["stopwords_emojis"]


def test_youtube_doc_tem_chaves_do_stm():
    _, c = get_corpus_config(load_params(), "youtube_doc")
    assert c["subdir"] == "youtube"
    assert c["text_column"] == "message"
    assert c["date_column"] == "upload_date"
    assert c["covariates"] == ["date", "product_category"]
    assert c["stm_prevalence_formula"] == "~ s(as.numeric(as.Date(date))) + product_category"
    assert c["stm_min_tokens_per_doc"] == 200
    assert isinstance(c["stm_best_k"], int) and 8 <= c["stm_best_k"] <= 25


def test_pipeline_final_tem_chaves_exigidas_pelos_scripts():
    f = load_params()["final_sentence_pipeline"]
    assert f["global_k"] == 20
    assert f["local"]["k_range"] == [2, 6]
    assert f["local"]["no_below"] == 5 and f["local"]["no_above"] == 0.5
    for k in ("global_run", "selected_topics", "local_subdir", "expected_sentences",
              "review_output", "top_k", "n_random"):
        assert k in f, k
