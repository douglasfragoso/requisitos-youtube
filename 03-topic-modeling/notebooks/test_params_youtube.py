"""Contrato dos params.yaml apos a limpeza: so corpora youtube, chaves exigidas pelo template STM."""
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
    assert "sample" not in c
    assert p["default_corpus"] == "youtube"


def test_topic_modeling_params_tem_youtube_doc_e_youtube_sent():
    p = load_params()
    assert list(p["corpora"]) == ["youtube_doc", "youtube_sent"]
    assert p["default_corpus"] == "youtube_doc"
    for k in ("bertopic", "lda", "nmf"):
        assert k not in p, f"bloco {k} deveria ter saido"
    assert "stm" in p and "evaluation" in p and "advisor" in p


def test_youtube_sent_tem_chaves_do_template_stm():
    _, c = get_corpus_config(load_params(), "youtube_sent")
    assert c["text_column"] == "sentence"
    assert c["post_id_column"] == "sent_id"
    assert c["covariates"] == ["product_category"]
    assert c["stm_prevalence_formula"] == "~ product_category"
    assert c["stm_min_tokens_per_doc"] == 5
    assert c["stm_no_below"] == 20
    assert c["stm_no_above"] == 0.5
    assert c["stm_k_range"] == [10, 15, 20, 25, 30, 35, 40]
    assert c["stm_grid_sample_size"] == 25000
    assert c["stm_best_k"] == 15
    assert c["language"] == "en"


def test_youtube_doc_tem_chaves_do_template_stm():
    _, c = get_corpus_config(load_params(), "youtube_doc")
    assert c["subdir"] == "youtube"
    assert c["text_column"] == "message"
    assert c["date_column"] == "upload_date"
    assert c["covariates"] == ["date", "product_category"]
    assert c["stm_prevalence_formula"] == "~ s(as.numeric(as.Date(date))) + product_category"
    assert c["stm_min_tokens_per_doc"] == 200
    assert c["language"] == "en"




def test_youtube_doc_tem_k_pinado_pelo_protocolo():
    _, c = get_corpus_config(load_params(), "youtube_doc")
    assert isinstance(c.get("stm_best_k"), int)
    assert 8 <= c["stm_best_k"] <= 25

def test_youtube_sent_tem_k_pinado_pelo_protocolo():
    _, c = get_corpus_config(load_params(), "youtube_sent")
    assert c["stm_best_k"] == 15
