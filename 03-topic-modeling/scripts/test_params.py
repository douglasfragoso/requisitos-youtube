"""Contrato do params.yaml: so o que os scripts e notebooks de NMF leem."""
from pathlib import Path

import yaml

RAIZ = Path(__file__).resolve().parents[2]


def _params():
    return yaml.safe_load((RAIZ / "03-topic-modeling/configs/params.yaml").read_text(encoding="utf-8"))


def test_preprocessing_params_so_tem_corpus_youtube():
    p = yaml.safe_load((RAIZ / "01-preprocessing/configs/params.yaml").read_text(encoding="utf-8"))
    assert list(p["corpora"]) == ["youtube"]
    c = p["corpora"]["youtube"]
    assert (c["language"], c["text_column"], c["id_column"]) == ("en", "message", "post_id")


def test_corpus_de_sentencas_tem_o_que_o_nmf_usa():
    p = _params()
    assert list(p["corpora"]) == ["youtube_sent"] and p["default_corpus"] == "youtube_sent"
    c = p["corpora"]["youtube_sent"]
    assert c["text_column"] == "sentence" and c["language"] == "en"
    assert c["token_pattern"] and c["stopwords_emojis"]


def test_pipeline_final_tem_chaves_exigidas_pelos_scripts():
    f = _params()["final_sentence_pipeline"]
    assert f["global_k"] == 20
    assert f["local"] == {"k_range": [2, 6], "no_below": 5, "no_above": 0.5}
    for k in ("global_run", "selected_topics", "local_subdir", "expected_sentences",
              "review_output", "top_k", "n_random"):
        assert k in f, k
    assert _params()["evaluation"]["top_n_keywords"] == 10


def test_run_pinado_e_selecao_sao_coerentes():
    f = _params()["final_sentence_pipeline"]
    if f["global_run"]:
        assert f["selected_topics"], "global_run fixado exige selected_topics"
        assert len(set(f["selected_topics"])) == len(f["selected_topics"])
        assert all(0 <= t < f["global_k"] for t in f["selected_topics"])


def test_blocos_removidos_nao_voltaram():
    p = _params()
    for k in ("stm", "llm", "advisor", "guided", "brett", "nmf_restrito"):
        assert k not in p, k
