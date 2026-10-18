import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_resultado import build_result  # noqa: E402


def _sentences():
    return pd.DataFrame({"sent_id": ["a_0", "a_1", "b_0"], "post_id": ["a", "a", "b"],
                         "sentence": ["Great screen.", "Hi.", "Battery is bad."],
                         "message_raw": ["x", "x", "y"]})


def _sentiment(ids=("b_0", "a_0", "a_1")):
    labels = {"a_0": "Positiva", "a_1": "Neutra", "b_0": "Negativa"}
    return pd.DataFrame({"sent_id": list(ids), "sentiment": [labels[i] for i in ids],
                         "score": 0.9, "score_positiva": 0.1, "score_neutra": 0.1,
                         "score_negativa": 0.8})


def test_rows_follow_corpus_order_and_drop_whole_transcript_columns():
    out = build_result(_sentences(), _sentiment())
    assert out["sent_id"].tolist() == ["a_0", "a_1", "b_0"]
    assert out["sentimento"].tolist() == ["Positiva", "Neutra", "Negativa"]
    assert "message_raw" not in out


def test_topics_are_optional_per_sentence():
    global_topics = pd.DataFrame({"sent_id": ["a_0", "b_0"], "global_topic_id": [11, 3],
                                  "global_topic_weight": [0.5, 0.4], "global_top_terms": ["screen", "battery"]})
    local_topics = pd.DataFrame({"sent_id": ["b_0"], "local_topic_id": [1], "local_topic_weight": [0.7],
                                 "local_top_terms": ["battery"], "lex_hit": [True], "intent_hint": ["queixa"]})
    out = build_result(_sentences(), _sentiment(), global_topics, local_topics).set_index("sent_id")
    assert out.loc["a_0", "global_topic_id"] == 11 and pd.isna(out.loc["a_0", "local_topic_id"])
    assert pd.isna(out.loc["a_1", "global_topic_id"])
    assert out["tema_selecionado"].tolist() == [False, False, True]


def test_missing_sentiment_is_an_error():
    with pytest.raises(ValueError):
        build_result(_sentences(), _sentiment(ids=("a_0", "b_0")))
