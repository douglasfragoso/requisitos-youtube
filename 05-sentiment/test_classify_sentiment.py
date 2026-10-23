import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from classify_sentiment import ScoreCache, classify_sentences, summarize, to_record  # noqa: E402


def _scores(pos, neu, neg):
    return [{"label": "positive", "score": pos}, {"label": "neutral", "score": neu},
            {"label": "negative", "score": neg}]


class FakeClassifier:
    def __init__(self):
        self.calls = []

    def __call__(self, texts):
        self.calls.extend(texts)
        return [_scores(0.8, 0.1, 0.1) if "great" in t else _scores(0.1, 0.2, 0.7) for t in texts]


def _df(sentences):
    return pd.DataFrame({"sent_id": [f"s_{i}" for i in range(len(sentences))], "sentence": sentences})


def test_to_record_maps_labels_and_picks_best():
    record = to_record(_scores(0.1, 0.3, 0.6))
    assert record["sentiment"] == "Negativa"
    assert record["score"] == 0.6
    assert record["score_positiva"] == 0.1 and record["score_neutra"] == 0.3


def test_each_distinct_sentence_is_classified_once_and_rows_keep_order():
    fake = FakeClassifier()
    out = classify_sentences(_df(["great screen", "bad battery", "great screen"]), fake,
                             batch_size=1, log=lambda _: None)
    assert sorted(fake.calls) == ["bad battery", "great screen"]
    assert out["sentiment"].tolist() == ["Positiva", "Negativa", "Positiva"]
    assert out["sent_id"].tolist() == ["s_0", "s_1", "s_2"]


def test_sentences_are_sent_to_the_classifier_sorted_by_length():
    fake = FakeClassifier()
    out = classify_sentences(_df(["a much longer great sentence", "bad", "great one"]), fake,
                             log=lambda _: None)
    assert fake.calls == ["bad", "great one", "a much longer great sentence"]
    assert out["sentiment"].tolist() == ["Positiva", "Negativa", "Positiva"]


def test_cache_skips_already_scored_sentences(tmp_path):
    cache = ScoreCache(tmp_path / "scores.sqlite")
    classify_sentences(_df(["great fan"]), FakeClassifier(), cache, log=lambda _: None)
    second = FakeClassifier()
    out = classify_sentences(_df(["great fan", "loud fan"]), second, ScoreCache(tmp_path / "scores.sqlite"),
                             log=lambda _: None)
    assert second.calls == ["loud fan"]
    assert out["sentiment"].tolist() == ["Positiva", "Negativa"]


def test_summarize_reports_counts_fractions_and_balance():
    df = pd.DataFrame({"group": ["a", "a", "a", "b"],
                       "sentiment": ["Positiva", "Negativa", "Neutra", "Negativa"]})
    out = summarize(df, ["group"]).set_index("group")
    assert out.loc["a", "n"] == 3 and out.loc["b", "Positiva"] == 0
    assert out.loc["a", "saldo"] == 0.0
    assert out.loc["b", "frac_negativa"] == 1.0
