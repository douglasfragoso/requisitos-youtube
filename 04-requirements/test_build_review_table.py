import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_review_table import join_stm_topics, make_review_table  # noqa: E402


def test_join_keeps_sentences_without_stm_document():
    sentences = pd.DataFrame([
        {"sent_id": "abcdefghijk_0", "post_id": "abcdefghijk", "sentence": "A sentence.", "n_words": 2},
        {"sent_id": "missing_id_0", "post_id": "missing_id", "sentence": "Missing sentence.", "n_words": 2},
    ])
    stm = pd.DataFrame([{"post_id": "abcdefghijk", "topic_id": 1, "topic_name": "Phones", "topic_prob_distribution": "[0.9]"}])
    out = join_stm_topics(sentences, stm)
    missing = out.loc[out.post_id.eq("missing_id")].iloc[0]
    assert pd.isna(missing.topic_id)
    assert len(out) == len(sentences)


def test_review_export_is_human_decision_template():
    linked = pd.DataFrame([{"sent_id": "abcdefghijk_0", "post_id": "abcdefghijk", "sentence": "A sentence.", "context": "A sentence.", "n_words": 2, "topic_id": 1, "topic_name": "Phones", "product_category": "phone"}])
    out = make_review_table(linked)
    assert out["requirement_candidate"].isna().all()
    assert out["review_decision"].isna().all()
    assert {"aspect", "review_notes", "topic_id", "sentence", "context"} <= set(out.columns)

def test_generated_review_preserves_sentence_provenance():
    from pathlib import Path
    base = Path(__file__).resolve().parent / "data" / "output"
    files = sorted(base.glob("youtube_sent_*/requirement_review.csv")) if base.exists() else []
    if not files:
        import pytest
        pytest.skip("requirement_review.csv ainda nao gerado")
    df = pd.read_csv(files[-1])
    assert df["sent_id"].is_unique
    assert df["post_id"].notna().all()
    assert df["sentence"].str.strip().ne("").all()