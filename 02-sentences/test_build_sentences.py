import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_sentences import build_sentence_corpus  # noqa: E402


class FakePunctuationModel:
    def __init__(self, output): self.output = output
    def restore_punctuation(self, text): return self.output


def _row(post_id, message, source="whisper", category="phone"):
    return {"post_id": post_id, "message": message, "source": source,
            "product_category": category, "upload_date": "2026-01-01",
            "year": 2026, "channel": "Test channel"}


def test_caption_preserves_raw_and_uses_restored_text_for_segmentation():
    df = pd.DataFrame([_row("abcdefghijk", "first sentence second sentence", "legenda")])
    out = build_sentence_corpus(df, punctuation_model=FakePunctuationModel("First sentence. Second sentence."))
    assert out["message_raw"].eq("first sentence second sentence").all()
    assert out["punctuation_restored"].all()
    assert out["sentence"].tolist() == ["First sentence.", "Second sentence."]


def test_context_never_crosses_post_id_boundaries():
    df = pd.DataFrame([
        _row("aaaaaaaaaaa", "First document. Last document."),
        _row("bbbbbbbbbbb", "Other document. Final other document."),
    ])
    out = build_sentence_corpus(df, context_window=1)
    assert "Other document" not in out.loc[out.sent_id.eq("aaaaaaaaaaa_0"), "context"].iloc[0]
    assert "First document" not in out.loc[out.sent_id.eq("bbbbbbbbbbb_0"), "context"].iloc[0]


def test_duplicate_and_fragment_flags_are_retained():
    df = pd.DataFrame([
        _row("aaaaaaaaaaa", "Okay. Same repeated sentence."),
        _row("bbbbbbbbbbb", "Same repeated sentence."),
    ])
    out = build_sentence_corpus(df)
    assert out["is_dup_exact"].sum() == 2
    assert out.loc[out.sentence.eq("Okay."), "is_fragment"].all()


def test_caption_requires_punctuation_model():
    df = pd.DataFrame([_row("abcdefghijk", "unpunctuated caption", "legenda")])
    with pytest.raises(RuntimeError, match="punctuation"):
        build_sentence_corpus(df)