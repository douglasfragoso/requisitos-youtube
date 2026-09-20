from pathlib import Path
import re

import pandas as pd
import pytest

BASE = Path(__file__).resolve().parent / "data" / "output" / "youtube_sent"
STAMP = re.compile(r"_\d{8}_\d{6}$")


def latest_sentence_csv() -> Path:
    runs = [d for d in BASE.iterdir() if d.is_dir() and STAMP.search(d.name) and (d / "corpus_sentencas.csv").exists()] if BASE.exists() else []
    if not runs:
        pytest.skip("corpus_sentencas.csv ainda nao gerado")
    return max(runs, key=lambda d: d.name) / "corpus_sentencas.csv"


def test_sentence_output_contract():
    df = pd.read_csv(latest_sentence_csv())
    assert df["sent_id"].is_unique
    assert df["post_id"].str.len().eq(11).all()
    assert {"sentence", "context", "punctuation_restored", "is_fragment"} <= set(df.columns)
    assert df.loc[df.source.eq("legenda"), "punctuation_restored"].all()