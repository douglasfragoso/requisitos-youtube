"""Contrato do corpus_limpo.csv do corpus youtube (roda sobre a versao mais recente em data/output)."""
import re
from pathlib import Path

import pandas as pd
import pytest

BASE = Path(__file__).resolve().parent / "data" / "output" / "youtube"
_STAMP = re.compile(r"_\d{8}_\d{6}$")


def _latest() -> Path:
    runs = [d for d in BASE.iterdir() if d.is_dir() and _STAMP.search(d.name) and (d / "corpus_limpo.csv").exists()] if BASE.exists() else []
    if not runs:
        pytest.skip("corpus_limpo.csv ainda nao gerado — rode o notebook 01")
    return max(runs, key=lambda d: d.name) / "corpus_limpo.csv"


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(_latest())


def test_colunas_de_contrato(df):
    for c in ["post_id", "message", "upload_date", "product_category", "source", "channel", "year"]:
        assert c in df.columns, c


def test_post_id_e_id_de_video_unico(df):
    assert df["post_id"].is_unique
    assert df["post_id"].str.len().eq(11).all()


def test_tamanho_e_idioma(df):
    assert 1000 <= len(df) <= 1200, len(df)
    assert (df["message"].str.split().str.len() >= 100).all()
    pt = df["message"].str.lower().str.count(r"\b(não|você|muito|também)\b")
    assert (pt <= 2).mean() > 0.99


def test_covariaveis_sem_na(df):
    assert df["product_category"].notna().all()
    assert df["upload_date"].notna().all()
    assert df["upload_date"].str.match(r"^\d{4}-\d{2}-\d{2}$").all()


def test_categorias_esperadas(df):
    vc = df["product_category"].value_counts()
    assert vc.index[0] == "laptop" and vc.index[1] == "phone"
    assert vc.get("other", 0) / len(df) <= 0.25