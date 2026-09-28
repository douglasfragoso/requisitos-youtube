"""Contrato dos resultados do STM em youtube_doc (run mais recente)."""
import re
from pathlib import Path

import pandas as pd
import pytest
import yaml

BASE = Path(__file__).resolve().parent.parent / "data" / "output" / "youtube_doc" / "stm"
STAMP = re.compile(r"_\d{8}_\d{6}$")


def _latest_run() -> Path:
    runs = [d for d in BASE.iterdir() if d.is_dir() and STAMP.search(d.name) and (d / "stm_results.csv").exists()] if BASE.exists() else []
    if not runs:
        pytest.skip("stm_results.csv ainda nao gerado")
    return max(runs, key=lambda d: d.name)


@pytest.fixture(scope="module")
def run_dir(): return _latest_run()

@pytest.fixture(scope="module")
def results(run_dir): return pd.read_csv(run_dir / "stm_results.csv")

@pytest.fixture(scope="module")
def metrics(run_dir): return pd.read_csv(run_dir / "stm_metrics.csv").iloc[0]


def test_resultados_tem_colunas_de_contrato(results):
    for c in ["post_id", "doc_id", "topic_id", "topic_name", "topic_prob_distribution", "topic_type", "granularity"]:
        assert c in results.columns, c


def test_post_id_e_id_de_video_unico(results):
    assert results["post_id"].is_unique
    assert results["post_id"].str.len().eq(11).all()


def test_k_pinado_por_maior_cv(metrics):
    assert int(metrics["n_topics"]) == 12


def test_topicos_tem_nomes_e_keywords(run_dir, metrics):
    topics = pd.read_csv(run_dir / "stm_topics_for_eval.csv")
    assert len(topics) == int(metrics["n_topics"])
    assert topics["topic_name"].str.strip().ne("").all()
    assert topics["keywords"].str.split(", ").apply(len).ge(5).all()


def test_topics_exclude_configured_stopwords(run_dir):
    with (Path(__file__).resolve().parent.parent / "configs" / "params.yaml").open(encoding="utf-8") as handle:
        params = yaml.safe_load(handle)
    stopwords = set(params["corpora"]["youtube_doc"]["stopwords_emojis"])
    topics = pd.read_csv(run_dir / "stm_topics_for_eval.csv")
    words = {word.strip().lower() for row in topics["keywords"] for word in row.split(",")}
    assert not words & stopwords, sorted(words & stopwords)


def test_efeitos_de_prevalencia_tem_product_category(run_dir):
    effects = pd.read_csv(run_dir / "stm_prevalence_effects.csv")
    for c in ["topic_id", "term", "estimate", "std_error", "p_value"]:
        assert c in effects.columns, c
    assert effects["term"].str.startswith("product_category").any()


def test_metricas_dentro_de_faixas_plausiveis(metrics):
    assert 0.0 <= float(metrics["exclusivity"]) <= 1.0
    assert 0.0 <= float(metrics["topic_diversity_dieng"]) <= 1.0
    assert 0.0 <= float(metrics["frex"]) <= 1.0
