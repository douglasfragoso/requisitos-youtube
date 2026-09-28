"""Contrato dos resultados do STM em youtube_sent (run pinado)."""
import re
from pathlib import Path

import pandas as pd
import pytest
import yaml

BASE = Path(__file__).resolve().parent.parent / "data" / "output" / "youtube_sent" / "stm"
STAMP = re.compile(r"_\d{8}_\d{6}$")


def test_run_pinado_tem_exports_obrigatorios():
    params_path = Path(__file__).resolve().parent.parent / "configs" / "params.yaml"
    with params_path.open(encoding="utf-8") as handle:
        run_name = yaml.safe_load(handle)["nmf_restrito"]["youtube_sent"]["stm_run"]
    pinned = BASE / run_name
    required = ("stm_results.csv", "stm_metrics.csv",
                "stm_prevalence_effects.csv", "stm_topics_for_eval.csv")
    assert pinned.is_dir(), pinned
    assert all((pinned / name).is_file() for name in required), pinned


def _pinned_run() -> Path:
    params_path = Path(__file__).resolve().parent.parent / "configs" / "params.yaml"
    with params_path.open(encoding="utf-8") as handle:
        run_name = yaml.safe_load(handle)["nmf_restrito"]["youtube_sent"]["stm_run"]
    run_dir = BASE / run_name
    assert STAMP.search(run_name) and run_dir.is_dir(), run_dir
    assert (run_dir / "stm_results.csv").is_file(), run_dir
    return run_dir


@pytest.fixture(scope="module")
def run_dir():
    return _pinned_run()


@pytest.fixture(scope="module")
def results(run_dir):
    return pd.read_csv(run_dir / "stm_results.csv")


@pytest.fixture(scope="module")
def metrics(run_dir):
    return pd.read_csv(run_dir / "stm_metrics.csv").iloc[0]


def test_resultados_tem_colunas_de_contrato(results):
    for c in ["post_id", "doc_id", "topic_id", "topic_name", "topic_prob_distribution", "topic_type", "granularity"]:
        assert c in results.columns, c


def test_post_id_e_sent_id_unico(results):
    assert results["post_id"].is_unique
    assert results["post_id"].str.match(r"^.+_\d+$").all()


def test_topicos_tem_nomes_e_keywords(run_dir, metrics):
    topics = pd.read_csv(run_dir / "stm_topics_for_eval.csv")
    assert len(topics) == int(metrics["n_topics"])
    assert topics["topic_name"].str.strip().ne("").all()
    assert topics["keywords"].str.split(", ").apply(len).ge(5).all()


def test_efeitos_de_prevalencia_tem_product_category(run_dir):
    effects = pd.read_csv(run_dir / "stm_prevalence_effects.csv")
    for c in ["topic_id", "term", "estimate", "std_error", "p_value"]:
        assert c in effects.columns, c
    assert effects["term"].str.startswith("product_category").any()


def test_metricas_dentro_de_faixas_plausiveis(metrics):
    assert 0.0 <= float(metrics["exclusivity"]) <= 1.0
    assert 0.0 <= float(metrics["topic_diversity_dieng"]) <= 1.0
    assert 0.0 <= float(metrics["frex"]) <= 1.0


def test_k_pinado_por_protocolo_de_selecao(metrics):
    assert int(metrics["n_topics"]) == 15
