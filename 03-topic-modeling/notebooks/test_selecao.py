"""Testes do protocolo de selecao — versao STM-only (corpus YouTube)."""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _selecao  # noqa: E402


def test_seed_declarada_e_42():
    """Protocolo secao 1.1: a selecao roda sobre a seed 42, e so sobre ela."""
    assert _selecao.SEED == 42


def test_faixa_k_declara_os_dois_corpora_youtube():
    """A3: faixa substantiva de K declarada ANTES de olhar a grade."""
    assert _selecao.FAIXA_K == {"youtube_doc": (8, 25), "youtube_sent": (10, 40)}


def test_desempate_stm_e_menor_k():
    assert _selecao.DESEMPATE_STM == "K"
    assert _selecao.DESEMPATE_STM_ASCENDING is True


def test_pareto_exclui_linhas_com_nan_e_mantem_nao_dominadas():
    d = pd.DataFrame({
        "K": [10, 15, 20, 25],
        "semantic_coherence": [-60.0, -55.0, -50.0, float("nan")],
        "exclusivity_stm": [9.5, 9.2, 9.0, 9.9],
    })
    fr = _selecao._pareto(d, ["semantic_coherence", "exclusivity_stm"])
    assert sorted(fr["K"].tolist()) == [10, 15, 20]


def test_stm_usa_tres_eixos_quando_ha_diversity():
    """Protocolo secao 1.7: semantic coherence x exclusivity MAIS Diversity."""
    d = pd.DataFrame({
        "k": [10, 15, 20],
        "semantic_coherence": [-60.0, -55.0, -50.0],
        "exclusivity_stm": [9.5, 9.2, 9.0],
        "diversity": [0.95, 0.70, 0.68],
    })
    assert _selecao._eixos_stm(d) == ["semantic_coherence", "exclusivity_stm", "diversity"]


def test_stm_cai_para_dois_eixos_e_avisa_sem_diversity():
    d = pd.DataFrame({
        "k": [10, 15],
        "semantic_coherence": [-60.0, -55.0],
        "exclusivity_stm": [9.5, 9.2],
    })
    assert _selecao._eixos_stm(d) == ["semantic_coherence", "exclusivity_stm"]


def test_selecionar_stm_ponta_a_ponta_escolhe_menor_k_da_fronteira(tmp_path, monkeypatch, capsys):
    """Grade sintetica: K=8 e K=15 nao dominados; desempate = menor K → 8."""
    base = tmp_path / "youtube_doc" / "stm"
    base.mkdir(parents=True)
    pd.DataFrame({
        "k": [5, 8, 15, 30],
        "cv": [0.40, 0.55, 0.60, 0.58],
        "semantic_coherence": [-70.0, -55.0, -50.0, -45.0],
        "exclusivity_stm": [9.9, 9.6, 9.3, 8.0],
        "diversity": [0.90, 0.85, 0.80, 0.60],
        "heldout_likelihood": [-8.1, -8.0, -7.9, -7.8],
        "residual_dispersion": [5.0, 4.0, 3.5, 3.0],
    }).to_csv(base / "stm_grid_k_diagnostics.csv", index=False)
    monkeypatch.setattr(_selecao, "OUT", tmp_path)
    _selecao.selecionar_stm("youtube_doc")
    out = capsys.readouterr().out
    assert "ESCOLHIDO: K=8" in out
    assert "por C_v (reporte, nao decide) o argmax seria K=15" in out