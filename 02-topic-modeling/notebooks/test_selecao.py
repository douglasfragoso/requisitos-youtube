"""Testes de `_selecao.py` — a regra de selecao do protocolo.

    venv\\Scripts\\python -m pytest 03-topic-modeling/notebooks/test_selecao.py

`_selecao.py` nao ajusta modelo: le CSV de grade e decide. Por isso da para
testar de verdade, com DataFrames sinteticos, sem corpus e sem fit.
"""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _selecao  # noqa: E402


def test_seed_declarada_e_42():
    """Protocolo secao 1.1: a selecao roda sobre a seed 42, e so sobre ela."""
    assert _selecao.SEED == 42


def test_grade_single_seed_e_apta():
    """Protocolo secao 1.1: A1 (K identico entre seeds) NAO existe mais.

    Recusar grade de 1 seed era correto sob o protocolo de 16/08; sob o de
    17/08 em diante e justamente o regime normal.
    """
    d = pd.DataFrame({
        "NPMI": [-0.19, -0.22],
        "Diversity": [0.89, 0.91],
        "cobertura": [0.86, 0.80],
        "n_raw": [13.0, 24.0],
        "n_seeds": [1, 1],
        "K": [11, 15],
    })
    impedimentos = _selecao._grade_apta(d)
    assert not any("seed" in i.lower() for i in impedimentos), impedimentos


def test_grade_sem_npmi_continua_inapta():
    """O eixo que ORDENA nao pode faltar — essa guarda permanece."""
    d = pd.DataFrame({
        "Diversity": [0.89],
        "cobertura": [0.86],
        "n_raw": [13.0],
        "n_seeds": [1],
        "K": [11],
    })
    impedimentos = _selecao._grade_apta(d)
    assert any("NPMI" in i for i in impedimentos), impedimentos


def test_curva_k_usa_pareto_npmi_e_diversity():
    """Protocolo secoes 1.5 e 1.6: LDA e NMF ordenam por NPMI E Diversity.

    K=12 tem NPMI menor que K=20 mas Diversity muito maior: nao e dominado, e
    o desempate por menor K o elege. Ordenacao por NPMI puro elegeria K=20.
    """
    curvas = {
        "NPMI": {12: -0.20, 16: -0.25, 20: -0.18},
        "Diversity": {12: 0.95, 16: 0.70, 20: 0.72},
    }
    escolhido, criterio = _selecao._decidir_curva_k(curvas, k_min=10, k_max=25)
    assert escolhido == 12, f"escolheu {escolhido} por {criterio}"
    assert criterio == "pareto NPMI x Diversity, desempate menor K"


def test_curva_k_sem_diversity_cai_para_npmi_e_avisa():
    curvas = {"NPMI": {12: -0.20, 20: -0.18}}
    escolhido, criterio = _selecao._decidir_curva_k(curvas, k_min=10, k_max=25)
    assert escolhido == 20
    assert "sem Diversity" in criterio


def test_curva_k_respeita_a_faixa_declarada():
    """K fora da faixa e inadmissivel mesmo sendo o melhor em tudo."""
    curvas = {
        "NPMI": {5: 0.10, 12: -0.20},
        "Diversity": {5: 0.99, 12: 0.95},
    }
    escolhido, _ = _selecao._decidir_curva_k(curvas, k_min=10, k_max=25)
    assert escolhido == 12


# --------------------------------------------------------------------------
# _porta_cobertura_nmf — protocolo secao 1.6: minimum_probability e
# hiperparametro de SELECAO e determinante de DESCARTE ao mesmo tempo.
# --------------------------------------------------------------------------

def test_porta_cobertura_vacua_quando_tudo_cobre():
    """Protocolo secao 1.6, caso (2): cobertura ~100% em toda a grade."""
    g = pd.DataFrame({"kappa": [1.0, 0.5], "cobertura": [1.0, 1.0], "NPMI": [-0.2, -0.3]})
    filtrada, nota = _selecao._porta_cobertura_nmf(g)
    assert len(filtrada) == 2
    assert "vacua" in nota.lower()


def test_porta_cobertura_ativa_filtra_e_declara():
    """Protocolo secao 1.6, caso (3): ha ponto que descarta documento."""
    g = pd.DataFrame({"kappa": [1.0, 0.5], "cobertura": [1.0, 0.82], "NPMI": [-0.3, -0.1]})
    filtrada, nota = _selecao._porta_cobertura_nmf(g)
    assert len(filtrada) == 1
    assert filtrada.iloc[0]["cobertura"] == 1.0
    assert "ativa" in nota.lower()


def test_porta_cobertura_sem_coluna_recusa():
    """Sem a coluna medida, o protocolo PROIBE comparar NMF com LDA."""
    g = pd.DataFrame({"kappa": [1.0], "NPMI": [-0.2]})
    with pytest.raises(SystemExit, match="cobertura"):
        _selecao._porta_cobertura_nmf(g)


# --------------------------------------------------------------------------
# carregar_bertopic — casing do RAW. `_bertopic_sweep_metrics` (_helpers.py)
# devolve npmi/cv/div/excl/frex em MINUSCULO; sweep_bertopic_grid_raw.csv
# herda essa casing sem alteracao (so o agregado renomeia). carregar_bertopic
# le o RAW por padrao (secao 1.1). Sem normalizar, "NPMI" nunca existe no `d`
# que chega em `_grade_apta`/`_pareto` — Critical A da revisao final.
# --------------------------------------------------------------------------

def _grade_raw_bertopic_real():
    """Colunas e nomes EXATOS que `sweep_bertopic_grid` grava no RAW hoje:
    n_neighbors, min_cluster_size, min_samples, cluster_selection_method,
    reduce_nr, seed, n_raw, outlier_pre, outlier_post, K, npmi, cv, div,
    excl, frex (npmi/cv/div/excl/frex MINUSCULOS — vem direto de
    `_bertopic_sweep_metrics`, sem renomear). Dois pontos, deliberadamente
    NAO dominados um pelo outro em NPMI x Diversity x cobertura: o primeiro
    ganha em NPMI e Diversity, o segundo em cobertura — obriga a fronteira
    de Pareto real a rodar (uma grade degenerada com 1 ponto so nao
    discriminaria entre "Pareto correto" e "caiu para cobertura sozinha").
    """
    return pd.DataFrame({
        "n_neighbors": [10, 15],
        "min_cluster_size": [10, 15],
        "min_samples": [None, None],
        "cluster_selection_method": ["leaf", "leaf"],
        "reduce_nr": [None, None],
        "seed": [42, 42],
        "n_raw": [30, 35],
        "outlier_pre": [0.20, 0.10],
        "outlier_post": [0.20, 0.10],
        "K": [20, 22],
        "npmi": [-0.15, -0.20],
        "cv": [0.30, 0.28],
        "div": [0.85, 0.80],
        "excl": [0.50, 0.55],
        "frex": [0.50, 0.55],
    })


def _grade_agg_bertopic_real():
    """Agregado correspondente (ja com a casing correta, como
    `sweep_bertopic_grid` grava hoje) — so precisa existir para o glob de
    `carregar_bertopic` achar o diretorio do run; o conteudo lido de fato,
    com seed 42 disponivel no raw, vem do raw.
    """
    g = _grade_raw_bertopic_real().rename(
        columns={"npmi": "NPMI", "cv": "C_v", "div": "Diversity",
                 "excl": "Exclus", "frex": "FREX"})
    g["cobertura"] = 1 - g["outlier_pre"]
    return g


def test_carregar_bertopic_normaliza_casing_do_raw(tmp_path, monkeypatch):
    """A grade lida do RAW tem que sair com NPMI/Diversity maiusculos, nao
    npmi/div — e a coluna que `_grade_apta` procura.
    """
    monkeypatch.setattr(_selecao, "OUT", tmp_path)
    base = tmp_path / "folha" / "bertopic" / "folha_20260823_120000"
    base.mkdir(parents=True)
    _grade_agg_bertopic_real().to_csv(base / "sweep_bertopic_grid.csv", index=False)
    _grade_raw_bertopic_real().to_csv(base / "sweep_bertopic_grid_raw.csv", index=False)

    d = _selecao.carregar_bertopic("folha")

    assert "NPMI" in d.columns, d.columns.tolist()
    assert "Diversity" in d.columns, d.columns.tolist()
    assert d["NPMI"].notna().all()
    assert d["Diversity"].notna().all()
    assert list(d["NPMI"]) == [-0.15, -0.20]


def test_grade_apta_aceita_grade_lida_do_raw(tmp_path, monkeypatch):
    """Reproduz o modo de falha da Critical A sem --forcar: `_grade_apta`
    NAO pode recusar uma grade que tem NPMI valido, so porque veio do raw
    com o nome errado.
    """
    monkeypatch.setattr(_selecao, "OUT", tmp_path)
    base = tmp_path / "folha" / "bertopic" / "folha_20260823_120000"
    base.mkdir(parents=True)
    _grade_agg_bertopic_real().to_csv(base / "sweep_bertopic_grid.csv", index=False)
    _grade_raw_bertopic_real().to_csv(base / "sweep_bertopic_grid_raw.csv", index=False)

    d = _selecao.carregar_bertopic("folha")
    impedimentos = _selecao._grade_apta(d)
    assert impedimentos == [], impedimentos


def test_selecionar_bertopic_ponta_a_ponta_usa_pareto_nao_so_cobertura(tmp_path, monkeypatch, capsys):
    """Reproduz o segundo modo de falha da Critical A, o de `--forcar`: sem a
    normalizacao, `_pareto` nao acha NPMI/Diversity em `d.columns`, os exclui
    silenciosamente do eixo e a fronteira degrada pra cobertura sozinha — o
    modo de falha que `_grade_apta` existe para impedir. Ponta a ponta (sem
    --forcar) prova que a grade e aceita E que a fronteira usa os tres eixos
    declarados em EIXOS, nao so cobertura.
    """
    monkeypatch.setattr(_selecao, "OUT", tmp_path)
    base = tmp_path / "folha" / "bertopic" / "folha_20260823_120000"
    base.mkdir(parents=True)
    _grade_agg_bertopic_real().to_csv(base / "sweep_bertopic_grid.csv", index=False)
    _grade_raw_bertopic_real().to_csv(base / "sweep_bertopic_grid_raw.csv", index=False)

    adm = _selecao.selecionar_bertopic("folha")

    saida = capsys.readouterr().out
    assert "GRADE NAO APTA" not in saida
    assert not adm.empty
    # os DOIS pontos sao nao-dominados (um ganha em NPMI/Diversity, o outro
    # em cobertura) — se a fronteira tivesse degradado pra cobertura sozinha,
    # so o de maior cobertura teria sobrevivido a `_pareto`.
    assert adm["na_fronteira"].sum() == 2
    # desempate declarado (maior cobertura) elege o segundo ponto
    escolhida = adm.loc[adm["escolhida"]].iloc[0]
    assert escolhida["cobertura"] == pytest.approx(0.90)


# --------------------------------------------------------------------------
# selecionar_nmf — ponta a ponta com a grade EXATA que grid_search_nmf_hparams
# emite hoje. Fix round 2 da Task 6, item 6 do fix round 1: a Critical da
# revisao (`_selecao.py:423`, KeyError: 'NPMI') so teria sido pega por um
# teste que exercita `selecionar_nmf` ate declarar vencedor, nao so
# `_porta_cobertura_nmf` isolada.
# --------------------------------------------------------------------------

def _grade_segundo_estagio_real():
    """Colunas e nomes EXATOS que `grid_search_nmf_hparams` devolve hoje:
    kappa, minimum_probability, cv, NPMI, Diversity, cobertura (NPMI/Diversity
    maiusculos, cv/cobertura minusculos — deliberado, ver _helpers.py).
    Porta ATIVA: min_prob=0.05 descarta documentos em ambos os kappa.
    """
    return pd.DataFrame({
        "kappa": [0.5, 0.5, 1.0, 1.0],
        "minimum_probability": [0.0, 0.05, 0.0, 0.05],
        "cv": [0.30, 0.30, 0.28, 0.28],
        "NPMI": [-0.15, -0.15, -0.20, -0.20],
        "Diversity": [0.80, 0.80, 0.85, 0.85],
        "cobertura": [1.0, 0.90, 1.0, 0.95],
    })


def test_selecionar_nmf_ponta_a_ponta_declara_vencedor(tmp_path, monkeypatch, capsys):
    """Reproduz a grade real (mesmas colunas que `grid_search_nmf_hparams`
    emite) e prova que `selecionar_nmf` roda ate declarar vencedor sem
    KeyError. E o teste que teria pego a Critical antes da revisao.
    """
    monkeypatch.setattr(_selecao, "OUT", tmp_path)
    base = tmp_path / "folha" / "nmf"
    base.mkdir(parents=True)
    # nmf_metrics.csv so precisa existir p/ _selecionar_por_curva_k nao abortar
    # com "ausente"; sem k_npmi_scores ele imprime aviso e retorna, sem lancar.
    pd.DataFrame({"placeholder": [1]}).to_csv(base / "nmf_metrics.csv", index=False)
    _grade_segundo_estagio_real().to_csv(base / "nmf_kappa_minprob_grid.csv", index=False)

    _selecao.selecionar_nmf("folha")  # nao pode levantar KeyError

    saida = capsys.readouterr().out
    assert ">>> kappa=0.5" in saida
    assert "minimum_probability=0.0" in saida


def test_selecionar_nmf_sem_npmi_reproduz_a_critical(tmp_path, monkeypatch):
    """Falsificacao inversa: a grade PRE-fix (sem NPMI, exatamente o que
    `grid_search_nmf_hparams` emitia antes desta task) tem de continuar
    quebrando `selecionar_nmf` com KeyError — prova que o teste acima
    discrimina de verdade, e nao passaria por acidente.
    """
    monkeypatch.setattr(_selecao, "OUT", tmp_path)
    base = tmp_path / "folha" / "nmf"
    base.mkdir(parents=True)
    pd.DataFrame({"placeholder": [1]}).to_csv(base / "nmf_metrics.csv", index=False)
    grade_pre_fix = _grade_segundo_estagio_real().drop(columns=["NPMI", "Diversity"])
    grade_pre_fix.to_csv(base / "nmf_kappa_minprob_grid.csv", index=False)

    with pytest.raises(KeyError):
        _selecao.selecionar_nmf("folha")


# --------------------------------------------------------------------------
# _eixos_stm — Task 11: Diversity como terceiro eixo do braco STM (protocolo
# secao 1.7). Grades antigas (sem a coluna) caem para a fronteira nativa de
# Roberts et al., que e a divergencia da spec que este eixo existe pra fechar.
# --------------------------------------------------------------------------

def test_stm_usa_tres_eixos_quando_ha_diversity():
    """Protocolo secao 1.7: semantic coherence x exclusivity MAIS Diversity."""
    d = pd.DataFrame({
        "k": [10, 15, 20],
        "semantic_coherence": [-60.0, -55.0, -50.0],
        "exclusivity_stm": [9.5, 9.2, 9.0],
        "diversity": [0.95, 0.70, 0.68],
    })
    eixos = _selecao._eixos_stm(d)
    assert eixos == ["semantic_coherence", "exclusivity_stm", "diversity"]


def test_stm_cai_para_dois_eixos_e_avisa_sem_diversity():
    """Grade antiga nao tem a coluna — cai para a fronteira de Roberts pura."""
    d = pd.DataFrame({
        "k": [10, 15],
        "semantic_coherence": [-60.0, -55.0],
        "exclusivity_stm": [9.5, 9.2],
    })
    eixos = _selecao._eixos_stm(d)
    assert eixos == ["semantic_coherence", "exclusivity_stm"]
