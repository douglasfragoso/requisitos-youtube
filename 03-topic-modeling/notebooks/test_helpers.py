"""Testes de `_helpers.py` — o modulo de onde saem os numeros publicados.

Escopo deliberadamente pequeno: este arquivo nasce da revisao do Plano 5
(`docs/revisao-2026-08-01.md` §M7), que apontou `_helpers.py` como o maior risco
descoberto do projeto — 2.6k linhas, zero teste, e todas as metricas da
dissertacao passando por ele. Comeca travando os dois bugs corrigidos em
2026-08-01 (C_v com vocabulario incompleto e resolucao latest-wins) mais o
comportamento das metricas que a revisao auditou.

    venv\\Scripts\\python -m pytest 03-topic-modeling/notebooks/test_helpers.py

Nao cobre o que exige rodar modelo (grid searches, train_*, sweeps): esses
dependem de corpus e de horas de fit. A lista do que falta, em ordem de risco,
esta no §M7 da revisao.
"""
import asyncio
import inspect
import json
import os
import sys

import httpx
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gensim.corpora import Dictionary  # noqa: E402

from _helpers import (  # noqa: E402
    _doc_topic_concentration,
    _h_sparsity,
    cache_protocolo_apto,
    compute_coherence_cv,
    compute_coherence_npmi,
    compute_ctfidf_scores,
    compute_embedding_coherence,
    compute_metrics_table,
    compute_nmf_reconstruction_error,
    diagnose_cv_window,
    compute_exclusivity_ctfidf,
    compute_frex_score,
    compute_jaccard,
    compute_topic_diversity,
    grid_search_alpha_eta,
    grid_search_k,
    grid_search_k_nmf,
    grid_search_k_stm,
    grid_search_nmf_hparams,
    make_run_output_dir,
    resolve_latest_dir,
    sweep_bertopic_grid,
    sweep_outlier_strategies,
    sweep_outlier_threshold,
    train_lda,
    train_nmf,
)


@pytest.fixture
def corpus():
    """Corpus minusculo com co-ocorrencia estavel o bastante para o c_v."""
    textos = [
        ["gato", "cachorro", "casa"],
        ["gato", "casa", "rua"],
        ["rua", "carro", "casa"],
        ["gato", "cachorro", "rua"],
    ] * 5
    return textos, Dictionary(textos)


# --------------------------------------------------------------------------
# compute_coherence_cv — regressao do C_v NaN (BERTopic-tweets top-10/15)
# --------------------------------------------------------------------------

def test_cv_topico_totalmente_fora_do_vocabulario_nao_derruba_o_conjunto(corpus):
    """O bug: 1 topico 100% OOV levantava ValueError e zerava TODOS os topicos.

    Era o caso do topico de emojis do BERTopic-tweets, que virava `NaN` para o
    modelo inteiro em top-10/15.
    """
    textos, d = corpus
    com_lixo = compute_coherence_cv(
        {0: ["gato", "cachorro", "casa"], 1: ["naoexiste", "tambemnao", "nem"]},
        textos, d)
    so_o_valido = compute_coherence_cv({0: ["gato", "cachorro", "casa"]}, textos, d)
    assert com_lixo == pytest.approx(so_o_valido)
    assert not np.isnan(com_lixo)


def test_cv_ignora_palavra_oov_solta_dentro_do_topico(corpus):
    textos, d = corpus
    assert compute_coherence_cv({0: ["gato", "cachorro", "OOV"]}, textos, d) == \
        pytest.approx(compute_coherence_cv({0: ["gato", "cachorro"]}, textos, d))


def test_cv_descarta_topico_que_sobra_com_menos_de_duas_palavras(corpus):
    """<2 palavras conhecidas nao sustenta coerencia (par minimo)."""
    textos, d = corpus
    assert compute_coherence_cv({0: ["gato", "OOV1", "OOV2"]}, textos, d) == 0.0


@pytest.mark.parametrize("topicos", [{}, {0: []}, {0: ["OOV1", "OOV2"]}])
def test_cv_sem_nada_valido_devolve_zero_e_nao_levanta(corpus, topicos):
    textos, d = corpus
    assert compute_coherence_cv(topicos, textos, d) == 0.0


def test_cv_tolera_none_e_string_vazia_nas_keywords(corpus):
    """MMR do BERTopic devolve '' em cluster degenerado."""
    textos, d = corpus
    assert compute_coherence_cv({0: ["gato", "", None, "cachorro"]}, textos, d) == \
        pytest.approx(compute_coherence_cv({0: ["gato", "cachorro"]}, textos, d))


# --------------------------------------------------------------------------
# compute_coherence_npmi — a coerencia do protocolo (P15)
# --------------------------------------------------------------------------

def test_npmi_herda_o_filtro_de_oov_do_cv(corpus):
    """O _compute_coherence e compartilhado: o fix de 2026-08-01 vale para as duas.

    O ARJ portou a funcao SEM o filtro; se alguem repetir isso aqui, este teste
    quebra antes de o C_v NaN voltar.
    """
    textos, d = corpus
    com_lixo = compute_coherence_npmi(
        {0: ["gato", "cachorro", "casa"], 1: ["naoexiste", "tambemnao", "nem"]},
        textos, d)
    so_o_valido = compute_coherence_npmi({0: ["gato", "cachorro", "casa"]}, textos, d)
    assert com_lixo == pytest.approx(so_o_valido)
    assert not np.isnan(com_lixo)


def test_npmi_esta_no_intervalo_canonico_e_difere_do_cv(corpus):
    """NPMI vive em [-1,1] e C_v em [0,1] — sao colunas distintas, nao substitutas."""
    textos, d = corpus
    topicos = {0: ["gato", "cachorro", "casa"], 1: ["rua", "carro", "casa"]}
    npmi = compute_coherence_npmi(topicos, textos, d)
    cv = compute_coherence_cv(topicos, textos, d)
    assert -1.0 <= npmi <= 1.0
    assert npmi != pytest.approx(cv)


def test_npmi_premia_par_que_sempre_coocorre(corpus):
    """Sanidade de direcao: co-ocorrencia forte deve pontuar acima de fraca."""
    textos = [["alfa", "beta"], ["alfa", "beta"], ["gama", "delta"], ["gama", "epsilon"]] * 5
    d = Dictionary(textos)
    juntos = compute_coherence_npmi({0: ["alfa", "beta"]}, textos, d)
    separados = compute_coherence_npmi({0: ["beta", "gama"]}, textos, d)
    assert juntos > separados


# --------------------------------------------------------------------------
# return_counts — instrumentacao do filtro de OOV (P0d)
# --------------------------------------------------------------------------

def test_return_counts_quantifica_topico_descartado_e_keyword_oov(corpus):
    """Sem isso o vies do filtro so e declaravel qualitativamente."""
    textos, d = corpus
    _, diag = compute_coherence_cv(
        {0: ["gato", "cachorro", "OOV"], 1: ["naoexiste", "tambemnao"]},
        textos, d, return_counts=True)
    assert diag["n_topicos"] == 2
    assert diag["n_topicos_avaliados"] == 1
    assert diag["n_topicos_descartados"] == 1
    assert diag["n_keywords"] == 5
    assert diag["n_keywords_oov"] == 3
    assert diag["taxa_oov"] == pytest.approx(3 / 5)
    assert diag["measure"] == "c_v"


def test_return_counts_nao_altera_o_score(corpus):
    """A instrumentacao e aditiva: o default segue devolvendo float puro."""
    textos, d = corpus
    topicos = {0: ["gato", "cachorro", "casa"]}
    puro = compute_coherence_cv(topicos, textos, d)
    com_diag, _ = compute_coherence_cv(topicos, textos, d, return_counts=True)
    assert puro == pytest.approx(com_diag)


def test_return_counts_com_tudo_oov_devolve_zero_e_diagnostico(corpus):
    textos, d = corpus
    score, diag = compute_coherence_npmi({0: ["OOV1", "OOV2"]}, textos, d, return_counts=True)
    assert score == 0.0
    assert diag["n_topicos_avaliados"] == 0
    assert diag["taxa_oov"] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# diagnose_cv_window — regime da janela deslizante (protocolo 2026-08-16 §3.0)
# --------------------------------------------------------------------------

def test_janela_corpus_longo_nao_degenera():
    longos = [["palavra"] * 200 for _ in range(10)]
    d = diagnose_cv_window(longos)
    assert d["frac_abaixo_janela"] == 0.0
    assert d["cv_degenerado"] is False


def test_janela_corpus_curto_degenera_como_nos_tweets():
    """Perfil dos tweets: mediana 9 lemas, 100% abaixo da janela de 110."""
    curtos = [["a", "b", "c"] * 3 for _ in range(10)]
    d = diagnose_cv_window(curtos)
    assert d["frac_abaixo_janela"] == 1.0
    assert d["cv_degenerado"] is True
    assert d["frac_abaixo_npmi"] == pytest.approx(1.0)


def test_janela_corpus_vazio_nao_divide_por_zero():
    d = diagnose_cv_window([])
    assert d["n_docs"] == 0 and d["cv_degenerado"] is False


# --------------------------------------------------------------------------
# compute_metrics_table — o helper unico (notebooks + script pos-hoc)
# --------------------------------------------------------------------------

def test_tabela_reproduz_as_funcoes_individuais(corpus):
    """O contrato central: a tabela nao pode ser uma segunda implementacao."""
    textos, d = corpus
    topicos = {0: ["gato", "cachorro", "casa"], 1: ["rua", "carro", "casa"]}
    t = compute_metrics_table(topicos, textos, d, top_n=3, verbose=False)
    tk = {tid: kws[:3] for tid, kws in topicos.items()}
    assert t["npmi"] == pytest.approx(compute_coherence_npmi(tk, textos, d))
    assert t["cv_bruto"] == pytest.approx(compute_coherence_cv(tk, textos, d))
    assert t["topic_diversity"] == pytest.approx(
        compute_topic_diversity(topicos, top_k=3))


def test_tabela_suprime_cv_quando_a_janela_degenera(corpus):
    """Nos tweets o C_v nao pode sair publicavel por descuido."""
    textos, d = corpus  # documentos de 3 tokens -> 100% degenerado
    t = compute_metrics_table({0: ["gato", "cachorro", "casa"]}, textos, d,
                              top_n=3, verbose=False)
    assert np.isnan(t["cv"])
    assert not np.isnan(t["cv_bruto"])
    assert t["cv_window_degenerado"] is True


def test_tabela_mantem_cv_quando_a_janela_e_valida():
    longos = [["gato", "cachorro"] * 60 + ["casa"] * 60 for _ in range(8)]
    d = Dictionary(longos)
    t = compute_metrics_table({0: ["gato", "cachorro", "casa"]}, longos, d,
                              top_n=3, verbose=False)
    assert not np.isnan(t["cv"])
    assert t["cv"] == pytest.approx(t["cv_bruto"])


def test_tabela_so_traz_exclusividade_e_frex_com_os_insumos(corpus):
    textos, d = corpus
    topicos = {0: ["gato", "cachorro"]}
    sem = compute_metrics_table(topicos, textos, d, top_n=2, verbose=False)
    assert "exclusivity" not in sem and "frex" not in sem

    scores = {0: {"gato": 0.9, "cachorro": 0.1}}
    com = compute_metrics_table(topicos, textos, d, top_n=2,
                                topic_word_scores=scores, verbose=False)
    assert "exclusivity" in com and "frex" not in com


def test_topn_do_gensim_acompanha_a_lista_e_nao_trava_em_20():
    """P0c/C10: CoherenceModel tem topn=20 por default e NAO segue o tamanho da lista.

    Sem passar topn explicitamente, pedir top-30 devolve o valor de top-20 — a
    tabela de robustez por top-N repetia o mesmo numero de top-25 em diante, o
    que a tornava exatamente o oposto do que ela alega demonstrar.
    """
    vocab = [f"w{i}" for i in range(30)]
    textos = [vocab[:] for _ in range(30)] + [vocab[:15] for _ in range(10)]
    d = Dictionary(textos)
    cv30 = compute_coherence_cv({0: vocab[:30]}, textos, d)
    cv20 = compute_coherence_cv({0: vocab[:20]}, textos, d)
    assert cv30 != pytest.approx(cv20), (
        "C_v em top-30 igual ao de top-20: o topn do CoherenceModel voltou a travar"
    )


def test_topn_efetivo_e_reportado_no_diagnostico(corpus):
    textos, d = corpus
    _, diag = compute_coherence_cv({0: ["gato", "cachorro", "casa"]}, textos, d,
                                   return_counts=True)
    assert diag["topn_efetivo"] == 3


# --------------------------------------------------------------------------
# _selecao.py — camada 1, Pareto e desempate (o nucleo do protocolo)
# --------------------------------------------------------------------------

def test_pareto_exclui_linha_com_nan_em_eixo_ativo():
    """NaN >= x e NaN > x sao ambos False: a linha nunca era dominada e sobrevivia.

    Uma configuracao pior em tudo, mas com NPMI nao medido, entrava na fronteira
    e podia ganhar o desempate.
    """
    from _selecao import _pareto
    df = pd.DataFrame({
        "NPMI": [0.5, 0.4, float("nan")],
        "Diversity": [0.6, 0.7, 0.1],
        "cobertura": [0.8, 0.9, 0.2],
    })
    fr = _pareto(df, ["NPMI", "Diversity", "cobertura"])
    assert 2 not in fr.index, "linha com NaN sobreviveu a fronteira"
    assert set(fr.index) == {0, 1}


def test_pareto_marca_dominado_corretamente():
    from _selecao import _pareto
    df = pd.DataFrame({"a": [1.0, 2.0, 0.5], "b": [1.0, 2.0, 0.4]})
    fr = _pareto(df, ["a", "b"])
    assert set(fr.index) == {1}  # linha 1 domina as outras duas


def test_pareto_sem_eixo_medido_devolve_vazio():
    from _selecao import _pareto
    df = pd.DataFrame({"a": [float("nan")] * 3, "b": [1.0, 2.0, 3.0]})
    assert _pareto(df, ["a"]).empty


@pytest.mark.skip(reason="BERTopic grade gate removed in STM-only selection")
def test_grade_pre_protocolo_e_recusada():
    pass


@pytest.mark.skip(reason="BERTopic grade gate removed in STM-only selection")
def test_grade_apta_nao_tem_impedimento():
    pass


def test_faixa_de_k_declarada_por_corpus():
    """A3 e substantiva e vem de fonte unica — os notebooks importam daqui."""
    from _selecao import FAIXA_K
    for corpus, (lo, hi) in FAIXA_K.items():
        assert lo < hi, f"{corpus}: faixa invertida"
        assert lo >= 2, f"{corpus}: limite inferior sem sentido"


def test_tabela_reporta_o_diagnostico_de_oov(corpus):
    textos, d = corpus
    t = compute_metrics_table(
        {0: ["gato", "cachorro", "OOV"], 1: ["naoexiste", "tambemnao"]},
        textos, d, top_n=3, verbose=False)
    assert t["n_topicos_descartados"] == 1
    assert t["oov_taxa"] == pytest.approx(3 / 5)


# --------------------------------------------------------------------------
# resolve_latest_dir — latest-wins
# --------------------------------------------------------------------------

def _cria(base, nome, arquivo="corpus_limpo.csv"):
    d = base / nome
    d.mkdir(parents=True)
    (d / arquivo).write_text("x", encoding="utf-8")
    return d


def test_resolve_escolhe_o_carimbo_mais_novo_entre_dias(tmp_path):
    for nome in ("c_20260102_090000", "c_20260630_235959",
                 "c_20260701_000001", "c_20261231_120000"):
        _cria(tmp_path, nome)
    escolhido = resolve_latest_dir(tmp_path, contains="corpus_limpo.csv", verbose=False)
    assert escolhido.name == "c_20261231_120000"


def test_resolve_ignora_diretorio_nao_carimbado_quando_ha_carimbado(tmp_path):
    """Ordenacao lexicografica deixava 'zzz_backup' vencer o carimbo ('z' > '2')."""
    _cria(tmp_path, "c_20260801_120000")
    _cria(tmp_path, "zzz_backup")
    escolhido = resolve_latest_dir(tmp_path, contains="corpus_limpo.csv", verbose=False)
    assert escolhido.name == "c_20260801_120000"


def test_resolve_aceita_nao_carimbado_se_nao_houver_carimbado(tmp_path):
    """Layout legado continua funcionando — degrada com aviso, nao com erro."""
    _cria(tmp_path, "versao_antiga")
    escolhido = resolve_latest_dir(tmp_path, contains="corpus_limpo.csv", verbose=False)
    assert escolhido.name == "versao_antiga"


def test_resolve_pula_subdir_mais_novo_que_nao_tem_o_arquivo(tmp_path):
    (tmp_path / "c_20260801_120000").mkdir()          # mais novo, mas vazio
    _cria(tmp_path, "c_20260701_000000")
    escolhido = resolve_latest_dir(tmp_path, contains="corpus_limpo.csv", verbose=False)
    assert escolhido.name == "c_20260701_000000"


def test_resolve_cai_no_layout_plano_legado(tmp_path):
    (tmp_path / "corpus_limpo.csv").write_text("x", encoding="utf-8")
    assert resolve_latest_dir(tmp_path, contains="corpus_limpo.csv",
                              verbose=False) == tmp_path


def test_resolve_cai_no_fallback_quando_a_base_esta_vazia(tmp_path):
    base, fb = tmp_path / "base", tmp_path / "input"
    base.mkdir(); fb.mkdir()
    (fb / "corpus_limpo.csv").write_text("x", encoding="utf-8")
    assert resolve_latest_dir(base, contains="corpus_limpo.csv",
                              fallback_dir=fb, verbose=False) == fb


def test_resolve_sem_nada_levanta_com_mensagem_acionavel(tmp_path):
    with pytest.raises(FileNotFoundError, match="corpus_limpo.csv"):
        resolve_latest_dir(tmp_path, contains="corpus_limpo.csv", verbose=False)


def test_carimbo_de_make_run_output_dir_e_reconhecido_por_resolve(tmp_path):
    """Contrato entre as duas funcoes: quem grava e quem le usam o mesmo formato."""
    run = make_run_output_dir(tmp_path, "folha")
    (run / "corpus_limpo.csv").write_text("x", encoding="utf-8")
    assert resolve_latest_dir(tmp_path, contains="corpus_limpo.csv",
                              verbose=False) == run


# --------------------------------------------------------------------------
# Metricas auditadas na revisao
# --------------------------------------------------------------------------

def test_topic_diversity_extremos():
    """TD = palavras unicas / (top_k * topicos)."""
    assert compute_topic_diversity({0: ["a", "b"], 1: ["c", "d"]}, top_k=2) == 1.0
    assert compute_topic_diversity({0: ["a", "b"], 1: ["a", "b"]}, top_k=2) == 0.5


def test_topic_diversity_ignora_outlier_e_topico_degenerado():
    td = compute_topic_diversity({-1: ["x", "y"], 0: ["a", "b"], 1: [""]}, top_k=2)
    assert td == 1.0   # so o topico 0 conta


def test_topic_diversity_com_top_k_maior_que_as_keywords_subestima():
    """Denominador usa top_k cheio: e por isso que TD despenca em top-50/100
    no sweep de robustez — artefato de definicao, nao perda de qualidade."""
    assert compute_topic_diversity({0: ["a", "b"]}, top_k=10) == pytest.approx(0.2)


def test_exclusividade_ctfidf_termo_exclusivo_vs_compartilhado():
    scores = {0: {"so_do_zero": 1.0, "dividido": 0.5},
              1: {"outro": 1.0, "dividido": 0.5}}
    _, por_topico = compute_exclusivity_ctfidf(
        {0: ["so_do_zero", "dividido"], 1: ["outro", "dividido"]}, scores, top_n=2)
    assert por_topico[0] == pytest.approx(0.75)   # (1.0 + 0.5) / 2


def test_ctfidf_scores_separa_termo_caracteristico_da_classe():
    scores = compute_ctfidf_scores({0: ["gato gato rato"], 1: ["carro carro rua"]})
    assert scores[0]["gato"] > scores[0].get("rato", 0)
    assert "carro" not in scores[0]


def test_frex_satura_nas_proprias_top_n_e_discrimina_palavra_aleatoria():
    """Achado M1: a saturacao (~0.98) e estrutural, nao bug — as top-N estao no
    topo do ranking por construcao. O controle com palavra fraca separa."""
    rng = np.random.RandomState(42)
    V = 500
    matriz = rng.rand(3, V) * 0.01
    vocab = {f"w{i}": i for i in range(V)}
    for t in range(3):
        matriz[t, t * 10:(t + 1) * 10] = 1.0        # 10 termos fortes por topico
    topicos = {t: [f"w{i}" for i in range(t * 10, (t + 1) * 10)] for t in range(3)}
    idx = {t: t for t in range(3)}
    forte, _ = compute_frex_score(topicos, matriz, vocab, idx, top_n=10)
    fraco, _ = compute_frex_score({t: ["w499"] for t in range(3)},
                                  matriz, vocab, idx, top_n=1)
    assert forte > 0.95
    assert fraco < forte


def test_frex_ignora_outlier_e_topico_fora_do_indice():
    matriz = np.array([[1.0, 0.0], [0.0, 1.0]])
    vocab, idx = {"a": 0, "b": 1}, {0: 0, 1: 1}
    _, por_topico = compute_frex_score(
        {-1: ["a"], 0: ["a"], 1: ["b"], 99: ["a"]}, matriz, vocab, idx, top_n=1)
    assert set(por_topico) == {0, 1}


def test_jaccard():
    assert compute_jaccard({1, 2}, {1, 2}) == 1.0
    assert compute_jaccard({1, 2}, {3, 4}) == 0.0
    assert compute_jaccard(set(), set()) == 0.0
    assert compute_jaccard({1, 2, 3}, {2, 3, 4}) == pytest.approx(0.5)


# --------------------------------------------------------------------------
# Reprodutibilidade do LDA — workers do LdaMulticore (auditoria de 2026-08-15)
#
# Medido: o LdaMulticore e deterministico para uma contagem FIXA de workers,
# mas o resultado DEPENDE dessa contagem (folha K=20: 0,5369 com 15 workers vs
# 0,5635 com 1; nos tweets a curva de C_v x K inverte). `batch=True` nao
# desacopla. Mecanismo: updateafter = chunksize * workers, que decide o regime
# online-vs-batch da inferencia.
#
# NAO ha teste travando um valor de `workers` porque a decisao de qual usar
# ainda esta em aberto — pinar 1 foi tentado e rejeitado (troca o algoritmo).
# O que da para travar hoje e o determinismo sob configuracao fixa, que e o
# que sustenta qualquer comparacao entre runs da mesma maquina.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("funcao", [grid_search_k, grid_search_alpha_eta, train_lda])
def test_lda_expoe_workers_como_parametro(funcao):
    """`workers` precisa ser controlavel pelo chamador — e o knob da reprodutibilidade."""
    assert "workers" in inspect.signature(funcao).parameters


def test_grid_lda_reproduz_entre_chamadas_com_workers_fixo(corpus):
    """Sob workers explicito, duas chamadas devolvem o mesmo numero."""
    textos, dic = corpus
    bow = [dic.doc2bow(t) for t in textos]
    cv1, _ = grid_search_k(bow, dic, textos, [2], seed=42, passes=2, workers=1)
    cv2, _ = grid_search_k(bow, dic, textos, [2], seed=42, passes=2, workers=1)
    assert cv1 == cv2


# --------------------------------------------------------------------------
# Diagnosticos do grid do NMF — instrumentacao do colapso K=8 (backlog P3)
#
# O NMF-tweets colapsa para K=8 e ate 2026-08-14 isso era so reportado, sem
# medida que explicasse. Estas duas funcoes sao os dois sinais do colapso:
# H densa (topicos que nao se especializam) e docs concentrados num topico so.
# --------------------------------------------------------------------------

def test_h_sparsity_matriz_toda_zero_e_toda_densa():
    assert _h_sparsity(np.zeros((3, 4))) == 1.0
    assert _h_sparsity(np.ones((3, 4))) == 0.0


def test_h_sparsity_conta_fracao_de_entradas_quase_zero():
    m = np.array([[1.0, 0.0, 0.0, 0.0],
                  [0.0, 1.0, 1.0, 1.0]])
    assert _h_sparsity(m) == pytest.approx(0.5)


def test_h_sparsity_usa_limiar_e_nao_igualdade_exata():
    """Fatoracao numerica quase nunca zera de verdade — 1e-12 e zero na pratica."""
    m = np.array([[1e-12, 1e-12], [0.5, 0.5]])
    assert _h_sparsity(m) == pytest.approx(0.5)
    # com limiar frouxo o bastante, o 0.5 tambem entra
    assert _h_sparsity(m, threshold=1.0) == 1.0


def test_concentracao_distribuicao_uniforme_entre_topicos():
    top_share, n_com_docs = _doc_topic_concentration([0, 1, 2, 3], k=4)
    assert top_share == pytest.approx(0.25)
    assert n_com_docs == 4


def test_concentracao_detecta_topico_dominante():
    """O sinal do colapso: quase tudo cai num topico so."""
    top_share, n_com_docs = _doc_topic_concentration([0] * 9 + [1], k=4)
    assert top_share == pytest.approx(0.9)
    assert n_com_docs == 2  # os outros 2 topicos ficaram vazios


def test_concentracao_conta_doc_sem_topico_no_denominador_mas_nao_como_topico():
    """-1 = nenhum topico acima do piso. Entra na base, nao vira topico."""
    top_share, n_com_docs = _doc_topic_concentration([0, 0, -1, -1], k=3)
    assert top_share == pytest.approx(0.5)
    assert n_com_docs == 1


def test_concentracao_sem_documentos_nao_divide_por_zero():
    assert _doc_topic_concentration([], k=5) == (0.0, 0)


def test_grid_nmf_sem_diagnosticos_mantem_o_contrato_antigo(corpus):
    """Default preservado: os 3 notebooks NMF dependem de receber um dict."""
    textos, dic = corpus
    bow = [dic.doc2bow(t) for t in textos]
    scores = grid_search_k_nmf(bow, dic, textos, [2], seed=42, passes=2)
    assert isinstance(scores, dict)
    assert set(scores) == {2}
    assert isinstance(scores[2], float)


def test_grid_nmf_com_diagnosticos_devolve_par_e_uma_linha_por_k(corpus):
    """Espelha o contrato de grid_search_k_stm: (scores, diag_df)."""
    textos, dic = corpus
    bow = [dic.doc2bow(t) for t in textos]
    scores, diag = grid_search_k_nmf(
        bow, dic, textos, [2, 3], seed=42, passes=2, return_diagnostics=True)

    assert set(scores) == {2, 3}
    assert list(diag["k"]) == [2, 3]
    for col in ("k", "cv", "h_sparsity", "top_topic_doc_share",
                "n_topics_with_docs", "elapsed_sec"):
        assert col in diag.columns

    # os C_v das duas vias sao o mesmo numero, nao dois calculos diferentes
    assert list(diag["cv"]) == [scores[2], scores[3]]

    assert ((diag["h_sparsity"] >= 0) & (diag["h_sparsity"] <= 1)).all()
    assert ((diag["top_topic_doc_share"] >= 0) & (diag["top_topic_doc_share"] <= 1)).all()
    assert (diag["n_topics_with_docs"] <= diag["k"]).all()


# --------------------------------------------------------------------------
# grid_search_k_nmf(return_npmi=...) — NPMI vira eixo de ordenacao do NMF
# (protocolo 2026-08-16, secao 4.2.2, mesma regra do LDA)
# --------------------------------------------------------------------------

def test_grid_nmf_com_npmi_devolve_par_scores_npmi(corpus):
    textos, dic = corpus
    bow = [dic.doc2bow(t) for t in textos]
    scores, npmi_scores = grid_search_k_nmf(
        bow, dic, textos, [2, 3], seed=42, passes=2, return_npmi=True)
    assert set(scores) == {2, 3}
    assert set(npmi_scores) == {2, 3}
    for v in npmi_scores.values():
        assert -1.0 <= v <= 1.0


def test_grid_nmf_com_npmi_e_diagnosticos_devolve_tripla_com_coluna_npmi(corpus):
    textos, dic = corpus
    bow = [dic.doc2bow(t) for t in textos]
    scores, npmi_scores, diag = grid_search_k_nmf(
        bow, dic, textos, [2, 3], seed=42, passes=2,
        return_npmi=True, return_diagnostics=True)
    assert set(scores) == {2, 3}
    assert set(npmi_scores) == {2, 3}
    assert "npmi" in diag.columns
    assert list(diag["npmi"]) == [npmi_scores[2], npmi_scores[3]]


def test_grid_nmf_sem_return_npmi_nao_tem_custo_de_npmi_no_contrato(corpus):
    """Default continua devolvendo so o dict — os 3 notebooks antigos nao quebram."""
    textos, dic = corpus
    bow = [dic.doc2bow(t) for t in textos]
    out = grid_search_k_nmf(bow, dic, textos, [2], seed=42, passes=2, return_diagnostics=True)
    scores, diag = out
    assert "npmi" not in diag.columns


# --------------------------------------------------------------------------
# compute_nmf_reconstruction_error — C4, analogo da perplexidade do LDA
# --------------------------------------------------------------------------

def test_reconstrucao_nmf_e_no_intervalo_e_finita(corpus):
    textos, dic = corpus
    bow = [dic.doc2bow(t) for t in textos]
    model = train_nmf(bow, dic, k=2, seed=42, passes=5)
    err = compute_nmf_reconstruction_error(model, bow)
    assert np.isfinite(err)
    assert err >= 0.0


def test_reconstrucao_nmf_pior_com_k_menor_em_corpus_com_estrutura(corpus):
    """K=1 forca todo documento numa mistura so — deve reconstruir pior que K=3
    num corpus com blocos tematicos distintos (o fixture `corpus` tem 3)."""
    textos, dic = corpus
    bow = [dic.doc2bow(t) for t in textos]
    modelo_k1 = train_nmf(bow, dic, k=1, seed=42, passes=10)
    modelo_k3 = train_nmf(bow, dic, k=3, seed=42, passes=10)
    err_k1 = compute_nmf_reconstruction_error(modelo_k1, bow)
    err_k3 = compute_nmf_reconstruction_error(modelo_k3, bow)
    assert err_k3 <= err_k1 + 1e-6


class _NmfPerfeito:
    """Stub minimo: H e W tais que W @ H reproduz exatamente as proporcoes de V
    normalizado por linha — erro de reconstrucao deve ser ~0."""

    def __init__(self, H, doc_topics):
        self._H = H
        self._doc_topics = doc_topics

    def get_topics(self):
        return self._H

    def get_document_topics(self, bow, minimum_probability=0.0):
        idx = self._bow_to_index[tuple(bow)]
        return list(enumerate(self._doc_topics[idx]))


def test_reconstrucao_nmf_stub_com_ajuste_perfeito_e_quase_zero():
    # V (2 docs x 3 termos), cada doc e um multiplo de uma unica linha de H
    H = np.array([[0.5, 0.5, 0.0], [0.0, 0.2, 0.8]])
    corpus_bow = [[(0, 1), (1, 1)], [(1, 1), (2, 4)]]  # doc0 ~ H[0]; doc1 ~ H[1]
    modelo = _NmfPerfeito(H, doc_topics=[[1.0, 0.0], [0.0, 1.0]])
    modelo._bow_to_index = {tuple(b): i for i, b in enumerate(corpus_bow)}
    err = compute_nmf_reconstruction_error(modelo, corpus_bow)
    assert err == pytest.approx(0.0, abs=1e-9)


def test_nmf_get_topics_normaliza_linhas():
    """Protocolo secao 1.6 (Exclusividade e FREX: nativas ou analogas): a pendencia
    esta fechada, mas o resultado e CONDICIONAL, nao uma propriedade incondicional
    do metodo. `Nmf.get_topics(self, normalize=None)` cai em `self.normalize`, e
    `Nmf.__init__` tem `normalize=True` como default do gensim — e o unico valor
    que este projeto usa (`train_nmf` o expoe como parametro sobrescrevivel, mas
    nenhum notebook de producao/template sobrescreve; verificado 2026-08-23 via
    grep por `train_nmf(` nos 3 notebooks NMF).

    Com `normalize=True` (default), as linhas de `get_topics()` somam 1.0 —
    distribuicao de probabilidade sobre o vocabulario, mesmo formato da matriz
    beta do LDA — logo Exclusividade e FREX no braco NMF sao NATIVAS (mesmo braco
    do LDA), nao analogas declaradas como no BERTopic, ENQUANTO o projeto continuar
    usando o default. Com `normalize=False` as linhas NAO somam 1 (verificado
    2026-08-23: [1.25427158, 1.25885925] no mesmo corpus de brinquedo) — prova de
    que a normalizacao e condicional a esse argumento, nao inerente ao metodo.

    Trava as duas observacoes de 2026-08-23 — se uma versao futura do gensim
    mudar o DEFAULT de `normalize` para `False`, este teste quebra (primeira
    metade) e a secao 1.6 do protocolo precisa ser reescrita (NMF passaria para o
    braco de analogas declaradas, como o BERTopic) antes de qualquer numero de
    Exclusividade/FREX do NMF sair no artigo. Se a segunda metade quebrar (i.e.
    `normalize=False` passar a normalizar tambem), a condicionalidade deixou de
    existir e a ressalva sobre `train_nmf` expor o parametro fica sem objeto —
    tambem exige revisão da §1.6.
    """
    from gensim.models import Nmf

    textos = [["a", "b", "c"], ["a", "b", "d"], ["c", "d", "e"], ["e", "f", "a"]] * 8
    dic = Dictionary(textos)
    bow = [dic.doc2bow(t) for t in textos]

    modelo_default = Nmf(bow, num_topics=2, id2word=dic, random_state=42)
    somas_default = modelo_default.get_topics().sum(axis=1)
    assert np.allclose(somas_default, 1.0, atol=1e-6), somas_default

    modelo_sem_norm = Nmf(bow, num_topics=2, id2word=dic, random_state=42, normalize=False)
    somas_sem_norm = modelo_sem_norm.get_topics().sum(axis=1)
    assert not np.allclose(somas_sem_norm, 1.0, atol=1e-6), somas_sem_norm


# --------------------------------------------------------------------------
# compute_embedding_coherence — renomeada de compute_semantic_coherence (C5):
# nome antigo colidia com a "semantic coherence" nativa do STM (Mimno et al.),
# quantidade diferente.
# --------------------------------------------------------------------------

def test_embedding_coherence_nome_novo_existe_e_antigo_nao():
    import _helpers
    assert hasattr(_helpers, "compute_embedding_coherence")
    assert not hasattr(_helpers, "compute_semantic_coherence")


def test_embedding_coherence_vetores_identicos_dao_similaridade_maxima():
    def _embed_constante(kws):
        return np.array([[1.0, 0.0]] * len(kws))
    score = compute_embedding_coherence({0: ["a", "b", "c"]}, _embed_constante)
    assert score == pytest.approx(1.0)


def test_embedding_coherence_topico_com_menos_de_duas_palavras_e_ignorado():
    def _embed(kws):
        return np.zeros((len(kws), 2))
    assert compute_embedding_coherence({0: ["so_uma"]}, _embed) == 0.0


# --------------------------------------------------------------------------
# _get_ollama_embeddings_batch — regressão de 2026-08-21: o BERTopic-folha
# travou na célula de Coerência Semântica porque cada keyword virava UMA
# requisição sequencial a /api/embeddings (~480 requisições p/ 24 tópicos,
# 3-16s cada = a sessão parecia travada). Ollama expõe /api/embed, que aceita
# uma lista de textos numa única requisição; o fix passa a usá-lo.
# --------------------------------------------------------------------------

def _client_com_transporte_falso(monkeypatch, handler):
    import _helpers
    transport = httpx.MockTransport(handler)
    real_async_client = httpx.AsyncClient
    monkeypatch.setattr(
        _helpers.httpx, "AsyncClient",
        lambda *a, **kw: real_async_client(transport=transport),
    )


def test_batch_usa_uma_unica_requisicao_para_varios_textos(monkeypatch):
    import _helpers
    calls = []

    def handler(request):
        calls.append(request.url.path)
        if request.url.path == "/api/embed":
            n = len(json.loads(request.content)["input"])
            return httpx.Response(200, json={"embeddings": [[0.1, 0.2]] * n})
        return httpx.Response(200, json={"embedding": [0.1, 0.2]})

    _client_com_transporte_falso(monkeypatch, handler)
    result = asyncio.run(_helpers._get_ollama_embeddings_batch(
        ["economia", "politica", "saude"], "qwen3-embedding:0.6b", dimension=None))

    assert calls == ["/api/embed"]
    assert len(result) == 3


def test_batch_cai_para_por_texto_se_api_embed_nao_existir(monkeypatch):
    """Servidor Ollama anterior ao /api/embed (404): cai no caminho antigo."""
    import _helpers
    calls = []

    def handler(request):
        calls.append(request.url.path)
        if request.url.path == "/api/embed":
            return httpx.Response(404)
        return httpx.Response(200, json={"embedding": [0.1, 0.2]})

    _client_com_transporte_falso(monkeypatch, handler)
    result = asyncio.run(_helpers._get_ollama_embeddings_batch(
        ["economia", "politica"], "qwen3-embedding:0.6b", dimension=None))

    assert calls[0] == "/api/embed"
    assert calls.count("/api/embeddings") == 2
    assert len(result) == 2


def test_batch_aplica_truncamento_matryoshka_na_resposta_em_lote(monkeypatch):
    import _helpers

    def handler(request):
        return httpx.Response(200, json={"embeddings": [[0.6, 0.8, 0.0, 0.0]]})

    _client_com_transporte_falso(monkeypatch, handler)
    result = asyncio.run(_helpers._get_ollama_embeddings_batch(
        ["x"], "modelo", dimension=2))

    arr = np.array(result[0])
    assert len(arr) == 2
    assert np.linalg.norm(arr) == pytest.approx(1.0)


def test_batch_retenta_em_5xx_e_conclui_apos_recuperar(monkeypatch):
    import _helpers
    attempts = {"n": 0}

    def handler(request):
        attempts["n"] += 1
        if attempts["n"] == 1:
            return httpx.Response(500)
        n = len(json.loads(request.content)["input"])
        return httpx.Response(200, json={"embeddings": [[0.1, 0.2]] * n})

    _client_com_transporte_falso(monkeypatch, handler)
    result = asyncio.run(_helpers._get_ollama_embeddings_batch(
        ["economia", "politica"], "qwen3-embedding:0.6b", dimension=None))

    assert attempts["n"] == 2
    assert len(result) == 2


# --------------------------------------------------------------------------
# sweep_bertopic_grid — regressão de 2026-08-22: o sweep multi-seed (108
# fits na Folha, 324 nos tweets) não imprime NADA durante a execução, só
# devolve no final. Rodando ~5h sem log foi impossível saber se travou ou
# se estava progredindo. Corpo do loop faz fit real (caro demais p/ testar
# aqui); testa só o contrato de progresso, com o fit e as métricas fakeados.
# --------------------------------------------------------------------------

def test_sweep_bertopic_grid_imprime_progresso_por_combo(monkeypatch, capsys):
    import _helpers

    class _FakeModel:
        def fit_transform(self, docs, embeddings=None):
            return None

    def _fake_build_model(seed, nn, mcs, ms, csm):
        return _FakeModel()

    monkeypatch.setattr(_helpers, "_bertopic_postprocess", lambda *a, **kw: (0.1, 0.1, 5))
    monkeypatch.setattr(
        _helpers, "_bertopic_sweep_metrics",
        lambda *a, **kw: (
            {"K": 3, "npmi": 0.1, "cv": 0.1, "div": 0.1, "excl": 0.1, "frex": 0.1},
            {0: ["a", "b"], 1: ["c", "d"], 2: ["e", "f"]},
        ),
    )

    _helpers.sweep_bertopic_grid(
        _fake_build_model, docs=["d1", "d2"], embeddings=None, tokenized=[["a"]],
        dictionary=None, seeds=[1, 2], n_neighbors_grid=[10], min_cluster_size_grid=[5],
    )

    out = capsys.readouterr().out
    assert "[1/2]" in out
    assert "[2/2]" in out


# --------------------------------------------------------------------------
# cache_protocolo_apto — C9, generaliza o check ad hoc de frescor de cache que
# os notebooks de grid (LDA/NMF/STM) reinventavam cada um do seu jeito.
# --------------------------------------------------------------------------

def test_cache_apto_series_com_todas_as_colunas_devolve_vazio():
    linha = pd.Series({"k_npmi_scores": "{}", "grid_passes": 20})
    assert cache_protocolo_apto(linha, ["k_npmi_scores"]) == []


def test_cache_apto_series_com_coluna_ausente_reporta():
    linha = pd.Series({"grid_passes": 20})
    assert cache_protocolo_apto(linha, ["k_npmi_scores"]) == ["k_npmi_scores"]


def test_cache_apto_series_com_coluna_nan_reporta():
    linha = pd.Series({"k_npmi_scores": float("nan")})
    assert cache_protocolo_apto(linha, ["k_npmi_scores"]) == ["k_npmi_scores"]


def test_cache_apto_dataframe_usa_a_ultima_linha():
    df = pd.DataFrame([
        {"k_npmi_scores": float("nan")},
        {"k_npmi_scores": "{}"},
    ])
    assert cache_protocolo_apto(df, ["k_npmi_scores"]) == []


def test_cache_apto_relata_todas_as_colunas_faltando_de_uma_vez():
    linha = pd.Series({"grid_passes": 20})
    faltando = cache_protocolo_apto(linha, ["k_npmi_scores", "reconstruction_error"])
    assert set(faltando) == {"k_npmi_scores", "reconstruction_error"}


# --------------------------------------------------------------------------
# _selecao.py — braco STM (fronteira semantic_coherence x exclusivity_stm,
# desempate = menor K). Mesmo padrao dos testes de _pareto/_grade_apta acima:
# import local, sem depender de CSV em disco.
# --------------------------------------------------------------------------

def test_desempate_stm_e_menor_k_ascendente():
    from _selecao import DESEMPATE_STM, DESEMPATE_STM_ASCENDING
    assert DESEMPATE_STM == "K"
    assert DESEMPATE_STM_ASCENDING is True


def test_fronteira_stm_com_desempate_por_menor_k():
    """Replica o que `selecionar_stm` faz: Pareto sobre semantic_coherence x
    exclusivity_stm, depois ordena pelo desempate declarado e pega o primeiro."""
    from _selecao import _pareto, DESEMPATE_STM, DESEMPATE_STM_ASCENDING
    df = pd.DataFrame({
        "K": [15, 20, 25],
        "semantic_coherence": [-60.0, -65.0, -70.0],  # K=15 e o melhor neste eixo
        "exclusivity_stm": [9.6, 9.7, 9.8],            # K=25 e o melhor neste eixo
    })
    fr = _pareto(df, ["semantic_coherence", "exclusivity_stm"])
    assert set(fr["K"]) == {15, 20, 25}  # nenhum domina os outros nos dois eixos
    escolhido = fr.sort_values(DESEMPATE_STM, ascending=DESEMPATE_STM_ASCENDING).iloc[0]
    assert int(escolhido["K"]) == 15


def test_fronteira_stm_k_dominado_sai_mesmo_fora_do_desempate():
    """K=20 pior nos dois eixos que K=15 — cai da fronteira antes do desempate entrar."""
    from _selecao import _pareto
    df = pd.DataFrame({
        "K": [15, 20],
        "semantic_coherence": [-60.0, -65.0],
        "exclusivity_stm": [9.8, 9.7],
    })
    fr = _pareto(df, ["semantic_coherence", "exclusivity_stm"])
    assert set(fr["K"]) == {15}


# --------------------------------------------------------------------------
# Checkpoint incremental — regressao de 2026-08-22: o sweep_bertopic_grid da
# Folha rodou 6h+ e foi morto sem deixar NENHUM resultado parcial (sem print,
# sem CSV incremental — so devolvia tudo no final). checkpoint_path=None
# preserva o comportamento antigo (sem custo); passado, grava CADA linha
# assim que ela e computada, para sobreviver a kill/crash em qualquer grid
# ou sweep longo (LDA, NMF, BERTopic — STM roda o grid inteiro num unico
# subprocesso R, fora do escopo deste checkpoint por linha).
# --------------------------------------------------------------------------

def test_checkpoint_grava_linha_e_so_escreve_o_header_uma_vez(tmp_path):
    import _helpers
    p = tmp_path / "chk.csv"
    _helpers._checkpoint_append_row(p, {"k": 2, "cv": 0.5})
    _helpers._checkpoint_append_row(p, {"k": 3, "cv": 0.6})
    df = pd.read_csv(p)
    assert list(df["k"]) == [2, 3]
    assert list(df["cv"]) == [0.5, 0.6]


def test_checkpoint_com_path_none_nao_grava_nada(tmp_path):
    import _helpers
    _helpers._checkpoint_append_row(None, {"k": 2, "cv": 0.5})
    assert list(tmp_path.iterdir()) == []


def test_grid_search_k_grava_checkpoint_por_k(corpus, tmp_path):
    textos, dic = corpus
    bow = [dic.doc2bow(t) for t in textos]
    p = tmp_path / "lda_k_checkpoint.csv"
    grid_search_k(bow, dic, textos, [2, 3], seed=42, passes=2, workers=1, checkpoint_path=p)
    df = pd.read_csv(p)
    assert list(df["k"]) == [2, 3]
    assert "cv" in df.columns and "perplexity" in df.columns


def test_grid_search_k_passa_topn_ao_gensim(monkeypatch):
    """A curva de K do LDA nao pode cair no topn=20 default do gensim.

    Protocolo secao 3.4: CoherenceModel trunca em silencio acima de 20, e a
    curva de K e o eixo que ORDENA os bracos LDA e NMF (secoes 1.5 e 1.6).

    Monkeypatch em ``_helpers.CoherenceModel``/``_helpers.LdaMulticore``, NAO
    em ``gensim.models.CoherenceModel``: ``_helpers.py`` importa esses nomes
    no nivel de modulo (``from gensim.models import CoherenceModel,
    LdaMulticore, Nmf``) e ``grid_search_k`` usa essas bindings diretamente —
    nao ha reimport local nesta funcao (so ``_compute_coherence`` reimporta).
    Patchear ``gensim.models.CoherenceModel`` nao intercepta nada aqui: o
    nome ja estava vinculado na importacao, e o teste treinaria um LDA de
    verdade e passaria pelo motivo errado.
    """
    import _helpers
    capturado = []

    class FakeCM:
        def __init__(self, **kwargs):
            capturado.append(kwargs)

        def get_coherence(self):
            return 0.5

    class FakeLda:
        def __init__(self, *a, **kw):
            pass

        def log_perplexity(self, corpus):
            return -7.0

    monkeypatch.setattr(_helpers, "CoherenceModel", FakeCM)
    monkeypatch.setattr(_helpers, "LdaMulticore", FakeLda)

    textos = [["a", "b", "c"], ["a", "b", "d"]] * 5
    dic = Dictionary(textos)
    bow = [dic.doc2bow(t) for t in textos]
    _helpers.grid_search_k(bow, dic, textos, k_range=[2], return_npmi=True, top_k=10)

    assert len(capturado) == 2, f"esperava 2 CoherenceModel (c_v + c_npmi), veio {len(capturado)}"
    for kwargs in capturado:
        assert kwargs.get("topn") == 10, f"topn ausente ou errado: {kwargs.get('topn')}"


def test_grid_search_alpha_eta_grava_checkpoint_por_combo(corpus, tmp_path):
    textos, dic = corpus
    bow = [dic.doc2bow(t) for t in textos]
    p = tmp_path / "lda_alpha_eta_checkpoint.csv"
    grid_search_alpha_eta(
        bow, dic, textos, k=2, alpha_grid=["symmetric"], eta_grid=[None, 0.1],
        seed=42, passes=2, workers=1, checkpoint_path=p,
    )
    df = pd.read_csv(p)
    assert len(df) == 2
    assert set(df["eta"].astype(str)) == {"nan", "0.1"}


def test_grid_search_k_nmf_grava_checkpoint_por_k(corpus, tmp_path):
    textos, dic = corpus
    bow = [dic.doc2bow(t) for t in textos]
    p = tmp_path / "nmf_k_checkpoint.csv"
    grid_search_k_nmf(bow, dic, textos, [2, 3], seed=42, passes=2, checkpoint_path=p)
    df = pd.read_csv(p)
    assert list(df["k"]) == [2, 3]


def test_grid_search_k_nmf_passa_topn_ao_gensim(monkeypatch):
    """A curva de K do NMF nao pode cair no topn=20 default do gensim (mesmo
    bug de ``grid_search_k``, protocolo secao 3.4 — a curva ORDENA o braco NMF).

    Mesma ressalva de monkeypatch: patchear ``_helpers.CoherenceModel`` e
    ``_helpers.Nmf`` (bindings de modulo), nao os nomes em ``gensim.models``.
    """
    import _helpers
    capturado = []

    class FakeCM:
        def __init__(self, **kwargs):
            capturado.append(kwargs)

        def get_coherence(self):
            return 0.5

    class FakeNmf:
        def __init__(self, *a, **kw):
            pass

    monkeypatch.setattr(_helpers, "CoherenceModel", FakeCM)
    monkeypatch.setattr(_helpers, "Nmf", FakeNmf)

    textos = [["a", "b", "c"], ["a", "b", "d"]] * 5
    dic = Dictionary(textos)
    bow = [dic.doc2bow(t) for t in textos]
    _helpers.grid_search_k_nmf(bow, dic, textos, k_range=[2], return_npmi=True, top_k=10)

    assert len(capturado) == 2, f"esperava 2 CoherenceModel (c_v + c_npmi), veio {len(capturado)}"
    for kwargs in capturado:
        assert kwargs.get("topn") == 10, f"topn ausente ou errado: {kwargs.get('topn')}"


# --------------------------------------------------------------------------
# grid_search_k_stm — Task 11 (protocolo secao 1.7): Diversity como terceiro
# eixo do braco STM, top_k compartilhado com C_v sobre a MESMA lista de top
# words por K. Seam monkeypatchada: `_helpers._run_stm_r`, a fronteira do
# subprocesso R (unico jeito de testar sem R instalado / sem fit real).
# `compute_coherence_cv` e `compute_topic_diversity` sao espionadas (chamada
# real preservada) para capturar os kwargs de cada chamada.
# --------------------------------------------------------------------------

def test_grid_search_k_stm_passa_top_k_as_duas_metricas_e_grava_diversity(monkeypatch, tmp_path):
    """`top_k` tem que chegar tanto a C_v (compute_coherence_cv) quanto a
    Diversity (compute_topic_diversity) — a MESMA lista de top words por K,
    truncada pelo mesmo `top_k`, alimenta as duas (Task 11: reusar, nao
    reconstruir). `diag_df` precisa trazer a coluna `diversity`.
    """
    import _helpers

    def fake_run_stm_r(mode, input_csv, work_dir, stm_cfg, seed, extra=None):
        assert mode == "grid_k"
        return {
            "results": [
                {
                    "k": 2,
                    "topics": {
                        # 5 palavras conhecidas cada, sem OOV — top_k=3 corta 2
                        "0": ["gato", "cachorro", "casa", "rua", "carro"],
                        "1": ["carro", "rua", "casa", "gato", "cachorro"],
                    },
                    "heldout_likelihood": -1.0,
                    "residual_dispersion": 1.2,
                    "semantic_coherence": -55.0,
                    "exclusivity_stm": 9.3,
                    "iterations": 12,
                    "elapsed_sec": 3.4,
                }
            ]
        }

    monkeypatch.setattr(_helpers, "_run_stm_r", fake_run_stm_r)
    monkeypatch.setattr(_helpers, "_validate_stm_formula", lambda *a, **kw: None)

    chamadas_cv, chamadas_div = [], []
    cv_real, div_real = _helpers.compute_coherence_cv, _helpers.compute_topic_diversity

    def spy_cv(*a, **kw):
        chamadas_cv.append(kw)
        return cv_real(*a, **kw)

    def spy_div(*a, **kw):
        chamadas_div.append(kw)
        return div_real(*a, **kw)

    monkeypatch.setattr(_helpers, "compute_coherence_cv", spy_cv)
    monkeypatch.setattr(_helpers, "compute_topic_diversity", spy_div)

    textos = [
        ["gato", "cachorro", "casa"],
        ["gato", "casa", "rua"],
        ["rua", "carro", "casa"],
        ["gato", "cachorro", "rua"],
    ] * 5
    dic = Dictionary(textos)
    stm_df = pd.DataFrame({"text": ["a b c"]})

    cv_scores, diag_df = _helpers.grid_search_k_stm(
        stm_df, textos, dic, k_range=[2],
        prevalence_formula="~ 1", stm_cfg={}, work_dir=tmp_path, top_k=3,
    )

    assert len(chamadas_cv) == 1, f"esperava 1 chamada a compute_coherence_cv, veio {len(chamadas_cv)}"
    assert chamadas_cv[0].get("top_k") == 3, f"top_k nao chegou ao C_v: {chamadas_cv[0]}"
    assert len(chamadas_div) == 1, f"esperava 1 chamada a compute_topic_diversity, veio {len(chamadas_div)}"
    assert chamadas_div[0].get("top_k") == 3, f"top_k nao chegou a Diversity: {chamadas_div[0]}"

    assert "diversity" in diag_df.columns
    # top_k=3 trunca cada topico para as 3 primeiras palavras: {gato,cachorro,casa}
    # e {carro,rua,casa} — 5 unicas / (3*2) = 0.8333...
    assert diag_df.loc[0, "diversity"] == pytest.approx(5 / 6)
    assert set(cv_scores) == {2}


def test_grid_search_k_stm_diversity_muda_com_top_k_mais_curto(monkeypatch, tmp_path):
    """Falsificacao inversa: se `top_k` NAO fosse repassado (ou fosse ignorado),
    a Diversity sairia sobre a lista inteira de 5 palavras — 0.5, nao 0.8333.
    Prova que o valor acima nao e coincidencia: o truncamento e real.
    """
    import _helpers

    def fake_run_stm_r(mode, input_csv, work_dir, stm_cfg, seed, extra=None):
        return {
            "results": [{
                "k": 2,
                "topics": {
                    "0": ["gato", "cachorro", "casa", "rua", "carro"],
                    "1": ["carro", "rua", "casa", "gato", "cachorro"],
                },
                "heldout_likelihood": -1.0, "residual_dispersion": 1.2,
                "semantic_coherence": -55.0, "exclusivity_stm": 9.3,
                "iterations": 12, "elapsed_sec": 3.4,
            }]
        }

    monkeypatch.setattr(_helpers, "_run_stm_r", fake_run_stm_r)
    monkeypatch.setattr(_helpers, "_validate_stm_formula", lambda *a, **kw: None)

    textos = [
        ["gato", "cachorro", "casa"],
        ["gato", "casa", "rua"],
        ["rua", "carro", "casa"],
        ["gato", "cachorro", "rua"],
    ] * 5
    dic = Dictionary(textos)
    stm_df = pd.DataFrame({"text": ["a b c"]})

    _, diag_curto = _helpers.grid_search_k_stm(
        stm_df, textos, dic, k_range=[2],
        prevalence_formula="~ 1", stm_cfg={}, work_dir=tmp_path / "curto", top_k=3,
    )
    _, diag_longo = _helpers.grid_search_k_stm(
        stm_df, textos, dic, k_range=[2],
        prevalence_formula="~ 1", stm_cfg={}, work_dir=tmp_path / "longo", top_k=5,
    )

    assert diag_curto.loc[0, "diversity"] == pytest.approx(5 / 6)
    assert diag_longo.loc[0, "diversity"] == pytest.approx(0.5)
    assert diag_curto.loc[0, "diversity"] != pytest.approx(diag_longo.loc[0, "diversity"])


def test_grid_search_nmf_hparams_grava_checkpoint_por_combo(corpus, tmp_path):
    textos, dic = corpus
    bow = [dic.doc2bow(t) for t in textos]
    p = tmp_path / "nmf_hparams_checkpoint.csv"
    grid_search_nmf_hparams(
        bow, dic, textos, k=2, kappa_grid=[0.5, 1.0], min_prob_grid=[0.0],
        seed=42, passes=2, checkpoint_path=p,
    )
    df = pd.read_csv(p)
    assert len(df) == 2
    assert set(df["kappa"]) == {0.5, 1.0}


def test_cobertura_nmf_conta_documento_sem_topico():
    """Documento sem nenhum topico acima do limiar NAO conta como coberto.

    Protocolo secao 1.6: e o mecanismo pelo qual o NMF pode ter cobertura
    < 100% sem que nada no grid avise.
    """
    from _helpers import _cobertura_doc_topico

    dists = [
        [(0, 0.9), (1, 0.1)],   # coberto
        [(0, 0.5), (1, 0.5)],   # coberto
        [],                      # descartado pelo minimum_probability
        [(2, 0.7)],             # coberto
    ]
    assert _cobertura_doc_topico(dists) == 0.75


def test_cobertura_nmf_vazia_e_zero():
    from _helpers import _cobertura_doc_topico
    assert _cobertura_doc_topico([]) == 0.0


def test_cobertura_nmf_exclui_bow_vazio_do_denominador():
    """Documento com BOW vazio nao conta contra a cobertura quando a mascara
    e passada -- achado 2026-08-24 (tweets_bre2022): 27 documentos sem
    nenhum token no vocabulario ficam sempre sem topico, mesmo com
    minimum_probability=0.0, e isso nao e descarte do hiperparametro.
    """
    from _helpers import _cobertura_doc_topico

    dists = [
        [(0, 0.9), (1, 0.1)],   # coberto, BOW nao vazio
        [],                      # BOW VAZIO -- nunca teria topico, exclui do denominador
        [],                      # descartado pelo minimum_probability, BOW nao vazio
        [(2, 0.7)],             # coberto, BOW nao vazio
    ]
    mask = [True, False, True, True]
    # denominador = 3 (indices 0, 2, 3); cobertos = 2 (indices 0, 3)
    assert _cobertura_doc_topico(dists, bow_nao_vazio=mask) == pytest.approx(2 / 3)


def test_cobertura_nmf_todos_bow_vazio_e_zero():
    from _helpers import _cobertura_doc_topico
    assert _cobertura_doc_topico([[], []], bow_nao_vazio=[False, False]) == 0.0


def test_grid_search_nmf_hparams_devolve_coluna_cobertura(corpus):
    """Wiring: cada linha da grade traz ``cobertura`` em [0, 1] (protocolo
    secao 1.6) — sem isso nada impede o minimum_probability vencedor de ser
    o que mais descarta documentos.
    """
    textos, dic = corpus
    bow = [dic.doc2bow(t) for t in textos]
    df, _, _ = grid_search_nmf_hparams(
        bow, dic, textos, k=2, kappa_grid=[1.0], min_prob_grid=[0.0, 0.99],
        seed=42, passes=2,
    )
    assert "cobertura" in df.columns
    assert df["cobertura"].between(0.0, 1.0).all()
    # minimum_probability=0.99 e um limiar quase impossivel de bater: deve
    # descartar estritamente mais documentos que 0.0 (prova que a cobertura
    # e recalculada por combo, e nao herdada do minimum_probability de
    # construcao do modelo).
    cov_by_min_prob = df.set_index("minimum_probability")["cobertura"]
    assert cov_by_min_prob.loc[0.99] < cov_by_min_prob.loc[0.0]


def test_grid_search_nmf_hparams_devolve_colunas_npmi_diversity(corpus):
    """Fix round 1 da Task 6: sem NPMI/Diversity na grade do segundo estagio,
    `_selecao.selecionar_nmf` quebra com KeyError ao tentar ordenar por NPMI
    (achado Critical da revisao). A grade precisa trazer as duas colunas, em
    MAIUSCULAS (mesmo nome que ``_selecao.EIXOS``/``_pareto`` esperam),
    lado a lado com ``cv``/``cobertura`` minusculos.
    """
    textos, dic = corpus
    bow = [dic.doc2bow(t) for t in textos]
    df, _, _ = grid_search_nmf_hparams(
        bow, dic, textos, k=2, kappa_grid=[1.0], min_prob_grid=[0.0, 0.99],
        seed=42, passes=2,
    )
    assert "NPMI" in df.columns
    assert "Diversity" in df.columns
    assert df["Diversity"].between(0.0, 1.0).all()
    assert df["NPMI"].notna().all()
    # NPMI/Diversity sao lidas via show_topic() (topico-palavra), que NAO
    # aceita minimum_probability (verificado via inspect.signature) — ao
    # contrario de cobertura (get_document_topics, onde o limiar importa de
    # verdade). Para o mesmo kappa, portanto, os dois valores tem de ser
    # IDENTICOS entre minimum_probability=0.0 e 0.99, enquanto cobertura
    # diverge (ja coberto pelo teste acima).
    npmi_by_min_prob = df.set_index("minimum_probability")["NPMI"]
    div_by_min_prob = df.set_index("minimum_probability")["Diversity"]
    assert npmi_by_min_prob.loc[0.0] == npmi_by_min_prob.loc[0.99]
    assert div_by_min_prob.loc[0.0] == div_by_min_prob.loc[0.99]


# --------------------------------------------------------------------------
# grid_search_nmf_hparams — score_by (Critical B da revisao final: os
# notebooks NMF treinavam o modelo publicado a partir de best_kappa/
# best_min_prob ordenados por cv, a UNICA metrica que a secao 1.3 proibe de
# decidir em qualquer braco). Espelha score_by de grid_search_alpha_eta
# (LDA): mesmo nome de parametro, mesmos dois valores aceitos, mesmo
# ValueError na entrada invalida.
# --------------------------------------------------------------------------

def test_grid_search_nmf_hparams_score_by_npmi_muda_o_vencedor(monkeypatch):
    """Falsificacao: kappa=0.5 vence por cv (0.9 > 0.1), kappa=1.0 vence por
    npmi (0.9 > 0.1) — os dois eixos DISCORDAM de proposito. Se score_by
    fosse ignorado (comportamento pre-fix: o parametro nem existia), os dois
    pedidos abaixo devolveriam o mesmo best_kappa.
    """
    import _helpers

    class FakeNmf:
        def __init__(self, *a, kappa=None, **kw):
            self.kappa = kappa

        def get_document_topics(self, doc, minimum_probability=None):
            return [(0, 0.9)]

        def show_topic(self, topicid, topn=10):
            return [(f"w{i}", 1.0 / topn) for i in range(topn)]

    class FakeCM:
        def __init__(self, **kwargs):
            self._coherence = kwargs["coherence"]
            self._kappa = kwargs["model"].kappa

        def get_coherence(self):
            valores = {
                (0.5, "c_v"): 0.9, (0.5, "c_npmi"): 0.1,
                (1.0, "c_v"): 0.1, (1.0, "c_npmi"): 0.9,
            }
            return valores[(self._kappa, self._coherence)]

    monkeypatch.setattr(_helpers, "CoherenceModel", FakeCM)
    monkeypatch.setattr(_helpers, "Nmf", FakeNmf)

    textos = [["a", "b", "c"], ["a", "b", "d"]] * 5
    dic = Dictionary(textos)
    bow = [dic.doc2bow(t) for t in textos]

    _, best_kappa_cv, _ = _helpers.grid_search_nmf_hparams(
        bow, dic, textos, k=2, kappa_grid=[0.5, 1.0], min_prob_grid=[0.0],
    )
    _, best_kappa_npmi, _ = _helpers.grid_search_nmf_hparams(
        bow, dic, textos, k=2, kappa_grid=[0.5, 1.0], min_prob_grid=[0.0],
        score_by="npmi",
    )

    assert best_kappa_cv == 0.5, "default score_by='cv' deveria continuar como estava"
    assert best_kappa_npmi == 1.0, "score_by='npmi' nao trocou o vencedor"


def test_grid_search_nmf_hparams_avisa_em_runtime_quando_score_by_e_cv(corpus, capsys):
    """C_v e a unica metrica que a secao 1.3 proibe de decidir em QUALQUER
    braco — ordenar por ela aqui tem de avisar em runtime, nao so em
    docstring (todo outro fallback deste projeto avisa; este nao era
    excecao).
    """
    textos, dic = corpus
    bow = [dic.doc2bow(t) for t in textos]
    grid_search_nmf_hparams(
        bow, dic, textos, k=2, kappa_grid=[1.0], min_prob_grid=[0.0],
        seed=42, passes=2,
    )
    saida = capsys.readouterr().out
    assert "AVISO" in saida
    assert "protocolo" in saida.lower()


def test_grid_search_nmf_hparams_score_by_npmi_nao_avisa(corpus, capsys):
    textos, dic = corpus
    bow = [dic.doc2bow(t) for t in textos]
    grid_search_nmf_hparams(
        bow, dic, textos, k=2, kappa_grid=[1.0], min_prob_grid=[0.0],
        seed=42, passes=2, score_by="npmi",
    )
    saida = capsys.readouterr().out
    assert "AVISO" not in saida


def test_grid_search_nmf_hparams_score_by_invalido_levanta_valueerror(corpus):
    textos, dic = corpus
    bow = [dic.doc2bow(t) for t in textos]
    with pytest.raises(ValueError, match="score_by"):
        grid_search_nmf_hparams(
            bow, dic, textos, k=2, kappa_grid=[1.0], min_prob_grid=[0.0],
            seed=42, passes=2, score_by="perplexidade",
        )


def test_grid_search_nmf_hparams_passa_topn_ao_gensim(monkeypatch):
    """Mesmo cuidado de `test_grid_search_k_nmf_passa_topn_ao_gensim`: sem
    ``topn=top_k`` explicito, cv/NPMI cairiam no default 20 do gensim e o
    segundo estagio mediria numa base diferente da curva de K (protocolo
    secao 1.8/3.4).
    """
    import _helpers
    capturado = []

    class FakeCM:
        def __init__(self, **kwargs):
            capturado.append(kwargs)

        def get_coherence(self):
            return 0.5

    class FakeNmf:
        def __init__(self, *a, **kw):
            pass

        def get_document_topics(self, doc, minimum_probability=None):
            return [(0, 0.9)]

        def show_topic(self, topicid, topn=10):
            return [(f"w{i}", 1.0 / topn) for i in range(topn)]

    monkeypatch.setattr(_helpers, "CoherenceModel", FakeCM)
    monkeypatch.setattr(_helpers, "Nmf", FakeNmf)

    textos = [["a", "b", "c"], ["a", "b", "d"]] * 5
    dic = Dictionary(textos)
    bow = [dic.doc2bow(t) for t in textos]
    _helpers.grid_search_nmf_hparams(
        bow, dic, textos, k=2, kappa_grid=[1.0], min_prob_grid=[0.0], top_k=7,
    )

    assert len(capturado) == 2, f"esperava 2 CoherenceModel (c_v + c_npmi), veio {len(capturado)}"
    for kwargs in capturado:
        assert kwargs.get("topn") == 7, f"topn ausente ou errado: {kwargs.get('topn')}"


class _FakeBertopicModel:
    def fit_transform(self, docs, embeddings=None):
        return None


def _fake_bertopic_metrics_patch(monkeypatch):
    import _helpers
    monkeypatch.setattr(_helpers, "_bertopic_postprocess", lambda *a, **kw: (0.1, 0.1, 5))
    monkeypatch.setattr(
        _helpers, "_bertopic_sweep_metrics",
        lambda *a, **kw: (
            {"K": 3, "npmi": 0.1, "cv": 0.1, "div": 0.1, "excl": 0.1, "frex": 0.1},
            {0: ["a", "b"], 1: ["c", "d"], 2: ["e", "f"]},
        ),
    )


def test_sweep_bertopic_grid_grava_checkpoint_por_linha(monkeypatch, tmp_path):
    _fake_bertopic_metrics_patch(monkeypatch)
    p = tmp_path / "sweep_bertopic_checkpoint.csv"
    sweep_bertopic_grid(
        lambda seed, nn, mcs, ms, csm: _FakeBertopicModel(),
        docs=["d1", "d2"], embeddings=None, tokenized=[["a"]], dictionary=None,
        seeds=[1, 2], n_neighbors_grid=[10], min_cluster_size_grid=[5],
        checkpoint_path=p,
    )
    df = pd.read_csv(p)
    assert len(df) == 2
    assert set(df["seed"]) == {1, 2}


def test_sweep_outlier_strategies_grava_checkpoint_por_linha(monkeypatch, tmp_path):
    _fake_bertopic_metrics_patch(monkeypatch)
    p = tmp_path / "sweep_outlier_checkpoint.csv"
    sweep_outlier_strategies(
        lambda seed: _FakeBertopicModel(),
        docs=["d1", "d2"], embeddings=None, tokenized=[["a"]], dictionary=None,
        seeds=[1, 2], strategies=("off", "c-tf-idf"), checkpoint_path=p,
    )
    df = pd.read_csv(p)
    assert len(df) == 4
    assert set(df["strategy"]) == {"off", "c-tf-idf"}


def test_sweep_outlier_threshold_grava_checkpoint_por_linha(monkeypatch, tmp_path):
    _fake_bertopic_metrics_patch(monkeypatch)
    p = tmp_path / "sweep_threshold_checkpoint.csv"
    sweep_outlier_threshold(
        lambda seed: _FakeBertopicModel(),
        docs=["d1", "d2"], embeddings=None, tokenized=[["a"]], dictionary=None,
        seeds=[1], grids={"c-tf-idf": [0.1, 0.2]}, checkpoint_path=p,
    )
    df = pd.read_csv(p)
    assert len(df) == 2
    assert set(df["threshold"]) == {0.1, 0.2}


def test_coerencia_respeita_top_k_declarado():
    """top_k explicito trunca as listas ANTES do CoherenceModel.

    Sem isso, dois bracos com listas de tamanhos diferentes sao medidos em
    profundidades diferentes e a camada 3 compara numeros incomparaveis
    (protocolo secao 1.8, condicao 1).
    """
    textos = [
        ["banco", "credito", "juros", "divida", "conta"],
        ["banco", "credito", "juros", "poupanca", "conta"],
        ["eleicao", "voto", "urna", "campanha", "partido"],
        ["eleicao", "voto", "urna", "debate", "partido"],
    ] * 5
    dic = Dictionary(textos)
    topicos = {
        0: ["banco", "credito", "juros", "divida", "conta"],
        1: ["eleicao", "voto", "urna", "campanha", "partido"],
    }
    _, diag5 = compute_coherence_npmi(topicos, textos, dic, return_counts=True)
    _, diag2 = compute_coherence_npmi(topicos, textos, dic, return_counts=True, top_k=2)
    assert diag5["topn_efetivo"] == 5
    assert diag2["topn_efetivo"] == 2
    assert diag2["top_k_declarado"] == 2
    assert diag5["top_k_declarado"] is None


def test_top_k_maior_que_a_lista_nao_inventa_palavra():
    """top_k acima do comprimento disponivel nao pode inflar topn_efetivo."""
    textos = [["a", "b", "c"], ["a", "b", "d"], ["a", "c", "d"]] * 5
    dic = Dictionary(textos)
    topicos = {0: ["a", "b"], 1: ["c", "d"]}
    _, diag = compute_coherence_npmi(topicos, textos, dic, return_counts=True, top_k=10)
    assert diag["topn_efetivo"] == 2
    assert diag["top_k_declarado"] == 10


def test_truncacao_antes_filtro_descarta_topico_pequeno():
    """Truncacao ANTES do filtro len(kws)>=2 e o aspecto critico da ordem.

    Discrimina entre truncar-antes e truncar-depois via topicos que passam
    o filtro (>= 2 palavras) ANTES de truncacao mas ficariam abaixo depois.

    Com top_k=1 sobre topicos de 3 e 2 palavras (todas conhecidas):
    - Correto (truncar-antes): ambos viram 1 palavra → ambos descartados
    - Errado (truncar-depois): filtro roda sobre originals (3>=2, 2>=2) →
      ambos passam → depois truncam para 1. Diferentes n_topicos_descartados.
    """
    textos = [["a", "b", "c"], ["a", "b", "d"], ["a", "c", "d"]] * 5
    dic = Dictionary(textos)
    # Topic 0: 3 words (todas conhecidas) → trunca a 1 → descartado (< 2)
    # Topic 1: 2 words (todas conhecidas) → trunca a 1 → descartado (< 2)
    topicos = {0: ["a", "b", "c"], 1: ["c", "d"]}
    _, diag = compute_coherence_npmi(topicos, textos, dic, return_counts=True, top_k=1)

    # Com truncacao-antes (correto): ambos truncam a 1 palavra, ambos descartados
    assert diag["n_topicos"] == 2
    assert diag["n_topicos_descartados"] == 2  # ambos filtrados pos-truncacao
    assert diag["n_topicos_avaliados"] == 0     # nenhum topico sobreviveu
    assert diag["top_k_declarado"] == 1
