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
import os
import sys

import httpx
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gensim.corpora import Dictionary  # noqa: E402

from _helpers import (  # noqa: E402
    compute_coherence_cv,
    compute_exclusivity_ctfidf,
    compute_frex_score,
    compute_jaccard,
    compute_topic_diversity,
    make_run_output_dir,
    resolve_latest_dir,
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


# --------------------------------------------------------------------------
# topn — o tamanho da lista chega ao CoherenceModel do gensim
# --------------------------------------------------------------------------

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
# train_nmf — linhas de get_topics() normalizadas (Exclusividade/FREX)
# --------------------------------------------------------------------------

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
