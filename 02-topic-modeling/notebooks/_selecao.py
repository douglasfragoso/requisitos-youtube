"""Selecao da configuracao — implementacao do protocolo de 2026-08-16.

    venv\\Scripts\\python 03-topic-modeling/notebooks/_selecao.py bertopic folha
    venv\\Scripts\\python 03-topic-modeling/notebooks/_selecao.py lda tweets_bre2022

NAO ajusta modelo: le o CSV de grade que ja existe e roda em segundos.

POR QUE POR RESTRICOES, E NAO POR ESCORE
----------------------------------------
O teste de invariancia rodado em 16/08 sobre as grades deste projeto
(``docs/protocolo-selecao-final-2026-08-16.md`` secao 3.2) mostrou duas coisas:

1. Sob **composto min-max** a escolha NAO e invariante a troca da familia de
   metricas — na folha a sobreposicao do top-5 cai a 2/5, 1/5 e **0/5** contra a
   familia Exclusividade+FREX. Ordenacao fina por escore nao e interpretavel.
2. Sob **Pareto + desempate declarado** a escolha e invariante ao conjunto de
   eixos nos dois corpora — MAS todas as vencedoras tinham K=9, o extremo da
   grade. Sem a camada 1 na frente, o desempate por cobertura sozinho puxa para
   a configuracao mais grossa.

Dai a ordem deste arquivo, que e a ordem do protocolo: **admissibilidade
primeiro, ordenacao depois, desempate por ultimo**. A faixa de K deixa de ser
prosa ("K=9 muito grosseiro para jornal generalista") e vira constante
declarada, auditavel e aplicada antes de qualquer olhada na grade.
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import pandas as pd


def _forcar_utf8_no_stdout() -> None:
    """Reconfigura o stdout para UTF-8 — chamado SO pelo __main__.

    No nivel do modulo isso e efeito colateral de import: quebra o pytest (que
    captura stdout com um arquivo proprio e o fecha ao final) e qualquer outro
    importador. Este arquivo precisa ser importavel por ser fonte unica de
    ``FAIXA_K`` para os notebooks de LDA.
    """
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass  # stdout ja e um wrapper sem .buffer (pytest, notebook) — nada a fazer

RAIZ = Path(__file__).resolve().parents[1]
OUT = RAIZ / "data" / "output"

# ---------------------------------------------------------------------------
# Parametros do protocolo — DECLARADOS, nunca ajustados post hoc
# ---------------------------------------------------------------------------
# Protocolo secao 1.1: a selecao roda sobre a seed 42 e APENAS sobre ela — e a
# mesma que gera o modelo publicado, entao selecao e publicacao incidem sobre o
# mesmo ajuste, sem camada de medias no meio. As demais seeds ficam nos CSVs
# como registro historico e NAO sao lidas aqui.
SEED = 42

# A3: faixa substantiva de K, por corpus. Justificativa e unidade de analise,
# nao resultado: a folha e um jornal generalista com 10 editorias balanceadas e
# precisa de granularidade fina (K=9 daria menos de um topico por editoria); os
# tweets cobrem um unico evento eleitoral e nao sustentam a mesma resolucao.
FAIXA_K = {
    "folha": (15, 30),
    "tweets_bre2022": (10, 25),
}
RAZAO_MAX = 2.0            # A4: n_raw <= 2K (so BERTopic — pressupoe reduce_topics)
DESEMPATE = "cobertura"    # declarado ANTES de olhar a fronteira
# Desempate do STM na fronteira semantic_coherence x exclusivity_stm (Roberts et al.):
# menor K, parcimonia estrutural — mesmo raciocinio do desempate secundario do
# BERTopic no protocolo do ARJ (K e a grandeza mais confiavel do conjunto).
# Decisao do usuario, 2026-08-19. ascending=True: ordena do menor K para o maior
# e pega o primeiro.
DESEMPATE_STM = "K"
DESEMPATE_STM_ASCENDING = True
# Estabilidade (Jaccard) REMOVIDA em 17/08/2026 (docs/protocolo-selecao-final-
# 2026-08-16.md §4.1, revisao): sem limiar de Jaccard consagrado, ela nao pode
# ser porta, e como eixo de Pareto nao ganhava poder de decisao — so alargava
# a fronteira sem separar candidatas. Decisao do usuario, mesma direcao do ARJ.
EIXOS = ["NPMI", "Diversity", "cobertura"]

CHAVES = ["n_neighbors", "min_cluster_size", "min_samples",
          "cluster_selection_method", "reduce_nr"]

# `_bertopic_sweep_metrics` (_helpers.py) devolve npmi/cv/div/excl/frex em
# MINUSCULO, e o RAW do sweep (sweep_bertopic_grid_raw.csv) herda essa
# casing sem alteracao — so o AGREGADO (sweep_bertopic_grid.csv) renomeia
# para NPMI/C_v/Diversity/Exclus/FREX na hora de montar `agg_rows`
# (`sweep_bertopic_grid`, _helpers.py). `carregar_bertopic` le o RAW por
# padrao (secao 1.1: selecao roda so sobre a seed 42, que so o raw tem por
# linha), entao sem esta normalizacao a coluna "NPMI" nunca existe no `d`
# que chega em `_grade_apta`/`_pareto` — a grade e recusada mesmo tendo NPMI
# valido, so com o nome errado. Aplicada logo apos qualquer um dos dois
# caminhos que produzem `d` (raw ou fallback pro agregado), para que os dois
# nunca possam divergir de novo.
_RENOMEIA_RAW_BERTOPIC = {"npmi": "NPMI", "cv": "C_v", "div": "Diversity",
                          "excl": "Exclus", "frex": "FREX"}


def _pareto(df: pd.DataFrame, eixos: list[str], verbose: bool = False) -> pd.DataFrame:
    """Pontos nao dominados: nenhum outro e >= em todos os eixos e > em algum.

    ATENCAO ao NaN, que aqui e armadilha silenciosa: ``NaN >= x`` e ``NaN > x``
    sao ambos False, entao uma linha com NaN em qualquer eixo NUNCA e marcada
    como dominada e sobrevive a fronteira mesmo sendo pior em tudo. Uma linha
    pode ter NaN legitimamente — ``_bertopic_sweep_metrics`` devolve
    ``npmi=nan`` quando o topico inteiro cai fora do dicionario. Por isso as
    linhas incompletas sao EXCLUIDAS da fronteira, e o numero delas e reportado
    em vez de silenciado: nao se pode afirmar que um ponto nao e dominado num
    eixo que nao foi medido.
    """
    eixos = [e for e in eixos if e in df.columns and df[e].notna().any()]
    if not eixos or df.empty:
        return df.iloc[0:0]
    completo = df.dropna(subset=eixos)
    n_fora = len(df) - len(completo)
    if n_fora and verbose:
        print(f"   ({n_fora} configuracoes fora da fronteira por NaN em "
              f"{', '.join(eixos)} — nao medidas, nao comparaveis)")
    if completo.empty:
        return df.iloc[0:0]
    A = completo[eixos].to_numpy()
    manter = [completo.index[i] for i in range(len(A))
              if not (((A >= A[i]).all(axis=1)) & ((A > A[i]).any(axis=1))).any()]
    return completo.loc[manter]


def carregar_bertopic(corpus: str) -> pd.DataFrame:
    """Grade da seed 42 (protocolo secao 1.1), lida do RAW; agregado so como fallback."""
    base = OUT / corpus / "bertopic"
    aggs = sorted(base.glob("*/sweep_bertopic_grid.csv"))
    if not aggs:
        raise SystemExit(f"nenhum sweep_bertopic_grid.csv em {base}")
    agg_p = aggs[-1]

    raw_p = agg_p.with_name("sweep_bertopic_grid_raw.csv")
    if raw_p.exists():
        bruto = pd.read_csv(raw_p)
        if "seed" in bruto.columns and (bruto["seed"] == SEED).any():
            d = bruto[bruto["seed"] == SEED].copy()
            print(f"grade: {raw_p.relative_to(OUT)}  (seed {SEED}, {len(d)} pontos)")
        else:
            d = pd.read_csv(agg_p)
            print(f"grade: {agg_p.relative_to(OUT)}  ({len(d)} pontos) "
                  f"— AVISO: raw sem coluna 'seed', usando agregado")
    else:
        d = pd.read_csv(agg_p)
        print(f"grade: {agg_p.relative_to(OUT)}  ({len(d)} pontos) "
              f"— AVISO: sem raw, usando agregado entre seeds (fora da secao 1.1)")

    # Casing canonica (NPMI/C_v/Diversity/Exclus/FREX) em qualquer um dos dois
    # caminhos acima — ver `_RENOMEIA_RAW_BERTOPIC`. No agregado isto e sempre
    # um no-op (as colunas ja chegam maiusculas); so tem efeito no raw.
    # Casing canonica (NPMI/C_v/Diversity/Exclus/FREX) em qualquer um dos dois
    # caminhos acima — ver `_RENOMEIA_RAW_BERTOPIC`. No agregado isto e sempre
    # um no-op (as colunas ja chegam maiusculas); so tem efeito no raw.
    d = d.rename(columns={k: v for k, v in _RENOMEIA_RAW_BERTOPIC.items()
                           if k in d.columns and v not in d.columns})

    if "cobertura" not in d.columns:
        d["cobertura"] = 1 - d["outlier_pre"]

    # K_min/K_max e n_raw so existem em grades geradas apos 16/08. Em grades
    # antigas, recupera do _raw.csv; se nem isso, as restricoes A1/A4 ficam
    # INAVALIAVEIS e o script diz isso em vez de fingir que passaram.
    raw_p = agg_p.with_name("sweep_bertopic_grid_raw.csv")
    if ("K_min" not in d.columns or "n_raw" not in d.columns) and raw_p.exists():
        raw = pd.read_csv(raw_p)
        g = raw.groupby(CHAVES, dropna=False).agg(
            K_min=("K", "min"), K_max=("K", "max"),
            n_raw=("n_raw", "mean"), n_seeds_raw=("K", "size"),
        ).reset_index()
        d = d.merge(g, on=CHAVES, how="left", suffixes=("", "_raw"))
    if "razao_nraw_k" not in d.columns and "n_raw" in d.columns:
        d["razao_nraw_k"] = d["n_raw"] / d["K"].clip(lower=1e-9)
    return d


def _grade_apta(d: pd.DataFrame) -> list[str]:
    """A grade suporta o protocolo? Devolve a lista de impedimentos.

    Existe porque o modo de falha mais perigoso deste script nao e errar a
    conta: e rodar sobre uma grade PRE-PROTOCOLO, produzir uma "escolhida"
    plausivel e grava-la em ``selecao_bertopic.csv`` — que a celula-guarda dos
    notebooks trata como fonte da verdade. O resultado seria trocar a
    configuracao de producao por uma eleita sem o eixo que decide.

    As grades de 06-07/2026 nao tem NPMI. Sem NPMI a fronteira de Pareto cai
    silenciosamente para Diversity x cobertura, que NAO e o criterio do
    protocolo (secao 1.4). Multi-seed deixou de ser exigencia: sob a secao 1.1
    a selecao roda em seed unica por decisao declarada, nao por limitacao da
    grade.
    """
    impedimentos = []
    if "NPMI" not in d.columns or not d["NPMI"].notna().any():
        impedimentos.append(
            "sem coluna NPMI — a grade e anterior ao protocolo de 2026-08-16; "
            "o eixo que ORDENA nao existe nela"
        )
    if "n_raw" not in d.columns or not d["n_raw"].notna().any():
        impedimentos.append("sem n_raw — A4 (n_raw <= 2K) e inavaliavel")
    return impedimentos


def selecionar_bertopic(corpus: str, forcar: bool = False) -> pd.DataFrame:
    d = carregar_bertopic(corpus)
    k_min, k_max = FAIXA_K[corpus]

    print("=" * 78)
    print(f"SELECAO — BERTopic / {corpus}   (faixa de K declarada: [{k_min},{k_max}])")
    print("=" * 78)

    impedimentos = _grade_apta(d)
    if impedimentos:
        print("\n!! GRADE NAO APTA AO PROTOCOLO:")
        for imp in impedimentos:
            print(f"   - {imp}")
        if not forcar:
            print("\nNADA foi gravado. Regere a grade com NPMI antes:")
            print("   o eixo que ORDENA (protocolo secao 1.4) nao existe na grade atual")
            print("\nSe quiser inspecionar mesmo assim (SEM gravar), use --forcar.")
            return d.iloc[0:0]
        print("\n--forcar: seguindo em modo INSPECAO. O resultado NAO e uma selecao")
        print("valida sob o protocolo e nao sera gravado em selecao_bertopic.csv.")

    cond = []
    cond.append((f"A3  K em [{k_min},{k_max}]", (d.K >= k_min) & (d.K <= k_max)))
    if "razao_nraw_k" in d.columns and d["razao_nraw_k"].notna().any():
        cond.append((f"A4  n_raw <= {RAZAO_MAX:g}K", d.razao_nraw_k <= RAZAO_MAX))
    else:
        print("!! n_raw ausente — A4 nao avaliada.")

    m = pd.Series(True, index=d.index)
    print("\nCascata de admissibilidade:")
    print(f"   grade completa: {len(d)}")
    for nome, c in cond:
        m &= c.fillna(False)
        print(f"   apos {nome}: {int(m.sum())}")
    adm = d[m].copy()
    if adm.empty:
        print("\nNENHUMA configuracao admissivel. Ou a grade nao cobre a faixa")
        print("declarada, ou a razao n_raw/K reprova tudo — os dois sao achados,")
        print("nao erros. Ver protocolo secao 3.2 resultado 3.")
        return adm

    fr = _pareto(adm, EIXOS, verbose=True)
    if fr.empty:
        print("\nFronteira vazia — nenhum ponto admissivel tem os eixos medidos.")
        return adm.iloc[0:0]
    adm["na_fronteira"] = adm.index.isin(fr.index)
    print(f"\nFronteira de Pareto ({' x '.join(e for e in EIXOS if e in adm.columns)}): "
          f"{len(fr)} de {len(adm)}")

    escolhida = fr.sort_values(DESEMPATE, ascending=False).index[0]
    adm["escolhida"] = adm.index == escolhida
    print(f"Desempate declarado: maior {DESEMPATE}")

    _invariancia(adm, fr, escolhida)

    cols = [c for c in CHAVES + ["K", "K_min", "K_max", "n_raw", "razao_nraw_k",
                                 "cobertura", "NPMI", "Diversity", "C_v",
                                 "Exclus"] if c in adm.columns]
    pd.set_option("display.width", 220)
    print("\n--- fronteira (ordenada por desempate) ---")
    print(fr.sort_values(DESEMPATE, ascending=False)[cols].head(8).round(4).to_string(index=False))
    e = adm.loc[escolhida]
    print(f"\n>>> ESCOLHIDA: " + " ".join(f"{k}={e[k]}" for k in CHAVES if k in adm.columns))
    return adm


def _invariancia(adm: pd.DataFrame, fr: pd.DataFrame, escolhida) -> None:
    """A escolha muda se trocarmos o conjunto de eixos? Custa segundos; reportar."""
    print("\n--- teste de invariancia (protocolo secao 3.2) ---")
    regimes = {
        "NPMI x cobertura": ["NPMI", "cobertura"],
        "os tres eixos (= EIXOS)": EIXOS,
        "Diversity x cobertura": ["Diversity", "cobertura"],
    }
    for nome, eixos in regimes.items():
        f = _pareto(adm, eixos)
        if f.empty:
            print(f"   {nome:32s} fronteira vazia")
            continue
        idx = f.sort_values(DESEMPATE, ascending=False).index[0]
        marca = "=" if idx == escolhida else "DIFERE"
        print(f"   {nome:32s} fronteira {len(f):3d} -> {marca}")


def _decidir_curva_k(curvas: dict, k_min: int, k_max: int) -> tuple[int | None, str]:
    """Escolhe K na curva — protocolo secoes 1.5 (LDA) e 1.6 (NMF).

    Dois eixos, nao um: NPMI e Diversity, por fronteira de Pareto, com
    desempate por MENOR K (parcimonia estrutural, mesma razao do desempate
    secundario do BERTopic na secao 1.4). Sem a curva de Diversity no cache,
    cai para NPMI puro e DIZ isso — ordenar por um eixo quando o protocolo
    manda dois e divergencia, e divergencia silenciosa e o que este projeto
    passou a auditoria inteira tentando eliminar.
    """
    ks = [k for k in sorted(curvas.get("NPMI", {})) if k_min <= k <= k_max]
    if not ks:
        return None, "nenhum K admissivel na faixa declarada"

    if "Diversity" not in curvas:
        melhor = max(ks, key=lambda k: curvas["NPMI"][k])
        return melhor, "NPMI puro (sem Diversity no cache — FORA do protocolo)"

    df = pd.DataFrame({
        "K": ks,
        "NPMI": [curvas["NPMI"][k] for k in ks],
        "Diversity": [curvas["Diversity"][k] for k in ks],
    })
    fr = _pareto(df, ["NPMI", "Diversity"])
    if fr.empty:
        melhor = max(ks, key=lambda k: curvas["NPMI"][k])
        return melhor, "NPMI puro (fronteira vazia)"
    melhor = int(fr.sort_values("K", ascending=True).iloc[0]["K"])
    return melhor, "pareto NPMI x Diversity, desempate menor K"


def _selecionar_por_curva_k(base: Path, arquivo_metrics: str, rotulo_modelo: str,
                             corpus: str) -> None:
    """LDA e NMF: K e ENTRADA (nao ha reduce_topics), entao A1/A4 sao vacuas.
    O eixo de ordenacao e NPMI x Diversity sobre a curva de K, por fronteira de
    Pareto (protocolo secoes 1.5 e 1.6). C_v e reportado por comparabilidade,
    nunca decide.
    """
    met = base / arquivo_metrics
    if not met.exists():
        raise SystemExit(f"ausente: {met}")
    d = pd.read_csv(met)
    k_min, k_max = FAIXA_K[corpus]
    print("=" * 78)
    print(f"SELECAO — {rotulo_modelo} / {corpus}   (faixa de K declarada: [{k_min},{k_max}])")
    print("=" * 78)

    import json
    linha = d.iloc[-1]
    curvas = {}
    for col, rot in [("k_npmi_scores", "NPMI"), ("k_diversity_scores", "Diversity"),
                     ("k_grid_scores", "C_v"), ("k_perplexity_scores", "Perplexidade")]:
        if col in d.columns and isinstance(linha.get(col), str):
            curvas[rot] = {int(k): v for k, v in json.loads(linha[col]).items()}

    if "NPMI" not in curvas:
        print(f"\n!! k_npmi_scores ausente — o grid foi gerado antes de 2026-08-16.")
        print(f"   Rode o grid com return_npmi=True e grave a curva; sem ela a")
        print(f"   ordenacao do {rotulo_modelo} continua sendo por C_v, fora do protocolo.")
        return

    print("\nCurva sobre a faixa de K (reportar a CURVA, nao so o argmax):")
    ks = sorted(curvas["NPMI"])
    cab = "   K     " + "".join(f"{r:>14s}" for r in curvas)
    print(cab)
    for k in ks:
        marca = " " if k_min <= k <= k_max else "x"
        print(f"  {marca}{k:>4d}  " + "".join(f"{curvas[r].get(k, float('nan')):>14.4f}" for r in curvas))
    print("   (x = fora da faixa declarada, inadmissivel)")

    melhor, criterio = _decidir_curva_k(curvas, k_min, k_max)
    if melhor is None:
        print(f"\n{criterio.upper()}.")
        return
    print(f"\n>>> K escolhido: {melhor} (NPMI {curvas['NPMI'][melhor]:+.4f})")
    print(f"    criterio: {criterio}")
    if "FORA do protocolo" in criterio:
        print("    !! regere o grid gravando tambem a curva de Diversity")
    adm = [k for k in ks if k_min <= k <= k_max]
    if "C_v" in curvas:
        melhor_cv = max(adm, key=lambda k: curvas["C_v"][k])
        if melhor_cv != melhor:
            print(f"    ATENCAO: por C_v o argmax seria K={melhor_cv}. Divergencia entre")
            print(f"    eixos e informacao, nao erro — reportar os dois.")
    if "Perplexidade" in curvas:
        print("    (perplexidade fica em eixo SEPARADO — Chang et al. 2009; nunca decide)")


def selecionar_lda(corpus: str) -> None:
    _selecionar_por_curva_k(OUT / corpus / "lda", "lda_metrics.csv", "LDA", corpus)


def _porta_cobertura_nmf(grade: pd.DataFrame,
                          tolerancia: float = 0.999) -> tuple[pd.DataFrame, str]:
    """Porta de admissibilidade do braco NMF — protocolo secao 1.6.

    ``minimum_probability`` e hiperparametro de SELECAO e determinante de
    DESCARTE ao mesmo tempo: um documento cujo peso caia abaixo do limiar em
    todos os topicos fica sem topico. Como NPMI, Diversity, Exclusividade e
    FREX melhoram quando os documentos ambiguos somem da conta, sem esta porta
    o grid pode premiar exatamente a configuracao que mais descarta.

    Incide sobre o SEGUNDO estagio da grade NMF (kappa x minimum_probability),
    que e onde o descarte pode ser introduzido — a curva de K (primeiro
    estagio) nao mexe em ``minimum_probability`` e por isso nao passa por
    esta porta.

    Tres desfechos, os tres da secao 1.6:
      (1) coluna ausente -> ABORTA (comparar NMF com LDA seria comparar
          populacoes diferentes de documentos);
      (2) cobertura ~100% em tudo -> porta vacua, registra e segue;
      (3) ha ponto abaixo -> porta ativa, filtra e declara.
    """
    if "cobertura" not in grade.columns:
        raise SystemExit(
            "grade do NMF sem coluna 'cobertura' — protocolo secao 1.6 exige "
            "medi-la ANTES de o grid decidir. Regere com grid_search_nmf_hparams "
            "atualizado (Task 3 do plano)."
        )
    minima = float(grade["cobertura"].min())
    if minima >= tolerancia:
        return grade, (
            f"porta de cobertura VACUA — minimo {minima:.4f} na grade inteira; "
            f"o braco NMF e o braco LDA sem perplexidade, sem mais nada"
        )
    filtrada = grade[grade["cobertura"] >= tolerancia]
    return filtrada, (
        f"porta de cobertura ATIVA — minimo {minima:.4f}; "
        f"{len(grade) - len(filtrada)} de {len(grade)} pontos reprovados. "
        f"A cobertura passa a ser coluna real na camada 3, nao constante"
    )


def selecionar_nmf(corpus: str) -> None:
    """NMF: curva de K como no LDA (secao 1.6), mais a porta de cobertura.

    A porta incide sobre o SEGUNDO estagio (kappa x minimum_probability), que e
    onde o descarte pode ser introduzido; a curva de K nao mexe em
    ``minimum_probability`` e por isso nao passa por ela.
    """
    base = OUT / corpus / "nmf"
    _selecionar_por_curva_k(base, "nmf_metrics.csv", "NMF", corpus)

    hp = base / "nmf_kappa_minprob_grid.csv"
    if not hp.exists():
        print(f"\n(sem {hp.name} — segundo estagio nao rodou ainda)")
        return
    grade = pd.read_csv(hp)
    filtrada, nota = _porta_cobertura_nmf(grade)
    print(f"\n{nota}")
    if filtrada.empty:
        print("NENHUM ponto admissivel apos a porta de cobertura.")
        return
    fr = _pareto(filtrada, ["NPMI", "Diversity"])
    alvo = fr if not fr.empty else filtrada
    escolhida = alvo.sort_values("NPMI", ascending=False).iloc[0]
    print(f">>> kappa={escolhida['kappa']} "
          f"minimum_probability={escolhida['minimum_probability']} "
          f"(NPMI {escolhida['NPMI']:+.4f}, cobertura {escolhida['cobertura']:.4f})")


def _eixos_stm(d: pd.DataFrame) -> list[str]:
    """Eixos da fronteira do STM — protocolo secao 1.7.

    A convencao de Roberts et al. e semantic coherence x exclusivity; o
    protocolo acrescenta a Diversity, para que o STM seja ordenado pelo mesmo
    eixo de repeticao de vocabulario que os outros tres bracos. Grades geradas
    antes de 2026-08-22 nao tem a coluna: cai para os dois eixos nativos, que e
    divergencia da spec e precisa ser dita em voz alta.
    """
    base = ["semantic_coherence", "exclusivity_stm"]
    if "diversity" in d.columns and d["diversity"].notna().any():
        return base + ["diversity"]
    return base


def selecionar_stm(corpus: str) -> None:
    """STM: ordenacao pela fronteira de Roberts et al. — semantic_coherence x
    exclusivity_stm (protocolo secao 4.2.3) MAIS Diversity quando o grid ja
    tem a coluna (secao 1.7) —, NAO por C_v (armadilha de rotulo: a coerencia
    nativa do STM e semantic coherence, familia UMass, nao NPMI).
    Fonte: stm_grid_k_diagnostics.csv, ja em disco — zero custo, nao ajusta
    modelo. Desempate declarado: menor K (DESEMPATE_STM, ver topo do arquivo).
    """
    base = OUT / corpus / "stm"
    diag_p = base / "stm_grid_k_diagnostics.csv"
    if not diag_p.exists():
        raise SystemExit(f"ausente: {diag_p}")
    d = pd.read_csv(diag_p)
    k_min, k_max = FAIXA_K[corpus]
    eixos_stm = _eixos_stm(d)

    print("=" * 78)
    print(f"SELECAO — STM / {corpus}   (faixa de K declarada: [{k_min},{k_max}])")
    print("=" * 78)
    if "diversity" not in eixos_stm:
        print("   !! sem coluna 'diversity' — fronteira de 2 eixos, FORA da secao 1.7")
        print("      regere o grid do STM (Task 11 do plano)")

    faltando = [c for c in eixos_stm if c not in d.columns or not d[c].notna().any()]
    if faltando:
        print(f"\n!! stm_grid_k_diagnostics.csv sem {faltando} — grid anterior ao")
        print("   protocolo ou o R nao emitiu semantic_coherence/exclusivity_stm.")
        return

    d = d.rename(columns={"k": "K"})
    m = (d.K >= k_min) & (d.K <= k_max)
    print("\nCascata de admissibilidade:")
    print(f"   grade completa: {len(d)}")
    print(f"   apos A3  K em [{k_min},{k_max}]: {int(m.sum())}")
    adm = d[m].copy()
    if adm.empty:
        print("\nNENHUM K admissivel na grade.")
        return

    fr = _pareto(adm, eixos_stm, verbose=True)
    if fr.empty:
        print("\nFronteira vazia — nenhum K admissivel tem os dois eixos medidos.")
        return
    print(f"\nFronteira de Pareto ({' x '.join(eixos_stm)}): {len(fr)} de {len(adm)}")

    escolhido = fr.sort_values(DESEMPATE_STM, ascending=DESEMPATE_STM_ASCENDING).iloc[0]
    print(f"Desempate declarado: menor K")

    cols = ["K", "cv", "semantic_coherence", "exclusivity_stm",
            "heldout_likelihood", "residual_dispersion"]
    cols = [c for c in cols if c in fr.columns]
    pd.set_option("display.width", 220)
    print("\n--- fronteira (ordenada por K) ---")
    print(fr.sort_values(DESEMPATE_STM, ascending=DESEMPATE_STM_ASCENDING)[cols]
          .round(4).to_string(index=False))
    print(f"\n>>> ESCOLHIDO: K={int(escolhido['K'])}")
    if "cv" in fr.columns:
        melhor_cv_k = int(adm.loc[adm["cv"].idxmax(), "K"])
        if melhor_cv_k != int(escolhido["K"]):
            print(f"    ATENCAO: por C_v (reporte, nao decide) o argmax seria K={melhor_cv_k}.")
            print("    Divergencia entre eixos e informacao, nao erro — reportar os dois.")
    print("\nNOTA: esta selecao NAO refita o modelo. Se K escolhido != stm_best_k")
    print("pinado em params.yaml, atualize o pin e rode a celula de treino final")
    print("do STM daquele corpus (nao precisa rodar o grid de K de novo).")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("modelo", choices=["bertopic", "lda", "nmf", "stm"])
    ap.add_argument("corpus", choices=sorted(FAIXA_K))
    ap.add_argument("--forcar", action="store_true",
                    help="inspecionar grade nao apta ao protocolo (NAO grava)")
    args = ap.parse_args()
    if args.modelo == "bertopic":
        adm = selecionar_bertopic(args.corpus, forcar=args.forcar)
        # So grava se a grade era apta: selecao_bertopic.csv e lido pela celula-guarda
        # dos notebooks como fonte da verdade, entao gravar a partir de grade
        # pre-protocolo trocaria a configuracao de producao por uma eleita sem o
        # eixo que decide.
        if not adm.empty and not args.forcar:
            dest = OUT / args.corpus / "bertopic" / "selecao_bertopic.csv"
            adm.sort_values(DESEMPATE, ascending=False).to_csv(dest, index=False)
            print(f"\nSalvo: {dest}")
        elif args.forcar:
            print("\n(modo --forcar: nada gravado, por desenho)")
    elif args.modelo == "lda":
        selecionar_lda(args.corpus)
    elif args.modelo == "nmf":
        selecionar_nmf(args.corpus)
    else:
        selecionar_stm(args.corpus)
    return 0


if __name__ == "__main__":
    _forcar_utf8_no_stdout()
    raise SystemExit(main())
