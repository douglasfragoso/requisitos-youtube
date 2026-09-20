# STM — Folha de São Paulo: Documentação Completa do Pipeline

> **⚠️ ERRATA 2026-08-13 — valores de referência superados nas tabelas comparativas.**
> As comparações deste documento citam o **BERTopic-Folha do run de junho**
> (`folha_20260628_221412`): C_v 0,620 / exclusividade 0,542 / TD 0,892 em top-10.
> O run vigente (`folha_20260725_203412`, reproduzido bit a bit em
> `folha_20260813_215259`) entrega **C_v 0,604 / exclusividade 0,547 / TD 0,896**
> na mesma base. A exclusividade do **LDA-tweets** também mudou de método:
> word-overlap 0,817 → **c-TF-IDF 0,512**.
>
> Os números medidos aqui **não foram reescritos**: são o registro do run analisado, e a
> leitura qualitativa em volta deles é propriedade desse run. Para citar valores, use a
> tabela autoritativa em `docs/tabela2-reconstruida-2026-08-13.md`.


> **Corpus:** Folha de São Paulo 2018-2024, PT-BR, textos jornalísticos longos
> **Notebook:** `04_stm_folha.ipynb`
> **Configuração:** `03-topic-modeling/configs/params.yaml`
> **Última revisão:** 2026-07-16: resultados do run de produção `folha_20260716_174430` (corpus `folha_20260628_185652`, pipeline completo, execução ponta a ponta); hiperparâmetros pinados no `params.yaml` (grids desligados)

---

## Índice

1. [O que é STM — conceitos fundamentais](#1-o-que-é-stm--conceitos-fundamentais)
2. [Arquitetura do pipeline](#2-arquitetura-do-pipeline)
3. [Corpus e estratégia 5k](#3-corpus-e-estratégia-5k)
4. [Lematização e vocabulário](#4-lematização-e-vocabulário)
5. [Seleção de K — grid search C_v + diagnósticos nativos](#5-seleção-de-k--grid-search-c_v--diagnósticos-nativos)
6. [Grid sigma.prior × gamma.prior](#6-grid-sigmaprior--gammaprior)
7. [Treino final](#7-treino-final)
8. [Nomeação de tópicos via LLM](#8-nomeação-de-tópicos-via-llm)
9. [Métricas de avaliação](#9-métricas-de-avaliação)
10. [Resultados da rodada de produção](#10-resultados-da-rodada-de-produção-folha_20260716_174430)
11. [Efeitos de prevalência (estimateEffect) — o diferencial do STM](#11-efeitos-de-prevalência-estimateeffect--o-diferencial-do-stm)
12. [Comparação com LDA, NMF e BERTopic](#12-comparação-com-lda-nmf-e-bertopic)
13. [Configuração final](#13-configuração-final)
14. [Arquivos de saída](#14-arquivos-de-saída)
15. [Pendências](#15-pendências-status-2026-07-16)

---

## 1. O que é STM — conceitos fundamentais

STM (Structural Topic Model, Roberts, Stewart & Airoldi 2016) é um modelo probabilístico de tópicos da família do LDA que incorpora **metadados dos documentos (covariáveis) na estrutura do modelo**. No uso deste projeto, as covariáveis entram na **prevalência**: a proporção esperada de cada tópico num documento deixa de vir de um prior Dirichlet global (como o α do LDA) e passa a ser função das covariáveis do documento, via prior logístico-normal:

```
θ_d ~ LogisticNormal( X_d · Γ , Σ )
```

- **X_d** — covariáveis do documento (aqui: data da publicação + editoria)
- **Γ** — coeficientes de prevalência estimados (regularizados por `gamma.prior`)
- **Σ** — covariância entre tópicos (regularizada por `sigma.prior`); permite tópicos correlacionados, diferente do Dirichlet do LDA que os assume ~independentes

Depois do treino, o `estimateEffect` faz regressão dos θ nas covariáveis com incerteza propagada do modelo — respondendo diretamente "**quanto a prevalência do tópico T varia com a editoria/data?**", com erro-padrão e p-valor. Esse é o diferencial do STM na triangulação: nenhum dos outros três modelos produz inferência sobre metadados.

A implementação é o pacote R `stm`, chamado por subprocess (`scripts/run_stm.R`, modos `grid_k`/`grid_hparams`/`train`). O Python é dono do protocolo: prepara o input, dispara o R e pontua os resultados com **as mesmas funções de C_v/exclusividade/diversidade dos outros modelos**.

### Diferenças em relação a LDA, NMF e BERTopic

| Aspecto | STM | LDA | NMF | BERTopic |
|---------|-----|-----|-----|----------|
| Natureza | Generativo probabilístico + covariáveis | Generativo probabilístico | Fatoração de matrizes | Embeddings + clustering |
| Prior de θ | Logístico-normal condicionado em X_d | Dirichlet (α) | — | — |
| Hiperparâmetros | `sigma.prior`, `gamma.prior` | `alpha`, `eta` | `kappa`, `min_probability` | UMAP/HDBSCAN |
| K | Fixo (grid search) | Fixo (grid search) | Fixo (grid search) | Emergente |
| Inicialização | **Spectral (determinística — sem multi-seed)** | Aleatória (seed) | Aleatória (seed) | UMAP (seed) |
| Outliers | 0% (docs vazios pós-filtro são removidos pelo `prepDocuments`) | 0% | 0% | Docs podem ser outliers |
| Diagnóstico held-out | **heldout likelihood nativo** | perplexidade in-sample | não exposto | — |
| Efeito de covariáveis | **`estimateEffect` (coef., EP, p-valor)** | — | — | — |
| pyLDAvis | Não integrado | Disponível | Não aplicável | — |

### Por que STM além dos outros três?

Fecha a triangulação com um **quarto paradigma**: tópicos condicionados a metadados. Para a dissertação, o STM transforma "o tópico X parece concentrado na editoria Y / no período Z" (leitura visual de cross-tabs nos outros modelos) em **estimativa estatística com incerteza** (§11).

---

## 2. Arquitetura do Pipeline

Espelha o protocolo de duas etapas do LDA/NMF, com o treino delegado ao R:

```
corpus_limpo.csv (resolve_latest_dir — latest-wins, sem cópia)
        │
        ▼
[Cell 3]  load_corpus() + subsample 5k (seed=42)
          • Versão do corpus: folha_20260628_185652
          • 4.939 docs → −1 (covariável NA) → −86 (< 200 tokens) = 4.852 docs
        │
        ▼
[Cell 5]  lemmatize_corpus(model_key="stm")
          • spaCy pt_core_news_lg (CPU, ~412s)
          • filter_extremes(no_below=5, no_above=0.5) — bloco stm: (= lda:/nmf:)
          • Vocabulário final: 17.290 palavras (17.285 após prepDocuments do R)
          • Média: 382,7 tokens/doc (243,8 únicos)
        │
        ▼
[Cell 7]  prepare_stm_input()
          • texto lematizado (join dos tokens do C_v) + covariáveis
          • colunas: text, post_id, date (renomeada de 'data'), category
          • → o vocabulário do STM é o MESMO BoW do LDA/NMF por construção
        │
        ▼
[Cell 8]  grid_search_k_stm()  [R: mode=grid_k, 1 fit por K, split heldout]
          • K ∈ {3,5,7,8,10,12,15,20,25,30}; seed=42; ~22min total
          • R devolve topics + diagnósticos; Python pontua C_v no BoW gensim
          • Cache: stm_metrics.csv (k_grid_scores) + stm_grid_k_diagnostics.csv
          • BEST_K = peak C_v → 25
        │
        ▼
[Cell 12] grid_search_stm_hparams()  [R: mode=grid_hparams, K=25 fixo]
          • sigma.prior {0, 0.3, 0.5, 0.7} × gamma.prior {Pooled, L1} = 8 fits
          • Cache: stm_sigma_gamma_grid.csv → vencedor sigma=0.7, gamma=L1
          • train_stm(K=25, sigma=0.7, gamma=L1) — 357s, 92 iterações EM
          • estimateEffect roda no mesmo processo R e devolve os efeitos
        │
        ▼
[Nomeação via LLM] (mesmo pipeline do LDA/NMF)
        │
        ▼
[Cells 14+] Wordclouds, bar charts, heatmap phi (beta), similaridade
            cosseno tópico×tópico, métricas, θ heatmap, t-SNE,
            cross-tab categoria×tópico, evolução temporal,
            efeitos de prevalência (forest plot + curva temporal)
        │
        ▼
[Cell 23] export_results() + export_topics_for_eval() + stm_metrics.csv
          + stm_prevalence_effects.csv + stm_topics_frex.csv
```

> **Pin de hiperparâmetros (2026-07-16):** `stm_best_k: 25`, `stm_sigma_prior: 0.7`, `stm_gamma_prior: "L1"` no bloco `folha:` do `params.yaml` — os dois grids ficam **desligados** em re-runs (as células caem direto nos valores pinados; remover as chaves reativa os grids, que ainda têm cache).

---

## 3. Corpus e Estratégia 5k

Mesmo subsample de 5k docs com `np.random.RandomState(42)` dos outros três modelos, com **dois filtros adicionais exigidos pelo STM**:

1. **Covariáveis sem NA** (matriz de desenho do `stm`/`estimateEffect` não aceita NA): −1 doc.
2. **`stm_min_tokens_per_doc: 200`** (tokens brutos): −86 docs. O STM degrada com docs muito curtos; em corpus longo como a Folha o custo é ~1,7% dos docs.

- N final: **4.852 docs** (vs 4.939 do LDA/NMF — diferença de 87 docs, mesma família de subsample)
- Comprimento: média 801,4 tokens brutos/doc (mediana ~723; mínimo 200 por filtro)
- Corpus resolvido via `resolve_latest_dir`: **`folha_20260628_185652`** (o mesmo dos runs definitivos de LDA/NMF/BERTopic)

---

## 4. Lematização e Vocabulário

Mesma `lemmatize_corpus` dos outros modelos, com `model_key="stm"` (bloco `stm:` de `no_below`/`no_above` — valores idênticos a `lda:`/`nmf:` por design):

| Métrica | STM | NMF (ref.) | LDA (ref.) |
|---|---|---|---|
| Tempo de lematização | 412s | 349s | ~448s |
| Vocabulário final | **17.290** (17.285 no R) | 17.331 | 17.303 |
| Tokens/doc após lematizar (média) | 382,7 | 377,1 | 376,4 |
| Tokens únicos/doc (média) | 243,8 | 240,5 | 240,1 |

> As diferenças de ~20-40 palavras entre vocabulários vêm de execuções independentes (empates no `filter_extremes`) e dos 87 docs a menos do STM; ruído sem impacto prático. O que vai ao R é o **texto já lematizado** (join dos tokens) — o `prepDocuments` do R remove 5 termos por corte próprio (17.290 → 17.285) e **nenhum documento** neste corpus.

---

## 5. Seleção de K — Grid Search C_v + diagnósticos nativos

`grid_search_k_stm`: o R treina 1 STM por K (com split heldout interno), o Python pontua **C_v no mesmo BoW lematizado do LDA/NMF** — K é escolhido por C_v (comparável entre os 4 modelos); os diagnósticos nativos do STM ficam como apoio.

| K | C_v | Heldout lik. | Dispersão resíduos | Iterações | Tempo |
|---|-----|--------------|--------------------|-----------|-------|
| 3 | 0,489 | −8,190 | 5,38 | 84 | 88s |
| 5 | 0,510 | −8,088 | 4,65 | 34 | 47s |
| 7 | 0,593 | −8,019 | 4,24 | 41 | 63s |
| 8 | 0,654 | −7,980 | 4,09 | 30 | 55s |
| 10 | 0,644 | −7,929 | 3,80 | 62 | 100s |
| 12 | 0,650 | −7,904 | 3,64 | 49 | 96s |
| 15 | 0,662 | −7,864 | 3,45 | 92 | 176s |
| 20 | 0,673 | −7,819 | 3,18 | 95 | 229s |
| **25** | **0,677** | −7,800 | 2,94 | 59 | 193s |
| 30 | 0,668 | −7,771 | 2,85 | 79 | 282s |

**BEST_K = 25** (peak C_v). Leituras:

- O **plateau K=15-30 (0,66-0,68)** espelha LDA (pico 20) e NMF (pico 25): granularidade da Folha satura em ~2-3 sub-tópicos por editoria. O STM concordou com o NMF em K=25.
- Os diagnósticos nativos **melhoram monotonicamente com K** (heldout ↑, resíduos ↓) — sozinhos escolheriam K=30+; o C_v é que impõe o custo de interpretabilidade. Mesmo trade-off documentado na literatura do stm.
- Grid completo: ~22min de fits R (todas as combinações convergiram dentro do teto de 150 iterações da época).

---

## 6. Grid sigma.prior × gamma.prior

Análogo de protocolo ao alpha×eta do LDA e kappa×min_prob do NMF: com K=25 fixo, varre os dois hiperparâmetros de prevalência do STM.

### O que são

- **`sigma.prior`** ∈ [0,1] — regularização da matriz Σ (covariância entre tópicos induzida pelas covariáveis). 0 = sem regularização; 1 = Σ diagonal.
- **`gamma.prior`** — prior dos coeficientes Γ de prevalência: `Pooled` (normal hierárquica) ou `L1` (esparso, via glmnet).

### Resultados (8 combinações, K=25)

| sigma.prior | gamma.prior | C_v |
|-------------|-------------|-----|
| **0.7** | **L1** | **0,6799 ← melhor** |
| 0.5 | Pooled | 0,6798 |
| 0.3 | Pooled | 0,6796 |
| 0.3 | L1 | 0,6796 |
| 0.7 | Pooled | 0,6795 |
| 0.0 | L1 | 0,6774 |
| 0.5 | L1 | 0,6773 |
| 0.0 | Pooled | 0,6764 |

**Escolhido: `sigma.prior=0.7, gamma.prior=L1`.** Leituras:

1. **Spread mínimo (0,0035)** — o STM da Folha é pouco sensível a sigma/gamma; como no LDA (spread alpha/eta ~0,013) e NMF (kappa ~0,02), o grid de hiperparâmetros é refinamento marginal frente à seleção de K (spread ~0,19). Padrão consistente nos 3 modelos BoW.
2. O único sinal estrutural: **`sigma.prior=0` ocupa o fundo da tabela** — alguma regularização da covariância entre tópicos sempre ajudou neste corpus (10 editorias correlacionadas).
3. ⚠ Não transfere entre corpora: nos tweets o vencedor foi `sigma=0.0` (ver `PIPELINE_TWEETS_STM.md`) — mesmo aviso do kappa do NMF.

---

## 7. Treino Final

```python
keywords, keywords_frex, theta, beta_df, effects, stm_meta = train_stm(
    stm_df, k=25,
    prevalence_formula="~ s(as.numeric(as.Date(date))) + category",
    stm_cfg=params["stm"], work_dir=out_dir,
    sigma_prior=0.7, gamma_prior="L1", seed=42,
)
```

- **357s, 92 iterações EM** (convergiu bem abaixo do teto `max_em_its: 300`)
- Init **Spectral**: determinístico — dois runs (`_162249` e `_174430`, 2026-07-16) reproduziram exatamente as mesmas métricas (C_v 0,6799, exclus. 0,5283, TD 0,804)
- A fórmula de prevalência usa **spline da data** (`s(as.numeric(as.Date(date)))`, 10 bases) + **editoria** (`category`, 9 dummies vs baseline `ambiente`)
- `prepDocuments` do R não removeu nenhum documento (docs longos, vocabulário denso)
- O treino devolve θ (4.852×25), β (25×17.285), keywords por probabilidade **e por FREX nativo do R**, e os efeitos do `estimateEffect`

---

## 8. Nomeação de Tópicos via LLM

Pipeline idêntico ao do LDA/NMF: 50 keywords por tópico + 3 docs representativos por θ, prompt PT-BR few-shot com anti-redundância, modelo `gemma2:2b-instruct-q4_K_M` (de `params.yaml > llm.model`), 25 tópicos nomeados.

> ⚠ **Nomes do LLM não são determinísticos entre runs** (o treino é; a nomeação não): `_162249` nomeou T4 "Eleições e política nacional", `_174430` nomeou "Eleições e governo federal" — mesmo tópico, mesmas keywords. Citar sempre o run canônico (`_174430`).

---

## 9. Métricas de Avaliação

Mesmas funções de `_helpers.py` dos outros modelos, aplicadas às keywords do STM:

- **C_v** — recomputada no modelo final sobre o BoW lematizado gensim → **diretamente comparável com LDA/NMF** (mesmo vocabulário, mesma função).
- **Exclusividade c-TF-IDF** (`compute_exclusivity_ctfidf`) — mesma escala de LDA/NMF/BERTopic na Folha.
- **Topic Diversity** (Dieng) e **diversity entropy** (θ) — idem.
- **FREX (`compute_frex_score`)** — ⚠ saturada (0,983), mesma suspeita de bug dos outros modelos; **não reportar sem auditar**. O STM tem, além dela, o **FREX nativo do R** (keywords `stm_topics_frex.csv`) — útil qualitativamente, mas é outra base de cálculo.
- **Perplexidade** — não aplicável (a *heldout likelihood* do grid cumpre o papel de diagnóstico preditivo).
- **Estabilidade Jaccard** — não aplicável: init Spectral é determinística (não existe variação por seed a medir).

---

## 10. Resultados da Rodada de Produção (`folha_20260716_174430`)

> Run de 2026-07-16: K=25, sigma.prior=0.7, gamma.prior=L1, corpus `folha_20260628_185652`, 4.852 docs, 92 iterações EM. Execução ponta a ponta (nbclient); reproduz exatamente as métricas do run `_162249` do mesmo dia (Spectral determinístico).

### Métricas

| Métrica | STM `174430` | LDA `004450` | NMF `000631` | BERTopic `221412` | Leitura |
|---|---|---|---|---|---|
| **C_v (recomputado)** | **0,680** | 0,657 | 0,655 | 0,620 | **Melhor C_v da Folha entre os 4 modelos** — e sobre o mesmo BoW do LDA/NMF (comparação limpa) |
| Exclusividade (c-TF-IDF) | 0,528 | 0,457 | 0,475 | 0,542 | 2º lugar, quase alcançando o BERTopic; bem acima dos outros BoW |
| Topic Diversity (Dieng) | 0,804 | 0,770 | 0,74 | 0,892 | 2º lugar |
| Diversity entropy (θ) | 0,979 | — | 0,960 | — | θ bem distribuído |
| FREX (`compute_frex_score`) | 0,983 | 0,974 | 0,982 | 0,982 | ⚠ saturado — não reportar |
| Iterações EM | 92 (convergiu) | — | — | — | teto 300 |
| Tempo de treino final | 357s | ~5-10min | 18s | ~min | — |

25 tópicos, **0% outliers** (todo doc recebe tópico dominante; `prepDocuments` não descartou nenhum).

### 25 tópicos produzidos

| ID | Nome LLM | Keywords (top-5) | Docs | % |
|---|---|---|---|---|
| T0 | Polícia e violência policial no Brasil | polícia, policial, segurança, crime, caso | 171 | 3,5% |
| T1 | Futebol e Seleção Brasileira | clube, copa, jogo, futebol, time | 284 | 5,9% |
| T2 | Educação no Brasil e ensino superior | educação, escola, aluno, ensino, professor | 323 | 6,7% |
| T3 | Olimpíadas e esporte brasileiro | atleta, esporte, jogos, brasileiro, olímpico | 115 | 2,4% |
| T4 | Eleições e governo federal | bolsonaro, presidente, lula, governo, partido | 344 | 7,1% |
| T5 | Corrupção e Justiça no STF | federal, caso, justiça, tribunal, processo | 266 | 5,5% |
| T6 | Saúde e doenças no Brasil | saúde, médico, paciente, hospital, doença | 178 | 3,7% |
| T7 | Política de governo e orçamento | governo, projeto, lei, ministério, proposta | 212 | 4,4% |
| T8 | Tecnologia e empresas digitais no Brasil ⚠ | empresa, **seta**, milhão, mercado, brasil | 102 | 2,1% |
| T9 | Chuvas na região sul de São Paulo ⚠ | cidade, paulo, região, rio, prefeitura | 219 | 4,5% |
| T10 | Guerra entre EUA e Rússia | eua, china, americano, guerra, trump | 263 | 5,4% |
| T11 | Desmatamento na Amazônia e governo | desmatamento, amazônia, indígena, ambiental, brasil | 287 | 5,9% |
| T12 | Aquecimento global e degelo polar | climático, temperatura, mudança, cientista, planeta | 163 | 3,4% |
| T13 | Vida familiar e pandemia ⚠ | pessoa, casa, família, **de**, gente | 134 | 2,8% |
| T14 | Economia brasileira e inflação | bilhão, economia, banco, mercado, preço | 254 | 5,2% |
| T15 | Paleontologia e descobertas de fósseis | museu, antigo, encontrar, espécie, milhão | 132 | 2,7% |
| T16 | Consumo e produção de alimentos no Brasil | água, alimento, produto, óleo, carne | 54 | 1,1% |
| T17 | Missões espaciais e lua | espacial, missão, lua, voo, nasa | 118 | 2,4% |
| T18 | Arte brasileira e artistas | livro, obra, artista, brasil, arte | 186 | 3,8% |
| T19 | Pandemia no Brasil e dados da Covid-19 | vacina, saúde, dose, caso, pandemia | 267 | 5,5% |
| T20 | Protestos e democracia venezuelana | protesto, governo, político, presidente, pessoa | 127 | 2,6% |
| T21 | Prêmios de cinema e música hollywoodianos | filme, cinema, música, série, show | 263 | 5,4% |
| T22 | Pesquisa científica e desenvolvimento de tratamentos | pesquisa, estudo, pesquisador, universidade, científico | 141 | 2,9% |
| T23 | Desinformação e fake news no Brasil ⚠ | social, negro, mulher, pessoa, rede | 182 | 3,8% |
| T24 | Pantanal e incêndio | animal, espécie, humano, peixe, pesquisador | 67 | 1,4% |

Distribuição equilibrada: do menor (T16, 1,1%) ao maior (T4, 7,1%) — mais próxima do LDA que do NMF (que concentrou 10,9% num tópico).

### Qualidade dos tópicos

**Limpos (~21 de 25):** T0-T7, T10-T12, T14-T22, T24 mapeiam com clareza as editorias da Folha e seus sub-temas (política, justiça, saúde/Covid, educação, esporte×2, geopolítica, ambiente×2+clima, economia, ciência×2, cultura×2, espaço, cotidiano urbano).

**Com ressalvas:**
- **T8** — `seta` (2ª keyword): resíduo de boilerplate de site, o mesmo token estrutural documentado no LDA (T19) e NMF (T9). Diferente do NMF (que isolou o boilerplate num fator próprio), o STM o **diluiu como keyword espúria dentro de um tópico temático** — comportamento igual ao do LDA. Causa estrutural conhecida: `stopwords_emojis` não alcança o BoW gensim.
- **T9** — nome do LLM restritivo demais ("Chuvas na região sul de São Paulo"): as keywords (cidade, prefeitura, município) apontam para **gestão urbana/cotidiano de SP** em geral; chuvas são um sub-evento. Usar com nome ajustado na análise.
- **T13** — `de` (4ª keyword): quase-stopword vazada pela lematização (mesma classe do `de` no NMF T14 do LDA); tópico de crônicas/cotidiano, coeso apesar do ruído.
- **T23** — leve grab-bag: mistura pauta identitária (negro, mulher) com redes sociais/desinformação. Núcleo real existe (checagem/redes), mas menos nítido que os demais.

---

## 11. Efeitos de Prevalência (estimateEffect) — o diferencial do STM

`stm_prevalence_effects.csv`: 500 linhas = 25 tópicos × 20 termos (intercepto + 10 bases da spline de data + 9 dummies de editoria, baseline = `ambiente`). **105 dos 475 coeficientes não-intercepto são significativos a p<0,05.**

### Validação estrutural: os efeitos recuperam as editorias

Os maiores coeficientes são exatamente os pareamentos tópico↔editoria esperados:

| Tópico | Termo | Coef. | p | Leitura |
|---|---|---|---|---|
| T1 Futebol | `esporte` | **+0,418** | ~0 | +42 p.p. de prevalência esperada vs baseline |
| T2 Educação | `educacao` | **+0,391** | ~0 | idem |
| T11 Desmatamento | todas as outras editorias | **−0,32 a −0,33** | ~0 | tópico concentrado na baseline `ambiente` — o sinal aparece como coeficientes negativos das demais |

Isso é o cross-tab tópico×editoria dos outros modelos **transformado em inferência estatística**: coeficiente, erro-padrão e p-valor por par tópico×editoria, com incerteza propagada do modelo (não de contagens pontuais).

### Efeito temporal

A spline da data (10 bases por tópico) captura tendências suaves 2018-2024 — visualizadas em `stm_prevalence_temporal.png` (curvas de prevalência esperada por tópico ao longo do tempo; ex.: T19 Covid concentrado em 2020-2021). Para a dissertação, é a versão com IC da "evolução temporal" que o notebook também plota descritivamente (`stm_topics_over_time.png`).

> **Forest plot:** `stm_prevalence_forest.png` — coeficientes categóricos (exclui intercepto e bases de spline) com IC95%, por tópico.

---

## 12. Comparação com LDA, NMF e BERTopic

| Dimensão | STM (K=25) | LDA (K=20) | NMF (K=25) | BERTopic (K=24) |
|----------|-----------|-----------|-----------|----------------|
| Corpus de treino | 4.852 docs (5k − filtros) | 4.939 | 4.939 | N completo |
| C_v | **0,680** | 0,657 | 0,655 | 0,620 |
| Exclusividade (c-TF-IDF) | 0,528 | 0,457 | 0,475 | **0,542** |
| Topic Diversity | 0,804 | 0,770 | 0,74 | **0,892** |
| Outliers | 0% | 0% | 0% | 21,7% |
| θ | Suave (logístico-normal) | Suave (Dirichlet) | Esparso | Hard |
| Reprodutibilidade | **Determinística (Spectral)** | por seed | por seed | por seed |
| Efeitos de covariáveis | **✅ estimateEffect** | — | — | — |
| Tempo de treino final | 357s | ~5-10min | **18s** | ~min |

### O que o STM capturou de diferente

- **Melhor C_v da Folha entre os 4 modelos (0,680)** — sobre o mesmo BoW lematizado do LDA/NMF, o que torna a comparação limpa. A leitura provável: condicionar a prevalência em data+editoria "desonera" as palavras de explicar sozinhas a estrutura documental, deixando os tópicos lexicalmente mais coesos.
- **Único modelo com inferência sobre metadados** (§11) — o argumento de uso do STM na dissertação não é o ranking de C_v, e sim responder estatisticamente às perguntas de prevalência.
- Determinismo do Spectral: substitui a discussão de estabilidade multi-seed por reprodutibilidade exata.

---

## 13. Configuração Final

### params.yaml (seções relevantes para o STM)

```yaml
corpora:
  folha:
    stm_prevalence_formula: "~ s(as.numeric(as.Date(date))) + category"
    stm_min_tokens_per_doc: 200
    # STM pinado 2026-07-16 (run folha_20260716_162249): grids concluídos e
    # desligados — remover as 3 chaves reativa os grids (com cache)
    stm_best_k: 25
    stm_sigma_prior: 0.7
    stm_gamma_prior: "L1"

stm:
  no_below: 5          # = lda:/nmf:
  no_above: 0.5
  max_em_its: 300      # subido de 150 em 2026-07-16 (tweets batia no teto)
  timeout_sec: 21600
  rscript_path: "Rscript"

evaluation:
  stm_sigma_prior_grid: [0, 0.3, 0.5, 0.7]
  stm_gamma_prior_grid: ["Pooled", "L1"]
```

### Decisões fixas

| Parâmetro | Valor | Justificativa |
|-----------|-------|---------------|
| `BEST_K` | **25 (pinado)** | peak C_v do grid; plateau 15-30 |
| `sigma.prior` | **0.7 (pinado)** | melhor C_v; sigma=0 foi o pior — específico da Folha |
| `gamma.prior` | **L1 (pinado)** | melhor C_v (margem mínima sobre Pooled) |
| Fórmula de prevalência | spline(data) + editoria | covariáveis disponíveis end-to-end no corpus |
| `stm_min_tokens_per_doc` | 200 | STM degrada com docs curtos; custo ~1,7% dos docs |
| Init | Spectral | determinístico, recomendação padrão do stm |

---

## 14. Arquivos de Saída

Cada execução gera `data/output/folha/stm/folha_<AAAAMMDD>_<HHMMSS>/`:

| Arquivo | Conteúdo |
|---------|----------|
| `stm_results.csv` | post_id, doc_id, text, topic_id, topic_name, distribuição θ completa |
| `stm_topics_for_eval.csv` | keywords (por probabilidade) + nomes LLM por tópico |
| `stm_topics_frex.csv` | keywords por **FREX nativo do R** (alternativa qualitativa) |
| `stm_metrics.csv` | métricas agregadas + K/sigma/gamma/em_iterations/corpus_version |
| `stm_theta.csv` / `stm_beta.csv` | matrizes θ (docs×K) e β (K×vocab) |
| `stm_prevalence_effects.csv` | **estimateEffect: tópico × termo × coef × EP × p-valor** |
| `stm_prevalence_forest.png` / `stm_prevalence_temporal.png` | visualizações dos efeitos |
| `stm_heatmap_phi.png`, `stm_topic_similarity.png` | heatmaps β e cosseno tópico×tópico |
| `stm_topic_category.png`, `stm_topics_over_time.png` | cross-tab editoria e evolução temporal descritiva |
| `stm_tsne_theta_topico.png` / `stm_tsne_theta_interactive.html` | t-SNE do espaço θ |
| `stm_input.csv`, `stm_grid_k.json`, `stm_hparams.json`, `stm_final.json` | input e artefatos brutos da ponte Python↔R |

**Caches (diretório-base `data/output/folha/stm/`):** `stm_metrics.csv` (k_grid_scores), `stm_grid_k_diagnostics.csv`, `stm_sigma_gamma_grid.csv`.

---

## Notas de execução

```bash
# 1. Ollama rodando (naming LLM) + R >= 4.4 com stm/jsonlite/glmnet no PATH
ollama serve

# 2. Executar notebook completo (04_stm_folha.ipynb) — Run All

# 3. Tempos observados (run 174430, com pins/caches):
#    - Lematização:            ~412s (7min)
#    - Grid K:                 pinado/cache (recomputar: ~22min de fits R)
#    - Grid sigma×gamma:       pinado/cache (recomputar: ~8 fits R)
#    - Treino final + effects: ~357s (6min, 92 iterações EM)
#    - Naming LLM:             ~min (25 tópicos)
#    - Total (com pins):       ~16min
```

---

## 15. Pendências (status 2026-07-16)

| Item | Status |
|---|---|
| Grid K + grid sigma×gamma | ✅ completos, cacheados e **pinados** no params.yaml |
| Run de produção ponta a ponta | ✅ `folha_20260716_174430` (reproduz `_162249` exatamente) |
| Efeitos de prevalência exportados | ✅ 105/475 coef. significativos; validação estrutural das editorias |
| **T8 `seta` / T13 `de` (ruído estrutural no BoW)** | ⚠ mesma pendência estrutural do LDA/NMF (`stopwords_emojis` não alcança o Dictionary gensim) |
| **T9 nome do LLM restritivo** | ⚠ ajustar nome na análise qualitativa ("gestão urbana paulista") |
| T23 leve grab-bag | ⚠ ressalva na análise |
| Nomes LLM não determinísticos entre runs | ℹ documentado (§8): citar sempre o run canônico |
| FREX (`compute_frex_score`) saturado | ⚠ aberto (compartilhado com os 4 modelos); o STM tem FREX nativo do R como alternativa qualitativa |
| Estabilidade Jaccard | ➖ não aplicável (Spectral determinístico) |

---

*Documentação gerada em 2026-07-16 a partir do run `folha_20260716_174430` (corpus `folha_20260628_185652`). Espelha a estrutura de `../nmf/PIPELINE_NMF_FOLHA.md` e `../lda/PIPELINE_LDA_FOLHA.md`.*
