# NMF — Folha de São Paulo: Documentação Completa do Pipeline

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
> **Notebook:** `03_nmf_folha.ipynb`
> **Configuração:** `03-topic-modeling/configs/params.yaml`
> **Última revisão:** 2026-07-14: resultados do run de produção `folha_20260714_000631` (corpus `folha_20260628_185652`, pipeline completo); T4 e T17 renomeados manualmente (nomes do LLM incorretos — ver §10)

---

## Índice

1. [O que é NMF — conceitos fundamentais](#1-o-que-é-nmf--conceitos-fundamentais)
2. [Arquitetura do pipeline](#2-arquitetura-do-pipeline)
3. [Corpus e estratégia 5k](#3-corpus-e-estratégia-5k)
4. [Lematização e vocabulário](#4-lematização-e-vocabulário)
5. [Seleção de K — grid search C_v](#5-seleção-de-k--grid-search-c_v)
6. [Grid kappa × minimum_probability](#6-grid-kappa--minimum_probability)
7. [Treino final](#7-treino-final)
8. [Nomeação de tópicos via LLM](#8-nomeação-de-tópicos-via-llm)
9. [Métricas de avaliação](#9-métricas-de-avaliação)
10. [Resultados da rodada de produção](#10-resultados-da-rodada-de-produção-folha_20260714_000631)
11. [Comparação com LDA e BERTopic](#11-comparação-com-lda-e-bertopic)
12. [Configuração final](#12-configuração-final)
13. [Arquivos de saída](#13-arquivos-de-saída)
14. [Pendências](#14-pendências-status-2026-07-14)

---

## 1. O que é NMF — conceitos fundamentais

NMF (Non-negative Matrix Factorization, Lee & Seung 1999) fatoriza a matriz documento-termo **V** (n_docs × vocab) em duas matrizes não-negativas de posto K:

```
V ≈ W · H
```

- **W** (n_docs × K): peso de cada tópico em cada documento — papel análogo ao θ do LDA
- **H** (K × vocab): peso de cada palavra em cada tópico — papel análogo ao φ do LDA

A restrição de não-negatividade força uma **decomposição aditiva por partes**: cada documento é uma soma ponderada de "partes" (tópicos), sem cancelamentos. É isso que torna os fatores interpretáveis como tópicos, diferente de SVD/LSA (que admite pesos negativos).

A implementação usada é `gensim.models.Nmf` (NMF *online*, Zhao & Tan 2016), que treina por mini-lotes com atualizações multiplicativas — mesma interface de `LdaMulticore` (corpus BoW + Dictionary), o que permite reusar todo o ferramental do LDA (`show_topic`, `get_topics`, `get_document_topics`).

### Diferenças em relação a LDA e BERTopic

| Aspecto | NMF | LDA | BERTopic |
|---------|-----|-----|----------|
| Natureza | Álgebra linear (fatoração) | Modelo generativo probabilístico | Embeddings + clustering |
| Priors | Nenhum (sem Dirichlet) | α (doc→tópico), η (palavra→tópico) | — |
| Hiperparâmetros | `kappa`, `minimum_probability` | `alpha`, `eta` | UMAP/HDBSCAN |
| K | Fixo (grid search) | Fixo (grid search) | Emergente (HDBSCAN) |
| Outliers | 0% (W nunca é toda zero na prática) | 0% | Docs podem ser outliers (id=-1) |
| Perplexidade | **Não exposta** pelo `Nmf` | `log_perplexity` disponível | — |
| pyLDAvis | Não aplicável | Disponível | — |
| θ (doc→tópico) | Linhas de W normalizadas (frequentemente **esparsas**: docs com P≈1.0 num único tópico) | Distribuição suave (todo tópico > 0) | Hard assignment |

### Por que NMF além de LDA e BERTopic?

Completa a **triangulação metodológica** da dissertação com um terceiro paradigma: fatoração de matrizes (NMF) × inferência bayesiana (LDA) × embeddings semânticos (BERTopic). NMF tende a produzir tópicos mais "esparsos" e nítidos que o LDA no mesmo BoW, ao custo de não ter interpretação probabilística formal.

---

## 2. Arquitetura do Pipeline

Espelha célula a célula o `02_lda_folha.ipynb`, trocando o modelo e o grid de hiperparâmetros:

```
corpus_limpo.csv (resolve_latest_dir — latest-wins, sem cópia)
        │
        ▼
[Cell 3]  load_corpus() + subsample 5k (seed=42)
          • Versão do corpus: folha_20260628_185652
          • N=4.939 docs (subsample seed=42, padronizado com LDA/BERTopic)
        │
        ▼
[Cell 5]  lemmatize_corpus(model_key="nmf")
          • spaCy pt_core_news_lg (CPU, ~349s)
          • filter_extremes(no_below=5, no_above=0.5) — bloco nmf: do params.yaml
          • Vocabulário final: 17.331 palavras
          • Média: 377.1 tokens/doc (240.5 únicos)
        │
        ▼
[Cell 7]  corpus_bow = [dictionary.doc2bow(d) for d in tokenized]
          • Média: 219.2 tokens únicos/doc no BoW
        │
        ▼
[Cell 8]  grid_search_k_nmf()
          • K ∈ {3, 5, 7, 8, 10, 12, 15, 20, 25, 30}; passes=20; seed=42
          • Cache: BASE_OUTPUT_DIR/nmf_metrics.csv (coluna k_grid_scores)
          • BEST_K = peak C_v (automático) → 25
        │
        ▼
[Cell 12] grid_search_nmf_hparams()
          • K=25 fixo; passes=10
          • 3 kappas × 3 minimum_probability = 9 combinações
          • Cache: BASE_OUTPUT_DIR/nmf_kappa_minprob_grid.csv
          • Treino final: train_nmf(K=25, kappa=2.0, min_prob=0.0, passes=20) — 18s
        │
        ▼
[Nomeação via LLM] (ANTES das visualizações — mesma ordem corrigida do LDA):
          • top_n_keywords_llm=50 keywords por tópico
          • 3 docs representativos por probabilidade theta
          • Prompt PT-BR few-shot + anti-redundância + TEMA GERAL
          • Deduplicação + ordering por tamanho
        │
        ▼
[Cells 13-14] Wordclouds + bar charts
        │
        ▼
[Cell 16] Métricas: C_v, Exclusividade c-TF-IDF, Diversity, FREX
          (SEM perplexidade — Nmf não expõe log_perplexity)
        │
        ▼
[Cells 17-22] c_v por tópico, estabilidade Jaccard (desativada), theta heatmap,
              t-SNE, cross-tab categoria×tópico
          (SEM pyLDAvis — não aplicável a NMF)
        │
        ▼
[Cell 23] export_results() + export_topics_for_eval() + nmf_metrics.csv
```

---

## 3. Corpus e Estratégia 5k

Idêntica ao LDA (`PIPELINE_LDA_FOLHA.md` §3): subsample de 5k documentos com `np.random.RandomState(42)` — **mesmos índices** do LDA e do sample de C_v do BERTopic, garantindo métricas diretamente comparáveis entre os três modelos.

- N final: **4.939 docs**, 10 editorias balanceadas (~482-500 docs cada)
- Comprimento: média 790 tokens brutos/doc (mediana 716)
- Corpus resolvido via `resolve_latest_dir` do output do 01-preprocessing: **`folha_20260628_185652`** (com `html.unescape` aplicado — mesmo corpus do run definitivo do LDA)

---

## 4. Lematização e Vocabulário

Mesma função `lemmatize_corpus` do LDA, com `model_key="nmf"` (lê o bloco `nmf:` de `no_below`/`no_above` do `params.yaml` — valores idênticos ao bloco `lda:`, por design, para comparabilidade):

| Métrica | NMF | LDA (referência) |
|---|---|---|
| Tempo de lematização | 349s | ~448s |
| Vocabulário final | **17.331** palavras | 17.303 palavras |
| Tokens/doc após lematizar (média) | 377.1 | 376.4 |
| Tokens únicos/doc (média) | 240.5 | 240.1 |
| Tokens únicos/doc no BoW (média) | 219.2 | 218.8 |

> A diferença de ~30 palavras no vocabulário (17.331 vs 17.303) vem de execuções independentes da lematização (ordem de inserção no Dictionary + empates no `filter_extremes`); é ruído sem impacto prático — as estatísticas por documento são idênticas às do LDA.

---

## 5. Seleção de K — Grid Search C_v

Mesmo protocolo do LDA: `grid_search_k_nmf` treina um NMF por K e pontua por C_v (`passes=20`, seed=42). Diferente do LDA, **não há perplexidade complementar** (o `Nmf` do gensim não expõe `log_perplexity`).

| K | C_v | Interpretação |
|---|-----|---------------|
| 3 | 0.429 | Muito grosseiro |
| 5 | 0.449 | — |
| 7 | 0.557 | Salto: começa a diferenciar editorias |
| 8 | 0.536 | — |
| 10 | 0.552 | — |
| 12 | 0.529 | — |
| 15 | 0.573 | — |
| 20 | 0.586 | (K escolhido pelo LDA neste corpus) |
| **25** | **0.609** | **Peak global** |
| 30 | 0.601 | Marginal; leve queda |

**BEST_K = 25** (peak C_v automático). O plateau K=20-30 (0.586-0.609) espelha o comportamento do LDA (que também formou plateau nessa faixa, com pico em 20): a granularidade adequada para as 10 editorias da Folha fica em ~2-3 sub-tópicos por editoria. O NMF pede K um pouco maior que o LDA (25 vs 20) — consistente com sua tendência a tópicos mais esparsos/específicos, que "cabem" em maior número antes de fragmentar.

---

## 6. Grid kappa × minimum_probability

Análogo **de protocolo** (não de semântica) ao grid alpha×eta do LDA: com K fixo, varre os dois hiperparâmetros expostos pelo `gensim.models.Nmf`.

### O que são kappa e minimum_probability

- **`kappa`** — taxa de aprendizado das atualizações multiplicativas do NMF online. Controla o tamanho do passo do gradiente: maior = convergência mais agressiva.
- **`minimum_probability`** — limiar abaixo do qual pesos de tópico são zerados **na saída** de `get_document_topics`. É um filtro de inferência, não um parâmetro de treino.

### Resultados do grid (9 combinações, K=25, passes=10)

| kappa | minimum_probability | C_v |
|-------|---------------------|-----|
| **2.0** | **0.00** | **0.6088 ← melhor** |
| 2.0 | 0.01 | 0.6088 |
| 2.0 | 0.05 | 0.6088 |
| 1.0 | 0.00 | 0.5920 |
| 1.0 | 0.01 | 0.5920 |
| 1.0 | 0.05 | 0.5920 |
| 0.5 | 0.00 | 0.5902 |
| 0.5 | 0.01 | 0.5902 |
| 0.5 | 0.05 | 0.5902 |

**Escolhido: `kappa=2.0, minimum_probability=0.0`.**

### Leituras importantes

1. **`minimum_probability` não afeta C_v** (linhas idênticas dentro de cada kappa). Esperado: a métrica é calculada sobre as keywords de H, que não passam pelo limiar de saída de θ. O parâmetro é mantido no grid para espelhar o protocolo de dois hiperparâmetros do LDA, mas **o único knob efetivo é o kappa**. Na dissertação, reportar o grid como sensibilidade a kappa.
2. **kappa alto vence na Folha:** 2.0 > 1.0 > 0.5, spread de 0.019 (3.2%). Passos maiores levaram a fatoração a um ótimo local melhor neste corpus (textos longos, sinal de co-ocorrência denso). ⚠ Não generaliza: nos tweets o vencedor foi kappa=1.0 e kappa=2.0 foi o **pior** (ver `PIPELINE_TWEETS_NMF.md`).
3. Assim como no LDA, o grid de hiperparâmetros é **refinamento marginal** (spread 0.019) comparado à seleção de K (spread 0.180 entre K=3 e K=25).

---

## 7. Treino Final

```python
model = train_nmf(
    corpus_bow, dictionary,
    k=BEST_K,            # 25
    seed=42,
    passes=20,
    kappa=2.0,
    minimum_probability=0.0,
)
```

Treino em **18s** — ordem de grandeza mais rápido que o LDA (~5-10min com `passes=20`): as atualizações multiplicativas do NMF online são muito mais baratas que a inferência variacional do LDA.

---

## 8. Nomeação de Tópicos via LLM

Pipeline **idêntico ao do LDA** (`PIPELINE_LDA_FOLHA.md` §8), portado para o NMF:

- `top_n_keywords_llm=50` keywords por tópico (de `model.show_topic(tid, topn=50)` — matriz H)
- 3 docs representativos por maior probabilidade θ (`_representative_docs_nmf`, mesmo mecanismo do `_representative_docs_lda`)
- Prompt PT-BR few-shot (6 exemplos positivos + 1 negativo) + anti-redundância acumulativa + regra TEMA GERAL
- Deduplicação pós-geração; nomeação em ordem decrescente de tamanho do tópico
- Modelo: `gemma2:2b-instruct-q4_K_M` (de `params.yaml > llm.model`)

**Execução:** 25 tópicos nomeados a ~3.1s/tópico (~80s total).

---

## 9. Métricas de Avaliação

Todas compartilhadas com o LDA (mesmas funções de `_helpers.py`), aplicadas à matriz H (`model.get_topics()`) no lugar do φ:

- **C_v** — recomputada no modelo final (`passes=20`); o valor do grid é comparativo.
- **Exclusividade c-TF-IDF** (`compute_exclusivity_ctfidf`) — construída da matriz H **completa**, `top_n=20`, exatamente como LDA e BERTopic → **diretamente comparável** entre os três modelos.
- **Topic Diversity** (Dieng et al.) e **diversity entropy** (θ) — idem LDA.
- **FREX** — calculada, mas ⚠ **saturada (~0.98), mesma suspeita de bug de `compute_frex_score` já registrada no LDA/BERTopic. Não reportar na dissertação sem auditar.**
- **Perplexidade** — **não disponível** (o `Nmf` não expõe `log_perplexity`). A comparação NMF×LDA fica restrita a C_v/exclusividade/diversity.
- **Estabilidade Jaccard** — DESATIVADA por design (`stability_seeds=[42]`, sem seed alternativo), mesmo status do LDA. As células fazem skip explícito.

---

## 10. Resultados da Rodada de Produção (`folha_20260714_000631`)

> Run de 2026-07-14: K=25, kappa=2.0, minimum_probability=0.0, passes=20, corpus `folha_20260628_185652`. Substitui o run exploratório `folha_20260713_200848` (mesma configuração; diferenças de métrica entre os dois runs vêm de estado de execução parcial do notebook no primeiro — o `000631` é o run "Run All" canônico).

### Métricas

| Métrica | NMF `000631` | LDA `004450` | BERTopic `221412` | Leitura |
|---|---|---|---|---|
| **C_v (recomputado)** | **0.655** | 0.657 | 0.620 | NMF ≈ LDA (diferença de 0.002 é ruído); ambos > BERTopic — triangulação convergente |
| Exclusividade (c-TF-IDF) | 0.475 | 0.457 | 0.542 | NMF levemente mais exclusivo que LDA (esparsidade da fatoração); BERTopic segue líder |
| Topic Diversity (Dieng) | 0.74 | 0.770 | 0.892 | NMF um pouco menos diverso que LDA (K maior → mais overlap nas top-25 listas) |
| Diversity entropy (θ) | 0.960 | — | — | θ bem distribuído entre documentos |
| FREX | 0.982 | 0.974 | 0.982 | ⚠ saturado: **não reportar sem auditar** |
| Perplexidade | n/d | 3431 | — | Nmf não expõe |

25 tópicos, **0% outliers** (como o LDA, todo doc recebe tópico dominante).

### 25 tópicos produzidos

| ID | Nome LLM | Keywords (top-5) | Docs | % |
|---|---|---|---|---|
| T0 | Justica e investigação política | caso, federal, justiça, ministro, polícia | 383 | 7,8% |
| T1 | Produção industrial brasileira ⚠ | brasil, milhão, brasileiro, país, dado | 87 | 1,8% |
| T2 | Educação e currículos nacionais | educação, ensino, mec, nacional, recurso | 117 | 2,4% |
| T3 | Mudanças climáticas globais e energia | país, climático, mudança, energia, global | 217 | 4,4% |
| T4 | Política ambiental do governo federal ✏ | governo, desmatamento, ministério, projeto, público | 294 | 6,0% |
| T5 | Defesa da Amazônia e indígenas | indígena, terra, garimpo, desmatamento, região | 79 | 1,6% |
| T6 | Vacinação e imunização no Brasil | vacina, dose, saúde, vacinação, caso | 143 | 2,9% |
| T7 | Política Nacional e Eleições ⚠ | senhor, jair, william, candidato, brasil | 49 | 1,0% |
| T8 | Política Nacional e Governo Bolsonaro ⚠ | bolsonaro, presidente, jair, ministro, brasil | 96 | 1,9% |
| T9 | Museu e estudantes ⚠ | seta, universidade, esquerda, país, federal | 65 | 1,3% |
| T10 | Vestibular e Enem 2023 | prova, enem, ensino, exame, candidato | 97 | 2,0% |
| T11 | Ciência e desenvolvimento de tecnologias biológicas | estudo, pesquisa, pesquisador, universidade, humano | 411 | 8,3% |
| T12 | Mercados financeiros e crises empresariais | empresa, americanas, bilhão, banco, mercado | 333 | 6,7% |
| T13 | Eleições e partidos políticos no Brasil | partido, eleição, candidato, deputado, voto | 127 | 2,6% |
| T14 | Família e saúde em tempos de pandemia | pessoa, casa, gente, família, falar | 432 | 8,7% |
| T15 | Educação e retorno escolar em São Paulo | escola, aluno, aula, professor, ensino | 136 | 2,8% |
| T16 | Guerra Internacional e Rússia | país, americano, eua, guerra, trump | 536 | 10,9% |
| T17 | Gestão pública paulista ✏ | paulo, obra, doria, governador, paulista | 131 | 2,7% |
| T18 | Eleições e Lula 2022 | lula, petista, bolsonaro, governo, turno | 68 | 1,4% |
| T19 | Missão espacial e exploração lunar | missão, espacial, lua, lunar, nasa | 78 | 1,6% |
| T20 | Saúde pública e pandemia no Brasil | saúde, paciente, médico, hospital, caso | 154 | 3,1% |
| T21 | Cinema e história do cinema | filme, animal, cinema, espécie, água | 354 | 7,2% |
| T22 | Tragédia e segurança em cidades brasileiras | cidade, região, rio, pessoa, morte | 334 | 6,8% |
| T23 | Feminismo e igualdade de gênero no Brasil | mulher, homem, negro, trabalho, gênero | 60 | 1,2% |
| T24 | Futebol e torcedores | sportv, compacto, espn, jogo, domingo | 158 | 3,2% |

> ✏ **Renomeados manualmente em 2026-07-14** (nomes originais do LLM incorretos frente às keywords/docs — ver "Qualidade dos tópicos" abaixo): T4 era "Política fiscal e reforma tributária"; T17 era "Política esportiva e futebol Paulista". Os CSVs do run (`nmf_topics_for_eval.csv`, `nmf_results.csv`) **mantêm os nomes originais do LLM** — a renomeação vale para os relatórios e para a dissertação; um re-run do notebook regeneraria os CSVs com novos nomes do LLM de qualquer forma.

Distribuição menos equilibrada que a do LDA: do menor (T7, 1,0%) ao maior (T16, 10,9%) — o NMF concentra mais massa nos tópicos grandes, reflexo do θ esparso (docs frequentemente com P≈1.0 num único tópico).

### Qualidade dos tópicos

**Limpos** (~18): T0 justiça, T2 MEC/Fundeb, T3 clima/energia, T5 Amazônia/garimpo, T6 vacinação, T10 Enem, T11 ciência, T12 Americanas/mercado, T13 eleições/partidos, T14 cotidiano/família, T15 escolas/volta às aulas, T16 geopolítica, T18 Lula/eleição, T19 espaço, T20 saúde/hospitais, T22 segurança/tragédias urbanas, T23 gênero, T24 futebol/TV.

**Com ressalvas:**
- **T1 "Produção industrial brasileira"**: grab-bag estatístico (`brasil`, `milhão`, `dado`, `total`) — os top docs por θ são de esporte e ilustrada com P baixa (0.61-0.67). Nome do LLM não sustentado pelos docs; tratar como tópico residual/genérico na análise.
- **T4** (LLM: "Política fiscal e reforma tributária"): **nome errado do LLM** — as keywords (`desmatamento`, `ambiental`, `amazônia`, `lei`) apontam para política ambiental do governo federal. ✅ **Renomeado para "Política ambiental do governo federal"** (2026-07-14).
- **T7 e T8**: par redundante — ambos capturam transcrições de entrevistas do Jornal Nacional com candidatos (`william`, `renata`, `vasconcellos`, `senhor` = William Bonner/Renata Vasconcellos). O LDA não formou esse tópico; o NMF isolou o gênero textual "sabatina JN" em dois fatores. Candidatos a merge na análise qualitativa.
- **T9 "Museu e estudantes"**: tópico de **boilerplate de site** (`seta`, `novamenteícone`, `pinterestpinterestícone` — resíduos de ícones de navegação da página). Mesma classe dos tokens `seta`/`the` já documentada no LDA (o filtro `stopwords_emojis` não alcança o BoW do gensim). É o exemplo mais claro de que o NMF, por ser mais esparso, **isola** o ruído estrutural num fator próprio em vez de diluí-lo — útil para diagnóstico, mas o tópico deve ser descartado da análise temática.
- **T17** (LLM: "Política esportiva e futebol Paulista"): nome parcialmente errado — keywords apontam para gestão pública paulista (Doria, obras, PSDB), sem futebol. ✅ **Renomeado para "Gestão pública paulista"** (2026-07-14).
- **T21**: `the` na lista (artigo inglês em citações, mesmo resíduo do LDA).

---

## 11. Comparação com LDA e BERTopic

| Dimensão | NMF (K=25) | LDA (K=20) | BERTopic (K=24) |
|----------|-----------|-----------|----------------|
| Corpus de treino | 5k docs | 5k docs | N completo |
| C_v | 0.655 | 0.657 | 0.620 |
| Exclusividade (c-TF-IDF) | 0.475 | 0.457 | 0.542 |
| Topic Diversity | 0.74 | 0.770 | 0.892 |
| Outliers | 0% | 0% | 21.7% |
| θ | Esparso (docs quase-puros, P≈1.0 comum) | Suave (multi-tópico) | Hard assignment |
| Tempo de treino final | **18s** | ~5-10min | ~min (com embeddings pré-computados) |
| Perplexidade | n/d | disponível | — |

### O que o NMF capturou de diferente

- **Isolou o boilerplate** (T9) e o **gênero textual "entrevista JN"** (T7/T8) em fatores próprios — o LDA diluiu esses sinais dentro de tópicos temáticos. Bom para diagnóstico de ruído do corpus; exige descarte/merge manual na análise.
- Granularidade preferida maior (K=25 vs 20) com C_v equivalente — consistente com fatoração esparsa.
- Empate técnico com o LDA em C_v (0.655 vs 0.657) valida a robustez dos temas encontrados: os dois paradigmas BoW convergem.

---

## 12. Configuração Final

### params.yaml (seções relevantes para NMF)

```yaml
nmf:
  no_below: 5               # filter_extremes — idêntico ao bloco lda:
  no_above: 0.5

evaluation:
  top_n_keywords: 10
  top_n_keywords_metrics: 20
  top_n_keywords_llm: 50
  stability_seeds: [42]     # = seed final → Jaccard DESATIVADO
  nmf_kappa_grid: [0.5, 1.0, 2.0]
  nmf_min_prob_grid: [0.0, 0.01, 0.05]
```

### Decisões fixas no notebook

| Parâmetro | Valor | Justificativa |
|-----------|-------|---------------|
| `BEST_K` | automático (peak C_v) | K=25; plateau K=20-30 confirma a faixa |
| `passes` (grid K) | 20 | Mesma convenção do LDA |
| `passes` (grid kappa×min_prob) | 10 | Estimativa rápida para comparação relativa |
| `passes` (treino final) | 20 | Convergência robusta |
| `kappa` | 2.0 | Melhor C_v no grid (+0.017 vs 1.0) — específico da Folha |
| `minimum_probability` | 0.0 | Sem efeito em C_v; 0.0 preserva θ completo para theta heatmap/t-SNE |
| `N_REPR_DOCS_NMF` | 3 | Top-3 docs por theta |
| `top_n_llm` | 50 | Lido de params.yaml |

---

## 13. Arquivos de Saída

Cada execução gera `data/output/folha/nmf/folha_<AAAAMMDD>_<HHMMSS>/` (subpasta `nmf/`, separada de `lda/` e `bertopic/`):

| Arquivo | Conteúdo |
|---------|----------|
| `nmf_results.csv` | post_id, doc_id, text, topic_id, topic_name, distribuição θ completa |
| `nmf_topics_for_eval.csv` | keywords + nomes LLM por tópico |
| `nmf_metrics.csv` | Métricas agregadas (C_v, Exclus, Diversity, FREX, kappa, min_prob, K, corpus_version) |
| `nmf_tsne_theta_topico.png` / `nmf_tsne_theta_interactive.html` | t-SNE do espaço θ |
| `nmf_topic_category.png` | Heatmap categoria × tópico (% por editoria) |
| `nmf_topics_seed<N>.csv` | Só gerado com estabilidade ativa (hoje: desativada) |

**Arquivos de cache (`data/output/folha/nmf/`, diretório-base, reutilizados entre runs):**

| Arquivo | Conteúdo |
|---------|----------|
| `nmf_metrics.csv` | Cache do grid K (coluna `k_grid_scores`) — evita refit ~13min |
| `nmf_kappa_minprob_grid.csv` | Cache do grid kappa×min_prob (✅ já computado) |

> Sem `nmf_pyldavis_*.html` (pyLDAvis não se aplica a NMF) e sem heatmaps phi/cosine no export deste run.

---

## Notas de execução

```bash
# 1. Ollama rodando (para naming LLM)
ollama serve

# 2. Executar notebook completo
# Run > Run All Cells  (03_nmf_folha.ipynb)

# 3. Tempos observados (CPU, run 000631):
#    - Lematização (Cell 5):       ~349s (6min)
#    - Grid K (Cell 8):            cache (recomputar: ~13min)
#    - Grid kappa/min_prob:        cache (recomputar: ~min; 9 combos com treino de 18s cada)
#    - Treino final:               ~18s
#    - Naming LLM:                 ~3.1s × 25 tópicos = ~80s
#    - t-SNE:                      ~2min
#    - Total (com caches):         ~12-15min
```

---

## 14. Pendências (status 2026-07-14)

| Item | Status |
|---|---|
| Grid K + grid kappa×min_prob cacheados | ✅ completos (pasta-base `data/output/folha/nmf/`) |
| Run de produção ponta a ponta | ✅ `folha_20260714_000631` |
| **T4 e T17 com nomes errados do LLM** | ✅ **renomeados nos relatórios em 2026-07-14** (T4 → "Política ambiental do governo federal"; T17 → "Gestão pública paulista"). Os CSVs do run mantêm os nomes originais do LLM |
| **T7/T8 redundantes (entrevistas JN)** | ⚠ decidir merge ou nota de ressalva |
| **T9 boilerplate (`seta`, ícones)** | ⚠ descartar da análise temática; mesmo problema estrutural do LDA (`stopwords_emojis` não alcança o BoW gensim) |
| T1 grab-bag genérico | ⚠ tratar como tópico residual |
| FREX saturado (~0.98) | ⚠ aberto (compartilhado com LDA/BERTopic): auditar `compute_frex_score` ou não reportar |
| Estabilidade Jaccard | ➖ desativada por design (`stability_seeds=[42]`) |
| `minimum_probability` sem efeito em C_v | ℹ documentado (§6): reportar o grid como sensibilidade a kappa |

---

*Documentação gerada em 2026-07-14 a partir do run `folha_20260714_000631` (corpus `folha_20260628_185652`). Espelha a estrutura de `../lda/PIPELINE_LDA_FOLHA.md`.*
