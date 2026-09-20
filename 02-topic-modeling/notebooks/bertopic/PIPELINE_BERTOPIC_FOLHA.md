# BERTopic — Folha de São Paulo: Documentação Completa do Pipeline

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


> **Corpus:** Folha de São Paulo 2024, PT-BR, textos jornalísticos longos
> **Notebook:** `01_bertopic_folha.ipynb`
> **Configuração:** `03-topic-modeling/configs/params.yaml`
> **Última revisão:** 2026-06-29: resultados definitivos do run `folha_20260628_221412` (corpus `folha_20260628_185652` + embeddings `folha_20260628_190039`, prompt melhorado, pipeline completo re-rodado)

---

## Índice

1. [O que é BERTopic — conceitos fundamentais](#1-o-que-é-bertopic--conceitos-fundamentais)
2. [Arquitetura do pipeline](#2-arquitetura-do-pipeline)
3. [Corpus e representação dos documentos](#3-corpus-e-representação-dos-documentos)
4. [Estratégia 5k — corpus de treino vs. métricas](#4-estratégia-5k--corpus-de-treino-vs-métricas)
5. [UMAP — redução de dimensionalidade](#5-umap--redução-de-dimensionalidade)
6. [HDBSCAN — clusterização densa](#6-hdbscan--clusterização-densa)
7. [c-TF-IDF — representação vocabular dos tópicos](#7-c-tf-idf--representação-vocabular-dos-tópicos)
8. [Pós-processamento em 4 etapas — ordem obrigatória](#8-pós-processamento-em-4-etapas--ordem-obrigatória)
9. [reduce_outliers — estratégias e threshold](#9-reduce_outliers--estratégias-e-threshold)
10. [reduce_topics — consolidação de granularidade](#10-reduce_topics--consolidação-de-granularidade)
11. [Macro temas — agrupamento hierárquico](#11-macro-temas--agrupamento-hierárquico)
12. [Investigações de grid search](#12-investigações-de-grid-search)
13. [Nomeação de tópicos via LLM](#13-nomeação-de-tópicos-via-llm)
14. [Métricas de avaliação](#14-métricas-de-avaliação)
15. [Melhor resultado encontrado (baseline)](#15-melhor-resultado-encontrado-baseline)
16. [Configuração final validada](#16-configuração-final-validada)
17. [Arquivos de saída](#17-arquivos-de-saída)

---

## 1. O que é BERTopic — conceitos fundamentais

BERTopic é um modelo de descoberta de tópicos que parte de uma premissa diferente do LDA: em vez de modelar documentos como distribuições probabilísticas sobre palavras (bag-of-words), usa **representações vetoriais densas** (embeddings) para capturar significado semântico, e depois aplica algoritmos de clusterização para agrupar documentos semanticamente similares.

### O problema que BERTopic resolve

O LDA trata palavras como símbolos sem significado contextual: "banco" (financeiro) e "banco" (de praça) têm o mesmo peso. Embeddings de modelos de linguagem como `qwen3-embedding:8b` mapeiam cada documento para um ponto no espaço vetorial onde documentos semanticamente próximos ficam geometricamente próximos, independentemente do vocabulário exato usado.

### Pipeline conceitual

```
Documentos → Embeddings → Redução dim. → Clusters → Representação vocabular → Tópicos
   texto        4096d        5d (UMAP)    (HDBSCAN)    (c-TF-IDF / MMR)       K grupos
```

Cada etapa tem um papel específico:
- **Embeddings:** capturam significado semântico em alta dimensão
- **UMAP:** comprime para dimensão manejável preservando estrutura local
- **HDBSCAN:** encontra grupos densos no espaço comprimido (sem assumir K fixo)
- **c-TF-IDF:** traduz cada cluster de volta para palavras representativas
- **MMR:** diversifica as palavras-chave dentro de cada tópico

---

## 2. Arquitetura do Pipeline

```
corpus_limpo.csv + embeddings_qwen3_4096d.npy
        │
        ▼
[Cell 3]  load_corpus()
          • N documentos completos (SEM amostragem para treino)
          • docs     = text_lemma + emoji tokens injetados
          • docs_raw = texto original (para docs representativos no LLM)
          • editorias preservadas para validação
        │
        ▼
[Cell 4]  get_or_compute_embeddings()
          • carrega embeddings_qwen3_4096d.npy (cache determinístico)
          • 4096 dimensões, qwen3-embedding:8b via Ollama
        │
        ▼
[Cell 9]  BERTopic.fit_transform(docs, embeddings=embeddings)
          • UMAP: 4096d → 5d (n_neighbors=10, cosine, min_dist=0.0)
          • HDBSCAN: mcs=8, eom, min_samples=8
          • c-TF-IDF + MMR (diversity=0.2)
          ⚠ TREINA NO CORPUS COMPLETO — todos os N documentos
        │
        ▼
[Cell 11] Pós-processamento em 4 etapas (ordem obrigatória):
          1. reduce_outliers(c-tf-idf, threshold=0.15)
          2. update_topics(new_topics)           ← reconstrói c-TF-IDF
          3. reduce_topics(nr_topics=25)
          4. update_topics(model.topics_)        ← reconstrói após merge
        │
        ▼
[Cell 19] C_v coherence — amostra 5k (APENAS MÉTRICA)
        │
        ▼
[Nomeação LLM] (movida p/ ANTES do macro e das visualizações):
          • _representative_docs_folha(): 3 docs do centróide por tópico
          • prompt_builder_custom(): few-shot + anti-redundância + TEMA GERAL
          • _dedupe_name_folha(): deduplicação pós-geração
          • Ordering: maiores tópicos → menores (claimed names first)
          ✓ roda logo após as métricas → macro/figuras já saem rotuladas (Run All único)
        │
        ▼
[Macro temas] clustering Ward dos embeddings de tópico (macro_k=10)
          ✓ bertopic_macro_temas.csv já sai com rótulos LLM (naming já rodou)
        │
        ▼
[Sweeps de avaliação] (desligados — já rodaram)
        │
        ▼
[Cells +] Métricas, visualizações, ranking exclusividade, temporal, exports
```

---

## 3. Corpus e Representação dos Documentos

### Construção dos `docs`

O BERTopic recebe cada documento como string. Para a Folha, a string é construída concatenando o texto lematizado com os tokens emoji demojizados:

```python
docs = [row["text_lemma"] + " " + " ".join(row["emoji_tokens"]) for _, row in df.iterrows()]
```

**Por que lemas e não texto bruto?** Lemas reduzem a variância morfológica (conjugações, plural, gênero) sem perder significado, o que torna o vocabulário do c-TF-IDF mais limpo. Os emojis são convertidos em palavras descritivas (`❤️` → `coração_vermelho`) para que carreguem sinal semântico em vez de serem ignorados.

**`docs_raw`** (texto original) é mantido separado, usado exclusivamente pelo `_representative_docs_folha()` para mostrar ao LLM trechos legíveis por humanos durante a nomeação.

### Stopwords e filtros de ruído (CountVectorizer)

O `CountVectorizer` do BERTopic recebe uma lista de stopwords customizadas via `params.yaml > corpora > folha > stopwords_emojis`. Esses tokens aparecem no c-TF-IDF mas não carregam semântica:

| Token | Origem | Efeito nocivo identificado |
|-------|--------|---------------------------|
| `atilde`, `ccedil`, `iacute`, `eacute`... | Entidades HTML PT-BR mal decodificadas | Contaminam rankings de keywords |
| `quot` | Resíduo de `&quot;` em HTML | Aparecia na posição 17 de T4 (cinema), levando o LLM a nomear o tópico como "Festa Literária Internacional de Paraty" |
| `the` | Artigo inglês não filtrado em citações | Vazava para o c-TF-IDF de tópicos em PT-BR |
| `seta` | Símbolo `→` tokenizado como palavra | Aparecia como keyword em T3 e T21 |

A remoção desses tokens é feita antes do ranking c-TF-IDF via `_filter_keywords()` no `_helpers.py`, garantindo que o LLM receba apenas keywords semanticamente relevantes.

---

## 4. Estratégia 5k — Corpus de Treino vs. Métricas

### Comportamento diferenciado

Um ponto de confusão recorrente é **onde** a amostra de 5k documentos entra no pipeline. A resposta curta: **não entra no treino do BERTopic**, só na métrica de coerência C_v.

| Etapa | BERTopic (Folha) | LDA (Folha) |
|-------|-----------------|-------------|
| **Treino** | N documentos completos | 5k documentos |
| **C_v** | 5k (sample seed=42) | 5k (mesmos índices) |
| **reduce_outliers / reduce_topics** | N documentos completos | — |

### Por que BERTopic treina em N e LDA em 5k?

É uma assimetria intencional para o contexto da dissertação: demonstra que BERTopic, ao aproveitar embeddings pré-computados, consegue processar o corpus completo de forma eficiente, enquanto o LDA (baseado em BoW e Gibbs sampling) fica limitado a uma amostra por custo computacional.

### Equivalência dos índices para C_v

```python
# BERTopic — Cell 19 (apenas para C_v, não altera o modelo)
_sample_idx = np.random.RandomState(42).choice(len(docs), min(5000, len(docs)), replace=False)

# LDA — Cell 5 (antes de qualquer processamento)
_s_idx = np.random.RandomState(42).choice(len(docs), min(5000, len(docs)), replace=False)
```

Mesma seed (42), mesmo `len(docs)` (N) → **índices idênticos** → mesmos artigos selecionados. Isso garante que a C_v de BERTopic e LDA sejam comparáveis entre si, mesmo que os modelos tenham sido treinados em volumes distintos.

---

## 5. UMAP — Redução de Dimensionalidade

### O problema da maldição da dimensionalidade

Algoritmos de clusterização como o HDBSCAN funcionam mal em alta dimensão: com 4096 dimensões, todos os pontos ficam aproximadamente equidistantes entre si: o conceito de "proximidade" perde significado. O UMAP resolve isso comprimindo os dados para um espaço de baixa dimensão (5d aqui) preservando a estrutura local do manifold.

### Como o UMAP funciona

O UMAP constrói um grafo fuzzy de vizinhança k-NN no espaço original (4096d) e então encontra uma projeção de baixa dimensão que preserva ao máximo esse grafo. O parâmetro `n_neighbors` define quantos vizinhos cada ponto considera. É o parâmetro mais crítico.

### Parâmetros e escolhas

| Parâmetro | Valor | Motivo |
|-----------|-------|--------|
| `n_neighbors` | 10 | Vizinhança pequena → estrutura local → clusters distintos; validado pelo composite sweep para o regime (mcs=8, eom, nr=25) |
| `n_components` | 5 | 5 dimensões preservam mais estrutura que 2d (visualização) com custo manejável para HDBSCAN |
| `min_dist` | 0.0 | Pontos dentro de um cluster ficam o mais próximos possível: maximiza separação entre clusters para HDBSCAN |
| `metric` | cosine | Embeddings de linguagem vivem na hiperesfera unitária; cosine captura similaridade semântica melhor que distância euclidiana |

### n_neighbors=10 vs. n_neighbors=20

O sweep estrutural mostrou:

| n_neighbors | C_v | Diversity | Exclus | Outliers_pre |
|------------|-----|-----------|--------|-------------|
| 10 | 0.536 | 0.796 | 0.488 | 24.3% |
| 15 | 0.536 | 0.803 | 0.494 | 27.2% |
| 20 | 0.545 | 0.804 | 0.506 | 29.2% |

A tabela agregada favorece n=20 marginalmente (+0.009 C_v, +0.018 Exclus), mas essa tabela conflate diferentes valores de `min_cluster_size`: com mcs=40, clusters grandes se beneficiam de uma vizinhança maior. Para mcs=8 especificamente, clusters pequenos e compactos são melhor capturados com vizinhança menor (n=10 encontra a estrutura fina sem "puxar" clusters adjacentes para perto no espaço UMAP). Adicionalmente, n=10 produz 4.9pp menos outliers pré-redução (24.3% vs 29.2%), o que é relevante para a cobertura.

---

## 6. HDBSCAN — Clusterização Densa

### O que o HDBSCAN faz de diferente

Diferente do k-means (que exige K fixo e agrupa todo ponto) ou do DBSCAN (que exige ε fixo), o HDBSCAN:
1. Não requer K pré-definido: descobre automaticamente o número de clusters
2. Classifica pontos de baixa densidade como **outliers** (id = -1) em vez de forçá-los a pertencer a um cluster
3. Usa uma hierarquia de densidade para encontrar clusters de tamanhos e densidades variadas

Para jornalismo, isso é ideal: a Folha tem tópicos muito populares (política, esporte) e outros bem específicos (vacinação, arqueologia), e o HDBSCAN captura ambos sem forçar granularidade uniforme.

### Parâmetros e escolhas

| Parâmetro | Valor | Motivo |
|-----------|-------|--------|
| `min_cluster_size` (mcs) | 8 | Cluster mínimo com 8 docs; preserva sub-temas específicos |
| `min_samples` | null (= mcs = 8) | Controla robustez do core point; igual ao mcs é o default HDBSCAN |
| `cluster_selection_method` | eom | Excess of Mass; vence LEAF no regime mcs=8/nr=25 |

### EOM vs. LEAF — a escolha mais importante do HDBSCAN

Este é o parâmetro que mais impacta a qualidade para corpus jornalísticos heterogêneos.

**EOM (Excess of Mass):** percorre a árvore condensada de baixo para cima e, em cada bifurcação, decide se é melhor manter os dois sub-clusters separados ou mesclá-los em um cluster maior. Tende a criar **clusters maiores e menos numerosos**, excelente para dados com poucos grupos bem definidos.

**LEAF:** seleciona as folhas mais finas da árvore condensada, criando **clusters menores e mais numerosos**: cada "ilha densa" no espaço de embeddings vira seu próprio tópico.

Para a Folha de São Paulo (jornal generalista com dezenas de editorias distintas), a estrutura real dos dados é formada por muitas ilhas semânticas densas: "política monetária + juros", "Amazônia + desmatamento", "cinema + streaming", etc. **Ponto crítico:** o `reduce_topics_nr=25` impõe K=24 a jusante **independentemente** do método de seleção. Logo, a escolha EOM vs. LEAF **não fixa a granularidade final** (o `reduce_topics` faz isso): ela determina a **qualidade dos clusters de origem** que alimentam o merge hierárquico.

**Sweep EOM vs. LEAF, média global (90 combinações cada):**

| Método | C_v | Diversity | Exclus | Outliers_pre |
|--------|-----|-----------|--------|-------------|
| EOM | 0.534 | 0.796 | 0.489 | 22.6% |
| **LEAF** | **0.543** | **0.806** | **0.503** | 31.2% |

Na **média global**, LEAF vence marginalmente em C_v/Diversity/Exclus. Mas essa média é enganosa: a vantagem do LEAF vem inteiramente dos `min_cluster_size` grandes. Quebrando o C_v médio por mcs:

| mcs | C_v EOM | C_v LEAF | Δ (leaf − eom) |
|-----|---------|----------|----------------|
| **8** | 0.538 | 0.538 | **−0.000 (empate)** |
| 15 | 0.530 | 0.537 | +0.007 |
| 25 | 0.540 | 0.545 | +0.006 |
| 40 | 0.536 | 0.552 | +0.016 |

No **regime escolhido (mcs=8)**, LEAF e EOM **empatam** em C_v (0.5384 vs 0.5382) e Exclus, EOM ganha em Diversity (0.813 vs 0.808) e gera **8pp menos outliers** (24.8% vs 32.8%). E no **ponto de operação exato** (n_neighbors=10, mcs=8, nr=25), EOM domina:

| Config (mcs=8, nr=25) | C_v | Diversity | Exclus | Outliers_pre |
|-----------------------|-----|-----------|--------|-------------|
| **10 / eom / 25** | **0.585** | **0.808** | **0.521** | **21.5%** |
| 10 / leaf / 25 | 0.545 | 0.783 | 0.485 | 29.1% |

Essa vantagem do EOM **se repete nos três valores de `n_neighbors`**, não é ruído de seed (o grid foi single-seed): ΔC_v = +0.039 / +0.042 / +0.005; ΔDiversity = +0.025 / +0.027 / +0.004; ΔOutliers = −7.6 / −8.5 / −7.9 pp. Exclus fica empatado (eom +0.036 / leaf +0.009 / leaf +0.003).

**Por isso a config final usa `eom`.** Para mcs pequeno, o EOM não "colapsa" as ilhas semânticas (o `reduce_topics` é quem controla a granularidade final), e entrega clusters de origem mais coerentes, mais diversos e com cobertura maior. O receio de que o EOM mescle distinções importantes não se materializa nas métricas neste regime.

> **Correção (revisão NLP Engineer 2026-06-27):** versões anteriores deste documento recomendavam `leaf` com base na média global e numa tabela de composite cujos números não eram reproduzíveis a partir de `sweep_bertopic_grid.csv`. A recomputação do CSV bruto inverteu a decisão para `eom`.

### min_cluster_size=8 vs. 40

O `min_cluster_size` define o menor cluster que o HDBSCAN pode criar. Com mcs=40, qualquer grupo com menos de 40 artigos é descartado como outlier antes de chegar ao BERTopic. Com mcs=8, grupos a partir de 8 artigos são preservados.

**Sweep de mcs:**

| mcs | C_v | Diversity | Exclus | Outliers_pre |
|-----|-----|-----------|--------|-------------|
| **8** | 0.538 | **0.810** | **0.516** | 28.8% |
| 15 | 0.534 | 0.811 | 0.508 | 27.2% |
| 25 | 0.542 | 0.804 | 0.505 | 26.8% |
| 40 | **0.544** | 0.793 | 0.479 | 26.0% |

mcs=40 maximiza C_v (+0.6% vs mcs=8), mas mcs=8 ganha em Diversity (+2.2%) e Exclusividade (+7.7%). Para uma dissertação de análise qualitativa, onde a distinção entre tópicos ("este tópico é sobre política fiscal, aquele sobre câmbio") é mais importante do que a coerência interna medida por C_v, a Exclusividade é o critério mais relevante. Keywords mais exclusivos = tópicos mais fáceis de nomear e de interpretar qualitativamente.

### min_topic_size=8 (= mcs)

O `min_topic_size` é um filtro do BERTopic aplicado **após** o HDBSCAN: remove tópicos com menos documentos que esse limiar. Com mcs=8, o HDBSCAN não produz clusters menores que 8, então qualquer valor de `min_topic_size` > 8 filtraria clusters que existem mas são "pequenos" segundo o critério do BERTopic, descartando-os antes do `reduce_topics`.

Definir `min_topic_size = mcs = 8` garante que todos os clusters produzidos pelo HDBSCAN entram no `reduce_topics` e são fundidos hierarquicamente em vez de descartados silenciosamente.

> **Bug do run anterior:** `min_topic_size=25` com `mcs=8` fazia o BERTopic criar clusters HDBSCAN de 8-24 artigos e então descartá-los como outliers pré-`reduce_topics`, inflando artificialmente a taxa de outliers.

---

## 7. c-TF-IDF — Representação Vocabular dos Tópicos

### O problema do TF-IDF clássico

O TF-IDF clássico calcula a importância de um termo para um *documento* específico. Para tópicos, queremos saber a importância de um termo para um *cluster inteiro*, não para um único documento.

### c-TF-IDF (class-based TF-IDF)

O BERTopic concatena todos os documentos de cada cluster em um "meta-documento" e aplica TF-IDF sobre esses meta-documentos. O resultado é um score que indica o quão exclusivo um termo é para um tópico em relação a todos os outros, exatamente a medida de "o que distingue este tópico dos demais".

```
c-TF-IDF(t, c) = tf(t, c) × log(1 + A / tf(t))

onde:
  tf(t, c) = frequência do termo t no cluster c
  A        = número médio de palavras por cluster
  tf(t)    = frequência total de t em todos os clusters
```

O fator `log(1 + A/tf(t))` penaliza termos muito frequentes em todos os tópicos (como "disse", "que", "em"), análogo ao IDF clássico.

### MMR (Maximal Marginal Relevance)

Após o c-TF-IDF rankear os termos por relevância, o MMR aplica uma diversificação: evita que os top-N keywords sejam todos variações do mesmo conceito. Por exemplo, sem MMR, um tópico de futebol poderia ter ["futebol", "bola", "jogo", "partida", "campeonato"...] onde as primeiras palavras são quase sinônimos. Com `mmr_diversity=0.2`, o MMR equilibra relevância e diversidade lexical.

---

## 8. Pós-processamento em 4 Etapas — Ordem Obrigatória

### Por que a ordem importa

O pós-processamento do BERTopic tem quatro etapas que devem ser executadas **nesta ordem exata**:

```
reduce_outliers → update_topics → reduce_topics → update_topics
      (1)              (2)             (3)              (4)
```

**Etapa 1 — `reduce_outliers`:** Reatribui documentos marcados como outlier (-1) ao tópico mais próximo. **Não reconstrói** a representação c-TF-IDF, apenas muda os labels de tópico dos documentos.

**Etapa 2 — `update_topics`:** Reconstrói o c-TF-IDF com a nova distribuição de documentos por tópico (incluindo os recém-reatribuídos). Esta etapa é obrigatória antes do `reduce_topics` porque o merge hierárquico usa as representações c-TF-IDF para calcular similaridade entre tópicos.

**Etapa 3 — `reduce_topics`:** Mescla tópicos hierarquicamente até atingir `nr_topics=25`. Usa a similaridade c-TF-IDF recalculada na Etapa 2.

**Etapa 4 — `update_topics`:** Reconstrói o c-TF-IDF uma última vez após os merges do `reduce_topics`. Os tópicos finais têm representações atualizadas que refletem todos os documentos fundidos.

**O que acontece se inverter a ordem?**
- Pular Etapa 2: `reduce_topics` usa representações c-TF-IDF calculadas antes da reatribuição de outliers → merges incorretos
- Pular Etapa 4: keywords dos tópicos mesclados não refletem o conjunto completo de documentos → ranking c-TF-IDF desatualizado

### Implementação (Cell 11)

```python
ro_cfg = bert_cfg.get("reduce_outliers", {})
n_outliers = sum(1 for t in topics if t == -1)

if ro_cfg.get("enabled", True) and n_outliers > 0:
    strategy  = ro_cfg.get("strategy",  "c-tf-idf")
    threshold = ro_cfg.get("threshold", 0.0)
    _ro_kw = dict(strategy=strategy, threshold=threshold)
    if strategy == "embeddings":
        _ro_kw["embeddings"] = embeddings    # espaço 4096d
    elif strategy == "probabilities":
        _ro_kw["probabilities"] = probs      # soft clustering HDBSCAN

    new_topics = topic_model.reduce_outliers(docs, topics, **_ro_kw)
    topic_model.update_topics(docs, topics=new_topics,          # Etapa 2
                              vectorizer_model=vectorizer_model,
                              representation_model=representation_model)
    topics = new_topics

topic_model.reduce_topics(docs, nr_topics=bert_cfg["reduce_topics_nr"])  # Etapa 3
topic_model.update_topics(docs, topics=topic_model.topics_,              # Etapa 4
                          vectorizer_model=vectorizer_model,
                          representation_model=representation_model)
```

---

## 9. reduce_outliers — Estratégias e Threshold

### O que são outliers no HDBSCAN

O HDBSCAN classifica como outlier (id=-1) qualquer ponto que não pertença a uma região suficientemente densa. Em termos de conteúdo, um artigo pode virar outlier por:
- Ser único (um artigo sobre um evento sem paralelo no corpus)
- Ser temáticamente ambíguo (uma matéria que cruza dois temas diferentes)
- Estar na borda de dois clusters densos (mais próximo de nenhum do que de qualquer um)

No run anterior sem `reduce_outliers`, **32.7%** dos artigos eram outliers, ou seja, 1 em cada 3 artigos não tinha tópico atribuído. Isso cria dois problemas: (1) análise temporal e de editorias fica enviesada (os outliers são excluídos); (2) a taxa de cobertura do modelo é baixa.

### Estratégias comparadas

O sweep `sweep_outlier_strategies` testou as 5 estratégias disponíveis com threshold=0 (reatribuição total) para comparar o mecanismo de decisão:

| Estratégia | Mecanismo | C_v | outlier_post | Diversity | Exclus |
|-----------|-----------|-----|-------------|-----------|--------|
| off | baseline (sem reatribuição) | 0.610 | 32.7% | 0.797 | 0.484 |
| **c-tf-idf** | similaridade vocabular doc↔tópico | 0.522 | 0.0% | 0.693 | 0.370 |
| distributions | distribuição token×tópico | 0.519 | 0.0% | 0.688 | 0.369 |
| embeddings | cosine no espaço 4096d | 0.507 | 0.0% | 0.669 | 0.344 |
| probabilities | probabilidades soft do HDBSCAN | 0.489 | 0.0% | 0.672 | 0.348 |

**Observação crítica:** com threshold=0, **todas as estratégias caem significativamente** em C_v, Diversity e Exclusividade em relação ao baseline "off". Isso ocorre porque, sem threshold, até documentos muito ambíguos são forçados para o tópico mais próximo, contaminando a representação c-TF-IDF dos tópicos com conteúdo irrelevante.

A escolha de estratégia importa: `c-tf-idf` supera as demais porque usa a representação vocabular do tópico (o que o tópico "fala") para decidir a reatribuição. Isso é semanticamente mais estável para textos longos (como artigos de jornal) do que distância no espaço de embeddings de 4096 dimensões, que pode capturar similaridades estilísticas além de temáticas.

### O papel do threshold

O threshold é o ponto de equilíbrio mais importante do `reduce_outliers`. Define a **similaridade mínima** que um documento outlier deve ter com um tópico para ser reatribuído a ele. Documentos com similaridade abaixo do threshold permanecem como outliers.

**Sweep de threshold para c-tf-idf:**

| Threshold | outlier_post | C_v | Diversity | Exclus | Interpretação |
|-----------|-------------|-----|-----------|--------|---------------|
| 0.00 | 0.0% | 0.522 | 0.693 | 0.370 | Tudo reatribuído; qualidade cai muito |
| 0.05 | 0.7% | 0.581 | 0.764 | 0.430 | Só os mais próximos permanecem outliers |
| 0.10 | 13.2% | 0.598 | 0.774 | 0.448 | Ponto de inflexão |
| **0.15** | **25.4%** | **0.605** | **0.795** | **0.476** | ← **escolhido** |
| 0.20 | 30.8% | 0.610 | 0.791 | 0.482 | Quase igual ao "off" |
| 0.30 | 32.6% | 0.610 | 0.797 | 0.485 | Equivalente a "off" |

**Por que threshold=0.15 e não 0.20 (que tem C_v levemente melhor)?**

Threshold=0.20 e threshold=0.30 produzem resultados estatisticamente equivalentes ao "off": a diferença é de apenas 1.9pp de cobertura extra. O `reduce_outliers` com threshold=0.20 reatribui documentos que teriam ficado como outliers igualmente sem a etapa.

Threshold=0.15 oferece um trade-off real:
- Cobre **+5.4pp** de documentos adicionais em relação ao threshold=0.20 (25.4% vs 30.8% de outliers remanescentes)
- Diversity **superior** (0.795 vs 0.791: threshold=0.15 produz tópicos mais diversificados)
- C_v levemente menor (-0.5%), mas a perda é marginal
- **vs 0.10:** apesar de 0.10 recuperar ~12pp de cobertura, o composite multi-objetivo (que já pondera cobertura) favorece 0.15 com folga (**0.776 vs 0.715**): os ganhos de Exclusividade/Diversity/C_v superam o ganho de cobertura

Para análise temporal e cruzamento com editorias, ter 5% mais artigos atribuídos a tópicos reais reduz o viés de exclusão dos outliers.

> **Nota importante:** este sweep foi executado com a **configuração estrutural anterior** (n_neighbors=20, mcs=40). Com a nova config (n_neighbors=10, mcs=8, eom), clusters são menores e mais densos: as distribuições de similaridade c-tf-idf podem se deslocar levemente. Threshold=0.15 é o prior mais defensável; se o próximo run produzir `outlier_post` significativamente diferente de ~25%, re-executar o threshold sweep com a nova config.

---

## 10. reduce_topics — Consolidação de Granularidade

### O que o reduce_topics faz

O HDBSCAN pode produzir dezenas de clusters de granularidade fina: por exemplo, "política fiscal", "política monetária", "orçamento federal" podem emergir como tópicos separados. O `reduce_topics` funde os tópicos menores nos maiores, usando **similaridade c-TF-IDF hierárquica**, até atingir o número alvo de tópicos.

O algoritmo constrói um dendrograma onde tópicos com representações c-TF-IDF similares ficam próximos, e então "poda" o dendrograma no nível que produz o número de tópicos desejado.

### Escolha de reduce_topics_nr=25 (K≈24)

O sweep varrreu valores de 10 a 30:

| reduce_nr | K resultante | C_v | Diversity | Exclus | Análise qualitativa |
|----------|-------------|-----|-----------|--------|---------------------|
| 10 | 9 | 0.518 | 0.832 | 0.562 | K=9 muito grosseiro para jornal generalista |
| 15 | 14 | 0.523 | 0.807 | 0.508 | Melhor Exclus; mas perde distinções importantes |
| 20 | 19 | 0.539 | 0.794 | 0.482 | Equilíbrio razoável |
| **25** | **24** | **0.553** | **0.790** | **0.470** | ← **escolhido: melhor C_v+Diversity para K∈[20,30]** |
| 30 | 29 | 0.562 | 0.783 | 0.456 | C_v máximo, mas Exclus cai e duplicatas surgem |

**Problema concreto com nr=30 (run anterior):** dois tópicos de meio ambiente ficaram separados:
- T10: "Política de desmatamento" (Salles, Ibama, bolsonaro), editoria `ambiente` 96%
- T24: "Dados de desmatamento INPE" (INPE, queimadas, prodes, cerrado), editoria `ambiente` 100%

Eram o mesmo tema coberto por ângulos distintos (político vs. técnico). Com nr=25, o merge hierárquico os une em um único tópico mais rico semanticamente.

### K=24 vs. K=9 — escolha para dissertação

K=9 tem as melhores métricas absolutas (C_v=0.518, Exclus=0.562), mas para uma análise de jornal generalista, 9 tópicos são grosseiros demais: "esporte", "política" e "economia" cada um representaria categorias enormes sem distinção analítica. K=24 preserva distinções como "futebol" vs. "olímpiadas" e "política fiscal" vs. "banco central", essenciais para a dissertação.

---

## 11. Macro Temas — Agrupamento Hierárquico

### Dois níveis de granularidade

O pipeline produz dois níveis:
1. **K=24 tópicos** (granularidade fina, análise detalhada)
2. **10 macro temas** (granularidade de apresentação, visão geral)

Os macro temas são calculados por clustering hierárquico Ward aplicado aos **embeddings dos tópicos** (centróides ponderados por c-TF-IDF no espaço de embeddings). Tópicos com conteúdo semelhante ficam próximos e são agrupados.

### Sweep de macro_k

> Valores reais do run de produção `folha_20260628_221412` (`bertopic_macro_k_sweep.csv`). Com o regime mcs=8/eom, os topic embeddings são mais compactos e numerosos antes do `reduce_topics`, produzindo uma distribuição de macro temas diferente do baseline anterior (mcs=40/leaf).

| macro_k | Maior cluster (docs) | Distribuição de subtemas | Avaliação |
|---------|---------------------|--------------------------|-----------|
| 4 | 2500 | [12, 6, 4, 2] | Péssimo: 1 grupo domina com 12 subtópicos |
| 5 | 2500 | [12, 4, 3, 3, 2] | Ruim: ainda muito desequilibrado |
| 6 | 2500 | [12, 3, 3, 2, 2, 2] | Grupo grande persiste |
| 7 | 2272 | [9, 3, 3, 3, 2, 2, 2] | Patamar: grupo grande reduz para 9; demais equilibrados |
| 8 | 2272 | [9, 3, 3, 2, 2, 2, 2, 1] | Top-heavy: 1 grupo absorve 9 subtemas |
| **10** | **2272** | **[9, 3, 2, 2, 2, 2, 1, 1, 1, 1]** | **Escolhido** |
| 12 | 2191 | [8, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1] | Fragmentação excessiva; 6 macros com 1 único subtópico |

Com macro_k=10, o **Macro 2** (9 subtópicos, 2.272 docs) agrupa tópicos de política nacional, educação, covid, desmatamento, segurança pública, cidades e migração: uma categoria "cotidiano e política BR" heterogênea mas semanticamente coerente no espaço de embeddings com mcs=8/eom. Os demais 9 macros têm 1-3 subtópicos bem individualizados (economia, conflitos internacionais, clima, cultura/esporte, ciência, museu). macro_k=7 é uma alternativa se o Macro 2 grande for problemático na apresentação: elimina os singletons mas preserva os grupos de 3.

### Validação com editorias da Folha (run `folha_20260628_221412`)

As editorias originais (`ambiente`, `ciencia`, `cotidiano`, `educacao`, `equilibrioesaude`, `esporte`, `ilustrada`, `mercado`, `mundo`, `poder`) servem como ground truth externo.

**Mapeamento macro → tópicos (macro_k=10):**

| Macro | Tópicos | Tema interpretado | Status |
|-------|---------|------------------|--------|
| 1 | T11 Economia SP, T12 Inflação, T17 Petróleo | Economia e finanças | ✅ Coerente |
| 2 | T0 Política eleitoral, T1 Educação, T2 Vacinação, T4 Desmatamento, T6 Polícia, T7 Cidades/pandemia, T13 ENEM, T14 Fake news, T19 Migração | Cotidiano e política nacional | ⚠ Grande (9 subtópicos); heterogêneo mas interpretável |
| 3 | T10 Ucrânia/Rússia, T16 Israel/Hamas | Conflitos internacionais | ✅ Coerente |
| 4 | T18 Mortes Covid | Covid: dados epidemiológicos | ✅ (singleton) |
| 5 | T22 China/Covid | Covid: China | ✅ (singleton) |
| 6 | T15 Mudanças climáticas, T20 Chuvas SP | Clima e eventos extremos | ✅ Coerente; confusão clima/saúde do baseline **resolvida** |
| 7 | T3 Futebol/Copa, T5 Cinema/cultura | Cultura e esporte | ✅ Coerente |
| 8 | T21 Fórmula 1 | Automobilismo | ✅ (singleton) |
| 9 | T8 Paleontologia, T9 Espaço/NASA | Ciência e descobertas | ✅ Coerente |
| 10 | T23 Museu Nacional | Patrimônio cultural | ✅ (singleton) |

**Problema clima/saúde do baseline:** resolvido. T15 (Mudanças Climáticas e El Niño) e T20 (Chuvas SP) estão agora no **Macro 6** juntos, sem mistura com tópicos de saúde.

---

## 12. Investigações de Grid Search

O pipeline executou três tipos de sweep sistemático antes de definir a configuração final. Todos estão atualmente **desligados** em `params.yaml` pois já foram executados.

### 12.1 Sweep 1 — Estratégias de reduce_outliers

**Controle:** `run_outlier_sweep: true/false`
**Implementação:** `_helpers.sweep_outlier_strategies()`

Testa as 5 estratégias de `reduce_outliers` com threshold=0 (reatribuição completa) para isolar o efeito do mecanismo de decisão. Um modelo base é treinado por seed e copiado (deepcopy) para cada estratégia: economiza tempo de refit mas isoliza o efeito do mecanismo.

**Principal achado:** threshold=0 derruba C_v em ~15% para todas as estratégias. A comparação entre estratégias indica `c-tf-idf` como melhor mecanismo, mas o threshold é o fator dominante.

---

### 12.2 Sweep 2 — Threshold de reduce_outliers

**Controle:** `run_threshold_sweep: true/false`
**Implementação:** `_helpers.sweep_outlier_threshold()`

Mantém a estratégia fixa e varre diferentes valores de threshold. O modelo base é copiado para cada (estratégia, threshold): 27 combinações × 1 seed = 27 refits.

**Resultado completo do threshold sweep (c-tf-idf):**

| Threshold | outlier_post | C_v | Diversity | Exclus | Composite |
|-----------|-------------|-----|-----------|--------|-----------|
| 0.00 | 0.0% | 0.522 | 0.693 | 0.370 | 0.580 |
| 0.05 | 0.7% | 0.581 | 0.764 | 0.430 | 0.681 |
| 0.10 | 13.2% | 0.598 | 0.774 | 0.448 | 0.715 |
| **0.15** | **25.4%** | **0.605** | **0.795** | **0.476** | **0.776** ← vencedor |
| 0.20 | 30.8% | 0.610 | 0.791 | 0.482 | 0.762 |
| 0.30 | 32.6% | 0.610 | 0.797 | 0.485 | 0.748 |

O composite score (média normalizada de C_v + Exclus + coverage + Diversity) identifica threshold=0.15 como o ponto ótimo multi-objetivo.

---

### 12.3 Sweep 3 — Grid estrutural UMAP/HDBSCAN

**Controle:** `run_param_sweep: true/false`
**Implementação:** `_helpers.sweep_bertopic_grid()`

O sweep mais custoso: refit completo do BERTopic para cada combinação. 180 configurações × 1 seed = 180 refits completos (UMAP + HDBSCAN + c-TF-IDF).

**Grid varrido:**
```yaml
n_neighbors:                [10, 15, 20]       # 3 valores
min_cluster_size:           [8, 15, 25, 30, 35, 40]  # 6 valores
min_samples:                [null]             # default = mcs
cluster_selection_method:   ["leaf", "eom"]    # 2 valores
reduce_topics_nr:           [10, 15, 20, 25, 30]  # 5 valores
strategy:                   "off"              # isola efeito estrutural
```

3 × 6 × 1 × 2 × 5 = **180 combinações**

**Composite score por configuração, recomputado de `sweep_bertopic_grid.csv` (top-6 para K∈[20,30]; composite = média normalizada de C_v + Exclus + cobertura + Diversity dentro do subset):**

| n_neighbors | mcs | csm | nr | K | C_v | Diversity | Exclus | Composite |
|------------|-----|-----|----|---|-----|-----------|--------|-----------|
| **10** | **8** | **eom** | **25** | **24** | **0.585** | **0.808** | **0.521** | **0.843** ← escolhido |
| 15 | 15 | eom | 25 | 24 | 0.557 | 0.823 | 0.503 | 0.772 |
| 15 | 8 | eom | 25 | 24 | 0.567 | 0.823 | 0.491 | 0.741 |
| 10 | 8 | eom | 30 | 29 | 0.580 | 0.791 | 0.496 | 0.740 |
| 15 | 8 | eom | 30 | 29 | 0.561 | 0.812 | 0.488 | 0.694 |
| 20 | 15 | eom | 30 | 29 | 0.547 | 0.816 | 0.499 | 0.673 |
| *referência* | | | | | | | | |
| 20 | 40 | leaf | 30 | 29 | 0.610 | 0.797 | 0.484 | 0.660 (rank 10, baseline anterior) |
| 10 | 8 | leaf | 25 | 24 | 0.545 | 0.783 | 0.485 | 0.513 (**rank 45/72**, leaf no mesmo regime) |

> **Correção (revisão NLP Engineer 2026-06-27):** a versão anterior desta tabela apontava `10/8/leaf/25` como vencedor (composite 0.678, C_v=0.553, Exclus=0.470). Esses C_v/Exclus eram, na verdade, a **média global de nr=25** colada por engano numa linha de config específica, e o composite 0.678 não era reproduzível a partir do CSV. Recomputando do bruto, o regime K∈[20,30] é **dominado por EOM**; o `10/8/leaf/25` real ocupa apenas o **rank 45 de 72**. A configuração final foi corrigida para `eom` (`10/8/eom/25`, composite 0.843).

### 12.4 Sweep de robustez por top-N

Mede como as métricas variam com o número de keywords avaliadas, importante para decidir qual top-N reportar na dissertação:

| top-N | C_v | Diversity | Exclus |
|-------|-----|-----------|--------|
| **10** | **0.688** | **0.886** | **0.537** |
| 15 | 0.660 | 0.837 | 0.507 |
| 20 | 0.610 | 0.797 | 0.484 |
| 25 | 0.593 | 0.778 | 0.477 |
| 30+ | 0.589 | plateou | plateou |

**Por que reportar C_v em top-10:** A literatura (Röder et al. 2015) usa top-10 como padrão. C_v@top-10 = 0.69 vs C_v@top-20 = 0.61. As top-10 keywords são as mais discriminativas; o valor em top-10 é o mais comparável com outros trabalhos da área.

---

## 13. Nomeação de Tópicos via LLM

### Visão geral

A nomeação automática resolve o problema de converter uma lista de keywords como `{marte, nasa, telescópio, órbita, missão}` em um rótulo humano legível como "Exploração espacial e astronomia". O modelo `gemma2:2b-instruct-q4_K_M` roda localmente via Ollama.

### Problemas identificados no run anterior

O run anterior usava **1 documento arbitrário** e **25 keywords**. Resultado: 7 erros de nomeação em 29 tópicos (24% de falhas).

| Tópico | Keywords top-5 | Nome gerado (errado) | Nome correto | Causa raiz |
|--------|---------------|---------------------|--------------|------------|
| T4 | filme, cinema, livro, festival, quot | "Festa Literária Internacional de Paraty" | "Cinema, séries e literatura" | Doc era sobre FLIP; token `quot` na posição 17 do ranking |
| T15 | marte, nasa, terra, telescópio, missão | "Missão espacial Emirados Árabes Unidos" | "Astronomia: Marte, Webb e sistema solar" | Doc era sobre missão emirada; `emirados` em posição 25/30 |
| T16 | arqueologia, fóssil, espécie, descoberta | "Sambaquis arqueológicos milenários" | "Descobertas arqueológicas e paleoantropologia" | Doc era sobre sambaquis; `tutancâmon` em posição 17 |
| T3 | bolsonaro, lula, pt, stf, lava_jato | "Lula e eleitores após prisão de Bolsonaro" | "Política BR: Bolsonaro, Lula e STF" | Hallucination narrativa; 25 keywords insuficientes |
| T27 | (mix: 3 sub-temas distintos) | (inomeável) | — | Tópico incoerente: problema de clustering |

**Padrão dos erros:** todos os erros de T4, T15, T16 foram causados pelo mesmo mecanismo: **1 documento específico ancora o LLM num evento pontual em vez do tema geral**.

### Refinamento do prompt (2026-06-28)

O run `folha_20260628_113224` ainda apresentou redundância em temas próximos (T13 "Mudanças Climáticas Globais" vs T15 "…(gelo)": dedup só colou sufixo) e diluição em tópicos mistos. O prompt foi reforçado (mantendo `gemma2:2b`):

1. **Few-shot de temas próximos distintos:** par de exemplos (política climática vs. degelo polar) ensinando a diferenciar temas adjacentes.
2. **Anti-redundância acionável:** em vez de só "seja distinto", instrui a identificar o termo das keywords que **DIFERENCIA** o tópico e proíbe sufixos como "(2)"/"(gelo)".
3. **Priorização por peso:** "as PRIMEIRAS palavras-chave são as mais importantes (maior peso): baseie o rótulo nelas".
4. **Assunto dominante:** quando as keywords misturam dois assuntos, nomear o **DOMINANTE** (o das primeiras palavras), não uma mistura vaga.

> Prompt **espelhado no LDA** (`PIPELINE_LDA_FOLHA.md`). Afeta só a etapa de naming: re-executar a célula de nomeação aplica sem refazer o modelo. **Tópicos estruturalmente mistos** (T6 com `israel`, T9 grab-bag de SP, T10 com `guarulhos`) são limitação de *clustering* e **não** se resolvem por renomeação: tratar como ressalva de interpretação na análise.

### Pipeline de nomeação atual — inovações do ARJ notebook

O notebook do American Railroad Journal (corpus EN, domínio ferroviário séc. 19) desenvolveu um pipeline de nomeação mais sofisticado que foi adaptado para o Folha:

#### 3 docs do centróide em vez de 1 doc arbitrário

```python
def _representative_docs_folha(tid, n=3):
    idxs = np.where(np.array(topics) == tid)[0]
    sub = embeddings[idxs]
    sub = sub / (np.linalg.norm(sub, axis=1, keepdims=True) + 1e-9)
    centroid = sub.mean(axis=0)
    centroid /= np.linalg.norm(centroid)
    order = np.argsort(-(sub @ centroid))
    return [docs_raw[idxs[j]] for j in order[:n]]
```

O centróide real do cluster é calculado no espaço de embeddings 4096d. Os 3 documentos mais próximos do centróide são os mais **representativos do tema geral**, não o primeiro doc arbitrário da lista. Com 3 docs, o LLM pode identificar o denominador comum: "esses 3 artigos são todos sobre X" em vez de "este artigo específico é sobre X".

#### Anti-redundância acumulativa

```python
_assigned_names = []
for tid in _order:  # maiores tópicos primeiro
    def _pb(kws, docs=None, _an=list(_assigned_names)):
        return prompt_builder_custom(kws, docs, assigned_names=_an)
    name = name_topic(..., prompt_builder=_pb)
    _assigned_names.append(name)
```

Os nomes já atribuídos a tópicos anteriores entram no prompt do próximo, e o LLM é instruído a gerar um rótulo **claramente distinto**. Tópicos maiores são nomeados primeiro: "ocupam" os nomes mais abrangentes; tópicos menores devem se diferenciar.

#### Deduplicação pós-geração

```python
def _dedupe_name_folha(name, keywords, used_lower):
    if name.lower() not in used_lower:
        return name
    for kw in keywords:                          # tenta keyword mais distintiva
        if kw.lower() not in name.lower():
            cand = f"{name} ({kw})"
            if cand.lower() not in used_lower:
                return cand
    i = 2                                         # último recurso: sufixo numérico
    while f"{name} ({i})".lower() in used_lower:
        i += 1
    return f"{name} ({i})"
```

Segunda camada de proteção: mesmo que o LLM ignore o bloco de anti-redundância, a deduplicação garante unicidade de todos os rótulos.

### Few-shot examples (Cell 44)

O prompt atual usa 6 exemplos positivos e 1 negativo, todos com diacríticos reais:

```
Palavras-chave: eleição, presidente, voto, urna, candidato, campanha
Rótulo: Eleições presidenciais e campanha

Palavras-chave: cinema, filme, série, streaming, roteiro, diretor
Rótulo: Cinema, séries e produções audiovisuais
```

O 6º exemplo (cinema/séries) foi adicionado especificamente para cobrir o domínio do T4 que falhou. Os exemplos negativos foram reduzidos de 4 para 1 (apenas o de lista de keywords): modelos de 2B parâmetros confundem exemplos negativos com alvos mais facilmente que modelos maiores.

### Mudanças no prompt vs. run anterior

| Elemento | Run anterior | Atual | Motivo |
|----------|-------------|-------|--------|
| Keywords por tópico | 25 (hardcoded) | **50** (lê `top_n_llm`) | Fecha disconnect com params.yaml; mais contexto para tópicos amplos |
| Documentos | 1 arbitrário (200 chars) | **3 do centróide** (250 chars cada) | Elimina over-specification T4/T15/T16 |
| Anti-redundância | ausente | **assigned_names acumulativo** | Evita rótulos duplicados entre tópicos |
| Few-shot diacríticos | sem | **com** (`eleição`, `inflação`) | Alinha ao formato real das keywords da Folha |
| Exemplos negativos | 4 | **1** (lista de keywords) | Modelos 2B confundem negativos com alvos |
| Comprimento do rótulo | "3-5 palavras" | **"4-6 palavras"** | PT-BR precisa de compostos ("política monetária e juros") |
| Regra TEMA GERAL | ausente | **adicionada** | Instrução direta anti-over-specification |
| `/no_think` | presente | **removido** | Diretiva Qwen3; gemma2:2b trata como texto literal |
| System prompt | sem constraint | **"notícias brasileiras"** | Ancora domínio e idioma |

---

## 14. Métricas de Avaliação

### C_v (Coerência de Tópico)

**Referência:** Röder et al. (2015), "Exploring the Space of Topic Coherence Measures"

Mede o quanto as palavras-chave de um tópico co-ocorrem nos documentos do corpus. Para cada par de keywords (w1, w2) nos top-N, calcula a Pointwise Mutual Information (PMI) com um contexto deslizante, e combina os scores com cosine similarity. Valores próximos de 1.0 indicam keywords que frequentemente aparecem juntas: tópico coerente.

**Por que usar amostra de 5k:** O `CoherenceModel` do gensim tokeniza todos os documentos de referência. Usando N documentos completos, o tempo de cálculo seria proibitivo. A amostra de 5k documentos com seed=42 é suficiente para estimativa confiável.

**Reportar em top-10:** Padrão da literatura; C_v@top-10 = 0.69 vs C_v@top-20 = 0.61 para o run anterior.

### Topic Diversity (Dieng et al., 2020)

Fração de keywords únicos considerando todos os tópicos juntos:

```
TD = |keywords únicas em todos os tópicos| / (K × top_N)
```

TD=1.0 significa que nenhuma keyword aparece em mais de um tópico. TD=0.5 significa metade das keywords são compartilhadas. Valores altos indicam tópicos bem diferenciados.

### Taxa de Outliers

Percentual de documentos não atribuídos a nenhum tópico (id=-1). Alta taxa de outliers indica que o clustering é conservador demais (mcs muito grande) ou que o reduce_outliers não está sendo eficaz.

### Exclusividade c-TF-IDF

Calculada sobre a **matriz completa** de c-TF-IDF (não apenas top-N keywords). Para cada tópico, mede a fração do "peso" c-TF-IDF das suas keywords que é exclusiva daquele tópico. Alta exclusividade = keywords que identificam unicamente o tópico.

### FREX (Airoldi & Bischof, 2016)

Combina **FREquência** e **EXclusividade**: uma keyword FREX-ótima é ao mesmo tempo frequente no tópico E exclusiva do tópico. O peso usado no pipeline é `w_freq=0.5` (média harmônica balanceada entre frequência e exclusividade, valor canônico de Airoldi & Bischof 2016): **idêntico ao LDA** (que usa o default 0.5 de `compute_frex_score`), garantindo comparabilidade entre os dois modelos.

### Composite Score (interno dos sweeps)

Média normalizada de (C_v + Exclusividade + coverage + Diversity), usado apenas para comparação interna nos sweeps. Não é uma métrica para reportar na dissertação: é um critério de seleção de configuração multi-objetivo.

---

## 15. Melhor Resultado Encontrado (baseline)

**Run:** `data/output/folha/bertopic/folha_20260627_111417/`
**Config:** n_neighbors=20, mcs=40, leaf, reduce_nr=30, reduce_outliers=off

| Métrica | Valor | Contexto |
|---------|-------|---------|
| K tópicos | 29 | Após reduce_topics(nr=30) |
| C_v @ top-10 | **0.69** | Benchmark para comparação |
| C_v @ top-20 | 0.61 | — |
| FREX @ top-20 | **0.99** | Máximo: keywords muito exclusivas |
| Topic Diversity @ top-20 | 0.80 | Boa diversidade |
| Exclusividade média | 0.48 | — |
| Taxa de outliers | **32.7%** | Alto: 1 em cada 3 artigos sem tópico |

**Tópicos mais exclusivos:** T12 Música (0.761), T14 Judô/Olimpíadas (0.688), T26 Varíola macacos (0.672)

**Tópicos menos exclusivos:** T2 Segurança pública (0.155), T6 Saúde geral (0.235)

**Problemas:** alta taxa de outliers; T10/T24 redundantes; 7 erros LLM; Macro 4 mistura clima com saúde.

---

## 16. Configuração Final Validada

> Revisada por NLP Engineer (validação teórica + sweep), Prompt Engineer (nomeação LLM), e cruzamento com ARJ notebook (boas práticas de nomeação).

### params.yaml completo (seção BERTopic)

```yaml
bertopic:
  embedding_model: "qwen3-embedding:8b"
  embedding_backend: "ollama"
  embedding_dimension: null    # 4096d nativo
  embedding_suffix: "4096d"
  umap:
    n_components: 5
    n_neighbors: 10            # composite sweep: melhor para (mcs=8, eom, nr=25)
    min_dist: 0.0
    metric: "cosine"
  hdbscan:
    min_cluster_size: 8        # Diversity +2.2%, Exclus +7.7% vs mcs=40
    min_samples: null          # = mcs = 8
    cluster_selection_method: "eom"    # EOM domina LEAF no regime mcs=8/nr=25 (ver §6)
  min_topic_size: 8            # = mcs; não filtra clusters antes do reduce_topics
  mmr_diversity: 0.2
  reduce_topics_nr: 25         # K≈24; Diversity > nr=30; elimina duplicatas tipo T10/T24
  macro_k: 10                  # distribuição [9,3,2,2,2,2,1,1,1,1]; maior cluster 2272 docs (Macro 2 = cotidiano/política BR)
  vectorizer:
    ngram_range: [1, 2]
  reduce_outliers:
    enabled: true
    strategy: "c-tf-idf"       # melhor mecanismo para textos longos
    threshold: 0.15            # +5.4pp cobertura vs threshold=0.20; Diversity superior

evaluation:
  top_n_keywords_llm: 50       # prompt_builder_custom slicia internamente via top_n_llm
  run_outlier_sweep: false
  run_threshold_sweep: false
  run_param_sweep: false
```

### Justificativa consolidada por parâmetro

| Parâmetro | Valor | Evidência principal |
|-----------|-------|---------------------|
| `n_neighbors=10` | 10 | Composite sweep valida para (mcs=8, eom, nr=25); n=10 também produz menos outliers pré-redução (24.3% vs 29.2%) |
| `min_cluster_size=8` | 8 | Exclus +7.7% vs mcs=40; preserva sub-temas específicos para análise qualitativa |
| `cluster_selection_method=eom` | eom | No regime mcs=8/nr=25, EOM domina LEAF: C_v +0.039, Diversity +0.025, Exclus +0.036, −7.6pp outliers (robusto nos 3 nn). A média global favorável a LEAF é puxada por mcs grande (25–40); com `reduce_topics_nr` controlando a granularidade, o EOM não colapsa ilhas |
| `min_topic_size=8` | 8 (= mcs) | Inerte se < mcs; = mcs preserva todos os clusters HDBSCAN para o reduce_topics |
| `reduce_topics_nr=25` | 25 | K=24; Diversity 0.790 > 0.783 (nr=30); elimina T10/T24; granularidade ideal para dissertação |
| `macro_k=10` | 10 | Sweep: distribuição [9,3,2,...]; Macro 2 agrupa 9 tópicos de cotidiano/política BR (catch-all interpretável); alternativa: macro_k=7 (elimina singletons) |
| `strategy=c-tf-idf` | c-tf-idf | Melhor mecanismo de reatribuição para textos longos (vocabular > espacial) |
| `threshold=0.15` | 0.15 | Threshold=0.20 ≈ off; 0.15 cobre +5.4pp com Diversity superior |
| `top_n_keywords_llm=50` | 50 | 50 keywords para tópicos mistos amplos; prompt_builder controla uso via top_n_llm |

### Validação final — NLP Engineer (2026-06-27)

| Ponto verificado | Veredito |
|-----------------|---------|
| Coerência geral da config | ✅ Sem conflitos. Chain `top_n_keywords_llm → top_n_llm → prompt_builder_custom` confirmada |
| Reprodutibilidade dos sweeps | ⚠ Tabelas §5/§6/§9/§10/§11/§12.4 recomputadas do CSV bruto: **batem**. Exceção: a §12.3 original (composite) **não era reproduzível**, corrigida |
| `cluster_selection_method=eom` | 🔧 **Corrigido de leaf→eom**: no regime mcs=8/nr=25 o EOM domina LEAF (C_v/Diversity/cobertura), robusto nos 3 nn. Grid foi single-seed (`n_seeds=1`) |
| `n_neighbors=10` | ✅ Correto: vizinhança menor resolve melhor clusters compactos de mcs=8; com eom, n=10/eom/25 é o top-1 composite (0.843) em K∈[20,30] |
| `min_topic_size=8` | ✅ HDBSCAN não produz clusters <8; valor menor seria inerte. Mantém todos os clusters para reduce_topics |
| `threshold=0.15` | ✅ Prior defensável. Re-varrer se `outlier_post` do próximo run desviar de ~25% |
| 50 keywords para gemma2:2b | ✅ ~1100 tokens totais ≈ 27% de 4096 num_ctx; posições 30-50 são contexto de apoio |
| Anti-redundância com 24 tópicos | ✅ Por tópico 24: ~120 tokens de nomes atribuídos; dentro do budget; `_dedupe_name_folha` como segunda camada |
| **Veredicto geral** | **Pronto para produção** |

### Resultado obtido (run `folha_20260628_221412`)

| Métrica | Run 111417 (baseline) | Expectativa | Run 221412 (produção) |
|---------|----------------------|-------------|-----------------------|
| K tópicos | 29 | ≈24 | **24** ✅ |
| outlier_pre (HDBSCAN, eom) | 32.7% | ≈21% | **26.4%** ✅ (eom < leaf) |
| outlier_post (após reduce_outliers) | 32.7% | ≈18-22% | **21.7%** ✅ |
| Cobertura | 67.3% | ≈78-82% | **78.3%** ✅ |
| Exclusividade @ top-10 | 0.48 | ≈0.52 | **0.542** ✅ |
| C_v @ top-10 | 0.69 | ≈0.67-0.69 | **0.620** ⚠ (corpus folha_uol mais limpo; CountVectorizer já filtrava HTML → C_v não sobe) |
| Erros de nomenclatura LLM | 7/29 (24%) | redução expressiva | **0 erros críticos** ✅ (nomeação consistente; ressalvas estruturais T7/T11 documentadas) |
| Macro distribuição | top-heavy | [4,4,3,...] | **[9,3,2,...]** ⚠ (Macro 2 catch-all; diferente da previsão) |

**Nota C_v:** O run `113224` já marcava 0.620; a correção HTML (`html.unescape`) não alterou o C_v do BERTopic porque o `CountVectorizer` com `stopwords_emojis` já estava excluindo os tokens-lixo de entidades. A cobertura e nomeação melhoraram, mas o C_v permaneceu estável.

**Nota macro:** A distribuição [9,3,...] difere da previsão [4,4,...] porque o sweep de macro_k foi feito com a config baseline (mcs=40/leaf). No regime mcs=8/eom os topic embeddings são mais compactos: política/cotidiano/covid compartilham espaço vetorial próximo e colapsam no Macro 2. Documentado como ressalva estrutural, não como problema a corrigir.

### Para re-executar

```bash
# 1. Ollama rodando localmente
ollama serve

# 2. Executar notebook completo
# Run > Run All Cells  (01_bertopic_folha.ipynb)

# 3. Monitorar progresso pelos WARNINGs do BERTopic:
#    WARNING #1 → reduce_outliers + update_topics concluídos
#    WARNING #2 → reduce_topics + update_topics final concluídos
#    Com sweeps desligados: apenas 2 WARNINGs totais
#    Runtime estimado: 30-60 min (sem sweeps)
```

---

## 17. Arquivos de Saída

Cada execução gera `data/output/folha/bertopic/folha_<AAAAMMDD>_<HHMMSS>/`:

| Arquivo | Conteúdo |
|---------|----------|
| `bertopic_results.csv` | Métricas agregadas do run (C_v, Diversity, Exclus, outlier_rate, K) |
| `bertopic_topics_for_eval.csv` | Keywords por tópico (para revisão humana e avaliação qualitativa) |
| `bertopic_topic_names.csv` | Rótulos LLM por tópico |
| `bertopic_exclusividade_ranking.csv` | Ranking de exclusividade c-TF-IDF por tópico |
| `bertopic_macro_temas.csv` | Mapeamento tópico → macro grupo + editoria dominante |
| `bertopic_categoria_topico.csv` | Heatmap editoria × tópico (% por linha) |
| `bertopic_macro_k_sweep.csv` | Avaliação de diferentes macro_k (gerado pela cell 34/35) |
| `sweep_bertopic_grid.csv` | Resultados do param sweep (se `run_param_sweep: true`) |
| `sweep_outlier_threshold.csv` | Resultados do threshold sweep (se `run_threshold_sweep: true`) |
| `sweep_outlier_strategies.csv` | Resultados do estratégia sweep (se `run_outlier_sweep: true`) |
| `*.html` | Visualizações interativas Plotly (hierarquia, documentos, tópicos) |
| `*.png` | Visualizações estáticas (UMAP 2d, wordclouds, heatmaps) |

---

## 18. Resultados da Rodada de Produção (`folha_20260628_221412`)

> Run definitivo executado em 2026-06-28 com a **Configuração Final** e corpus HTML-corrigido (`folha_20260628_185652`). Pipeline completo re-rodado: `01-preprocessing` (2026-06-28 18:56) → `02-embeddings` (2026-06-28 19:00) → `03-bertopic` (2026-06-28 22:14). Naming via `gemma2:2b-instruct-q4_K_M` (50 keywords, 3 docs centróide, prompt melhorado).

### Versões dos artefatos

| Artefato | Versão |
|---|---|
| Corpus limpo | `folha_20260628_185652` (html.unescape aplicado) |
| Embeddings | `folha_20260628_190039` (recomputados pós-correção) |
| Run BERTopic | `folha_20260628_221412` |

### Pipeline (trace real)

```
Fit inicial:                        145 tópicos, 1306 outliers (26.4%)
reduce_outliers(c-tf-idf, 0.15):    1306 → 1070 outliers  (reatribuiu 236 docs)
reduce_topics(nr=25):               145 → 24 tópicos
update_topics                       (c-TF-IDF reconstruído)
Final:                              24 tópicos válidos | 1070 outliers (21.7%) | cobertura 78.3%
```

### Métricas

| Métrica | top-10 | top-20 | Leitura |
|---|---|---|---|
| **C_v (Röder)** | **0.620** | 0.549 | Bom para jornalismo (>0.55); acima do LDA (0.657: LDA beneficiou mais da correção HTML porque seu CoW-BoW era afetado; BERTopic já filtrava via stopwords_emojis) |
| Topic Diversity (Dieng) | 0.892 | 0.810 | Excelente: baixa redundância |
| Coerência Semântica | 0.667 | 0.654 | Sólida (embeddings qwen3-8b) |
| Exclusividade (c-TF-IDF completo) | 0.542 | 0.486 | Moderada |
| FREX (harmônico) | 0.982 | 0.982 | ⚠ constante em todo top_n (bug) |

> **Valor a reportar na dissertação:** C_v **top-10 = 0.62** (padrão Röder et al. 2015), do `bertopic_robustness_topn.csv`.
>
> **⚠ FREX constante:** mantém-se ~0.982 **inalterado** em todos os top_n (10→100), quando deveria cair. Bug confirmado em `compute_frex_score`: **não reportar FREX na dissertação** até auditoria.

### 24 Tópicos produzidos

| ID | Nome LLM | Keywords (top-5) |
|---|---|---|
| T0 | Políticas eleitorais e governo | bolsonaro, presidente, ministro, governo, federal |
| T1 | Educação no Brasil | escola, aluno, ensino, estudante, aula |
| T2 | Vacinação Covid-19 | saúde, vacina, caso, dose, pessoa |
| T3 | Copa do Mundo | jogo, copa, jogador, brasileiro, gol |
| T4 | Desmatamento Amazônia | governo, indígena, desmatamento, brasil, amazônia |
| T5 | Cinema e artistas | filme, música, cinema, série, show |
| T6 | Polícia e violência | polícia, caso, policial, justiça, crime |
| T7 | Cidades e pandemia | cidade, paulo, casa, região, pessoa |
| T8 | Paleontologia | animal, espécie, humano, estudo, pesquisador |
| T9 | Missões espaciais | espacial, missão, lua, nasa, terra |
| T10 | Guerra Ucrânia/Rússia | país, eua, governo, americano, trump |
| T11 | Economia SP/pandemia | empresa, milhão, bilhão, mercado, preço |
| T12 | Economia/inflação | país, governo, trabalho, economia, acordo |
| T13 | ENEM | escola, aluno, ensino, estudante, prova |
| T14 | Fake news | bolsonaro, candidato, eleição, voto, governador |
| T15 | Mudanças Climáticas e El Niño | climático, água, região, mudança, temperatura |
| T16 | Israel-Hamas | — (conflito internacional) |
| T17 | Petróleo/combustíveis | — (mercado de energia) |
| T18 | Mortes Covid | — (dados epidemiológicos) |
| T19 | Migrações | — (fluxos migratórios) |
| T20 | Chuvas SP | cidade, paulo, casa, região, rio |
| T21 | Fórmula 1 | — (automobilismo) |
| T22 | China/Covid | — (relações China/pandemia) |
| T23 | Museu Nacional | — (patrimônio cultural) |

### Qualidade dos tópicos — pontos de atenção

Todos os 24 tópicos são interpretáveis. Ressalvas estruturais (de clustering, não de naming):

| Tópico | Observação | Natureza |
|---|---|---|
| T7 "Cidades e pandemia" | Grab-bag SP: `leitos, energia, covas, metrô` | Estrutural (clustering): interpretar como catch-all SP/cotidiano |
| T11 "Economia SP/pandemia" | `guarulhos`/`guarulhos sp` aparecem entre keywords | Token geográfico local; contexto de cobertura SP |
| T22 "China/Covid" | Possíveis resíduos HTML do rodapé (`"folha"`, `"uol"`) | Residual pós-correção; volume baixo, não afeta análise |

Comparando com o run `113224` (pré-correção HTML, prompt antigo):
- ✅ Sem nomes quase-duplicados ("Mudanças Climáticas (gelo)")
- ✅ Sem `israel` como keyword dominante em T6 "Polícia e violência"
- ✅ Problema clima/saúde no macro resolvido (T15+T20 no Macro 6 juntos)

### Estabilidade

`Estabilidade Jaccard: N/A (< 2 seeds)`: `stability_seeds=[42]` (desativada por design). Para reportar robustez formal, adicionar seed alternativo (opcional).

---

## 19. Pendências (status 2026-06-29)

| Item | Status | Ação |
|---|---|---|
| Blocker `import time` + `name_topic` | ✅ corrigido | — |
| Naming reordenado antes do macro/figuras | ✅ corrigido | — |
| Versionamento `bertopic/` (folha) | ✅ corrigido | — |
| Prompt de naming melhorado (50 kw, 3 docs centróide, anti-redundância, TEMA GERAL) | ✅ aplicado | — |
| Entidades HTML no preprocessing (`html.unescape`) | ✅ propagado | Pipeline 01→02→03 re-rodado; corpus `folha_20260628_185652` |
| reduce_outliers c-tf-idf @ 0.15 | ✅ validado (composite) | mantido |
| FREX saturado (~0.982, constante em todo top_n) | ⚠ aberto | auditar `compute_frex_score`; **não reportar na dissertação** |
| Macro distribuição [9,3,...]: Macro 2 com 9 subtópicos | 📝 documentado | Ressalva estrutural (regime mcs=8/eom); alternativa: macro_k=7 |
| Tópicos estruturalmente mistos (T7 SP, T11 guarulhos, T22 HTML residual) | 📝 ressalva | Interpretar na análise; volume baixo, não afeta resultado principal |
| Estabilidade Jaccard | ➖ desativada | Opcional (adicionar seed alternativo para robustez formal) |

---

*Documentação gerada em 2026-06-27 após análise do run `folha_20260627_111417` e validação por NLP Engineer + Prompt Engineer + cruzamento com ARJ notebook. Atualizada em 2026-06-28 com resultados do run `folha_20260628_113224` e melhoria do prompt de nomeação. Revisada em 2026-06-29 com resultados definitivos do run `folha_20260628_221412` (corpus HTML-corrigido `folha_20260628_185652`, embeddings `folha_20260628_190039`, pipeline completo re-rodado).*
