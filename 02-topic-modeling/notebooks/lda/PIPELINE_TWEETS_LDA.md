# Tweets BR 2022 — Documentação do Corpus e Estratégia de Amostragem

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


> **Corpus:** Tweets eleições presidenciais Brasil 2022, PT-BR, textos curtos  
> **Notebooks:** `lda/02_lda_tweets_bre2022.ipynb` · `bertopic/01_bertopic_tweets_bre2022.ipynb`  
> **Configuração:** `01-preprocessing/configs/params.yaml` → corpus `tweets_bre2022`  
> **Última revisão:** 2026-07-01: run de produção do LDA concluído (`tweets_bre2022_20260701_214134`), K=20 via grid search, resultados analisados neste documento

---

## Índice

1. [Origem do dataset](#1-origem-do-dataset)
2. [Distribuição bruta por mês](#2-distribuição-bruta-por-mês)
3. [Por que setembro está ausente](#3-por-que-setembro-está-ausente)
4. [Estratégia de amostragem e decisões](#4-estratégia-de-amostragem-e-decisões)
5. [Pipeline de pré-processamento e atrito](#5-pipeline-de-pré-processamento-e-atrito)
6. [Corpus final](#6-corpus-final)
7. [Características para topic modeling](#7-características-para-topic-modeling)
8. [Configuração params.yaml](#8-configuração-paramsyaml)
9. [Lematização e vocabulário](#9-lematização-e-vocabulário)
10. [Seleção de K — grid search C_v](#10-seleção-de-k--grid-search-c_v)
11. [Grid alpha × eta e treino final](#11-grid-alpha--eta-e-treino-final)
12. [Nomeação de tópicos via LLM](#12-nomeação-de-tópicos-via-llm)
13. [Resultados da Rodada de Produção (`tweets_bre2022_20260701_214134`)](#13-resultados-da-rodada-de-produção-tweets_bre2022_20260701_214134)
14. [Comparação com Folha e com BERTopic (tweets)](#14-comparação-com-folha-e-com-bertopic-tweets)
15. [Pendências e arquivos de saída](#15-pendências-e-arquivos-de-saída)

---

## 1. Origem do dataset

**Fonte:** Zenodo 14834749: *Brazilian Elections 2022 Twitter Dataset*  
**Dataset original:** ~7 milhões de tweets coletados durante o período eleitoral brasileiro de 2022  
**Amostra disponível localmente:** 200.000 tweets (`data/raw/tweets_bre2022/tweets_bre2022.csv`, 163 MB)

A amostra de 200k foi extraída via amostragem uniforme aleatória (seed=42) do dataset completo do Zenodo. O arquivo bruto contém as colunas:

| Coluna original | Renomeada para | Descrição |
|---|---|---|
| `tweet_content` | `message` | Texto do tweet |
| `created_at` | `data` | Data/hora UTC |
| `tweet_id` | `tweet_id` | ID original do Twitter |
| `post_id` | `post_id` | ID interno do dataset Zenodo |
| `user`, `user_info` | — | Metadados do autor |
| `has_mention`, `mentions`, etc. | — | Metadados de interação |

---

## 2. Distribuição bruta por mês

A coluna `category` (YYYY-MM) é derivada de `data` via `derive_category_from_date` no pré-processamento.

Distribuição nos 200k tweets brutos:

| Mês | Tweets brutos | % do total válido |
|---|---|---|
| 2022-08 | 29.489 | 15,5% |
| **2022-09** | **0** | **0%: AUSENTE** |
| 2022-10 | 21.690 | 11,4% |
| 2022-11 | 64.940 | 34,1% |
| 2022-12 | 74.372 | 39,1% |
| Datas inválidas (NaT) | 9.509 | — |
| **Total válido** | **190.491** | |

> A concentração em novembro e dezembro reflete o calendário eleitoral: o 2º turno foi em 30/10/2022, e a polarização continuou intensa até a posse em 01/01/2023.

---

## 3. Por que setembro está ausente

Setembro de 2022 tem **zero tweets** na amostra de 200k, apesar de ser um mês de alta atividade política (campanha pré-1º turno, 02/10/2022).

A causa provável é a amostragem uniforme sobre o dataset do Zenodo: se o dataset original não cobriu setembro de forma sistemática (por exemplo, por falha de coleta no período), nenhuma instância de setembro aparece na amostra de 200k.

**Decisão:** setembro foi **removido das categorias-alvo** no `params.yaml`. O corpus final cobre os 4 meses disponíveis: agosto, outubro, novembro e dezembro de 2022.

---

## 4. Estratégia de amostragem e decisões

### Objetivo

Produzir um corpus balanceado de ~10k tweets para topic modeling, com representação dos principais momentos da campanha:

| Mês | Evento eleitoral |
|---|---|
| 2022-08 | Campanha presidencial: início oficial (16/08) |
| 2022-10 | 1º turno (02/10) e 2º turno (30/10) |
| 2022-11 | Pós-eleição, contestações e transição |
| 2022-12 | Acampamentos pró-golpe, posse prevista |

### Problema: atrito no pré-processamento

A primeira tentativa usou `n_per_category=2000`, resultando em apenas **4.240 tweets** finais. O motivo foi que o pré-processamento (deduplicação de retweets + detecção de idioma) descarta ~40–70% dos tweets por mês antes de atingir o corpus limpo:

| Mês | `n=2.000` amostrados | Sobreviveram | Taxa de retenção |
|---|---|---|---|
| 2022-08 | 2.000 | 1.210 | 60,5% |
| 2022-10 | 2.000 | 684 | 34,2% |
| 2022-11 | 2.000 | 1.052 | 52,6% |
| 2022-12 | 2.000 | 1.294 | 64,7% |

### Solução: sobre-amostragem no pré-processamento

Para compensar o atrito, `n_per_category` foi aumentado para **5.000**, valor que alimenta o pipeline com mais tweets brutos antes dos filtros, sem risco de esgotar nenhum mês (o menor pool, outubro, tem 21.690 disponíveis):

| Mês | `n=5.000` amostrados | Sobreviveram | Taxa de retenção |
|---|---|---|---|
| 2022-08 | 5.000 | 2.621 | 52,4% |
| 2022-10 | 5.000 | 1.381 | 27,6% |
| 2022-11 | 5.000 | 2.070 | 41,4% |
| 2022-12 | 5.000 | 2.739 | 54,8% |

> Outubro tem retenção consistentemente baixa (~28–34%) mesmo com mais amostras. Isso indica que os tweets de outubro têm alta proporção de retweets e/ou texto não-português detectado pelo `langdetect`, possivelmente reflexo do pico de engajamento (período de votação gera muito retweet de conteúdo institucional).

---

## 5. Pipeline de pré-processamento e atrito

```
200.000 tweets brutos (tweets_bre2022.csv)
        │
        ▼
[1.2] Rename: tweet_content→message, created_at→data
[1.3] add_constants: platform=twitter, election=br_2022
        │
        ▼
[1.5a] derive_category_from_date: data → category (YYYY-MM)
        Filtro implícito: datas inválidas (NaT) → sem categoria → excluídos na amostragem
        190.491 tweets com categoria válida
        │
        ▼
[1.5]  Amostragem estratificada (n_per_category=5.000, seed=42)
        Categorias: ["2022-08", "2022-10", "2022-11", "2022-12"]
        → 20.000 tweets selecionados (5.000 × 4 meses)
        │
        ▼
[2]    Deduplicação de texto exato
        Principal fonte de atrito: retweets têm o prefixo "RT @user: " removido
        pelo demojize, mas o texto citado é frequentemente idêntico
        │
        ▼
[3]    Limpeza: demojize PT-BR, collapse de caracteres repetidos
        URLs e @mentions preservados (relevantes para contexto eleitoral)
        │
        ▼
[4]    Detecção de idioma (langdetect)
        Tweets curtos têm classificação instável — fonte significativa de perdas
        em outubro (campanha com muito conteúdo compartilhado de outras línguas)
        │
        ▼
[5]    Filtros de comprimento: min_doc_length=20 chars, min_words=3
        │
        ▼
[6]    Atribuição de post_id sequencial
        │
        ▼
corpus_limpo.csv  →  8.811 tweets
```

---

## 6. Corpus final

**Versão de produção:** `tweets_bre2022_20260629_215159`  
**Arquivo:** `01-preprocessing/data/output/tweets_bre2022/tweets_bre2022_20260629_215159/corpus_limpo.csv`

### Distribuição por mês

| Mês | Tweets | % do corpus | Palavras médias | Mediana |
|---|---|---|---|---|
| 2022-08 | 2.621 | 29,7% | 19,3 | 16 |
| 2022-10 | 1.381 | 15,7% | 21,6 | 18 |
| 2022-11 | 2.070 | 23,5% | 23,5 | 20 |
| 2022-12 | 2.739 | 31,1% | 26,1 | 24 |
| **Total** | **8.811** | | **22,8 média** | **19 mediana** |

> Outubro está sub-representado (15,7% vs ~25% esperado em distribuição uniforme). Isso é estrutural: reflexo do alto atrito desse mês no pré-processamento, não de viés na amostragem.

### Estatísticas de comprimento (palavras por tweet)

```
count   8.811
mean    22,8
std     13,1
min      5
P25     12
P50     19
P75     33
max     88
```

Distribuição característica de textos curtos: mediana de 19 palavras vs ~300 palavras/doc na Folha. Isso tem impacto direto nas escolhas do BERTopic (`min_cluster_size` menor, `min_samples` menor para lidar com ruído) e no LDA (vocabulário mais enxuto por `filter_extremes`).

---

## 7. Características para topic modeling

| Característica | Valor | Impacto |
|---|---|---|
| Comprimento médio | 22,8 palavras | Embeddings qwen3 de textos curtos têm menor densidade semântica |
| Setembro ausente | 0 tweets | Cobertura temporal: ago–out–nov–dez 2022 |
| Outubro sub-representado | 1.381 (15,7%) | Análise temporal comparativa deve considerar peso diferente |
| @mentions e #hashtags preservados | sim | Enriquecem semântica eleitoral no BERTopic |
| Retweets | removidos por dedup | Reduz redundância, mas remove sinal de amplificação |
| Idioma | PT-BR (langdetect) | ~40–70% de atrito por mês inclui ruído multilíngue |
| Contexto temporal | 4 meses eleitorais | `category` disponível para validação temporal no 03 |

### Ajustes esperados nos modelos em relação à Folha

**BERTopic:**
- `min_cluster_size` menor (textos curtos formam clusters menores)
- `min_samples` menor (reduz outliers, que são especialmente altos em textos curtos)
- `reduce_outliers` pode precisar de sweep específico: parâmetros ótimos da Folha não generalizam diretamente

**LDA:**
- `filter_extremes` com `no_below` provavelmente menor que 5 (corpus menor, vocab mais restrito)
- K provavelmente menor que 20 (menos diversidade temática esperada num corpus focado numa eleição)
- Tópicos de ruído (URLs, @mentions repetidos) mais prováveis

---

## 8. Configuração params.yaml

```yaml
# 01-preprocessing/configs/params.yaml
tweets_bre2022:
  derive_category_from_date:
    date_column: data        # após rename (created_at → data)
    format: "%Y-%m"
    output_column: category

  sample:
    n_per_category: 5000     # sobre-amostragem para compensar atrito (~28-55% por mês)
    category_column: category
    categories:
      - "2022-08"
      # "2022-09" ausente do dataset bruto (0 tweets nos 200k)
      - "2022-10"
      - "2022-11"
      - "2022-12"
```

```yaml
# 03-topic-modeling/configs/params.yaml  (preenchido — LDA não usa bertopic_overrides, ver §9-11)
tweets_bre2022:
  text_column: message
  id_column: post_id
  subdir: tweets_bre2022
  language: pt
  date_column: data
```

> Nota (2026-07-01): esse segundo bloco estava marcado como "a preencher" numa revisão anterior deste doc. Já está preenchido e em uso (`03-topic-modeling/configs/params.yaml`); o LDA não tem um mecanismo de override por corpus análogo ao `bertopic_overrides` do BERTopic (`PIPELINE_TWEETS_BERTOPIC.md`) porque não há UMAP/HDBSCAN para calibrar: os únicos hiperparâmetros do LDA (K, alpha, eta) já são resolvidos por grid search por corpus nas células do próprio notebook, sem necessidade de bifurcação no `params.yaml`.
---

## 9. Lematização e vocabulário

Mesma função `lemmatize_corpus` (spaCy `pt_core_news_lg`, single-process) usada pela Folha, mas o resultado é bem mais enxuto, o esperado para textos curtos:

| Métrica | Tweets | Folha (referência) |
|---|---|---|
| Tempo de lematização | 24s | ~448s (7min) |
| Vocabulário final (após `filter_extremes`) | 2.617 palavras | (corpus maior, vocab bem mais amplo) |
| Tokens/doc após lematizar (média) | 10,9 | — |
| Tokens únicos/doc (média) | 10,4 | — |
| Tokens únicos/doc no BoW (média) | 8,8 | — |

O vocabulário de 2.617 palavras para 8.811 tweets é compatível com a expectativa registrada na Seção 7 ("vocabulário mais enxuto por `filter_extremes`"): textos curtos convergem rápido para um vocabulário compacto e repetitivo (hashtags, nomes próprios, gírias eleitorais).

---

## 10. Seleção de K — grid search C_v

Grid `K ∈ [3, 5, 7, 8, 10, 12, 15, 20, 25, 30]`, `passes=20`, seed=42, mesmo procedimento da Folha (`PIPELINE_LDA_FOLHA.md`). Primeira execução bem-sucedida para este corpus: sem cache prévio, rodou os 10 valores por completo (~13min).

| K | C_v | Perplexidade |
|---|---|---|
| 3 | 0,336 | 970 |
| 5 | 0,361 | 964 |
| 7 | 0,341 | 993 |
| 8 | 0,343 | 1.007 |
| 10 | 0,332 | 1.032 |
| 12 | 0,356 | 1.035 |
| 15 | 0,382 | 1.033 |
| **20** | **0,430 ← pico** | 1.058 |
| 25 | 0,416 | 1.111 |
| 30 | 0,397 | 1.180 |

K=20 é um pico interior (cai para 25 e 30, não é artefato de borda do range testado), mas contraria a expectativa registrada na Seção 7 ("K provavelmente menor que 20... corpus focado numa eleição"). Na prática, o corpus eleitoral parece ter granularidade temática comparável à da Folha nesse critério (que também converge em K=20), possivelmente porque a polarização gera múltiplos sub-discursos específicos (STF, orçamento secreto, caminhoneiros, Capes, etc.) em vez de um punhado de macro-temas.

Perplexidade cresce monotonicamente com K (esperado: mais tópicos = mais parâmetros = menor perplexidade *in-sample* seria o esperado, mas aqui sobe; é a métrica secundária, C_v é a que direciona a escolha, mesma convenção da Folha).

---

## 11. Grid alpha × eta e treino final

Grid `alpha ∈ [symmetric, asymmetric, 0.1, 0.5, 0.01]` × `eta ∈ [None, 0.01, 0.1]` (15 combos), `passes=10`, K=20 fixo. Cache já existia (`lda_alpha_eta_grid.csv`, de uma execução anterior que não chegou a terminar por causa de bugs corrigidos depois) e foi reaproveitado sem recomputar:

| alpha | eta | C_v | Perplexidade |
|---|---|---|---|
| **0.1** | **None** | **0,3883 ← melhor** | 1.308 |
| symmetric | None | 0,3867 | 1.233 |
| 0.1 | 0.01 | 0,3854 | 1.742 |
| 0.1 | 0.1 | 0,3770 | 1.250 |
| symmetric | 0.1 | 0,3768 | 1.170 |
| 0.01 | None | 0,3763 | 1.242 |
| symmetric | 0.01 | 0,3752 | 1.657 |
| 0.01 | 0.01 | 0,3746 | 1.679 |
| 0.5 | 0.1 | 0,3726 | 1.524 |
| asymmetric | 0.1 | 0,3647 | 1.186 |
| 0.5 | 0.01 | 0,3639 | 2.343 |
| 0.5 | None | 0,3618 | 1.608 |
| 0.01 | 0.1 | 0,3599 | 1.174 |
| asymmetric | 0.01 | 0,3547 | 1.635 |
| asymmetric | None | 0,3498 | 1.254 |

**Escolhido: `alpha=0.1, eta=None`** (C_v mais alto do grid). Nota: os valores de C_v deste grid (~0,35-0,39) são mais baixos que o pico do grid de K (0,4297) porque rodam com `passes=10` (exploratório) em vez de `passes=20` (produção), mesma convenção de custo/benefício já usada na Folha.

**Treino final:** `K=20, alpha=0.1, eta=None, passes=20` (27s).

---

## 12. Nomeação de tópicos via LLM

Mesmo mecanismo da Folha (`name_all_topics`, modelo em `params.yaml > llm.model`): **`gemma2:2b-instruct-q4_K_M`**, 20 tópicos nomeados em 63s (~3s/tópico, bem mais rápido que os ~9s/tópico estimados no comentário do notebook, provavelmente por keywords mais curtas/repetitivas do que artigos longos).

---

## 13. Resultados da Rodada de Produção (`tweets_bre2022_20260701_214134`)

> Primeiro run do LDA a rodar do início ao fim sem erros para este corpus: as duas tentativas anteriores (`tweets_bre2022_20260701_204329`/`213822`) travaram nos bugs de ordem de células e do caso single-seed, ambos corrigidos. Corpus `tweets_bre2022_20260629_215159`.

### Configuração final aplicada

| Parâmetro | Valor | Evidência |
|---|---|---|
| K | **20** | pico C_v (0,4297), interior ao range testado |
| alpha | **0.1** | melhor C_v (0,3883) no grid alpha×eta |
| eta | **None** | idem |
| passes (treino final) | 20 | fixo, mesma convenção da Folha |

### Métricas

| Métrica | Valor |
|---|---|
| C_v (recomputado no modelo final) | **0,527** |
| Exclusividade | 0,817 |
| Topic Diversity (Dieng) | 0,71 |
| Diversity entropy (θ) | 0,994 |
| Outliers | **0%**: LDA sempre atribui um tópico dominante (θ nunca é totalmente uniforme na prática) |
| Estabilidade Jaccard | N/A: `stability_seeds=[42]` = mesmo valor do `seed` global, desativada por design |

### 20 tópicos produzidos

| ID | Nome LLM | Keywords (top-5) | Docs | % |
|---|---|---|---|---|
| T0 | Debate Bolsonaro Lula e Bilhão | debatenaband, bolsonaro, lula, bilhão, deixar | 618 | 7,0% |
| T1 | Bolsonaro e Mentiras | forabolsonaro, bolsonaronojn, bonner, renata, bolsonaro | 399 | 4,5% |
| T2 | Alvorada Bolsonaro Palácio | alvorada, bolsonaro, palácio, brasília, presidente | 306 | 3,5% |
| T3 | Manifestação em Brasília | brasília, terrorista, bolsonarista, ato, esquerda | 492 | 5,6% |
| T4 | Lula e Democracia no Brasil | lula, presidente, país, deixar, governo | 420 | 4,8% |
| T5 | Saque em Brasil e Bolsonaro | brasil, jair, quebrou, lulapresidente, saqueou | 329 | 3,7% |
| T6 | Ataque a Lula | dia, lula, bolsonaro, ataque, pra | 337 | 3,8% |
| T7 | Bolsonarismo Eleitoral | bolsonaroreeleito, candidato, deus, liberdade, nao | 266 | 3,0% |
| T8 | Manifestação Bolsonaro | infiltrar, bolsonaro, gritar, fogo, colocar | 435 | 4,9% |
| T9 | Lula e Bolsonaro ⚠ | lulapresidente, vencer, amor, ódio, hoje | 359 | 4,1% |
| T10 | Bolsonaro e Jovem Brasília | nacional, xandão, jornal, bolsonaro, ladraonojn | 270 | 3,1% |
| T11 | Lula e Diplomação | lula, corte, presidente, bobo, bolsonaro | 715 | 8,1% |
| T12 | Eleições e Nordeste | votar, federal, polícia, nordeste, rodoviária | 618 | 7,0% |
| T13 | Lula e Bolsonaro ⚠ | lula, brasil, pra, ficar, tar | 323 | 3,7% |
| T14 | Bolsonaro e Bolsa Família | capes, bolsonaro, pagar, bolsa, pagueminhabolsa | 278 | 3,2% |
| T15 | Mané Eleitoral | mané, perdeu, amolar, perder, eleição | 553 | 6,3% |
| T16 | Prisão de Cacique Bolsonarista | brasília, polícia, federal, cacique, carro | 380 | 4,3% |
| T17 | Alexandre Moraes e Cacique | alexandre, moraes, prender, milhão, cacique | 407 | 4,6% |
| T18 | Globolixo e Bolsonaro | globolixo, bolsonaronojn, secreto, orçamento, presidente | 729 | 8,3% |
| T19 | Bolsonaro e Catar | bolsonaro, eduardo, catar, ladrão, brasil | 577 | 6,5% |

Distribuição bem mais equilibrada que o BERTopic: do menor (T7, 3,0%) ao maior (T18, 8,3%), sem nenhum tópico grab-bag dominante como o T0 do BERTopic (9,5% e exclusividade muito abaixo da média).

### Qualidade dos tópicos — pontos de atenção

| Tópico | Observação | Natureza |
|---|---|---|
| **T9 e T13 ⚠** | Ambos nomeados **"Lula e Bolsonaro"** pelo LLM | Keywords são de fato diferentes (T9: `vencer/amor/ódio`, tom emocional; T13: `ficar/querer/roubar`, tom acusatório), mas o nome genérico não captura a diferença. Provável limitação de granularidade da nomeação (LLM convergiu pro rótulo mais óbvio dado que ambos compartilham `lula/bolsonaro/pra/presidente`). Recomenda-se revisão manual antes de citar esses dois tópicos separadamente na dissertação |
| T13 | Keywords incluem `tar` (provável artefato de lematização de "estar/tá") | Ruído de lematização em texto informal, mesma classe de problema já esperada em textos curtos |
| T2/T16 | Ambos citam `alvorada`/`palácio`/`brasília` | Tematicamente próximos (cobertura de Brasília) mas LLM diferenciou corretamente (T2 = residência oficial; T16 = prisão/polícia) |

Nenhum tópico "morto" (sem massa de probabilidade): K=20 não fragmentou o vocabulário esparso a ponto de gerar tópicos degenerados.

---

## 14. Comparação com Folha e com BERTopic (tweets)

### Vs. Folha (`PIPELINE_LDA_FOLHA.md`, ambos K=20)

| Métrica | Folha | Tweets | Leitura |
|---|---|---|---|
| C_v (recomputado) | 0,657 | **0,527** | Folha mais coerente, inverso do observado no BERTopic (onde tweets superava a Folha); vocabulário jornalístico da Folha parece se beneficiar mais do BoW que o vocabulário de hashtags/gírias dos tweets |
| Exclusividade | 0,457 | **0,817** | Tweets bem mais exclusivo: vocabulário eleitoral específico gera fronteiras lexicais mais nítidas |
| Topic Diversity (Dieng) | 0,770 | 0,71 | Folha um pouco mais diverso |
| Outliers | 0% | 0% | LDA nunca deixa doc sem tópico em nenhum dos dois corpora, diferença estrutural central vs. BERTopic |

### Vs. BERTopic tweets (`PIPELINE_TWEETS_BERTOPIC.md`, run `tweets_bre2022_20260701_201916`, config calibrada)

| Métrica | LDA (K=20) | BERTopic (K=24) | Leitura |
|---|---|---|---|
| Coerência (C_v) | 0,527 | 0,682 (top-20 kw) | ⚠ não comparável diretamente: bases de cálculo diferentes (LDA: `coherence_cv_recomputed` sobre keywords do modelo final; BERTopic: `top_n=20` do sweep de robustez). Direção geral (BERTopic mais coerente) é consistente com o padrão já visto na Folha |
| Exclusividade | 0,817 (word-overlap simples) | 0,795 (c-TF-IDF) | Também não é apples-to-apples: métodos de cálculo distintos; ambos "altos" no seu próprio referencial |
| Outliers/cobertura | 0% / 100% | 37,6% / 62,4% | Diferença estrutural mais informativa: LDA força atribuição probabilística total, BERTopic permite "sem tópico" via HDBSCAN, trade-off clássico entre as duas famílias de modelo |
| Distribuição de docs | equilibrada (3,0%-8,3%) | concentrada (T0 sozinho = 9,5% dos docs atribuídos, mais outliers) | LDA distribui a massa de forma mais uniforme entre tópicos |

**Leitura geral:** LDA e BERTopic capturam recortes distintos do mesmo corpus: não dá para dizer que um "venceu" o outro sem fixar a métrica e a base de cálculo. A comparação mais robusta viria de rodar as duas métricas de exclusividade/coerência com a mesma implementação nos dois modelos (pendência).

---

## 15. Pendências e arquivos de saída

### Concluído (2026-07-01)

- **Bug de ordem de células (naming antes de visualização):** o notebook tinha "§7 Visualização" antes de "§8 Rotulagem via LLM", ao contrário do `02_lda_folha.ipynb`, que já corrigiu essa ordem. Cells reordenadas (naming agora vem primeiro), heatmaps phi/cosseno (que dependem de `names`) passaram a funcionar.
- **Bug do caso single-seed na Estabilidade Jaccard:** com `stability_seeds=[42]` (== `seed` global), `_other_seeds` fica vazio e o `groupby("topic_id")` quebrava com `KeyError` num DataFrame sem colunas. Portado o mesmo guard já usado no `02_lda_folha.ipynb` (`if not _other_seeds: print("...desativada...")`) para as duas células afetadas (treino dos seeds alternativos e boxplot Jaccard).
- **Grid de K e grid alpha/eta:** ambos completos e cacheados (`lda_metrics.csv` e `lda_alpha_eta_grid.csv` na pasta-base `data/output/tweets_bre2022/lda/`). Próximos runs pulam os dois grids automaticamente.

### Pendente (próximos passos)

1. **Revisar manualmente T9 vs. T13** ("Lula e Bolsonaro" duplicado): decidir se são de fato tópicos distintos (renomear com mais especificidade) ou candidatos a merge.
2. **Métricas de exclusividade/coerência com implementação unificada** entre LDA e BERTopic, para permitir comparação direta na dissertação (atualmente cada modelo usa seu próprio método de cálculo).
3. **Estabilidade multi-seed:** adicionar seeds alternativos em `params.yaml > evaluation.stability_seeds` (hoje só `[42]`) para validar robustez do K=20 e dos tópicos, mesma pendência já registrada para o BERTopic (`PIPELINE_TWEETS_BERTOPIC.md`).
4. **Cross-tab tópico × mês, t-SNE, pyLDAvis, top documentos por tópico:** gerados neste run mas não analisados neste documento (os HTML/PNG estão na pasta de output), candidatos a uma seção qualitativa futura se necessário para a dissertação.

### Arquivos de saída

Cada execução gera `data/output/tweets_bre2022/lda/tweets_bre2022_<AAAAMMDD>_<HHMMSS>/`:

| Arquivo | Conteúdo |
|---|---|
| `lda_results.csv` | Tópico × documento (atribuições finais + distribuição θ completa) |
| `lda_topics_for_eval.csv` | Keywords + nomes LLM por tópico |
| `lda_metrics.csv` | Métricas agregadas do run (também cacheia o grid de K na pasta-base) |
| `lda_topics_seed42.csv` | Atribuições do (único) seed rodado, presente mesmo com estabilidade desativada, pois o loop de treino roda para todo `s` em `stability_seeds`, incluindo o seed principal |
| `lda_heatmap_phi.png` / `lda_heatmap_topicos_cosine.png` | Heatmaps palavra×tópico e tópico×tópico |
| `lda_tsne_theta_topico.png` / `lda_tsne_theta_interactive.html` | t-SNE do espaço θ |
| `lda_pyldavis_K20.html` | Visualização interativa pyLDAvis |
| `lda_alpha_eta_grid.csv` (pasta-base) | Cache do grid alpha×eta, reutilizado entre runs |
