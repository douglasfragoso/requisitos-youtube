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


> **Corpus:** Tweets eleições presidenciais Brasil 2022: PT-BR, textos curtos  
> **Notebooks:** `lda/02_lda_tweets_bre2022.ipynb` · `bertopic/01_bertopic_tweets_bre2022.ipynb`  
> **Configuração:** `01-preprocessing/configs/params.yaml` → corpus `tweets_bre2022`  
> **Última revisão:** 2026-07-03. `reduce_outliers` recalibrado sob `nn=15/mcs=30/leaf`: sweep de estratégias/threshold re-executado (run `tweets_bre2022_20260703_094835`) e novo run de produção (`tweets_bre2022_20260703_101639`) com `threshold=0.05`. outlier_post cai de 37,6% para **16,5%** (cobertura 83,5%) com C_v/Diversity/Exclusividade praticamente inalterados. Pendência da recalibração do `reduce_outliers` encerrada.

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
9. [Pipeline BERTopic — arquitetura e diferenças em relação ao Folha](#9-pipeline-bertopic--arquitetura-e-diferenças-em-relação-ao-folha)
10. [Config compartilhada vs. override por corpus](#10-config-compartilhada-vs-override-por-corpus)
11. [Bug de investigação: ValueError no C_v do sweep (gensim)](#11-bug-de-investigação-valueerror-no-c_v-do-sweep-gensim)
12. [Sweep estrutural UMAP/HDBSCAN (108 combinações)](#12-sweep-estrutural-umaphdbscan-108-combinações)
13. [Sweep de reduce_outliers (estratégias e threshold)](#13-sweep-de-reduce_outliers-estratégias-e-threshold)
14. [Configuração final aplicada](#14-configuração-final-aplicada)
15. [Resultado baseline (config antiga, herdada do Folha)](#15-resultado-baseline-config-antiga-herdada-do-folha)
16. [Pendências e arquivos de saída](#16-pendências-e-arquivos-de-saída)
17. [Resultados da Rodada de Produção Calibrada (`tweets_bre2022_20260701_201916`)](#17-resultados-da-rodada-de-produção-calibrada-tweets_bre2022_20260701_201916)
18. [Recalibração de reduce_outliers e Resultado Definitivo (`tweets_bre2022_20260703_101639`)](#18-recalibração-de-reduce_outliers-e-resultado-definitivo-tweets_bre2022_20260703_101639)

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
# 03-topic-modeling/configs/params.yaml  (preenchido — ver §9-14 abaixo)
tweets_bre2022:
  text_column: message
  id_column: post_id
  subdir: tweets_bre2022
  language: pt
  date_column: data
  bertopic_overrides:      # ver §10
    umap: { n_neighbors: 15 }
    hdbscan: { min_cluster_size: 30, cluster_selection_method: leaf }
```

---

## 9. Pipeline BERTopic — arquitetura e diferenças em relação ao Folha

> Esta seção documenta a execução do `01_bertopic_tweets_bre2022.ipynb` no mesmo nível de profundidade do `PIPELINE_BERTOPIC_FOLHA.md`. Os **conceitos fundamentais de BERTopic** (o que é UMAP, HDBSCAN, c-TF-IDF, MMR, a ordem obrigatória do pós-processamento em 4 etapas, as métricas C_v/Diversity/Exclusividade/FREX) são **idênticos** ao Folha e não são repetidos aqui na íntegra; ver `PIPELINE_BERTOPIC_FOLHA.md` para a explicação pedagógica completa. Esta seção foca no que **é específico dos tweets**: construção dos `docs`, e os resultados dos sweeps rodados para este corpus.

### Arquitetura (idêntica ao Folha, mesmo `_helpers.py`)

```
corpus_limpo.csv + embeddings_qwen3_4096d.npy
        │
        ▼
[Cell 3]  load_corpus() → docs = df["message"] (SEM lematização/emoji injetados — ver abaixo)
        │
        ▼
[Cell 5]  get_or_compute_embeddings() → embeddings_qwen3_4096d.npy (8.811 × 4096d)
        │
        ▼
[Cell 6]  bert_cfg = params["bertopic"]; bert_cfg = apply_bertopic_overrides(bert_cfg, cfg)  ← NOVO (§10)
        │
        ▼
[Cell 7]  BERTopic.fit_transform(docs, embeddings=embeddings)
          • UMAP: 4096d → 5d (n_neighbors=15 ← override, cosine, min_dist=0.0)
          • HDBSCAN: mcs=30 ← override, leaf ← override, min_samples=null
          • c-TF-IDF + MMR (diversity=0.2)
        │
        ▼
[Cell 8]  Pós-processamento em 4 etapas (idêntico ao Folha — ordem obrigatória)
        │
        ▼
[Cell 9]  Extrair keywords (cache até top-100 por tópico)
        │
        ▼
[Cell 15] FREX + Exclusividade (c-TF-IDF completo) + Coerência Semântica
          ⚠ Coerência Semântica usa modelo de embedding DEDICADO, menor
            (qwen3-embedding:0.6b) — ver nota abaixo; independe do sweep estrutural
        │
        ▼
[Cell 18] Nomeação LLM — MESMO mecanismo do Folha (centróide 3 docs, anti-
          redundância acumulativa, dedupe), few-shot adaptado ao domínio eleitoral
        │
        ▼
[Cell 23] Sweeps de avaliação (outlier strategies/threshold/grid estrutural) — §12-13
        │
        ▼
[Cells +] Métricas, visualizações, heatmap mês×tópico, hierarquia, export
```

### Diferenças na construção dos `docs`

Ao contrário do Folha (que concatena `text_lemma + emoji_tokens`, ver `PIPELINE_BERTOPIC_FOLHA.md`), os tweets usam o texto limpo diretamente, sem lematização nem injeção de tokens emoji separados:

```python
# Cell 3 do notebook de tweets
docs     = df[TEXT_COL].astype(str).tolist()               # message (já limpo no 01-preprocessing)
docs_raw = df.get("message_raw", df[TEXT_COL]).astype(str).tolist()  # para LLM naming
```

Emojis já são demojizados em texto (`❤️` → `coração_vermelho`) durante o `01-preprocessing`, então já chegam como tokens no `message`. Não há uma etapa de lematização separada para tweets (textos curtos e informais respondem mal à lematização agressiva, que pode distorcer gírias e hashtags).

### Por que a config estrutural do Folha não se transferia diretamente

O comprimento médio dos tweets (22,8 palavras, mediana 19) é ~15× menor que o dos artigos da Folha (~300 palavras/doc). Isso afeta diretamente:

- **Densidade dos embeddings:** textos curtos produzem embeddings menos "carregados" semanticamente, então a vizinhança UMAP (`n_neighbors`) que funciona bem para artigos longos não necessariamente serve para tweets.
- **Tamanho natural dos clusters:** um `min_cluster_size=8` (calibrado para as "ilhas semânticas" de editorias jornalísticas) pode ser pequeno demais ou grande demais para tópicos eleitorais discutidos em milhares de tweets curtos e repetitivos (retweets, hashtags virais).
- **Taxa de outliers:** textos curtos e ruidosos (gírias, abreviações, hashtags) tendem a gerar mais outliers no HDBSCAN, e por isso o `params.yaml` já observava que `min_samples` é o parâmetro mais crítico para este corpus.

Por isso este corpus recebeu sua **própria rodada completa de `sweep_bertopic_grid`**, em vez de herdar os valores do Folha.

---

## 10. Config compartilhada vs. override por corpus

### O problema

O bloco `bertopic:` em `params.yaml` (umap/hdbscan/vectorizer/reduce_outliers/reduce_topics_nr) é **compartilhado entre todos os corpora**: não há bifurcação por `CORPUS_ID` nas células que o consomem (`bert_cfg = params["bertopic"]`, idêntico nas células 6 dos notebooks de Folha e tweets). Os valores atuais desse bloco são a calibração feita **para o Folha** (ver `PIPELINE_BERTOPIC_FOLHA.md`).

Isso significa que aplicar diretamente o resultado do `sweep_bertopic_grid` dos tweets nesse bloco compartilhado **sobrescreveria silenciosamente a calibração do Folha**, já que os dois notebooks leem o mesmo `params["bertopic"]`.

### Solução: `bertopic_overrides` por corpus

Foi adicionado um mecanismo de override em `_helpers.py`:

```python
def apply_bertopic_overrides(bert_cfg: dict, corpus_cfg: dict) -> dict:
    """Overlay corpus_cfg['bertopic_overrides'] sobre o bloco `bertopic:`
    compartilhado, um nível de profundidade (chaves-dict como umap/hdbscan
    são mescladas chave-a-chave; chaves escalares são substituídas).
    Retorna um dict NOVO — não muta bert_cfg nem corpus_cfg.
    """
    import copy
    merged = copy.deepcopy(bert_cfg)
    overrides = corpus_cfg.get("bertopic_overrides") or {}
    for key, val in overrides.items():
        if isinstance(val, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **val}
        else:
            merged[key] = val
    return merged
```

Aplicado na célula 6 do notebook de tweets, logo após ler o bloco compartilhado:

```python
bert_cfg = params["bertopic"]
from _helpers import apply_bertopic_overrides
bert_cfg = apply_bertopic_overrides(bert_cfg, cfg)   # cfg = corpora.tweets_bre2022
```

`cfg` (o dict de configuração do corpus, retornado por `get_corpus_config`) agora carrega:

```yaml
# corpora.tweets_bre2022.bertopic_overrides (params.yaml)
bertopic_overrides:
  umap:
    n_neighbors: 15
  hdbscan:
    min_cluster_size: 30
    cluster_selection_method: leaf
```

**Verificado:** o notebook do Folha não chama `apply_bertopic_overrides` (não precisa: `corpora.folha` não tem `bertopic_overrides`), e mesmo que chamasse, a ausência da chave faz a função devolver `bert_cfg` inalterado. Testado explicitamente: `bert_cfg` original (retornado por `params["bertopic"]`) permanece com `n_neighbors=10`/`min_cluster_size=8`/`eom` depois da chamada para o corpus `folha`. O merge é seguro e não tem efeito colateral entre corpora.

---

## 11. Bug de investigação: ValueError no C_v do sweep (gensim)

Ao rodar o `sweep_bertopic_grid` pela primeira vez para este corpus (108 combinações estruturais), a célula quebrou com:

```
ValueError: unable to interpret topic as either a list of tokens or a list of ids
```

### Causa raiz

1. O vetorizador do BERTopic usa `ngram_range: [1, 2]` (unigrama + bigrama): um tópico pode ter keywords como `"eduardo bolsonaro"`.
2. O `dictionary` do gensim usado para C_v é construído só com **unigramas** (`lemmatize_corpus`/`.split()`, célula "Métricas quantitativas completas").
3. Quando um tópico cai com **todas** as keywords sendo bigramas fora desse vocabulário unigrama, o que aconteceu em pelo menos uma das 540 combinações do grid (`n_neighbors=10, min_cluster_size=30, min_samples=None, eom, reduce_nr=30`), o `gensim.CoherenceModel` não consegue interpretar o tópico e levanta a exceção.
4. A célula principal do notebook (fora do sweep) já tinha proteção (`try/except Exception as e: cv_score = None`), mas a função interna compartilhada pelos **3 sweeps** (`sweep_bertopic_grid`, `sweep_outlier_strategies`, `sweep_outlier_threshold`), `_bertopic_sweep_metrics()` em `_helpers.py`, chamava `compute_coherence_cv()` sem essa proteção.

### Reprodução mínima (confirmou a hipótese antes de corrigir)

```python
from gensim.corpora import Dictionary
tokenized = [["eduardo", "bolsonaro", "brasil", "bandeira"], ["lula", "presidente", "voto"]]
dictionary = Dictionary(tokenized)
topics_keywords = {0: ["eduardo bolsonaro", "brasil bandeira", "bandeira brasil"]}  # 100% bigramas
compute_coherence_cv(topics_keywords, tokenized, dictionary)
# ValueError: unable to interpret topic as either a list of tokens or a list of ids
```

### Correção aplicada

Em `_bertopic_sweep_metrics()` (`_helpers.py`), o mesmo padrão já usado na célula principal:

```python
try:
    cv = compute_coherence_cv(tk_metrics, tokenized, dictionary)
except ValueError:
    # gensim exige >=1 token do topico no dictionary (unigramas); com
    # ngram_range=(1,2) um topico pode cair 100% em bigramas fora do
    # vocabulario unigrama — mesmo modo de falha ja tratado no pipeline
    # principal (celula "Metricas quantitativas completas").
    cv = float("nan")
```

`NaN` é ignorado automaticamente pelo `.mean()` do pandas na agregação (`agg_grid`), então uma única combinação degenerada não contamina a média nem derruba o sweep. Corrige os **3** sweeps simultaneamente (`sweep_bertopic_grid`, `sweep_outlier_strategies`, `sweep_outlier_threshold` compartilham essa função interna).

**Verificado em produção:** no re-run completo do grid (540 linhas), exatamente **1 linha** ficou com `C_v = NaN` (`nn=10, mcs=30, ms=None, eom, nr=30`), o que confirma que o fix foi exercitado e não mascarou nenhum outro erro.

---

## 12. Sweep estrutural UMAP/HDBSCAN (108 combinações)

**Run:** `data/output/tweets_bre2022/bertopic/tweets_bre2022_20260630_234301/sweep_bertopic_grid.csv`
**Controle:** `evaluation.run_param_sweep: true`
**Grid varrido:**

```yaml
n_neighbors:               [10, 15, 20]                 # 3 valores
min_cluster_size:          [8, 15, 25, 30, 35, 40]       # 6 valores
min_samples:                [null, 5, 15]                # 3 valores (null = default HDBSCAN)
cluster_selection_method:  [leaf, eom]                   # 2 valores
reduce_topics_nr:          [10, 15, 20, 25, 30]           # 5 valores (pós-processamento, barato)
seed:                      [42]                           # 1 seed (Stability = NaN — ver ressalva abaixo)
outlier_strategy:          "off"                          # isola efeito estrutural
```

3 × 6 × 3 × 2 = **108 refits completos** (UMAP + HDBSCAN + c-TF-IDF) × 5 valores de `reduce_topics_nr` (pós-processamento barato, sem refit) = **540 linhas** no CSV agregado.

> **Ressalva sobre Stability:** `param_sweep_seeds: [42]` usa apenas 1 seed: `Stability`/`Stab_std` ficam `NaN` para todas as linhas (a função `compute_stability` exige ≥2 seeds para calcular Jaccard entre execuções). A seleção abaixo usa apenas C_v/Diversity/Exclusividade/FREX. Rodar com 2+ seeds (dobra o tempo: ~110-140 min) validaria se a escolha é robusta a variação de seed, seguindo a mesma ressalva já documentada no Folha (`PIPELINE_BERTOPIC_FOLHA.md`).

### Escolha estrutural — composite médio sobre os 5 `reduce_topics_nr`

Isolando o efeito estrutural (média de C_v/Diversity/Exclus/FREX normalizados sobre os 5 valores de `reduce_nr`, mesmo método do Folha):

| n_neighbors | min_cluster_size | min_samples | cluster_selection_method | C_v (méd.) | Diversity (méd.) | Exclus (méd.) | FREX (méd.) | Composite |
|---|---|---|---|---|---|---|---|---|
| **15** | **30** | **null** | **leaf** | 0.666 | 0.931 | 0.834 | 0.985 | **0.841 ← escolhido** |
| 20 | 8 | null | leaf | 0.613 | 0.934 | 0.839 | 0.985 | 0.828 |
| 10 | 30 | null | leaf | 0.634 | 0.936 | 0.830 | 0.984 | 0.807 |
| 15 | 35 | null | leaf | 0.646 | 0.928 | 0.826 | 0.985 | 0.766 |
| 20 | 40 | null | leaf | 0.627 | 0.930 | 0.822 | 0.986 | 0.745 |

O `leaf` domina estruturalmente para tweets, o oposto do Folha, onde `eom` vence no regime `mcs=8/nr=25` (`PIPELINE_BERTOPIC_FOLHA.md`). Isso é consistente com a natureza do corpus: textos curtos e repetitivos (muitos retweets/hashtags virais) formam "ilhas" densas mais numerosas e menos hierárquicas. O `leaf` (que seleciona as folhas mais finas da árvore condensada) captura melhor essa estrutura do que o `eom` (que tende a mesclar em clusters maiores, adequado a poucas categorias jornalísticas bem definidas).

### Granularidade (`reduce_topics_nr`) para o combo escolhido (nn=15, mcs=30, leaf)

Mesmo critério do Folha ("melhor equilíbrio C_v/Diversity"):

| reduce_nr | K | outlier_pre | C_v | Diversity | Exclus | FREX |
|---|---|---|---|---|---|---|
| 10 | 9 | 41,6% | 0,659 | 0,956 | 0,902 | 0,971 |
| 15 | 14 | 41,6% | 0,611 | 0,946 | 0,874 | 0,983 |
| 20 | 19 | 41,6% | 0,682 | 0,937 | 0,841 | 0,988 |
| **25** | **24** | **41,6%** | **0,696** | **0,915** | **0,798** | **0,990** ← **escolhido** |
| 30 | 29 | 41,6% | 0,681 | 0,902 | 0,755 | 0,992 |

`reduce_topics_nr=25` (K=24) tem o melhor C_v entre os 5 pontos e Diversity ainda alta (0,915 vs. máximo 0,956 em nr=10), o mesmo valor já usado por herança do bloco compartilhado, então não precisou de override: o ponto ótimo do Folha (nr=25) também é o ponto ótimo para tweets, por coincidência de critério (não de valor absoluto de K: K=24 tópicos para 8.811 tweets é uma granularidade bem mais fina proporcionalmente do que K=24 para ~50k artigos da Folha).

> **outlier_pre alto (41,6%):** medido com `outlier_strategy="off"` (isola o efeito estrutural, sem `reduce_outliers`). Não é a taxa final do pipeline, que aplica `reduce_outliers` depois.

---

## 13. Sweep de reduce_outliers (estratégias e threshold)

**Run:** mesma execução (`tweets_bre2022_20260630_234301`)
**Controles:** `run_outlier_sweep: true`, `run_threshold_sweep: true`

> **Ressalva importante:** estes dois sweeps rodaram sobre a estrutura antiga/compartilhada (`n_neighbors=10, min_cluster_size=8, eom`, o default herdado do Folha antes de o override existir). Ainda não foram recalibrados para a nova estrutura (`n_neighbors=15, min_cluster_size=30, leaf`). Mesma situação documentada no Folha para o seu próprio threshold sweep (`PIPELINE_BERTOPIC_FOLHA.md`, nota final).

### 13.1 Estratégias (threshold=0, isola o mecanismo de decisão)

| Estratégia | K | outlier_pre | outlier_post | C_v | Diversity | Exclus | FREX |
|---|---|---|---|---|---|---|---|
| off | 24 | 27,9% | 27,9% | 0,561 | 0,900 | 0,772 | 0,986 |
| **c-tf-idf** | 25 | 27,9% | 0,0% | 0,466 | 0,908 | 0,723 | 0,980 |
| embeddings | 25 | 27,9% | 0,0% | 0,441 | 0,872 | 0,687 | 0,979 |
| probabilities | 25 | 27,9% | 0,0% | 0,436 | 0,894 | 0,689 | 0,980 |
| distributions | 25 | 27,9% | 0,0% | 0,477 | 0,908 | 0,719 | 0,980 |

Mesmo padrão do Folha: threshold=0 (reatribuição total) derruba C_v/Exclus para todas as estratégias vs. `off`. Entre as estratégias reais, `c-tf-idf` e `distributions` empatam em Diversity (0,908); `c-tf-idf` tem leve vantagem em Exclus (0,723 vs 0,719).

### 13.2 Threshold (estratégia `c-tf-idf`, fixa)

| Threshold | K | outlier_post | C_v | Diversity | Exclus | FREX |
|---|---|---|---|---|---|---|
| 0,00 | 25 | 0,0% | 0,466 | 0,908 | 0,723 | 0,980 |
| 0,05 | 24 | 13,2% | 0,525 | 0,902 | 0,762 | 0,983 |
| 0,10 | 24 | 22,8% | 0,560 | 0,904 | 0,767 | 0,985 |
| **0,15** | **24** | **25,9%** | **0,582** | **0,900** | **0,770** | **0,985** ← **atual, herdado** |
| 0,20 | 24 | 26,8% | 0,595 | 0,902 | 0,772 | 0,985 |
| 0,30 | 24 | 27,5% | 0,573 | 0,902 | 0,774 | 0,985 |

`threshold=0,20` tem C_v marginalmente melhor (+0,013) que `0,15`, com Diversity/Exclus praticamente empatados. A diferença é pequena e, como esta tabela foi computada sob a estrutura antiga (ressalva acima), não há evidência forte o suficiente para mudar o valor herdado do Folha antes de recalibrar sob `nn=15/mcs=30/leaf`.

> **⚠ Tabela histórica: superada em 2026-07-03.** Este sweep (`threshold=0,15`) rodou sob a estrutura antiga. O re-sweep sob `nn=15/mcs=30/leaf` mudou a decisão para `threshold=0,05`.

---

## 14. Configuração final aplicada

### `params.yaml` — bloco `corpora.tweets_bre2022` (override)

```yaml
tweets_bre2022:
  # ... (demais chaves inalteradas — ver §8)
  bertopic_overrides:
    umap:
      n_neighbors: 15   # ← 10 (herdado do Folha); §12: leaf+mcs alto domina estruturalmente p/ tweets
    hdbscan:
      min_cluster_size: 30           # ← 8 (Folha); §12: melhor composite, robusto nos 5 reduce_nr
      cluster_selection_method: leaf # ← eom (Folha); §12: leaf domina p/ tweets (oposto do Folha)
    # reduce_topics_nr NAO sobrescrito: nr=25 (compartilhado) já é o ótimo p/ tweets tb (§12)
    reduce_outliers:
      threshold: 0.05   # RECALIBRADO 2026-07-03 sob nn=15/mcs=30/leaf — ver §18
```

### Justificativa consolidada

| Parâmetro | Valor | Evidência principal |
|---|---|---|
| `umap.n_neighbors` | **15** (override) | composite 0,841, melhor entre os 108 combos estruturais |
| `hdbscan.min_cluster_size` | **30** (override) | robusto nos 5 `reduce_nr`; melhor Exclus/Diversity médios |
| `hdbscan.cluster_selection_method` | **leaf** (override) | `leaf` domina estruturalmente p/ textos curtos/repetitivos, oposto do Folha |
| `hdbscan.min_samples` | `null` (herdado) | Não testado ganho vs. `null` no top-1; ver grid completo p/ outros valores |
| `reduce_topics_nr` | **25** (herdado) | mesmo critério do Folha, ponto ótimo coincide |
| `reduce_outliers.strategy` | `c-tf-idf` (herdado, reconfirmado) | reconfirmado sob a estrutura nova: C_v 0,492 vs 0,42-0,46 das outras 4 estratégias |
| `reduce_outliers.threshold` | **`0,05`** (override, recalibrado) | composite 0,832 vs 0,724 do `0,15` herdado; cobertura 62,4%→83,5% com C_v/Diversity/Exclus quase inalterados |

---

## 15. Resultado baseline (config antiga, herdada do Folha)

**Run:** `data/output/tweets_bre2022/bertopic/tweets_bre2022_20260630_230717/`
**Config:** `n_neighbors=10, min_cluster_size=8, eom, reduce_nr=25, reduce_outliers=c-tf-idf/0,15` (bloco compartilhado, sem override: o override ainda não existia quando este run foi feito)

| Métrica | Valor |
|---|---|
| K tópicos | 24 |
| Outlier rate (pós reduce_outliers) | 25,9% |
| Cobertura | 74,1% |
| C_v | 0,519 |
| Topic Diversity (Dieng) | 0,90 |
| Exclusividade c-TF-IDF | 0,770 |
| Coerência Semântica (top-20, `qwen3-embedding:0.6b`) | 0,652 |
| FREX | 0,985 |

Este é o resultado de referência antes da calibração estrutural. Serve como baseline de comparação, análogo ao "Melhor Resultado Encontrado (baseline)" do Folha (`PIPELINE_BERTOPIC_FOLHA.md`), mas aqui a config baseline é literalmente a config compartilhada herdada, não uma tentativa anterior descartada.

---

## 16. Pendências e arquivos de saída

### Concluído (2026-07-01)

- **Wiring do override corrigido:** a célula 5 do notebook (`bert_cfg = params["bertopic"]`) estava sem a chamada a `apply_bertopic_overrides`: verificado por grep no `.ipynb`, a função não aparecia em nenhuma célula. O notebook rodava com a config antiga (`nn=10/mcs=8/eom`, herdada da Folha) mesmo com `bertopic_overrides` já declarado em `params.yaml`. Corrigido: a célula agora chama `apply_bertopic_overrides(bert_cfg, cfg)` logo após ler o bloco compartilhado (código idêntico ao já documentado acima).
- **Sweeps desligados:** `run_param_sweep`, `run_outlier_sweep` e `run_threshold_sweep` voltaram a `false` em `params.yaml`: os 3 sweeps já rodaram (run `tweets_bre2022_20260630_234301`) e os resultados estão consolidados. Reativar só para recalibrar.
- **`param_sweep_macro_k`** (evaluation) é uma chave morta: não é lida por nenhuma função em `_helpers.py`. O sweep de macro_k que gera `bertopic_macro_k_sweep.csv` roda inline no notebook (cél. macro-temas), independente dessa chave. Não remover, mas não esperar que ela afete o sweep estrutural.

### Pendente (próximos passos)

1. ~~**Re-executar o notebook completo** agora que o override está de fato ligado~~. ✅ **concluído**: run `tweets_bre2022_20260701_201916`.
2. ~~**Recalibrar `reduce_outliers` (estratégia/threshold)** sob a nova estrutura~~. ✅ **concluído 2026-07-03**: sweep re-executado sob `nn=15/mcs=30/leaf` (run `tweets_bre2022_20260703_094835`) e novo run de produção (`tweets_bre2022_20260703_101639`) com `threshold=0.05`. outlier_post: 37,6% → **16,5%**.
3. **Opcional:** rodar o `sweep_bertopic_grid` com 2+ seeds para validar `Stability`/`Stab_std` (atualmente `NaN`, single-seed), mesma ressalva metodológica já aceita no Folha.
4. ~~**Macro temas:** ainda não recalculado sob a nova estrutura~~. ✅ **concluído** (`bertopic_macro_k_sweep.csv` e `bertopic_macro_temas.csv` gerados sob `nn=15/mcs=30/leaf`), com ressalva de grab-bag no Macro 9; recomputado no run definitivo, mesma estrutura de macros.

### Arquivos de saída

Cada execução gera `data/output/tweets_bre2022/bertopic/tweets_bre2022_<AAAAMMDD>_<HHMMSS>/` (mesmo layout do Folha):

| Arquivo | Conteúdo |
|---|---|
| `bertopic_metrics.csv` | Métricas agregadas do run (C_v, Diversity, Exclus, semantic_coherence, FREX, outlier_rate, K) |
| `bertopic_results.csv` | Tópico × documento (atribuições finais) |
| `bertopic_topics_for_eval.csv` | Keywords por tópico |
| `bertopic_topic_names.csv` | Rótulos LLM por tópico |
| `bertopic_exclusividade_ranking.csv` | Ranking de exclusividade c-TF-IDF |
| `bertopic_macro_temas.csv` | Mapeamento tópico → macro grupo |
| `bertopic_macro_k_sweep.csv` | Avaliação de diferentes `macro_k` |
| `bertopic_robustness_topn.csv/.png` | Sweep de robustez de métricas vs. top-N keywords |
| `sweep_bertopic_grid.csv` / `_raw.csv` | Resultados do sweep estrutural |
| `sweep_outlier_strategies.csv` / `_raw.csv` | Resultados do sweep de estratégias |
| `sweep_outlier_threshold.csv` / `_raw.csv` | Resultados do sweep de threshold |
| `*.html` | Visualizações interativas Plotly (hierarquia, documentos, tópicos, sankey) |
| `*.png` | Wordclouds, heatmaps, UMAP 2D, árvore macro |

---

## 17. Resultados da Rodada de Produção Calibrada (`tweets_bre2022_20260701_201916`)

> Primeiro run em que a config calibrada (`n_neighbors=15, min_cluster_size=30, cluster_selection_method=leaf`) foi **de fato exercitada**: o wiring bug do `apply_bertopic_overrides` impedia isso em todos os runs anteriores, incluindo o baseline. Mesma estrutura de análise do `PIPELINE_BERTOPIC_FOLHA.md` ("Resultados da Rodada de Produção"), para permitir comparação direta entre os dois corpora.

### Versões dos artefatos

| Artefato | Versão |
|---|---|
| Corpus limpo | `tweets_bre2022_20260629_215159` |
| Embeddings | `tweets_bre2022_20260629_220920` |
| Run BERTopic | `tweets_bre2022_20260701_201916` |

### Pipeline (trace real)

```
Fit inicial:                        66 tópicos, 3663 outliers (41.6%)
reduce_outliers(c-tf-idf, 0.15):    3663 → 3312 outliers (reatribuiu 351 docs, 9.6% dos outliers)
reduce_topics(nr=25):               66 → 24 tópicos
update_topics                       (c-TF-IDF reconstruído)
Final:                              24 tópicos válidos | 3312 outliers (37.6%) | cobertura 62.4%
```

O `outlier_pre=41.6%` e o `K=24` batem com o previsto (`reduce_topics_nr=25` já herdado do Folha): confirma que este run de fato usou a estrutura calibrada, ao contrário do baseline.

### Métricas — calibrado vs. baseline (mesmo corpus, mesmos `top_n`)

Comparação direta com o run `tweets_bre2022_20260630_230717` (baseline, config antiga herdada do Folha: `nn=10/mcs=8/eom`), via `bertopic_robustness_topn.csv` de cada run:

| `top_n` | C_v baseline | C_v calibrado | Δ | Diversity baseline | Diversity calibrado | Exclus. baseline | Exclus. calibrado |
|---|---|---|---|---|---|---|---|
| 10 | 0.667 | **NaN** ⚠ | — | 0.900 | 0.921 | 0.776 | 0.797 |
| 15 | 0.634 | **0.751** | +0.116 | 0.903 | 0.911 | 0.774 | 0.800 |
| 20 | 0.582 | **0.682** | +0.100 | 0.900 | 0.913 | 0.770 | 0.795 |
| 25 | 0.545 | **0.634** | +0.088 | 0.898 | 0.913 | 0.769 | 0.783 |
| 30 | 0.519 | **0.573** | +0.054 | 0.892 | 0.900 | 0.762 | 0.771 |

A calibração estrutural melhorou C_v de forma consistente em todos os `top_n` comparáveis (+0.05 a +0.12), confirmando que o sweep de 108 combinações valeu a pena, a mesma direção do que se esperava ao trocar `eom`→`leaf` e `mcs=8`→`30` para textos curtos e repetitivos.

> **⚠ `top_n=10` retorna `NaN` no run calibrado.** É o mesmo bug já documentado: com `ngram_range=(1,2)`, um tópico cujas top-10 keywords caem 100% em bigramas fora do dicionário gensim (unigramas) quebra o `CoherenceModel`. O `try/except` protege os **sweeps**, mas a célula principal de métricas (fora do sweep) ainda pode propagar `NaN` para o `top_n` mais restritivo do `bertopic_robustness_topn.csv`. Não afetou o baseline (`top_n=10` válido lá), então é sensível à config: com `leaf`/`mcs=30` os tópicos ficam mais compactos e concentram mais bigramas nas primeiras posições. **Ação sugerida:** aplicar o mesmo `try/except ValueError → NaN` já usado em `_bertopic_sweep_metrics()` também na célula principal de robustez (`bertopic_robustness_topn`), se ainda não estiver lá.

### Outlier rate: regressão vs. baseline — evidencia a pendência do §16 item 2

| Métrica | Baseline (`230717`) | Calibrado (`201916`) | Δ |
|---|---|---|---|
| outlier_rate (pós `reduce_outliers`) | 25.9% | **37.6%** | +11.7pp (pior) |
| Cobertura | 74.1% | **62.4%** | −11.7pp (pior) |

A calibração estrutural melhora C_v/Diversity/Exclusividade, mas **piora bastante a taxa de outliers**: `strategy=c-tf-idf`/`threshold=0.15` foi calibrado sob a estrutura antiga (`nn=10/mcs=8/eom`) e nunca foi revalidado sob `nn=15/mcs=30/leaf`. Isso confirma empiricamente a ressalva já registrada: falta rodar `sweep_outlier_strategies`/`sweep_outlier_threshold` sob a nova estrutura antes de considerar a config totalmente calibrada.

### FREX — mesmo bug do Folha, confirmado num segundo corpus

`FREX` varia entre 0.9889 (`top_n=10`) e 0.9897 (`top_n=100`) no run calibrado: variação de **0.0008** ao decuplicar o número de keywords, quando deveria cair de forma visível (mais keywords → menos exclusivas → FREX menor). É o mesmo padrão relatado em `PIPELINE_BERTOPIC_FOLHA.md` (`~0.982` constante para a Folha): a reprodução em dois corpora distintos reforça que o bug está em `compute_frex_score`, não é peculiaridade de um corpus. **Não reportar FREX na dissertação** até a função ser auditada (mesma recomendação do Folha).

### 24 tópicos produzidos

| ID | Nome LLM | Keywords (top-5) | Docs | Exclus. |
|---|---|---|---|---|
| T0 | Bolsonaristas comemoram vitória de Bolsonaro | bolsonaroreeleito, bandeira_brasil, forabolsonaro, não trabalha, trabalha | 838 | 0.472 ⚠ menor |
| T1 | Votação e eleição de Lula | lulapresidente, diplomação, votar, nordeste, deixem | 616 | 0.740 |
| T2 | Terrorismo e atos de vandalismo em Brasília | terroristas, infiltrados, brasília, em brasília, ônibus | 571 | 0.644 |
| T3 | GloboLixo e ataques ao jornalismo | globolixo, globolixo https, em casa, rosto_vomitando, puder | 525 | 0.822 |
| T4 | Barroso e mané na disputa eleitoral | mané, perdeu mané, amola, não amola, mané não | 392 | 0.841 |
| T5 | Debate político na Band | debatenaband, ciro, debate, simone, tebet | 280 | 0.895 |
| T6 | Apoio a Xandão e ataques ao STF | xandão, ladraonojn, stfvergonhamundial, ladraonojn https, stfvergonhamundial https | 234 | 0.716 |
| T7 | Polícia Rodoviária e impedimento de voto | rodoviária, rodoviária federal, polícia rodoviária, polícia, da polícia | 228 | 0.735 |
| T8 | Bolsonaristas e ataques a Lula | bobo, bobo da, da corte, corte, parece | 196 | 0.819 |
| T9 | Bolsonaro no Catar e jogo do Brasil | eduardo, eduardo bolsonaro, catar, no catar, copa | 193 | 0.889 |
| T10 | Rombo e desabamento do orçamento brasileiro | bilhões, rombo, quebrou, de bilhões, quebrou brasil | 177 | 0.762 |
| T11 | Bolsistas e pagamento da Capes | capes, pagueminhabolsa, da capes, bolsistas, bolsas | 157 | 0.840 |
| T12 | Manifestações em Brasília | alvorada, brasília, palácio, brasília df, do alvorada | 151 | 0.885 |
| T13 | Prisão de Cacique Xavante | cacique, índio, xavante, de moraes, cacique xavante | 146 | 0.725 |
| T14 | Eleições e resultados do turno eleitoral | imóveis, tá eleito, lula tá, eleito, bolsonaro imóveis | 126 | 0.810 |
| T15 | Saqueio e crise no governo Bolsonaro | brasil jair, jair saqueou, saqueou, saqueou brasil, quebrou brasil | 121 | 0.754 |
| T16 | Orçamento secreto e Bolsonaro | orçamento, orçamento secreto, secreto, secreto não, não orçamento | 120 | 0.627 |
| T17 | Multa ao PL por fraude eleitoral | multa, multa de, moraes, do pl, de milhões | 104 | 0.760 |
| T18 | Caminhoneiros e greve de protesto | caminhoneiros, caminhoneiros https, co caminhoneiros, os caminhoneiros, braziliansprings | 90 | 0.876 |
| T19 | Multa e apoio a Xandão | multa, faz milhões, de multa, milhões de, co faz | 64 | 0.713 |
| T20 | Coração Vermelho e Memes Políticos | coração_vermelho, co faz, faz https, faz faz, coração_verde | 46 | 0.882 |
| T21 | Links e vídeos com Bolsonaro | link, seu vídeo, co forabolsonaro, aqui link, link com | 44 | 0.973 |
| T22 | Renúncia e cadeia de Bolsonaro | braga, na cadeia, braga neto, neto, neto na | 44 | 0.908 |
| T23 | Intolerância religiosa e racismo online | não vote, vote em, vote, racista, em racista | 36 | **0.993** máx. |

Todos os 24 tópicos são interpretáveis e tematicamente coerentes com o contexto eleitoral 2022 (o mesmo veredito qualitativo do Folha: nenhum tópico-lixo).

### Macro-temas (K=10, `bertopic_macro_k_sweep.csv`/`bertopic_macro_temas.csv`)

| Macro | Sub-tópicos | Docs | Tema dominante |
|---|---|---|---|
| 9 | T0,T3,T4,T5,T6,T8,T9 (7) | 2.658 | Bolsonarismo/pró-Bolsonaro difuso (debate, STF, Lula, futebol): grab-bag |
| 10 | T1,T12,T14,T21,T22 (5) | 981 | Votação/eleição de Lula e desdobramentos |
| 5 | T2,T7,T18 (3) | 889 | Vandalismo, bloqueios e atos golpistas |
| 1 | T10,T15 (2) | 298 | Crise fiscal/orçamentária (narrativa bolsonarista) |
| 8 | T19,T20 (2) | 110 | Memes/multas: miscelânea |
| 2 | T11 (1) | 157 | Bolsistas Capes |
| 3 | T16 (1) | 120 | Orçamento secreto |
| 4 | T17 (1) | 104 | Multa eleitoral ao PL |
| 6 | T13 (1) | 146 | Prisão de indígena (Cacique Xavante) |
| 7 | T23 (1) | 36 | Intolerância religiosa/racismo |

### Qualidade dos tópicos — pontos de atenção

| Tópico/Macro | Observação | Natureza |
|---|---|---|
| T0 (838 docs) | Maior tópico e **menos exclusivo** (0.472, bem abaixo do 2º pior, T16=0.627): mistura torcida bolsonarista, ataques à imprensa e memes num só cluster | Estrutural (`leaf` ainda deixa um cluster denso central absorver variações de discurso pró-Bolsonaro genérico) |
| Macro 9 (7 sub-tópicos, 2.658 docs, 30% do corpus) | Agrupa T0/T3/T4/T5/T6/T8/T9: temas tão diversos quanto "debate na Band" e "Bolsonaro no Catar" acabam no mesmo macro-grupo | Catch-all análogo ao "Macro 2" do Folha, mesma limitação metodológica do agrupamento hierárquico de tópicos vetorialmente próximos |
| Keywords com tokens de emoji (`coração_vermelho`, `rosto_chorando_de_rir`, `rosto_de_palhaço`) | Aparecem como keywords de pleno direito, não como ruído | Esperado e correto: tweets não passam por lematização; emojis demojizados carregam sinal semântico real (T20 é literalmente sobre isso) |
| T20 "Coração Vermelho e Memes Políticos" (46 docs) | Cluster dominado por variações de emojis de coração | É um tópico real de "reação emotiva/meme", não resíduo, mas de baixa densidade lexical para nomeação LLM |
| Hashtags/handles como keywords (`debatenaband`, `ladraonojn`, `stfvergonhamundial`) | Presentes em T5, T6, T8, T16 | Esperado para tweets: hashtags carregam a própria semântica do evento, ao contrário de resíduo HTML (caso do Folha T22) |

Comparado ao baseline (`230717`), não há tópicos redundantes visíveis na matriz de similaridade de cosseno (par mais próximo: T10↔T15, sim=0.285, bem abaixo do limiar de fusão usual de ~0.5), sugerindo que a granularidade `reduce_topics_nr=25` está bem calibrada para este corpus.

### Comparação síntese com a Folha (`PIPELINE_BERTOPIC_FOLHA.md §18`, run `folha_20260628_221412`)

| Métrica | Folha | Tweets (calibrado) | Leitura |
|---|---|---|---|
| K tópicos | 24 | 24 | Coincidência de critério (`reduce_topics_nr=25`→K=24 em ambos), não de granularidade real: 24 tópicos para ~50k artigos (Folha) é muito mais grosso que 24 para 8.811 tweets |
| outlier_post | 21.7% | **37.6%** | Tweets muito pior: reduce_outliers não recalibrado |
| Cobertura | 78.3% | **62.4%** | Idem |
| C_v @ top-20 | 0.549 | **0.682** | Tweets mais coerente no mesmo `top_n`: corpus eleitoral tem universo temático mais estreito (poucos temas reais) vs. editorias heterogêneas da Folha |
| Topic Diversity @ top-20 | 0.810 | 0.913 | Tweets mais diverso: menos redundância entre tópicos |
| Exclusividade (c-TF-IDF, `top_n=20`) | 0.486 | 0.795 | Tweets bem mais exclusivo: vocabulário de hashtags/gírias específicas gera fronteiras lexicais mais nítidas que o vocabulário jornalístico genérico da Folha |
| Coerência Semântica | 0.654 | 0.660 | Praticamente empatado |
| FREX | ⚠ não reportar (bug) | ⚠ não reportar (bug) | Confirmado em ambos os corpora |

**Leitura geral:** a calibração estrutural específica por corpus valeu a pena: sem ela, os tweets herdariam cegamente `eom`/`mcs=8` da Folha e teriam métricas piores em quase todas as dimensões. O ponto fraco remanescente era exclusivamente o outlier rate, resolvido na recalibração seguinte.

### Estabilidade

`Estabilidade Jaccard: N/A (< 2 seeds)`: `param_sweep_seeds=[42]` no sweep estrutural e sem seed alternativo no run de produção. Mesma limitação metodológica aceita no Folha; validação formal de robustez fica como próximo passo opcional.

---

## 18. Recalibração de reduce_outliers e Resultado Definitivo (`tweets_bre2022_20260703_101639`)

> Fecha a pendência de recalibração do `reduce_outliers`: os sweeps de estratégia/threshold rodaram sob a estrutura antiga (`nn=10/mcs=8/eom`); com a estrutura calibrada em produção desde 2026-07-01, o `threshold=0,15` herdado da Folha vinha reatribuindo só 9,6% dos outliers, deixando 37,6% de docs sem tópico. Esta seção documenta o re-sweep sob a estrutura correta e o run de produção definitivo.

### Sequência executada (2026-07-03)

1. `evaluation.run_outlier_sweep` e `evaluation.run_threshold_sweep` religados em `params.yaml` (`run_param_sweep` continuou `false`: o grid estrutural de 108 combos não precisa ser refeito).
2. Run All → sweeps executados sob `nn=15/mcs=30/leaf` (via `bertopic_overrides`, wiring já corrigido): run `tweets_bre2022_20260703_094835`, ~7 min para os dois sweeps.
3. Escolha de configuração a partir dos CSVs brutos (abaixo).
4. `bertopic_overrides.tweets_bre2022.reduce_outliers.threshold` atualizado para `0.05`; os 2 flags de sweep desligados de volta.
5. Novo Run All de produção → run `tweets_bre2022_20260703_101639`, ~20 min.

### 18.1 Sweep de estratégias (threshold=0, isola o mecanismo) — reconfirma `c-tf-idf`

| Estratégia | K | outlier_pre | C_v | Diversity | Exclus | FREX |
|---|---|---|---|---|---|---|
| off | 24 | 41,57% | 0,696 | 0,915 | 0,798 | 0,990 |
| **c-tf-idf** | 25 | 41,57% | **0,492** | 0,868 | 0,676 | 0,981 |
| distributions | 25 | 41,57% | 0,456 | 0,884 | 0,692 | 0,982 |
| probabilities | 25 | 41,57% | 0,458 | 0,856 | 0,631 | 0,981 |
| embeddings | 25 | 41,57% | 0,420 | 0,856 | 0,650 | 0,981 |

`c-tf-idf` continua o melhor mecanismo mesmo sob a estrutura nova, mesmo padrão do sweep anterior e do Folha (`PIPELINE_BERTOPIC_FOLHA.md`): decidir por similaridade vocabular é mais estável que distância no espaço de embeddings ou probabilidade soft do HDBSCAN.

### 18.2 Sweep de threshold (`c-tf-idf` fixo) — composite recalculado

Composite = média normalizada (min-max) de C_v + Diversity + Exclusividade + Cobertura, mesmo método de `PIPELINE_BERTOPIC_FOLHA.md`:

| Threshold | outlier_post | Cobertura | C_v | Diversity | Exclus | Composite |
|---|---|---|---|---|---|---|
| 0,00 | 0,0% | 100% | 0,492 | 0,868 | 0,676 | 0,250 |
| **0,05** | **16,5%** | **83,5%** | 0,679 | 0,917 | 0,778 | **0,832 ← vencedor** |
| 0,10 | 32,1% | 67,9% | 0,686 | 0,919 | 0,794 | 0,791 |
| 0,15 (antigo, herdado da Folha) | 37,6% | 62,4% | 0,682 | 0,913 | 0,795 | 0,724 |
| 0,20 | 39,7% | 60,3% | 0,686 | 0,910 | 0,793 | 0,700 |
| 0,30 | 41,1% | 58,9% | 0,696 | 0,915 | 0,794 | 0,727 |

**Por que `threshold=0,05` e não `0,15`?** Sob a estrutura nova (clusters `leaf`/`mcs=30` maiores e mais densos), os outliers ficam vocabularmente muito mais próximos dos tópicos vizinhos do que sob a estrutura antiga: um threshold baixo já captura a maioria das reatribuições corretas. `0,05` recupera **+21,1pp de cobertura** vs. `0,15` (83,5% vs 62,4%) com C_v praticamente igual (0,679 vs 0,682, Δ=−0,003) e Diversity/Exclus ligeiramente melhores. É o oposto do padrão observado na Folha, onde threshold baixo derrubava C_v ~15%: reforça que a curva de similaridade c-TF-IDF depende inteiramente da geometria dos clusters upstream (ressalva já registrada), não é uma constante do algoritmo.

### 18.3 Run de produção definitivo — antes vs. depois da recalibração

Comparação `tweets_bre2022_20260701_201916` (baseline calibrado, `threshold=0,15`) vs. `tweets_bre2022_20260703_101639` (recalibrado, `threshold=0,05`):

| Métrica | `threshold=0,15` | `threshold=0,05` | Δ |
|---|---|---|---|
| K tópicos | 24 | 24 | — |
| outlier_rate | 37,6% | **16,5%** | **−21,1pp** |
| Cobertura | 62,4% | **83,5%** | **+21,1pp** |
| C_v (top-20) | 0,682 | 0,679 | −0,003 (ruído) |
| Topic Diversity (Dieng) | 0,913 | 0,929 | +0,016 |
| Exclusividade (c-TF-IDF) | 0,795 | 0,778 | −0,017 |
| Coerência Semântica | 0,660 | 0,656 | −0,004 |
| FREX | 0,990 | 0,986 | −0,004 (bug já conhecido, ambos inválidos p/ reportar) |

Confirma a previsão do sweep: ganho grande de cobertura, custo desprezível em qualidade. Os 24 tópicos permanecem os mesmos temas eleitorais interpretáveis; T0 ("Bolsonaristas comemoram vitória") continua o menos exclusivo (0,459, antes 0,472) e cresce de 838 para 1.089 docs (esperado: é o catch-all que absorve parte dos outliers reatribuídos pelo threshold mais permissivo). Nenhum tópico novo de ruído surgiu; macro-temas (`bertopic_macro_temas.csv`) mantêm a mesma estrutura de 10 grupos do run anterior, com o mesmo catch-all documentado (Macro 5/9 concentrando temas pró/contra-Bolsonaro difusos).

> **`C_v @ top-10` continua `NaN`** neste run: mesmo bug já documentado (bigramas 100% fora do dicionário unigrama gensim na célula principal de robustez, fora do `try/except` que já protege os 3 sweeps). Não bloqueia a leitura em top-20 (valor reportável), mas seguiria valendo aplicar o mesmo guard na célula `bertopic_robustness_topn`.

### Conclusão da recalibração

A pendência de recalibração está **encerrada**. A configuração final de `reduce_outliers` para os tweets diverge da Folha tanto na estratégia de decisão (`c-tf-idf` em ambos, coincidência) quanto no threshold (`0,05` tweets vs. `0,15` Folha): mais uma confirmação de que a calibração por corpus precisa se estender a **todo** hiperparâmetro sensível à estrutura, não só a UMAP/HDBSCAN.

---
