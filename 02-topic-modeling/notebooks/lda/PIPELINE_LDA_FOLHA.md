# LDA — Folha de São Paulo: Documentação Completa do Pipeline

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
> **Notebook:** `02_lda_folha.ipynb`
> **Configuração:** `03-topic-modeling/configs/params.yaml`
> **Última revisão:** 2026-06-29: resultados definitivos do run `folha_20260629_004450` (corpus corrigido, prompt melhorado, pipeline completo re-rodado)

---

## Índice

1. [O que é LDA — conceitos fundamentais](#1-o-que-é-lda--conceitos-fundamentais)
2. [Arquitetura do pipeline](#2-arquitetura-do-pipeline)
3. [Corpus e estratégia 5k](#3-corpus-e-estratégia-5k)
4. [Lematização e vocabulário (Gensim Dictionary)](#4-lematização-e-vocabulário-gensim-dictionary)
5. [Seleção de K — grid search C_v](#5-seleção-de-k--grid-search-cv)
6. [Grid alpha × eta — hiperparâmetros Dirichlet](#6-grid-alpha--eta--hiperparâmetros-dirichlet)
7. [Treino final](#7-treino-final)
8. [Nomeação de tópicos via LLM](#8-nomeação-de-tópicos-via-llm)
9. [Métricas de avaliação](#9-métricas-de-avaliação)
10. [Comparação com BERTopic](#10-comparação-com-bertopic)
11. [Configuração final](#11-configuração-final)
12. [Arquivos de saída](#12-arquivos-de-saída)

---

## 1. O que é LDA — conceitos fundamentais

LDA (Latent Dirichlet Allocation, Blei et al. 2003) é um **modelo generativo probabilístico** que assume que cada documento foi gerado por um processo de dois estágios:

1. **Escolha de tópicos:** Para cada documento d, amostrar uma distribuição θ_d sobre K tópicos a partir de uma Dirichlet(α). Isso define "quanto" do documento pertence a cada tópico.
2. **Escolha de palavras:** Para cada posição no documento, amostrar um tópico z a partir de θ_d, e então amostrar uma palavra w a partir da distribuição φ_z (Dirichlet(η)) do tópico escolhido.

O modelo inverte esse processo: dado um corpus observado, usa inferência variacional (ou Gibbs sampling) para estimar as distribuições latentes θ (doc→tópico) e φ (tópico→vocabulário).

### Diferença fundamental em relação ao BERTopic

| Aspecto | LDA | BERTopic |
|---------|-----|----------|
| Representação | Bag-of-Words (frequência de tokens) | Embeddings densos (significado semântico) |
| "banco" (financeiro) vs "banco" (praça) | Mesmo token, mesmo peso | Vetores diferentes no espaço semântico |
| K tópicos | Fixo (escolhido antes do treino) | Emergente (HDBSCAN descobre automaticamente) |
| Outliers | Todos os docs têm tópico (θ sempre > 0) | Documentos podem ser outliers (id=-1) |
| Interpretação θ | Distribuição suave: doc pertence parcialmente a vários tópicos | Tópico dominante único por doc |
| Vocabulário | Token lemmatizado | Token lemmatizado + contexto semântico |

### Por que usar LDA além do BERTopic?

A dissertação usa **triangulação metodológica**: LDA e BERTopic são paradigmas distintos de topic modeling. Convergência nos resultados (tópicos similares identificados por ambos) fortalece a validade das conclusões. Divergências revelam o que cada método captura que o outro não.

---

## 2. Arquitetura do Pipeline

```
corpus_limpo.csv
        │
        ▼
[Cell 3]  load_corpus() + subsample 5k (seed=42)
          • N=4.939 docs (subsample seed=42, padronizado com BERTopic)
          • docs = df["message"].astype(str).tolist()
          • 10 categorias editoriais balanceadas (~490-500 docs cada)
        │
        ▼
[Cell 5]  lemmatize_corpus()
          • spaCy pt_core_news_lg (CPU, ~448s)
          • Descarta: stopwords spaCy + tokens não-alfa + len ≤ 2
          • Gensim Dictionary: filter_extremes(no_below=5, no_above=0.5)
          • Vocabulário final: 17.303 palavras
          • Média: 376.4 tokens/doc (240.1 únicos)
        │
        ▼
[Cell 7]  corpus_bow = [dictionary.doc2bow(d) for d in tokenized]
          • Representação Bag-of-Words: lista de (word_id, count)
          • Média: 218.8 tokens únicos/doc no BoW
        │
        ▼
[Cell 8]  grid_search_k()
          • K ∈ {3, 5, 7, 8, 10, 12, 15, 20, 25, 30}
          • passes=20; seed=42
          • Cache: BASE_OUTPUT_DIR/lda_metrics.csv (reutilizado entre runs)
          • BEST_K = peak C_v (automático)
        │
        ▼
[Cell 12] grid_search_alpha_eta()
          • K=BEST_K fixo; passes=10
          • 5 alphas × 3 etas = 15 combinações
          • Cache: BASE_OUTPUT_DIR/lda_alpha_eta_grid.csv
          • Treino final: train_lda(K=BEST_K, alpha=best_alpha, eta=best_eta, passes=20)
        │
        ▼
[Nomeação LDA via LLM] (movida p/ ANTES das visualizações — ver Seção 8):
          • top_n_keywords_llm=50 keywords por tópico
          • 3 docs representativos por probabilidade theta
          • Prompt PT-BR few-shot + anti-redundância + TEMA GERAL
          • Deduplicação + ordering por tamanho
          ✓ roda logo após o treino → TODAS as figuras abaixo já saem rotuladas
            num único "Run All" (não precisa re-executar célula nenhuma)
        │
        ▼
[Cells 13-14] Wordclouds + bar charts (visualização exploratória)
        │
        ▼
[Phi heatmap + cosine similarity] Visualizações de palavras×tópicos (✓ com nomes)
        │
        ▼
[Cell 16] Métricas: C_v, Exclusividade, Diversity, FREX, Perplexidade
          • dominant, full = compute_doc_distributions()
          • theta_arr = np.array(full)  # (n_docs × K)
        │
        ▼
[Cells 17-22] c_v por tópico, estabilidade Jaccard, theta heatmap,
              t-SNE, cross-tab categoria×tópico, pyLDAvis
        │
        ▼
[Cell 23] export_results() + export_topics_for_eval() + lda_metrics.csv
```

---

## 3. Corpus e Estratégia 5k

### Por que 5k e não o corpus completo?

O LDA usa **Gibbs sampling** (ou inferência variacional), algoritmos iterativos que escalam quadraticamente com o número de documentos em versões básicas. O `LdaMulticore` do gensim é eficiente, mas treinado em corpus grande com passes=20 pode levar horas em CPU.

A amostra de 5k documentos é estrategicamente escolhida para:
- Tornar o treino viável em CPU (~13min para o grid K, ~30min para o treino final)
- Manter comparabilidade com BERTopic (mesmos índices via seed=42)
- Cobrir todas as 10 editorias com ~490-500 docs cada (amostragem bem balanceada)

### Mesmos índices que BERTopic

```python
# LDA — Cell 3
_s_idx = np.random.RandomState(42).choice(len(docs_all), min(5000, len(docs_all)), replace=False)

# BERTopic — Cell 19 (apenas para C_v, não afeta treino)
_sample_idx = np.random.RandomState(42).choice(len(docs), min(5000, len(docs)), replace=False)
```

Mesma seed (42) → mesmos artigos → **métricas de coerência C_v diretamente comparáveis** entre os dois modelos.

### Diferença de escala no treino

| | LDA (Folha) | BERTopic (Folha) |
|--|-------------|-----------------|
| **Documentos de treino** | **5.000** | **N completo** |
| Documentos para C_v | 5.000 (= treino) | 5.000 (amostra) |
| Justificativa | Custo computacional BoW | Embeddings pré-computados |

---

## 4. Lematização e Vocabulário (Gensim Dictionary)

### O que `lemmatize_corpus` faz

```
texto bruto → spaCy pt_core_news_lg → lemma de cada token → filtros → Dictionary
```

**Filtros aplicados por token:**
1. `token.is_stop` → descarta stopwords da lista spaCy PT
2. `not token.is_alpha` → descarta números, pontuação, símbolos
3. `len(lemma) <= 2` → descarta tokens muito curtos (artigos, preposições não filtrados)

**Por que lematizar e não usar texto bruto?**
LDA é sensível a variância morfológica: "eleição", "eleições", "eleicional" seriam 3 tokens distintos no BoW, inflando o vocabulário e espalhando o sinal de co-ocorrência. Com lematização, todos viram "eleição", um único token com contagem acumulada.

**Gensim Dictionary com `filter_extremes`:**

```python
dictionary.filter_extremes(no_below=5, no_above=0.5)
```

| Parâmetro | Valor | Significado | Tokens removidos |
|-----------|-------|-------------|-----------------|
| `no_below` | 5 | Remove token que aparece em < 5 docs | Hapax e erros de OCR |
| `no_above` | 0.5 | Remove token em > 50% dos docs | "disse", "afirmou", "país" (genéricos) |

**Resultado:** 17.303 palavras no vocabulário final (de ~80k antes do filtro). Essa redução de ~78% elimina ruído e acelera o Gibbs sampling.

### Por que `no_above=0.5` e não mais baixo?

50% é o padrão da literatura. Abaixar para 30% removeria termos como "governo" e "brasil" que, apesar de frequentes, são discriminativos quando combinados com outros tokens menos frequentes. Subir para 80% incluiria stopwords funcionais que o spaCy não capturou.

---

## 5. Seleção de K — Grid Search C_v

### O que é o grid search de K

O LDA precisa de K fixo antes do treino. Não há mecanismo automático como o HDBSCAN do BERTopic para descobrir K. A abordagem padrão é treinar modelos para vários K e avaliar por C_v coherence.

**K testados e resultados:**

| K | C_v | Interpretação qualitativa |
|---|-----|--------------------------|
| 3 | 0.370 | Muito grosseiro: política, cultura, resto |
| 5 | 0.407 | Ainda grosseiro |
| 7 | 0.491 | Começa a diferenciar editorias |
| **8** | **0.532** | Pico local: 1 tópico por editoria aproximadamente |
| 10 | 0.519 | Leve queda (fragmentação inicial) |
| 12 | 0.515 | — |
| 15 | 0.500 | — |
| **20** | **0.552** | **Peak global**: 2 sub-tópicos por editoria em média |
| 25 | 0.543 | Marginal vs K=20 |
| 30 | 0.548 | Marginal; risco de fragmentação |

**BEST_K = 20** (peak C_v automático via `max(scores, key=scores.get)`).

### Por que K=20 faz sentido para a Folha?

O corpus tem **10 editorias balanceadas**: ilustrada, equilibrioesaude, poder, cotidiano, mercado, esporte, mundo, educacao, ciencia, ambiente. K=20 corresponde a ~2 sub-tópicos por editoria em média, captura subdivisões como:
- `mercado` → "política monetária/Selic" + "câmbio/comércio exterior"
- `mundo` → "conflitos geopolíticos" + "eleições internacionais"
- `ambiente` → "Amazônia/desmatamento" + "mudanças climáticas"

K=8 produziria tópicos que misturam sub-editorias distintas. K=30 começa a fragmentar tópicos coesos.

### Limitação conhecida do C_v

C_v penaliza K alto (Stevens et al. 2012): com mais tópicos, as top-10 palavras de cada tópico tendem a ter co-ocorrências mais fracas porque os tópicos ficam mais específicos. O pico em K=20 (e não K=30) pode refletir essa penalidade: o plateau K=20-30 (C_v: 0.543-0.552) indica que nessa faixa a granularidade é adequada.

**Recomendação para dissertação:** reportar K=20 com nota de que o plateau K=20-30 sugere robustez da escolha nessa faixa.

---

## 6. Grid alpha × eta — Hiperparâmetros Dirichlet

### O que são alpha e eta

**α (alpha):** Prior Dirichlet sobre a distribuição tópico→documento (θ_d). Controla quão "concentrado" é cada documento num ou poucos tópicos.

| Alpha | Efeito | Interpretação |
|-------|--------|---------------|
| Baixo (0.01) | Documentos concentrados em 1-2 tópicos | Cada artigo fala de 1 tema principal |
| Alto (0.5, symmetric) | Documentos distribuídos entre muitos tópicos | Cada artigo é multi-temático |
| symmetric (= 1/K = 0.05) | Prior uniforme, sem assunção a priori | Neutro epistemológico |
| asymmetric (= 1/[K-i]) | Alguns tópicos a priori mais prevalentes | Assume estrutura hierárquica |

**η (eta):** Prior Dirichlet sobre a distribuição palavra→tópico (φ_z). Controla quão "exclusivo" é o vocabulário de cada tópico.

| Eta | Efeito | Interpretação |
|-----|--------|---------------|
| Baixo (0.01) | Vocabulário esparso e exclusivo por tópico | Cada tópico usa poucas palavras muito específicas |
| Alto (0.1) | Vocabulário difuso, palavras compartilhadas | Tópicos com overlap lexical |
| None (1/K) | Default gensim | Neutro |

### Resultados do grid (15 combinações, K=20, passes=10)

| alpha | eta | C_v | Perplexidade | Interpretação |
|-------|-----|-----|--------------|---------------|
| **symmetric** | **0.01** | **0.5101** | 3979 | **Vencedor** |
| 0.01 | 0.01 | 0.5092 | 4031 | Quase igual; alpha concentrado |
| 0.1 | 0.01 | 0.5080 | 3992 | — |
| 0.01 | None | 0.5062 | 3429 | eta default |
| symmetric | None | 0.5060 | 3427 | eta default |
| 0.1 | None | 0.5059 | 3448 | — |
| 0.5 | 0.01 | 0.5019 | 4220 | alpha alto degrada |
| 0.01 | 0.1 | 0.5002 | 3303 | eta difuso: pior |
| asymmetric | 0.01 | 0.4994 | 3963 | — |
| 0.1 | 0.1 | 0.4993 | 3330 | — |
| 0.5 | None | 0.4992 | 3634 | — |
| symmetric | 0.1 | 0.4985 | 3304 | eta difuso |
| asymmetric | 0.1 | 0.4971 | 3295 | — |
| 0.5 | 0.1 | 0.4968 | 3519 | — |
| asymmetric | None | 0.4967 | 3419 | **Pior** |

### Análise do NLP Engineer (2026-06-27)

**Padrão eta:** `eta=0.01` domina as 3 primeiras posições. Um prior muito concentrado força cada tópico a depender de poucas palavras altamente específicas, exatamente o que aumenta a co-ocorrência das top-10 keywords medida pelo C_v. Para jornalismo com vocabulário técnico por editoria ("vacina"/"imunização" para equilibrioesaude; "bolsa"/"juros" para mercado), esse comportamento é semanticamente correto: cada editoria tem um léxico próprio.

**Trade-off C_v vs Perplexidade:** `eta=0.01` eleva a perplexidade em ~15-25% relativo a `eta=None` (ex: 3979 vs 3427 para alpha=symmetric). Distribuições esparsas encaixam pior o modelo probabilístico globalmente, mas produzem tópicos mais legíveis. Para dissertação, C_v é a métrica de interpretabilidade humana: o trade-off favorece eta=0.01.

**alpha=symmetric:** Delta de 0.0009 em C_v vs alpha=0.01 é ruído estatístico. Documentos jornalísticos da Folha já são naturalmente focados em poucos tópicos por artigo: o prior alpha pouco importa porque os dados impõem concentração organicamente. `symmetric` é metodologicamente preferível: é o prior não-informativo canônico, sem assumir estrutura preexistente.

**Spread do grid:** Range C_v = 0.0134 (2.6%). O grid alpha/eta é **refinamento marginal**: a seleção de K foi o lever dominante (delta C_v = 0.182 entre K=3 e K=20). Isso é esperado: quando K está bem calibrado, a estrutura topica é robusta ao prior.

### Configuração selecionada

```
alpha = symmetric    (prior uniforme, não-informativo)
eta   = 0.01         (vocabulário esparso, exclusivo por tópico)
passes = 20          (treino final; grid usou passes=10)
```

---

## 7. Treino Final

```python
model = train_lda(
    corpus_bow, dictionary,
    k=BEST_K,          # 20
    seed=42,
    passes=20,
    alpha="symmetric",
    eta=0.01,
)
```

`LdaMulticore` usa todos os cores disponíveis. `passes=20` significa que o algoritmo percorre o corpus completo 20 vezes, necessário para convergência em corpus de tamanho médio. O modelo produz:
- **φ (phi):** matriz K × vocab_size com P(palavra | tópico)
- **θ (theta):** matriz n_docs × K com P(tópico | documento), computada em `Cell 16`

---

## 8. Nomeação de Tópicos via LLM

### Problema com o naming original

O Cell 15 original chamava `name_all_topics(topics_keywords, model=..., base_url=...)` com defaults:
- Apenas 10 keywords por tópico (de `extract_topics_keywords(top_n=10)`)
- 0 documentos representativos
- Prompt básico sem few-shot e sem anti-redundância
- `/no_think` no prompt (diretiva Qwen3, que o gemma2:2b trata como texto literal)

Para tópicos LDA amplos com K=20, apenas 10 keywords são insuficientes para distinguir sub-temas de editorias grandes como `mundo` ou `cotidiano`.

### Refinamento do prompt (2026-06-28)

Espelhado do BERTopic (`PIPELINE_BERTOPIC_FOLHA.md`), para reduzir rótulos redundantes em temas próximos e diluição em tópicos mistos (mantendo `gemma2:2b`):

1. **Few-shot de temas próximos distintos** (política climática vs. degelo polar): ensina a diferenciar temas adjacentes.
2. **Anti-redundância acionável**: identificar o termo que diferencia o tópico; proíbe sufixos "(2)"/"(gelo)".
3. **Priorização por peso**: as PRIMEIRAS palavras-chave (maior probabilidade no φ) guiam o rótulo.
4. **Assunto dominante** quando as keywords misturam dois temas.

> Afeta só a etapa de naming: re-executar a célula de nomeação aplica sem refazer o modelo.

### Pipeline melhorado (Cell 15 reescrita)

#### top_n_keywords_llm = 50

```python
top_n_llm = params["evaluation"].get("top_n_keywords_llm", 50)
topics_keywords_llm = {
    tid: [w for w, _ in model.show_topic(tid, topn=top_n_llm)]
    for tid in range(BEST_K)
}
```

Separado de `topics_keywords` (usado para métricas, top_n=10): o LLM recebe 50 keywords para ter mais contexto sobre o tema do tópico.

#### Docs representativos por theta

LDA não tem embeddings: não é possível calcular centróide no espaço vetorial como no BERTopic. O equivalente natural são os **documentos com maior probabilidade θ para o tópico**:

```python
def _representative_docs_lda(tid, n=3):
    top_idx = np.argsort(theta_arr[:, tid])[::-1][:n]
    return [docs[i] for i in top_idx]
```

`theta_arr[:, tid]` é o vetor de probabilidades de todos os documentos para o tópico `tid`. Os 3 primeiros do ranking são os documentos mais "puros": falam principalmente do tema do tópico. Isso cumpre o mesmo papel dos 3 docs do centróide do BERTopic: mostrar ao LLM amplitude do tema sem ancorar num evento específico.

> **Theta inline:** O cell calcula theta inline se `theta_arr` ainda não está no namespace (execução fora de ordem), garantindo robustez.

#### Anti-redundância acumulativa

Idêntico ao BERTopic (Cell 45): nomes já atribuídos entram no prompt do próximo tópico. Maiores tópicos são nomeados primeiro: "ocupam" os nomes mais abrangentes; menores se diferenciam.

```python
_order_lda = sorted(range(BEST_K), key=lambda t: -_topic_counts_lda.get(t, 0))
```

#### Deduplicação pós-geração

Mesmo mecanismo do BERTopic: tenta annexar a keyword mais distintiva ausente do nome; último recurso é sufixo numérico.

#### Prompt few-shot PT-BR

Mesmos 6 exemplos positivos do BERTopic (eleição, juros, futebol, STF, clima, cinema) + 1 exemplo negativo de lista de keywords. A regra "TEMA GERAL" evita que o LLM ancore num evento específico de um único documento.

### Diferenças BERTopic vs LDA no naming

| Elemento | BERTopic | LDA |
|----------|---------|-----|
| Docs representativos | Centróide do cluster (espaço 4096d) | Maior P(tópico\|doc) em theta |
| Keywords para LLM | top_n_llm do c-TF-IDF | top_n_llm do phi de gensim |
| Lógica de representação | Distância geométrica | Probabilidade bayesiana |
| Resultado esperado | Docs no "núcleo" do cluster | Docs "mais puros" do tópico |

---

## 9. Métricas de Avaliação

### C_v (Coerência)

Calculada duas vezes: no grid (passes=10, estimativa rápida) e após o treino final (passes=20, valor definitivo). A C_v do grid é usada para comparação relativa entre K; a C_v recomputada é reportada na dissertação.

**Interpretação da escala:**
- < 0.40: tópicos incoerentes (keywords não co-ocorrem)
- 0.40-0.55: razoável para corpus genérico (jornalismo coberto)
- > 0.55: boa coerência (esperado em corpus técnico especializado)

O corpus Folha (jornalismo generalista) atinge C_v~0.55 @ K=20, adequado para o tipo de dado.

### Perplexidade

```python
perplexity = float(np.exp(-model.log_perplexity(corpus_bow)))
```

Mede o quanto o modelo "se surpreende" com os dados, proxy de fit probabilístico. Menor = melhor. Não é comparável com C_v (trade-off intrínseco: eta=0.01 eleva perplexidade para ganhar coerência). Reportar como diagnóstico secundário na dissertação.

> **Nota:** é calculada **in-sample** (sobre `corpus_bow`, o próprio conjunto de treino), não sobre dado retido. Mede fit no dado visto, não generalização: modelos com mais tópicos tendem a perplexidade de treino menor por sobreajuste. Descrever na dissertação como "perplexidade de treino".

### Exclusividade, Diversity, FREX

Definições idênticas ao BERTopic (`PIPELINE_BERTOPIC_FOLHA.md`). A **exclusividade** usa a mesma função `compute_exclusivity_ctfidf` (versão contínua: para cada keyword top-N, fração da massa de score concentrada no tópico), aplicada à matriz φ (phi) do gensim no lugar do c-TF-IDF do BERTopic. Os scores `topic_word_scores` são construídos da matriz φ **completa** (todas as palavras com P(w|t) > 0), `top_n = top_n_keywords_metrics` (20), exatamente como no BERTopic. Os valores de exclusividade LDA e BERTopic são, portanto, **diretamente comparáveis** (mesma escala). FREX e Diversity também são calculadas sobre φ.

> **⚠ FREX suspeito (verificar antes de reportar):** nas rodadas de 2026-06-28 o FREX ficou ~0.97 (LDA) / ~0.98 (BERTopic) e constante em todos os top_n. FREX deveria variar conforme a lista de keywords: o valor achatado indica **saturação/possível bug em `compute_frex_score`**. Não reportar esse número na dissertação sem antes auditar a função.

> Versões anteriores deste notebook usavam `compute_exclusivity` (binária: conta keywords que aparecem em um único tópico), que não era comparável ao BERTopic. Corrigido na auditoria de 2026-06-27.

### Estabilidade Jaccard

Para cada seed alternativo em `stability_seeds`, treina um LDA independente (com `alpha=best_alpha, eta=best_eta`, idêntico ao modelo final) e calcula o índice de Jaccard entre as top-N keywords de cada par de tópicos alinhados. LDA é mais instável que BERTopic para tópicos de baixa coerência: o Jaccard por tópico identifica os mais frágeis.

> ⚠ **DESATIVADA na configuração de produção.** Jaccard requer ao menos um seed diferente do seed final (42). Com `stability_seeds: [42]` (config atual), não há seed alternativo: as células de estabilidade fazem *skip* explícito (imprimem "DESATIVADA"), não treinam o modelo duplicado nem geram `lda_stability_boxplot.png`. Para ativar, adicionar ao menos um seed (ex.: `stability_seeds: [42, 123]`) em `params.yaml > evaluation`. **Não reportar Jaccard na dissertação** sem antes ativar com seeds adicionais.

### Distribuição θ (doc→tópico)

O LDA atribui probabilidade a **todos os K tópicos** para cada documento (nenhum é zero). O "tópico dominante" é `argmax(θ_d)`. O heatmap θ (doc × tópico) e o t-SNE no espaço θ visualizam como os documentos se organizam no espaço de tópicos.

---

## 10. Comparação com BERTopic

| Dimensão | LDA (K=20) | BERTopic (K≈24) |
|----------|-----------|----------------|
| **Corpus de treino** | 5k docs | N docs (completo) |
| **C_v** | **0.657** (recomputado) | 0.620 (top-10) |
| **Outliers** | 0% (todos atribuídos) | 21.7% (após reduce_outliers) |
| **Granularidade de tópico** | 1 tópico = mistura de temas | 1 tópico = cluster semântico denso |
| **Distribuição θ** | Suave (doc em múltiplos tópicos) | Hard (tópico dominante único) |
| **Vocabulário** | BoW, lemas | Embeddings + c-TF-IDF |
| **Reprodutibilidade** | seed=42 → determinístico | seed=42 → determinístico |
| **Outliers temporais** | Análise sobre todos os docs | Análise exclui outliers remanescentes |

### Quando LDA captura o que BERTopic não captura

- Documentos multi-temáticos (matérias de opinião que cruzam política e economia): LDA distribui a probabilidade entre ambos; BERTopic force-classifica no tópico dominante.
- Tópicos com vocabulário sobreposto: LDA usa co-ocorrência; BERTopic usa distância semântica e pode separar mesmo com vocabulário parecido.

### Quando BERTopic captura o que LDA não captura

- Polissemia: "banco" financeiro vs "banco" de praça, embeddings distinguem; BoW não.
- Sub-temas com vocabulário muito similar: HDBSCAN pode separar clusters densos que o LDA funde num único tópico.
- Documentos atípicos: BERTopic os isola como outliers; LDA os distribui em tópicos existentes.

---

## 11. Configuração Final

### params.yaml (seções relevantes para LDA)

```yaml
lda:
  k_range: [5, 30]          # grid K; BEST_K = automático (peak C_v)
  no_below: 5               # filter_extremes: palavra em ≥ 5 docs
  no_above: 0.5             # filter_extremes: palavra em ≤ 50% dos docs

evaluation:
  top_n_keywords: 10        # keywords para métricas (C_v, Exclus, FREX)
  top_n_keywords_metrics: 20
  top_n_keywords_llm: 50    # keywords passadas ao LLM para naming
  stability_seeds: [42]     # = seed final → Jaccard DESATIVADO (sem seed alternativo).
                            #   Para ativar: [42, 123, ...]
  lda_alpha_grid:
    - symmetric             # ← vencedor: prior uniforme não-informativo
    - asymmetric
    - 0.01
    - 0.1
    - 0.5
  lda_eta_grid:
    - null                  # None = gensim default (1/K)
    - 0.01                  # ← vencedor: vocabulário esparso por tópico
    - 0.1
```

### Decisões fixas no notebook

| Parâmetro | Valor | Justificativa |
|-----------|-------|---------------|
| `BEST_K` | automático (peak C_v) | K=20; 2 sub-tópicos/editoria; plateau K=20-30 confirma |
| `passes` (grid K) | 20 | Seleção de K via C_v (Cell 8 / `grid_search_k`) |
| `passes` (grid alpha×eta) | 10 | Refinamento de prior (Cell 12); estimativa rápida para comparação relativa |
| `passes` (treino final) | 20 | Convergência robusta |
| `alpha` | symmetric | Prior não-informativo; delta vs 0.01 é ruído (0.0009 C_v) |
| `eta` | 0.01 | Vocabulário esparso e exclusivo; maximiza C_v (+0.013 vs eta=None) |
| `N_REPR_DOCS_LDA` | 3 | Top-3 docs por theta; amplitude sem over-specification |
| `top_n_llm` | 50 | Lido de params.yaml; contexto rico para tópicos mistos |

---

## 12. Arquivos de Saída

Cada execução gera `data/output/folha/lda/folha_<AAAAMMDD>_<HHMMSS>/` (subpasta `lda/`, separada do BERTopic que grava em `data/output/folha/bertopic/`):

| Arquivo | Conteúdo |
|---------|----------|
| `lda_results.csv` | doc_id, topic_id, topic_prob, topic_name, text |
| `lda_topics_for_eval.csv` | keywords por tópico (para revisão humana) |
| `lda_metrics.csv` | Métricas agregadas (C_v, Exclus, Diversity, FREX, Perplexidade, alpha, eta, K) |
| `lda_pyldavis_K20.html` | Visualização canônica pyLDAvis interativa |
| `lda_heatmap_phi.png` | Palavras × tópicos (distribuição φ) |
| `lda_heatmap_topicos_cosine.png` | Similaridade cosine entre tópicos |
| `lda_tsne_theta_topico.png` | t-SNE do espaço θ por tópico dominante |
| `lda_tsne_theta_interactive.html` | t-SNE interativo (Plotly, hover com texto) |
| `lda_topic_category.png` | Heatmap categoria × tópico (% por editoria) |
| `lda_stability_boxplot.png` | Estabilidade Jaccard por tópico |
| `lda_topics_seed<N>.csv` | Distribuição tópicos para cada seed de estabilidade |

**Arquivos de cache (`BASE_OUTPUT_DIR = data/output/folha/lda/`, reutilizados entre runs: vivem no diretório-base, pai dos run dirs carimbados):**

| Arquivo | Conteúdo |
|---------|----------|
| `data/output/folha/lda/lda_metrics.csv` | Cache do grid K (evita refit demorado) |
| `data/output/folha/lda/lda_alpha_eta_grid.csv` | Cache do grid alpha×eta (✅ já computado) |

> O `lda_alpha_eta_grid.csv` foi movido para `data/output/folha/lda/` na reorganização de 2026-06-27. Se o cache do grid K (`lda_metrics.csv`) não estiver presente, a Cell 8 o recomputa (~13min).

---

## Notas de execução

```bash
# 1. Ollama rodando (para naming LLM)
ollama serve

# 2. Executar notebook completo
# Run > Run All Cells  (02_lda_folha.ipynb)

# 3. Tempos esperados (CPU):
#    - Lematização (Cell 5):     ~448s (7min)
#    - Grid K (Cell 8):          ~13min (apenas se lda_metrics.csv ausente)
#    - Grid alpha/eta (Cell 12): ~30min × 15 combos (já computado → ~1s do cache)
#    - Treino final (Cell 12):   ~5-10min
#    - Naming LLM (Cell 15):     ~9s × 20 tópicos = ~3min
#    - t-SNE (Cell 20):          ~2min
#    - Total (sem grid):         ~20-30min

# 4. Cache disponível:
#    lda_alpha_eta_grid.csv → best: alpha=symmetric, eta=0.01
#    Não precisa re-rodar o grid alpha/eta.
```

---

## 13. Resultados da Rodada de Produção (`folha_20260629_004450`)

> Run de 2026-06-29: K=20, alpha=symmetric, eta=0.01, passes=20. Corpus `folha_20260628_185652` (`html.unescape` aplicado: entidades HTML corrigidas). Naming via `gemma2:2b` (prompt melhorado: 50 keywords + 3 docs representativos por theta + anti-redundância acumulativa + regra TEMA GERAL).

### Métricas

| Métrica | LDA `004450` | LDA anterior `122734` | BERTopic `221412` | Leitura |
|---|---|---|---|---|
| **C_v (recomputado)** | **0.657** | 0.618 | 0.620 | Melhoria +6.3% com corpus limpo; triangulação LDA ≈ BERTopic |
| Exclusividade (c-TF-IDF) | 0.457 | 0.446 | 0.542 | LDA < BERTopic (BoW menos exclusivo que embeddings) |
| Topic Diversity (Dieng) | 0.770 | 0.765 | 0.892 | BERTopic mais diverso |
| Perplexidade | 3431 | 3496 | — | Melhora com corpus limpo; diagnóstico secundário (in-sample) |
| FREX | 0.974 | 0.972 | 0.982 | ⚠ saturado: **não reportar sem auditar** |

20 tópicos, **0% outliers** (LDA atribui todos os docs, contraste com 21.7% do BERTopic).

> **Por que C_v melhorou +6.3%?** O corpus anterior tinha entidades HTML não decodificadas (`&eacute;` → `eacute` como token) que inflavam o vocabulário com ruído. Com o corpus limpo, as co-ocorrências nas top-10 keywords são reais, elevando a C_v. O BERTopic não ganhou nessa dimensão porque o `stopwords_emojis` já filtrava esses tokens do CountVectorizer.

### Qualidade dos tópicos

**Limpos** (~15): T0 Economia BR/finanças, T1 Exploração espacial, T2 Desmatamento/Bolsonaro, T3 Conflitos internacionais, T5 Copa do Mundo, T6 Políticas Públicas/Justiça, T7 Economia global/inflação, T9 Genética/evolução, T10 Eleições, T11 Mudanças climáticas, T12 Vacinação/Covid, T15 Pesquisa científica, T16 Enem, T17 Educação/financiamento, T18 Polícia/segurança.

**Com ressalvas** (característico de LDA: BoW distribui co-ocorrência):
- **T4** "Paleontologia e fósseis": keyword `gelo` na posição #1 (co-ocorrência espúria com tópicos de clima). Keywords seguintes (`fóssil`, `dinossauro`, `cientista`) confirmam a temática; nome correto.
- **T8** "Literatura e história do homem": grab-bag (pessoa, livro, vida, mundo, negro, pai, mulher). Tópico misto; tratar como ressalva na análise.
- **T13** "Cinema e música": `the` em posição #8. Artigo inglês em citações PT-BR não filtrado pelo lematizador spaCy.
- **T14** "Chuvas no litoral paulista": muito específico; candidato a merge com T11 (clima) na análise qualitativa.
- **T19** "Olímpicos e Atletas": `seta` em posição #6. Símbolo `→` de listas tokenizado pelo gensim.

> **Tokens residuais `seta`/`the` no LDA:** o filtro `stopwords_emojis` do `params.yaml` só é aplicado ao CountVectorizer do BERTopic, não ao dicionário gensim do LDA. Esses tokens passam pelo `filter_extremes`. Sem impacto nos nomes dos tópicos; ruído interno nas keywords. Sem fix disponível sem reescrever a etapa de lematização.

### Melhorias em relação ao run anterior (`122734`)

| Problema (run `122734`) | Solução (run `004450`) |
|---|---|
| T11 `**Pós-graduação e ciência brasileira.**`: markdown no nome | T15 "Pesquisa científica e saúde no Brasil" ✅ |
| T12 "Crise no Ministério da Educação": nome errado para keywords de ciência/CAPES | T12 "Vacinação e Covid no Brasil" ✅ |
| `eacute`, `ccedil` em keywords | Ausentes ✅ (corpus corrigido) |
| C_v 0.618 | C_v **0.657** (+6.3%) ✅ |

---

## 14. Pendências (status 2026-06-29)

| Item | Status |
|---|---|
| Blocker `name_topic`/`dominant`/reorder naming | ✅ corrigido |
| Exclusividade LDA → c-TF-IDF (comparável) | ✅ corrigido |
| Versionamento `lda/` vs `bertopic/` (folha) | ✅ corrigido |
| Prompt de naming melhorado (anti-redundância, assunto dominante) | ✅ aplicado e re-rodado |
| **Entidades HTML no preprocessing** (`html.unescape`) | ✅ **propagado**: pipeline 01→02→03 re-rodado (corpus `folha_20260628_185652`) |
| `seta`/`the` em keywords LDA | ⚠ estrutural: `stopwords_emojis` não alcança gensim BoW; sem fix sem reescrever lematização |
| FREX saturado (~0.97) | ⚠ aberto: auditar `compute_frex_score` ou não reportar na dissertação |
| Estabilidade Jaccard | ➖ desativada por design (`stability_seeds=[42]`), opcional (adicionar seed p/ ativar) |

---

*Documentação gerada em 2026-06-27. Atualizada em 2026-06-28: resultados do run `folha_20260628_122734`, exclusividade c-TF-IDF, melhoria do prompt, e correção de entidades HTML no preprocessing. Atualizada em 2026-06-29: resultados definitivos do run `folha_20260629_004450` (corpus `folha_20260628_185652`), pendências encerradas.*
