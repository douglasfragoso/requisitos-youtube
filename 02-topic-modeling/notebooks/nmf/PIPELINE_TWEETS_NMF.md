# NMF — Tweets BR 2022: Documentação do Pipeline e Resultados

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
> **Notebook:** `03_nmf_tweets_bre2022.ipynb`
> **Configuração:** `03-topic-modeling/configs/params.yaml` → corpus `tweets_bre2022`
> **Última revisão:** 2026-07-14: run de produção do NMF concluído (`tweets_bre2022_20260714_002347`), K=8 via grid search, resultados analisados neste documento
>
> A documentação do **corpus** (origem Zenodo, ausência de setembro, estratégia de amostragem, atrito do pré-processamento) está em `../lda/PIPELINE_TWEETS_LDA.md` §1-8 e não é repetida aqui. Este documento cobre o pipeline NMF e seus resultados.

---

## Índice

1. [Escopo e hipótese H2](#1-escopo-e-hipótese-h2)
2. [Corpus utilizado](#2-corpus-utilizado)
3. [Lematização e vocabulário](#3-lematização-e-vocabulário)
4. [Seleção de K — grid search C_v](#4-seleção-de-k--grid-search-c_v)
5. [Grid kappa × minimum_probability e treino final](#5-grid-kappa--minimum_probability-e-treino-final)
6. [Nomeação de tópicos via LLM](#6-nomeação-de-tópicos-via-llm)
7. [Resultados da rodada de produção](#7-resultados-da-rodada-de-produção-tweets_bre2022_20260714_002347)
8. [Comparação com Folha (NMF) e com LDA/BERTopic (tweets)](#8-comparação-com-folha-nmf-e-com-ldabertopic-tweets)
9. [Pendências e arquivos de saída](#9-pendências-e-arquivos-de-saída)

---

## 1. Escopo e hipótese H2

Aplica `gensim.models.Nmf` ao corpus de tweets seguindo a mesma estrutura do `03_nmf_folha.ipynb` (conceitos de NMF em `PIPELINE_NMF_FOLHA.md` §1) e do `02_lda_tweets_bre2022.ipynb`.

**Hipótese H2:** assim como observado no LDA, espera-se que topic modeling BoW em texto curto/informal degrade a coerência vs corpus formal (Folha). Este notebook documenta empiricamente esse comportamento para o NMF.

**Resultado antecipado:** H2 confirmada — C_v recomputado de **0.559** (tweets) vs **0.655** (Folha), queda de ~15%, na mesma direção do LDA (0.527 vs 0.657, queda de ~20%). O NMF degradou *menos* que o LDA em texto curto.

Diferenças estruturais vs LDA mantidas da Folha: sem perplexidade (o `Nmf` não expõe `log_perplexity`) e sem pyLDAvis.

---

## 2. Corpus utilizado

- **Versão (latest-wins):** `tweets_bre2022_20260629_215159` — mesma do run de produção do LDA
- **N:** 8.811 tweets (corpus completo, **sem subsample** — diferente da Folha, onde se usa 5k)
- Comprimento: média 22,8 palavras/tweet (mediana 19)
- `category` = mês de coleta (`YYYY-MM`): proxy **temporal**, não rótulo temático — cross-tab tópico×mês é checagem de consistência temporal, não validação de ground-truth
- Distribuição mensal: 2022-08 (2.621), 2022-10 (1.381), 2022-11 (2.070), 2022-12 (2.739)

---

## 3. Lematização e vocabulário

Mesma `lemmatize_corpus` (spaCy `pt_core_news_lg`) com `model_key="nmf"`:

| Métrica | Tweets NMF | Tweets LDA | Folha NMF |
|---|---|---|---|
| Tempo de lematização | 23s | 24s | 349s |
| Vocabulário final (após `filter_extremes`) | **2.617** | 2.617 | 17.331 |
| Tokens/doc após lematizar (média) | 10,9 | 10,9 | 377,1 |
| Tokens únicos/doc no BoW (média) | 8,8 | 8,8 | 219,2 |

Vocabulário e estatísticas **idênticos aos do LDA tweets** (mesmos `no_below=5`/`no_above=0.5`): a comparação NMF×LDA neste corpus é sobre exatamente o mesmo BoW.

---

## 4. Seleção de K — grid search C_v

Grid `K ∈ [3, 5, 7, 8, 10, 12, 15, 20, 25, 30]`, `passes=20`, seed=42, via `grid_search_k_nmf` (cache em `data/output/tweets_bre2022/nmf/nmf_metrics.csv`):

| K | C_v |
|---|-----|
| 3 | 0,331 |
| 5 | 0,395 |
| 7 | 0,414 |
| **8** | **0,492 ← pico** |
| 10 | 0,452 |
| 12 | 0,447 |
| 15 | 0,421 |
| 20 | 0,447 |
| 25 | 0,403 |
| 30 | 0,391 |

**BEST_K = 8** — pico interior nítido (cai de forma consistente para K≥10). Contraste forte com o LDA no mesmo corpus, que escolheu **K=20** (C_v 0,430): o NMF prefere granularidade **muito mais grossa** nos tweets. Leitura: com vocabulário esparso (2.617 palavras, ~9 tokens únicos/doc), a fatoração não sustenta 20+ fatores com co-ocorrência forte — 8 macro-discursos eleitorais maximizam a coerência. Curiosamente, o C_v do pico do NMF (0,492) supera o do pico do LDA (0,430) no mesmo BoW.

Também é o inverso do padrão da Folha, onde o NMF pediu K *maior* que o LDA (25 vs 20): a direção do ajuste de granularidade do NMF depende da densidade do corpus.

---

## 5. Grid kappa × minimum_probability e treino final

Grid `kappa ∈ [0.5, 1.0, 2.0]` × `minimum_probability ∈ [0.0, 0.01, 0.05]` (9 combos), `passes=10`, K=8 fixo (cache em `nmf_kappa_minprob_grid.csv`):

| kappa | minimum_probability | C_v |
|-------|---------------------|-----|
| **1.0** | **0.00** | **0,4822 ← melhor** |
| 1.0 | 0.01 | 0,4822 |
| 1.0 | 0.05 | 0,4822 |
| 0.5 | 0.00 | 0,4742 |
| 0.5 | 0.01 | 0,4742 |
| 0.5 | 0.05 | 0,4742 |
| 2.0 | 0.00 | 0,3792 |
| 2.0 | 0.01 | 0,3792 |
| 2.0 | 0.05 | 0,3792 |

**Escolhido: `kappa=1.0, minimum_probability=0.0`.**

Duas observações (a primeira compartilhada com a Folha, a segunda oposta):

1. **`minimum_probability` não afeta C_v** (linhas idênticas por kappa) — é limiar de inferência sobre θ, não parâmetro de treino; as keywords vêm de H. O knob efetivo é só o kappa (ver `PIPELINE_NMF_FOLHA.md` §6).
2. **kappa=2.0 é o pior aqui** (−21% vs 1.0), sendo o **vencedor na Folha**. Em corpus esparso, o passo agressivo desestabiliza a fatoração. Confirma que o grid de kappa precisa ser rodado por corpus — não há valor universal.

**Treino final:** `K=8, kappa=1.0, minimum_probability=0.0, passes=20` — **3s** (vs 27s do LDA tweets com K=20).

---

## 6. Nomeação de tópicos via LLM

Mesmo mecanismo (`name_all_topics`, modelo `gemma2:2b-instruct-q4_K_M` de `params.yaml > llm.model`): 8 tópicos nomeados em **27s** (~3,4s/tópico).

> Nota: diferente do `03_nmf_folha.ipynb` (que usa o pipeline melhorado com 50 keywords + 3 docs por theta + anti-redundância), este notebook usa a chamada padrão `name_all_topics(topics_keywords, ...)` com as top-10 keywords. Com só 8 tópicos bem separados, não houve colisão de nomes; se K crescer numa revisão futura, portar o pipeline melhorado da Folha.

---

## 7. Resultados da Rodada de Produção (`tweets_bre2022_20260714_002347`)

> Run de 2026-07-14, corpus `tweets_bre2022_20260629_215159`. Substitui o run exploratório `tweets_bre2022_20260713_202436` (mesma configuração; o `002347` é o "Run All" canônico).

### Configuração final aplicada

| Parâmetro | Valor | Evidência |
|---|---|---|
| K | **8** | pico C_v (0,4920), interior ao range testado |
| kappa | **1.0** | melhor C_v (0,4822) no grid kappa×min_prob |
| minimum_probability | **0.0** | sem efeito em C_v; preserva θ completo |
| passes (treino final) | 20 | mesma convenção da Folha |

### Métricas

| Métrica | Valor |
|---|---|
| C_v (recomputado no modelo final) | **0,559** |
| Exclusividade (word-overlap simples, `compute_exclusivity`) | 0,855 |
| Topic Diversity (Dieng) | 0,8625 |
| Diversity entropy (θ) | 0,913 |
| Outliers | **0%** (NMF sempre produz um tópico dominante) |
| FREX | não calculada neste notebook (só na Folha, onde está saturada) |
| Estabilidade Jaccard | N/A — `stability_seeds=[42]`, desativada por design |

> ⚠ **Exclusividade não-comparável com a Folha NMF:** este notebook usa `compute_exclusivity` (binária/word-overlap, mesma do LDA tweets — 0,817), enquanto o `03_nmf_folha.ipynb` usa `compute_exclusivity_ctfidf` (contínua, 0,475). Comparações válidas: tweets NMF (0,855) × tweets LDA (0,817) ✅; folha NMF (0,475) × folha LDA (0,457) ✅. Não cruzar as duas escalas.

### 8 tópicos produzidos

| ID | Nome LLM | Keywords (top-5) | Docs | % |
|---|---|---|---|---|
| T0 | Brasília e Terrorismo ⚠ | brasília, ser, polícia, federal, bolsonarista | 2.969 | 33,7% |
| T1 | Orçamento Secreto Presidencial | orçamento, secreto, presidente, hoje, dia | 200 | 2,3% |
| T2 | Polícia e Rodoviária em Nordeste | votar, nordeste, deixem, federal, polícia | 383 | 4,3% |
| T3 | Bolsonaro e Governo | bolsonaro, ladrão, eduardo, cadeia, catar | 1.205 | 13,7% |
| T4 | Posse de Lula | lula, presidente, lulapresidente, diplomação, hoje | 1.357 | 15,4% |
| T5 | Mané e Debatenaband | mané, perdeu, pra, debatenaband, amolar | 1.264 | 14,3% |
| T6 | Jair Rombo Bilionário | brasil, jair, quebrou, saqueou, rombo | 320 | 3,6% |
| T7 | Globo e Bolsonaro | globolixo, bonner, bolsonaronojn, renata, presidente | 1.113 | 12,6% |

Os 8 fatores mapeiam macro-discursos reconhecíveis da eleição de 2022: atos golpistas de dezembro em Brasília (T0), orçamento secreto/diplomação (T1), bloqueios da PRF no Nordeste no 2º turno (T2), ataques a Bolsonaro (T3), vitória/posse de Lula (T4), provocação "mané" pós-eleição (T5), narrativa econômica anti-Bolsonaro (T6) e o embate com o Jornal Nacional/Globo (T7).

### Qualidade dos tópicos — pontos de atenção

| Tópico | Observação | Natureza |
|---|---|---|
| **T0 ⚠** | **33,7% dos docs** — 3× o segundo maior. Keywords incluem `ser` e `dia` (quase-stopwords), sintoma de fator residual: além do tema real (atos de dezembro em Brasília), absorve tweets genéricos que não encaixam nos outros 7 fatores. Com θ esparso do NMF, isso é o análogo funcional de um "tópico lixo" | Estrutural do K baixo; ao citar T0 na dissertação, separar o núcleo temático (terrorismo/Brasília, docs com P≈1.0) da cauda genérica |
| T5 | `pra` nas keywords (coloquialismo não filtrado) | Ruído de lematização de texto informal, mesma classe do `tar` no LDA tweets |
| T1 | Apenas 2,3% dos docs, mas altamente coeso (top docs P=1.0 todos "ORÇAMENTO SECRETO NÃO") | Fator pequeno e nítido — exemplo da esparsidade do NMF isolando uma campanha de hashtag específica |

Nenhum tópico degenerado (sem massa de probabilidade) — a guarda de "tópico morto" do notebook (wordclouds) não disparou com K=8.

### Distribuição vs LDA tweets

LDA (K=20) distribuiu de forma equilibrada (3,0%–8,3% por tópico); o NMF (K=8) concentrou (2,3%–33,7%). São retratos diferentes do mesmo corpus: o LDA fragmenta os macro-discursos em sub-tópicos de tamanho parecido; o NMF agrega e deixa a assimetria real de volume aparecer (com a ressalva do componente residual em T0).

---

## 8. Comparação com Folha (NMF) e com LDA/BERTopic (tweets)

### Vs. Folha NMF (`PIPELINE_NMF_FOLHA.md`) — teste da H2

| Métrica | Folha (K=25) | Tweets (K=8) | Leitura |
|---|---|---|---|
| C_v (recomputado) | 0,655 | **0,559** | H2 confirmada: texto curto degrada (−15%); menos que no LDA (−20%) |
| K escolhido | 25 | 8 | Direções opostas vs LDA (Folha: NMF>LDA; tweets: NMF≪LDA) — granularidade do NMF segue a densidade do corpus |
| kappa vencedor | 2.0 | 1.0 (2.0 é o pior) | Hiperparâmetro não transfere entre corpora |
| Topic Diversity (Dieng) | 0,74 | 0,8625 | Vocabulários eleitorais específicos por fator |
| Outliers | 0% | 0% | Estrutural do NMF |

### Vs. LDA tweets (`../lda/PIPELINE_TWEETS_LDA.md`, run `214134`) — mesmo BoW

| Métrica | NMF (K=8) | LDA (K=20) | Leitura |
|---|---|---|---|
| C_v (recomputado) | **0,559** | 0,527 | NMF mais coerente no mesmo vocabulário — e com pico de grid também superior (0,492 vs 0,430) |
| Exclusividade (word-overlap, mesma função) | **0,855** | 0,817 | ✅ comparável: fatoração esparsa gera fronteiras lexicais mais nítidas |
| Topic Diversity (Dieng) | 0,8625 | 0,71 | NMF mais diverso (menos tópicos → menos overlap nas listas) |
| Diversity entropy (θ) | 0,913 | 0,994 | θ do NMF bem mais concentrado (docs quase-puros) |
| Distribuição de docs | concentrada (2,3%–33,7%) | equilibrada (3,0%–8,3%) | Trade-off granularidade × equilíbrio |
| Tempo de treino final | 3s | 27s | NMF ~9× mais rápido |
| Perplexidade | n/d | disponível | Limitação do `Nmf` |

### Vs. BERTopic tweets (`PIPELINE_TWEETS_BERTOPIC.md`, run definitivo `tweets_bre2022_20260703_101639`, K=24)

| Métrica | NMF (K=8) | BERTopic (K=24) | Leitura |
|---|---|---|---|
| C_v | 0,559 | 0,570 | ⚠ indicativo, não estrito: BERTopic calcula sobre keywords c-TF-IDF/tokenização própria, NMF sobre o BoW lematizado |
| Exclusividade word-overlap | 0,855 | 0,913 | BERTopic mais exclusivo também na métrica simples |
| Topic Diversity (Dieng) | 0,8625 | 0,929 | BERTopic mais diverso |
| Outliers / cobertura | **0% / 100%** | 16,5% / 83,5% | Diferença estrutural central: NMF atribui tudo; BERTopic rejeita ruído |

O BERTopic (após a recalibração de 2026-07-03, que reduziu outliers de 37,6% para 16,5%) supera o NMF por margens pequenas nas métricas de qualidade, ao custo de não classificar 16,5% dos tweets. O NMF fica no meio-termo de granularidade (8) entre a visão macro e os 24 clusters semânticos do BERTopic.

**Leitura geral para a dissertação:** nos tweets, o NMF entrega a visão mais parcimoniosa (8 macro-discursos com C_v superior ao LDA no mesmo BoW), o LDA a visão mais granular e equilibrada (20 sub-discursos), e o BERTopic a visão semântica com rejeição de ruído. A triangulação é complementar, não competitiva.

---

## 9. Pendências e arquivos de saída

### Concluído (2026-07-14)

- **Grid K e grid kappa×min_prob:** completos e cacheados (`nmf_metrics.csv` e `nmf_kappa_minprob_grid.csv` na pasta-base `data/output/tweets_bre2022/nmf/`). Próximos runs pulam os dois grids automaticamente.
- **Run de produção ponta a ponta:** `tweets_bre2022_20260714_002347` (Run All sem erros, com naming LLM, t-SNE, cross-tab mensal e exports).
- Guarda de **tópico degenerado** nas wordclouds (NaN em `show_topic` com K alto em texto curto) implementada — não disparou com K=8.

### Pendente (próximos passos)

1. **T0 (33,7%):** análise qualitativa separando o núcleo "atos de Brasília" da cauda residual genérica antes de citar o tópico na dissertação.
2. **Unificar a métrica de exclusividade** com a versão c-TF-IDF usada na Folha NMF (e no LDA/BERTopic da Folha), para permitir comparação cross-corpus — mesma pendência do LDA tweets.
3. **Portar o pipeline de naming melhorado** da Folha NMF (50 keywords + 3 docs por theta + anti-redundância) se K for revisado para cima.
4. **Estabilidade multi-seed:** adicionar seeds em `params.yaml > evaluation.stability_seeds` (hoje `[42]`) — NMF é sabidamente sensível à inicialização, a validação multi-seed é mais importante aqui do que no LDA.
5. **Cross-tab tópico × mês / t-SNE / top docs:** gerados no run mas não analisados neste documento — candidatos a seção qualitativa futura (ex.: T0 e T1 devem concentrar em 2022-12; T2 em 2022-10/11).

### Arquivos de saída

Cada execução gera `data/output/tweets_bre2022/nmf/tweets_bre2022_<AAAAMMDD>_<HHMMSS>/`:

| Arquivo | Conteúdo |
|---|---|
| `nmf_results.csv` | post_id, doc_id, text, topic_id, topic_name, distribuição θ completa |
| `nmf_topics_for_eval.csv` | Keywords + nomes LLM por tópico |
| `nmf_metrics.csv` | Métricas agregadas do run (também cacheado na pasta-base com `k_grid_scores`) |
| `nmf_tsne_theta_topico.png` / `nmf_tsne_theta_interactive.html` | t-SNE do espaço θ |
| `nmf_kappa_minprob_grid.csv` (pasta-base) | Cache do grid kappa×min_prob, reutilizado entre runs |

> Sem pyLDAvis (não aplicável a NMF) e sem `nmf_topic_category.png` neste corpus (o heatmap tópico×mês é exibido no notebook; `category` aqui é proxy temporal, não editoria).

---

*Documentação gerada em 2026-07-14 a partir do run `tweets_bre2022_20260714_002347` (corpus `tweets_bre2022_20260629_215159`). Espelha a estrutura de `../lda/PIPELINE_TWEETS_LDA.md`; conceitos de NMF e pipeline detalhado em `PIPELINE_NMF_FOLHA.md`.*
