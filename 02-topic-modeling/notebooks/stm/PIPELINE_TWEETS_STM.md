# STM — Tweets BR Eleições 2022: Documentação Completa do Pipeline

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


> **Corpus:** Tweets PT-BR sobre as eleições brasileiras de 2022 (Zenodo 14834749), texto curto/informal
> **Notebook:** `04_stm_tweets_bre2022.ipynb`
> **Configuração:** `03-topic-modeling/configs/params.yaml`
> **Última revisão:** 2026-07-16: resultados do run de produção `tweets_bre2022_20260716_175301` (corpus `tweets_bre2022_20260629_215159`, retreino convergido após `max_em_its` 150→300); hiperparâmetros pinados no `params.yaml`
>
> Método, conceitos e arquitetura do pipeline: ver `PIPELINE_STM_FOLHA.md` §1-2 (idênticos; este documento cobre o que difere e os resultados do corpus).

---

## Índice

1. [Corpus e filtros](#1-corpus-e-filtros)
2. [Lematização e vocabulário](#2-lematização-e-vocabulário)
3. [Seleção de K — grid search C_v](#3-seleção-de-k--grid-search-c_v)
4. [Grid sigma.prior × gamma.prior](#4-grid-sigmaprior--gammaprior)
5. [Treino final — o episódio do teto de EM](#5-treino-final--o-episódio-do-teto-de-em)
6. [Resultados da rodada de produção](#6-resultados-da-rodada-de-produção-tweets_bre2022_20260716_175301)
7. [Efeitos de prevalência por mês (estimateEffect)](#7-efeitos-de-prevalência-por-mês-estimateeffect)
8. [Comparação com LDA, NMF e BERTopic + H2](#8-comparação-com-lda-nmf-e-bertopic--h2)
9. [Configuração final](#9-configuração-final)
10. [Pendências](#10-pendências-status-2026-07-16)

---

## 1. Corpus e Filtros

- **8.811 tweets** (corpus completo — sem subsample; mesmo N do LDA/NMF)
- Corpus resolvido via `resolve_latest_dir`: **`tweets_bre2022_20260629_215159`** (o mesmo dos runs definitivos de LDA/NMF)
- Comprimento: média 22,8 tokens brutos/doc (mediana 19) — o desafio H2
- Filtros do STM: **0 docs removidos** por covariável NA; **0 docs** por `stm_min_tokens_per_doc: 5` (relaxado de propósito — texto curto é a característica do corpus; 200 como na Folha descartaria quase tudo)
- **Covariável de prevalência:** `category` = mês de coleta (`YYYY-MM`) — proxy temporal eleitoral, não rótulo de assunto. Fórmula: `~ category`. Distribuição: 2022-08 (2.621), 2022-10 (1.381), 2022-11 (2.070), 2022-12 (2.739) — não há setembro no corpus.
- No treino, o `prepDocuments` do R removeu **27 docs** (0,3%) que ficaram vazios após o corte de vocabulário → N final treinado = **8.784** (o notebook realinha df/θ automaticamente).

---

## 2. Lematização e Vocabulário

`lemmatize_corpus(model_key="stm")` — bloco `stm:` com `no_below=5`/`no_above=0.5`, idêntico a `lda:`/`nmf:`:

| Métrica | STM | LDA/NMF (referência) |
|---|---|---|
| Tempo de lematização | 29s | ~30s |
| Vocabulário final | **2.617** palavras (2.616 no R) | 2.617 (idêntico) |
| Tokens/doc após lematizar (média) | 10,9 | ~10,9 |
| Tokens únicos/doc (média) | 10,4 | ~10,4 |

O STM roda sobre **exatamente o mesmo BoW lematizado** do LDA/NMF (o texto enviado ao R é o join dos tokens do C_v) — a comparação de C_v entre os três é a mais limpa possível.

---

## 3. Seleção de K — Grid Search C_v

| K | C_v | Heldout lik. | Dispersão resíduos | Iterações |
|---|-----|--------------|--------------------|-----------|
| 3 | 0,400 | −6,671 | 14,22 | 150 ⚠ |
| 5 | 0,358 | −6,509 | 12,24 | 150 ⚠ |
| 7 | 0,477 | −6,433 | 11,16 | 150 ⚠ |
| 8 | 0,430 | −6,428 | 10,48 | 150 ⚠ |
| 10 | 0,533 | −6,305 | 9,39 | 150 ⚠ |
| 12 | 0,578 | −6,293 | 8,87 | 150 ⚠ |
| 15 | 0,581 | −6,209 | 8,53 | 150 ⚠ |
| **20** | **0,611** | −6,135 | 8,13 | **127 ✓** |
| 25 | 0,560 | −6,133 | 8,11 | 150 ⚠ |
| 30 | 0,590 | −6,125 | 8,14 | 150 ⚠ |

**BEST_K = 20** (peak C_v, com queda nítida em K=25 — pico genuíno, não borda). Leituras:

- Grid total: ~53min de fits R (bem mais lento que a Folha por doc — EM converge devagar em corpus esparso).
- ⚠ **9 dos 10 fits bateram no teto de 150 iterações da época** (grid rodado antes do aumento de `max_em_its`). K=20 foi o único que convergiu — o C_v relativo entre K segue informativo (todos sob o mesmo teto), mas os valores absolutos do grid subestimam levemente os de modelos convergidos (§5).
- Diferente do NMF (que colapsou para K=8 neste corpus), o STM sustentou K=20 — mesma granularidade escolhida pelo LDA, com C_v muito superior (0,611 vs 0,430 no pico do grid do LDA).

---

## 4. Grid sigma.prior × gamma.prior

8 combinações, K=20 fixo:

| sigma.prior | gamma.prior | C_v |
|-------------|-------------|-----|
| **0.0** | **L1** | **0,5725 ← melhor** |
| 0.7 | L1 | 0,5720 |
| 0.0 | Pooled | 0,5688 |
| 0.5 | L1 | 0,5676 |
| 0.5 | Pooled | 0,5609 |
| 0.3 | L1 | 0,5598 |
| 0.3 | Pooled | 0,5594 |
| 0.7 | Pooled | 0,5570 |

**Escolhido: `sigma.prior=0.0, gamma.prior=L1`.** Leituras:

1. **Inverte a Folha** (que preferiu sigma=0.7 e teve sigma=0 no fundo): com uma única covariável categórica de 4 níveis (mês), regularizar a covariância entre tópicos não ajuda. Mesmo padrão do kappa do NMF: **hiperparâmetro de prevalência não transfere entre corpora**.
2. `gamma.prior=L1` em 3 das 4 primeiras posições — prior esparso consistente com efeitos concentrados (cada tópico "pertence" a poucos meses, ver §7).
3. Spread 0,0155 — maior que o da Folha (0,0035), mas ainda refinamento marginal vs grid de K (spread ~0,25).

---

## 5. Treino Final — o episódio do teto de EM

O primeiro treino final (run `_161613`, 2026-07-16) parou em **exatamente 150 iterações = o teto `max_em_its` da época, sem convergir**. Correção no mesmo dia:

- `max_em_its: 150 → 300` no `params.yaml` (a Folha, que converge em 92, não é afetada)
- Retreino (run `_175301`): **convergiu em 222 iterações** (331s)
- Efeito da convergência: **C_v 0,572 → 0,588** (+0,016), exclusividade e diversidade estáveis

> Lição registrada: em corpus esparso o EM do STM precisa de ~2-4× mais iterações que em corpus denso. Verificar sempre `em_iterations < max_em_its` no `stm_metrics.csv` — igualdade significa teto batido, não convergência.

Run canônico: **`tweets_bre2022_20260716_175301`** (o `_161613` fica como registro do episódio).

---

## 6. Resultados da Rodada de Produção (`tweets_bre2022_20260716_175301`)

> K=20, sigma.prior=0.0, gamma.prior=L1, corpus `tweets_bre2022_20260629_215159`, 8.784 docs treinados, 222 iterações EM (convergido).

### Métricas

| Métrica | STM `175301` | LDA `214134` | NMF `002347` | BERTopic `101639` | Leitura |
|---|---|---|---|---|---|
| **C_v (recomputado)** | **0,588** | 0,527 | 0,559 | 0,570 ⚠ | **Melhor C_v dos tweets entre os 4** — e sobre o mesmo BoW do LDA/NMF; o do BERTopic é em base própria (indicativo) |
| Exclusividade (word-overlap) | 0,815 | 0,817 | 0,855 | **0,913** | empate com LDA; NMF e BERTopic acima |
| Topic Diversity (Dieng) | 0,93 | 0,71 | 0,8625 | 0,929 | empate técnico com o BERTopic na liderança |
| Diversity entropy (θ) | 0,993 | 0,994 | 0,913 | — | θ suave, bem distribuído |
| Outliers / cobertura | 0% / 99,7%* | 0% / 100% | 0% / 100% | 16,5% / 83,5% | *27 docs (0,3%) removidos por pré-processo do R, não por rejeição do modelo |
| Iterações EM | 222 (convergido) | — | — | — | teto 300 |
| Tempo de treino final | 331s | 27s | **3s** | — | STM paga o preço da ponte R + EM |

### 20 tópicos produzidos

| ID | Nome LLM | Keywords (top-5) | Docs | % |
|---|---|---|---|---|
| T0 | Ministro STF e manifestantes | mané, perdeu, perder, amolar, barroso | 611 | 7,0% |
| T1 | Bolsonarismo e ataques ao governo ⚠ | bolsonaro, acabar, trabalha, contar, preso | 35 | 0,4% |
| T2 | Bolsonarismo e manifestação política | eduardo, catar, frente, quartel, copa | 487 | 5,5% |
| T3 | Bolsonarismo e jornalismo no Brasil | forabolsonaro, falar, bonner, cara, renata | 321 | 3,7% |
| T4 | Crime organizado e quadrilhas no Brasil ⚠ | **de**, **tar**, ladrão, ele, ladraonojn | 179 | 2,0% |
| T5 | Saqueio e crise política no Brasil | brasil, jair, quebrou, deixar, rombo | 381 | 4,3% |
| T6 | Votação no Nordeste | votar, federal, polícia, nordeste, rodoviária | 667 | 7,6% |
| T7 | Política e Fake News no Brasil ⚠ | ficar, ver, ano, achar, querer | 57 | 0,6% |
| T8 | Eleições e posse presidencial | presidente, dia, diplomação, alvorada, posse | 519 | 5,9% |
| T9 | Política e cultura no governo Lula ⚠ | lula, hoje, hora, passar, democracia | 79 | 0,9% |
| T10 | Eleições e polarização política | lulapresidente, hoje, democracia, vencer, voto | 722 | 8,2% |
| T11 | Golpes e violência política ⚠ | pessoa, pedir, país, golpe, rua | 93 | 1,1% |
| T12 | Bolsonarismo e notícias falsas | globolixo, bolsonaronojn, presidente, entrevista, globo | 1.082 | 12,3% |
| T13 | Incêndios e depredações em Brasília | bolsonarista, terrorista, prender, cacique, fogo | 1.170 | 13,3% |
| T14 | Políticas eleitorais e Justiça | alexandre, moraes, xandão, milhão, cadeia | 513 | 5,8% |
| T15 | Orçamento secreto e política | corte, bobo, orçamento, secreto, chamar | 389 | 4,4% |
| T16 | Science e educação em crise ✏ | ter, governo, capes, dinheiro, pagar | 343 | 3,9% |
| T17 | Infiltrados e vandalismo em Brasília | brasília, infiltrar, patriota, sirene, manifestante | 447 | 5,1% |
| T18 | Campanha de Ciro Gomes | debatenaband, ciro, debate, governo, lulanaband | 680 | 7,7% |
| T19 | Crescimento e fé no Brasil ⚠ | pra, ser, gente, dizer, pro | 9 | 0,1% |

### Qualidade dos tópicos

**Limpos (~13 de 20)** e mapeando macro-eventos reconhecíveis de 2022: T0 provocação "mané"/Barroso (nov), T2 atos pró-Bolsonaro/quartéis + Copa do Catar, T3 embate com o Jornal Nacional (entrevista Bonner/Renata, ago), T5 narrativa econômica ("quebrou o Brasil"), T6 bloqueios da PRF no Nordeste no 2º turno (out), T8 diplomação/posse (dez), T10 celebração lulista pós-vitória, T12 hashtags anti-Globo/anti-Bolsonaro, T13+T17 atos de dezembro em Brasília (duas facetas: depredação e narrativa do "infiltrado"), T14 Moraes/TSE, T15 campanha "orçamento secreto", T18 debates/Ciro.

**Com ressalvas:**
- **T19 (9 docs, 0,1%)** — resíduo de quase-stopwords (`pra`, `ser`, `gente`): análogo funcional de tópico-lixo. Descartar da análise.
- **T4** — `de`/`tar` como top-keywords (mesmo ruído de texto informal do `tar` no LDA tweets); há núcleo real ("ladrão"/ladraonojn) mas contaminado.
- **T1, T7, T9, T11 (< 1,1% cada)** — tópicos minúsculos e genéricos; θ suave do STM dilui a atribuição dominante deles. Usar com cautela (ou agrupar) na análise.
- **T16** ✏ — nome do LLM ruim ("Science e educação em crise", com anglicismo): as keywords/docs são a campanha **#PagueMinhaBolsa / cortes CAPES-CNPq**. Renomear na análise para "Bolsas e financiamento da ciência (#PagueMinhaBolsa)".
- **T13 vs T17** — par temático próximo (atos de dezembro em Brasília); manter separados é defensável (depredação × narrativa de infiltração), mas citar com a distinção explícita.

---

## 7. Efeitos de Prevalência por Mês (estimateEffect)

`stm_prevalence_effects.csv`: 80 linhas = 20 tópicos × 4 termos (intercepto + 3 dummies de mês, baseline = **2022-08**). **47 dos 60 coeficientes não-intercepto são significativos a p<0,05** — quase todos os tópicos têm assinatura temporal.

Os maiores efeitos reconstroem a **cronologia eleitoral de 2022** com inferência estatística:

| Tópico | Mês | Coef. | Evento |
|---|---|---|---|
| T6 Votação no Nordeste | 2022-10 | **+0,308** | bloqueios da PRF no 2º turno |
| T10 Polarização/vitória | 2022-10 | **+0,259** | eleição e celebração do resultado |
| T13 Depredações em Brasília | 2022-12 | **+0,234** | atos de 12/dez (queima de carros/ônibus) |
| T0 "Mané"/Barroso | 2022-11 | **+0,186** | fala de Barroso ("perdeu, mané") em nov |
| T17 Infiltrados/vandalismo | 2022-12 | **+0,142** | narrativa pós-atos |
| T12 Anti-Globo/JN | 2022-10/11/12 | **−0,20 a −0,196** | concentrado na baseline ago (entrevista de Bolsonaro ao JN em 22/ago) |

Nos outros modelos, essa leitura existia só como cross-tab descritivo tópico×mês; no STM ela vem com coeficiente, EP e p-valor — e é a covariável de prevalência do próprio modelo (o cross-tab descritivo `stm_topic_category.png` continua no notebook como contraparte visual).

---

## 8. Comparação com LDA, NMF e BERTopic + H2

| Dimensão | STM (K=20) | LDA (K=20) | NMF (K=8) | BERTopic (K=24) |
|----------|-----------|-----------|----------|----------------|
| C_v | **0,588** | 0,527 | 0,559 | 0,570 ⚠ base própria |
| Exclusividade (word-overlap) | 0,815 | 0,817 | 0,855 | **0,913** |
| Topic Diversity | 0,93 | 0,71 | 0,8625 | 0,929 |
| Cobertura | 99,7% | 100% | 100% | 83,5% |
| Granularidade escolhida | 20 | 20 | 8 | 24 |
| Distribuição de docs | T13=13,3% máx; 5 tópicos <1,1% | equilibrada (3,0-8,3%) | concentrada (T0=33,7%) | intermediária |

**Leituras:**

- **Melhor C_v dos tweets entre os 4 modelos** — e na comparação limpa (mesmo BoW), a vantagem sobre o LDA é grande (+0,061) e sobre o NMF relevante (+0,029). Condicionar a prevalência no mês ajudou justamente onde a co-ocorrência BoW é fraca: o mês carrega parte da estrutura documental.
- **Sustentou K=20 onde o NMF colapsou para 8** — o STM manteve granularidade fina com coerência superior, sem o trade-off de cobertura do BERTopic.
- Custo: 5 tópicos minúsculos (<1,1%) e 1 residual (T19) — o θ suave do STM não "aposenta" fatores como a esparsidade do NMF faz.

### H2 — degradação formal → curto (atualizada com STM)

| Modelo | C_v Folha | C_v Tweets | Δ |
|---|---|---|---|
| LDA | 0,657 | 0,527 | −20% |
| NMF | 0,655 | 0,559 | −15% |
| **STM** | **0,680** | **0,588** | **−13,5%** |

**H2 confirmada também no STM**, com a menor degradação entre os modelos BoW — as covariáveis de prevalência amortecem parte da perda (o mês "explica" estrutura que as palavras raras não conseguem).

---

## 9. Configuração Final

```yaml
corpora:
  tweets_bre2022:
    stm_prevalence_formula: "~ category"   # mês YYYY-MM
    stm_min_tokens_per_doc: 5
    # STM pinado 2026-07-16 (run tweets_bre2022_20260716_161613; canônico _175301)
    stm_best_k: 20
    stm_sigma_prior: 0.0
    stm_gamma_prior: "L1"

stm:
  max_em_its: 300   # subido de 150 em 2026-07-16 — o treino dos tweets batia no teto
```

| Parâmetro | Valor | Justificativa |
|-----------|-------|---------------|
| `BEST_K` | **20 (pinado)** | peak C_v com queda nítida em 25 |
| `sigma.prior` | **0.0 (pinado)** | melhor C_v; inverte a Folha (1 covariável categórica) |
| `gamma.prior` | **L1 (pinado)** | melhor C_v; consistente com efeitos concentrados por mês |
| `stm_min_tokens_per_doc` | 5 | texto curto é a característica do corpus |
| `max_em_its` | 300 | tweets convergem em ~222 its (a 150 batia no teto) |

---

## 10. Pendências (status 2026-07-16)

| Item | Status |
|---|---|
| Grid K + grid sigma×gamma | ✅ completos, cacheados e **pinados** |
| Run de produção convergido | ✅ `tweets_bre2022_20260716_175301` (222 its < 300) |
| Grid K rodado sob teto de 150 its | ℹ documentado (§3): ranking relativo válido; se algum dia refizer o grid, será sob teto 300 |
| **T19 residual (9 docs, quase-stopwords)** | ⚠ descartar da análise temática |
| **T16 nome do LLM errado** | ⚠ renomear na análise → "Bolsas e financiamento da ciência (#PagueMinhaBolsa)" |
| T4 `de`/`tar` nas keywords | ⚠ ruído de texto informal (mesma classe do LDA tweets); estrutural |
| T1/T7/T9/T11 minúsculos (<1,1%) | ⚠ usar com cautela ou agrupar |
| T13×T17 par temático próximo | ℹ manter separados com distinção explícita (depredação × infiltração) |
| FREX (`compute_frex_score`) saturado | ⚠ aberto (compartilhado); FREX nativo do R disponível em `stm_topics_frex.csv` |
| Estabilidade Jaccard | ➖ não aplicável (Spectral determinístico) |

---

*Documentação gerada em 2026-07-16 a partir do run `tweets_bre2022_20260716_175301` (corpus `tweets_bre2022_20260629_215159`). Espelha a estrutura de `PIPELINE_STM_FOLHA.md`; método completo lá.*
