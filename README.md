# STM em transcrições de reviews do YouTube para apoiar a elicitação de requisitos

Pipeline de pesquisa: transcrições de vídeos de review de produtos (YouTube) → pré-processamento →
Structural Topic Model (STM, Roberts et al.) em dois níveis (documento e sentença), com sentimento
e categoria de produto como covariáveis → aspectos problemáticos → requisitos candidatos.

Desenho completo: `docs/specs/2026-09-19-youtube-stm-requirements-design.md`.
Literatura: `articles/artigos_mapeados.txt` (por relevância) e `docs/related-works.md` (por eixo).

---

## 1. Objetivo

Avaliar se o STM, aplicado em duas granularidades (documento e sentença) e condicionado à
polaridade de sentimento e à categoria de produto, sobre transcrições de reviews de produtos no
YouTube, recupera **(a)** a estrutura de aspectos de produto e **(b)** os problemas reportados
pelos revisores, de forma que apoie a elicitação de requisitos.

**Premissa** (herdada de Ferreira et al., SBTI 2026): topic modeling no nível documento recupera
*sobre o que se fala*; conteúdo transversal — aqui, a crítica negativa, que atravessa todos os
aspectos — só aparece no nível da sentença. Por isso o desenho é em dois níveis: o aspecto vem do
tópico; o requisito vem da sentença negativa dentro do aspecto.

## 2. Questões de pesquisa

| RQ | Pergunta | Nível | Evidência |
|----|----------|-------|-----------|
| RQ1 | Que estrutura de aspectos de produto o STM recupera nas transcrições? | documento | tópicos nomeados, C_v, efeito de `product_category` e tempo |
| RQ2 | Com sentimento e categoria como covariáveis, quais aspectos têm prevalência significativamente maior em sentenças negativas? | sentença | `estimateEffect`: coeficiente, EP, p-valor por tópico |
| RQ3 | As sentenças negativas dos tópicos com efeito significativo se traduzem em requisitos candidatos? | sentença | precisão por anotação manual; recall vs. baseline lexical |
| RQ4 | Os problemas de RQ2 são visíveis nas keywords do nível documento? (esperado: não) | ambos | comparação de keywords RQ1 × RQ2 |

## 3. Corpus

- Fonte: `transcricoes_youtube_metadados.json` — 1.563 vídeos, 1.496 com transcrição, metadados
  via yt-dlp (canal, data, views, likes, duração), coleta em 11/09/2026.
- Recorte: **inglês** (detecção na transcrição, não no metadado `language`) ≈ 1.100 vídeos.
  Canais: Dave2D (470), Marques Brownlee (387), Just Josh (244), MinistryTech (79).
- Mediana ~1.800 palavras por transcrição (max ~11k). 1.206 via Whisper (com pontuação),
  274 via legenda automática (sem pontuação — exige restauração antes de segmentar sentenças).
- Categoria de produto derivada do título (regex + override manual): laptop (~620), phone (~300),
  tablet, headphone, watch, tv, console, camera, other.
- Covariáveis STM: `product_category`, `year`; `sentiment` (só no nível sentença).
  `channel` e `product_category` são colineares (cada canal cobre uma família) → só a categoria
  entra na fórmula; canal fica em análise descritiva.

## 4. Metodologia prévia (pipeline)

```
00-dataset/            build_corpus.py — JSON → CSV (video_id, message, title, channel,
                       upload_date, year, views, likes, duration, source, product_category)
01-preprocessing/      corpus `youtube`: limpeza conservadora, filtro de idioma (langdetect),
                       filtro de comprimento, post_id = video_id → corpus_limpo.csv
02-sentences/          restauração de pontuação (só docs `legenda`) → segmentação spaCy →
                       sentimento por sentença (transformer EN) → corpus de sentenças com
                       covariáveis herdadas + amostra de 300 para anotação manual (kappa ≥ 0,70)
03-topic-modeling/     STM em dois corpora (mesmo protocolo, ver §5):
                         youtube_doc   prevalence ~ product_category + s(year)
                         youtube_sent  prevalence ~ sentiment * product_category
04-requirements/       findThoughts nas sentenças negativas por tópico → requisitos candidatos →
                       anotação (precisão) → recall vs. baseline lexical (problem/issue/wish/should…)
```

Ordem de execução: 00 → 01 → 02 → 03 (doc e sent) → 04. Cada módulo lê `data/output/` do
anterior via `resolve_latest_dir` (latest-wins) e grava em `data/output/<corpus>/<run_id>/`.

### 4.1 Pré-processamento (01)
Limpeza conservadora (HTML, URLs, espaços), sem remover pontuação (necessária para segmentação
no módulo 02); dedup exata; filtro de idioma na transcrição; `min_words = 100`; sem amostragem.

### 4.2 Sentenças e sentimento (02)
- Pontuação: `oliverguhr/fullstop-punctuation-multilang-large` nos 274 docs sem pontuação
  (se a qualidade for baixa, o nível sentença exclui `source = legenda` e reporta).
- Segmentação: spaCy `en_core_web_sm`; descarta sentenças com < 4 tokens.
- Sentimento: classificador transformer em inglês (decisão entre `siebert/sentiment-roberta-large-english`
  e `cardiffnlp/twitter-roberta-base-sentiment-latest` após teste em 50 sentenças).
  Validação: 300 sentenças, 2 anotadores cegos, kappa ≥ 0,70; F1 por classe reportado.
- Volume esperado: ~100k sentenças.

### 4.3 Elicitação (04)
Para cada tópico com efeito `sentiment = neg` significativo (p < 0,05): top-N sentenças negativas
por θ → redação de requisitos candidatos (manual ou LLM-assistida com validação humana) →
planilha de anotação → precisão. Baseline lexical de problema/desejo para medir recall e
complementaridade. RQ4: confronto com as keywords do nível documento.

## 5. Protocolo do STM (herdado e calibrado no projeto anterior)

O motor é o pacote R `stm` (v1.3.8), chamado por subprocess (`scripts/run_stm.R`, modos
`grid_k` / `grid_hparams` / `train`). O Python é dono do protocolo: prepara o input, dispara o R e
pontua os resultados.

```
corpus_limpo.csv
   │
   ▼
lemmatize_corpus(lang="en")          spaCy; remove stopwords, não-alfabéticos, tokens ≤ 2;
                                     gensim Dictionary.filter_extremes(no_below, no_above)
   │
   ▼
prepare_stm_input()                  texto = join dos lemas (mesmo BoW do C_v) + covariáveis
   │
   ▼
grid_search_k_stm()   [R grid_k]     1 STM por K, init Spectral, split heldout interno;
                                     Python pontua C_v no BoW gensim; R devolve heldout
                                     likelihood e dispersão de resíduos por K
   │                                 → BEST_K = pico de C_v (diagnósticos nativos só apoiam:
   │                                   sobem monotonicamente com K e sozinhos escolheriam K=30+)
   ▼
grid_search_stm_hparams() [R grid_hparams]
                                     K fixo; sigma.prior {0, 0.3, 0.5, 0.7} × gamma.prior {Pooled, L1}
                                     → vencedor por C_v (spread esperado pequeno; K é o lever dominante)
   │
   ▼
train_stm()           [R train]      treino final + estimateEffect no mesmo processo;
                                     devolve θ (doc×K), β (K×V), keywords (prob e FREX), efeitos
   │
   ▼
name_all_topics()                    nomeação via LLM (50 keywords + 3 docs representativos);
                                     não determinística → citar sempre o run canônico
   │
   ▼
métricas + export                    C_v, exclusividade c-TF-IDF, topic diversity, θ-entropy;
                                     stm_metrics.csv, stm_theta.csv, stm_beta.csv, stm_effects.csv
```

**Parâmetros por corpus (ponto de partida; calibrar e pinar no `params.yaml`):**

| Parâmetro | youtube_doc | youtube_sent | Nota |
|-----------|-------------|--------------|------|
| `k_range` | 5–30 | 10–40 | grid com passo variável (3,5,7,8,10,12,15,20,25,30,…) |
| `no_below` / `no_above` | 5 / 0.5 | 20 / 0.5 | sentença: corpus grande, cortar raros |
| `stm_min_tokens_per_doc` | 200 | 5 | STM degrada com docs curtos; sentença já é curta por natureza |
| `max_em_its` | 300 | 300 | `em_iterations == max_em_its` no metrics = teto batido, não convergência |
| `prevalence` | `~ product_category + s(year)` | `~ sentiment * product_category` | `sentiment` factor com referência `neu`/`pos` |
| `sigma.prior` / `gamma.prior` | grid | grid | não transferem entre corpora — grid obrigatório |
| init / seed | Spectral / 42 | Spectral / 42 | determinístico: reprodutibilidade exata substitui multi-seed |

**Regras fixas do protocolo**
1. K é escolhido por C_v (comparável entre modelos e corpora); heldout/resíduos são apoio.
2. Grid de hiperparâmetros só depois de fixar K; spread esperado ~0,005–0,015 em C_v.
3. Vocabulário do STM = BoW lematizado do C_v, por construção (o R não re-tokeniza).
4. Covariável com NA é erro: `dropna` antes de lematizar para manter alinhamento df ↔ tokens.
5. Após calibrar, pinar `stm_best_k`, `stm_sigma_prior`, `stm_gamma_prior` no `params.yaml`
   (grids desligam; remover as chaves reativa com cache).
6. Nomes de tópicos do LLM não são determinísticos; métricas e θ/β são. Citar o run canônico.

## 6. Critérios de sucesso

1. RQ1: K selecionado por C_v com ≥ 80% dos tópicos nomeáveis sem ressalva.
2. RQ2: subconjunto de tópicos com efeito `sentiment = neg` significativo e interpretável como aspecto problemático.
3. RQ3: precisão ≥ 0,7 dos requisitos candidatos; recall superior ou complementar ao baseline lexical.
4. RQ4: aspectos problemáticos de RQ2 não aparecem como tópico próprio em RQ1.

## 7. Ambiente

- Python 3.12 — `requirements.txt` (pandas, numpy, pyyaml, spacy + `en_core_web_sm`, langdetect,
  gensim, transformers, torch, deepmultilingualpunctuation, matplotlib, seaborn, wordcloud, plotly,
  jupyter, pytest).
- R ≥ 4.4 com `stm`, `jsonlite`, `glmnet`; `Rscript` no PATH.
- LLM para nomeação de tópicos: OpenAI (`gpt-4.1-mini`) com fallback Ollama Cloud (`params.yaml > advisor`).

## 8. Estrutura do repositório

```
00-dataset/            build_corpus.py, product_category_overrides.csv
01-preprocessing/      configs/params.yaml, notebooks/01_preprocessing.ipynb, data/{raw,output}
02-sentences/          configs/, notebooks/, data/{output}
03-topic-modeling/     configs/params.yaml, configs/advisor_prompts/{shared.yaml,stm/},
                       notebooks/{_helpers.py,_selecao.py,_advisor.py,stm/,advisor/}, scripts/run_stm.R
04-requirements/       (fase 2)
articles/              PDFs + artigos_mapeados.txt          (não versionado)
docs/                  specs/, related-works.md             (não versionado)
```

`.gitignore` exclui `articles/`, `docs/`, `data/raw`, `data/output`, datasets `.json`, logs e `.venv`.

## 9. Proveniência

`01-preprocessing` e `03-topic-modeling` derivam do pipeline multi-corpus da dissertação
(`D:\Documentos\master`, corpora Folha/tweets). Este repositório mantém apenas o motor STM e
os corpora `youtube_*`; os achados metodológicos herdados (teto de EM, não-transferência de
`sigma.prior`, K como lever dominante) estão registrados em §5.
