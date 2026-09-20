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
- Recorte: **inglês** (detecção na transcrição, não no metadado `language`): 1.109 vídeos.
  Canais: Dave2D (467), Marques Brownlee (319), Just Josh (242), MinistryTech (79), CNBC (1) e CNBC Television (1).
- 869 via Whisper (com pontuação) e 240 via legenda automática (sem pontuação — exige restauração antes de segmentar sentenças).
- Categoria de produto derivada do título (regex + override manual): laptop (629), phone (276),
  other (57), tablet (42), headphone (36), watch (20), camera (14), desktop (12), console (9), monitor (9) e vr (5).
- Covariáveis STM: `product_category`, `year`; `sentiment` (só no nível sentença).
  `channel` e `product_category` são colineares (cada canal cobre uma família) → só a categoria
  entra na fórmula; canal fica em análise descritiva.

## 4. Pipeline de sentenças e evidências

```
00-dataset/            JSON → CSV bruto de transcrições
01-preprocessing/      limpeza, filtro EN e corpus_limpo.csv (1.109 documentos)
03-topic-modeling/     STM no documento: K=12, aspectos por categoria de produto
02-sentences/          130.016 sentenças com contexto e proveniência;
                       `legenda` recebe restauração de pontuação, `whisper` é preservado
04-requirements/       join por post_id com o STM → tabela de revisão humana
```

`02-sentences` mantém `message_raw`, `message_punctuated`, `punctuation_restored`,
`is_fragment` e `is_dup_exact`. Nenhuma sentença é excluída automaticamente.

`04-requirements` não infere requisitos automaticamente: cria uma linha por evidência com
aspecto/tópico, sentença, contexto e campos vazios para `requirement_candidate`,
`review_decision` e `review_notes`. Evidências positivas, negativas ou neutras podem ser
avaliadas por humanos.

Ordem de execução atual: `00 → 01 → 03 (STM documento) → 02 → 04`.
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
                                     R devolve semantic_coherence, exclusivity_stm, heldout
                                     likelihood e dispersão de resíduos; Python adiciona C_v
                                     e diversity → stm_grid_k_diagnostics.csv
   │
   ▼
_selecao.py stm <corpus>             protocolo 2026-08-17: (1) admissibilidade — K dentro da
                                     faixa declarada FAIXA_K[corpus]; (2) fronteira de Pareto
                                     semantic_coherence × exclusivity_stm × diversity (Roberts
                                     et al.); (3) desempate declarado = menor K.
                                     C_v é REPORTADO ao lado, nunca decide. Heldout/resíduos
                                     sobem monotonicamente com K e sozinhos escolheriam K=30+.
   │                                 → pinar o K escolhido em params.yaml (stm_best_k)
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
| grid de K | 3,5,7,8,10,12,15,20,25,30 | 10,15,20,25,30,40 | 1 fit R por K |
| `FAIXA_K` (admissibilidade, `_selecao.py`) | (8, 25) | (10, 40) | declarada antes de olhar a grade; ajustar com justificativa substantiva, nunca post hoc |
| `no_below` / `no_above` | 5 / 0.5 | 20 / 0.5 | sentença: corpus grande, cortar raros |
| `stm_min_tokens_per_doc` | 200 | 5 | STM degrada com docs curtos; sentença já é curta por natureza |
| `max_em_its` | 300 | 300 | `em_iterations == max_em_its` no metrics = teto batido, não convergência |
| `prevalence` | `~ product_category + s(year)` | `~ sentiment * product_category` | `sentiment` factor com referência `neu`/`pos` |
| `sigma.prior` / `gamma.prior` | grid | grid | não transferem entre corpora — grid obrigatório |
| init / seed | Spectral / 42 | Spectral / 42 | determinístico: reprodutibilidade exata substitui multi-seed |

**Regras fixas do protocolo**
1. K é escolhido por `_selecao.py` (Pareto semantic_coherence × exclusivity_stm × diversity,
   faixa declarada, desempate = menor K); C_v e heldout/resíduos são reporte, não decisão.
2. Grid de hiperparâmetros (sigma × gamma, por C_v) só depois de fixar K; spread esperado
   ~0,005–0,015 em C_v.
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
00-dataset/            build_corpus.py, test_build_corpus.py
01-preprocessing/      configs/, notebooks/, test_corpus_limpo.py, data/{raw,output}
02-sentences/          build_sentences.py, notebooks/, testes, data/output/
03-topic-modeling/     STM de documento, configs/, notebooks/, scripts/run_stm.R
04-requirements/       build_review_table.py, notebook/, testes, data/output/
articles/              materiais de referência (não versionado)
docs/                  planos e especificações (não versionado)
```

`.gitignore` exclui `articles/`, `docs/`, `data/raw`, `data/output`, datasets `.json`, logs e `.venv`.

## 9. Proveniência

`01-preprocessing` e `03-topic-modeling` derivam do pipeline multi-corpus da dissertação
(`D:\Documentos\master`, corpora Folha/tweets). Este repositório mantém apenas o motor STM e
os corpora `youtube_*`; os achados metodológicos herdados (teto de EM, não-transferência de
`sigma.prior`, K como lever dominante) estão registrados em §5.
