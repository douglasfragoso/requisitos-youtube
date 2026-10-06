# Tópicos em reviews do YouTube para elicitação de requisitos

Este projeto analisa transcrições de reviews de produtos no YouTube (1.109 vídeos em inglês após o filtro de idioma). Usa **NMF global → NMF local** nas sentenças para organizar evidências de requisitos que serão julgadas por pessoas. Nenhuma frase é rotulada como requisito automaticamente.

O documento de pesquisa é [docs/idealizacao-artigo.txt](docs/idealizacao-artigo.txt).

## Pipeline, do dado bruto à revisão humana

| Etapa | Onde | O que faz |
|---|---|---|
| 0 | `00-dataset/` | JSON de transcrições + metadados → CSV bruto e categoria de produto |
| 1 | `01-preprocessing/notebooks/01_preprocessing.ipynb` | filtro de idioma e limpeza → `corpus_limpo.csv` |
| 2 | `02-sentences/build_sentences.py` | segmentação em frases, com uma vizinha de cada lado como contexto |
| 3a | `03-topic-modeling/notebooks/nmf/01_nmf_global.ipynb` | NMF global (K=20) na frase central; escolha dos temas de aspecto |
| 3b | `03-topic-modeling/notebooks/nmf/02_nmf_topicos.ipynb` | NMF local nos temas escolhidos; monta a amostra cega |
| 4 | `04-requirements/` | rankings, amostra cega, guia de anotação e pontuação dos rótulos humanos |

Os dois notebooks de NMF chamam os scripts `03-topic-modeling/scripts/run_global_nmf_sentences.py` e `run_restricted_nmf_gensim.py` (com `nmf_utils.py`), que também rodam pela linha de comando. Os notebooks versionados trazem as saídas do run pinado.

## Como rodar do zero

```powershell
python -m venv .venv ; .venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python -m spacy download en_core_web_sm

.venv\Scripts\python 00-dataset/build_corpus.py --input 00-dataset/transcricoes_youtube_metadados.json --output 01-preprocessing/data/raw/youtube/youtube_reviews.csv
# 01-preprocessing/notebooks/01_preprocessing.ipynb  (gera corpus_limpo.csv)
.venv\Scripts\python 02-sentences/build_sentences.py
# 03-topic-modeling/notebooks/nmf/01_nmf_global.ipynb   (RUN_EXISTENTE = None treina um run novo)
# 03-topic-modeling/notebooks/nmf/02_nmf_topicos.ipynb  -> amostra cega em 04-requirements/data/output/
```

Com a seed fixa (42), o pipeline é reprodutível: o rerun completo gerou os mesmos temas, K locais e contagens do artigo (116.111 frases filtradas, 10 temas, 57.952 frases, 1.620 marcas lexicais, 212 frases cegas). O run global e os temas escolhidos estão fixados em [params.yaml](03-topic-modeling/configs/params.yaml), seção `final_sentence_pipeline`. Se o corpus ou o vocabulário mudarem, os IDs de tema mudam: refaça a escolha no notebook 01 e atualize `global_run`, `selected_topics` e `expected_sentences`.

Para pontuar uma **cópia preenchida por anotadores humanos** de `amostra_cega.csv`:

```powershell
.venv\Scripts\python 04-requirements/score_topic_requirement_review.py --run CAMINHO_DA_PASTA_DA_REVISAO --annotations CAMINHO_DA_COPIA_PREENCHIDA.csv
```

O [guia de anotação](04-requirements/guia_anotacao_rq4.md) aceita pedidos de mudança e elogios a capacidades concretas a preservar. O léxico de pedido/queixa e o peso temático só ordenam a leitura; **não confirmam requisito**.

## Testes

```powershell
.venv\Scripts\python -m pytest 00-dataset 01-preprocessing 02-sentences 03-topic-modeling 04-requirements
```

Os testes não ajustam o NMF real; os de `nmf_utils` usam corpora minúsculos.

## Dados

O dataset de transcrições está em `00-dataset/transcricoes_youtube_metadados.json` e é versionado. `.gitignore` exclui `docs/` (exceto `docs/idealizacao-artigo.txt`, já rastreado), `articles/` e todos os diretórios `data/raw` e `data/output`; runs e amostras são locais e devem ser arquivados separadamente.
