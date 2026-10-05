# Tópicos em reviews do YouTube para elicitação de requisitos

Este projeto analisa 1.109 transcrições de reviews de produtos. O modelo final usa **STM nos documentos** para descrever a estrutura temática por categoria e data. Nas sentenças, usa **NMF global → NMF local** para organizar evidências de requisitos que serão julgadas por pessoas. O STM de sentenças → NMF é preservado como baseline histórico.

O documento de pesquisa atualizado é [docs/idealizacao-artigo.txt](docs/idealizacao-artigo.txt). A comparação dos pipelines está em [docs/analises/comparacao_stm_nmf_20261003/relatorio.md](docs/analises/comparacao_stm_nmf_20261003/relatorio.md).

## Fluxo final de sentenças

1. `02-sentences/` segmenta as transcrições e guarda a frase central, uma vizinha de cada lado, categoria, vídeo e `sent_id`.
2. `03-topic-modeling/scripts/run_global_nmf_sentences.py` ajusta NMF sobre TF-IDF da **frase central**. O run final pinado é `nmf_global_sentence_k20_20261003_134035`: 116.111 frases filtradas e K=20.
3. `03-topic-modeling/scripts/run_restricted_nmf_gensim.py` ajusta NMF local nos dez temas globais `[1,3,5,7,9,10,11,14,15,17]`. Foram selecionadas 57.952 frases; 711 sem BoW local continuam no resultado com subtópico `-1`.
4. `04-requirements/run_topic_requirement_review.py` une os dois níveis por `sent_id`, aplica o léxico fixo de pedidos/queixas e gera rankings, evidências e uma amostra cega de 212 frases. O contexto é usado na **leitura humana**, não no treino do NMF final.

O run, os temas, as contagens esperadas e os parâmetros de revisão estão em [params.yaml](03-topic-modeling/configs/params.yaml), seção `final_sentence_pipeline`. Os artefatos de revisão estão em `04-requirements/data/output/topicos_requisitos/nmf_global_k20_20261003_final/`.

## Comandos principais

Com os runs de NMF já presentes no workspace:

```powershell
python 04-requirements/run_topic_requirement_review.py --help
```

O comando sem argumentos usa a configuração congelada. O diretório final já foi gerado; para refazer a exportação, escolha outro caminho com `--output`, pois o comando recusa sobrescrever um run existente. Para pontuar uma **cópia preenchida por anotadores humanos** de `amostra_cega.csv`:

```powershell
python 04-requirements/score_topic_requirement_review.py --run 04-requirements/data/output/topicos_requisitos/nmf_global_k20_20261003_final --annotations CAMINHO_DA_COPIA_PREENCHIDA.csv
```

O [guia de anotação](04-requirements/guia_anotacao_rq4.md) aceita pedidos de mudança e elogios a capacidades concretas a preservar. O léxico é apenas uma forma de ordenar a leitura; uma marca lexical ou peso temático **não confirma requisito**. A amostra original permanece sem rótulos humanos.

## Comparadores e limites

- STM de documento: análise paralela de RQ1; sua saída não é usada no ranking de requisitos.
- STM de sentenças → NMF: baseline histórico, reproduzível por `04-requirements/run_stm_baseline_requirement_review.py` com caminhos explícitos. Seus 56.669 registros e 205 frases cegas não são a amostra final.
- Guided NMF e BRETT: pilotos separados, sem participação no pipeline final.
- `run_restricted_nmf_sentences.py`: alternativa exploratória com TF-IDF local; o run final usa `run_restricted_nmf_gensim.py` com BoW lematizado.
- A escolha do NMF final foi metodológica. Em corpus comum, o NMF global teve NPMI médio maior que o STM de sentenças (0,0485 × 0,0310), porém menor diversidade de termos (0,635 × 0,939). Nos grupos locais, a média da grade NMF foi menor após NMF global do que após STM (0,0355 × 0,0684). Essas métricas não medem precisão de elicitação; a validação humana continua pendente.

## Ambiente e dados

Python 3.12, dependências de `requirements.txt` e modelo spaCy `en_core_web_sm` são usados no NMF local. R com o pacote `stm` é necessário para refazer o modelo de documentos ou o baseline histórico. O projeto também contém etapas de coleta (`00-dataset/`) e pré-processamento (`01-preprocessing/`).

`.gitignore` exclui `docs/`, `articles/` e diretórios `data/output`, mas `docs/idealizacao-artigo.txt` já era rastreado pelo Git e suas alterações aparecem normalmente. Os runs e o relatório de comparação são locais e devem ser arquivados separadamente para reprodução fora deste workspace.
