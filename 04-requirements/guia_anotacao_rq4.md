# Guia de revisão de requisitos candidatos nos tópicos

## Unidade e decisão

Leia `sentence` e `context`. Marque `requirement_candidate` como `sim` quando a
frase avalia uma característica do produto que pode ser traduzida em requisito:
um defeito, limitação, necessidade ou desejo sugere uma mudança; um elogio a
uma capacidade ou qualidade concreta sugere algo a preservar. Uma comparação
favorável ou um resultado de teste apresentado como bom também pode servir de
evidência positiva. Sentimento positivo, negativo e neutro são elegíveis.
Marque `nao` quando a frase apenas enumera especificações ou medidas sem
avaliação, faz um elogio vago sem característica identificável, ou trata de
assunto sem relação com o produto. Use `incerto` quando não é possível
identificar a característica valorizada ou a direção da avaliação. O contexto
desambigua a frase; a evidência deve estar na frase central.

Exemplos sintéticos de `sim`: “I wish this laptop had an Ethernet port”
(adicionar porta); “The fan stays loud even when the computer is idle”
(reduzir ruído); “The colors are accurate and the screen stays bright outdoors”
(preservar fidelidade e brilho); “Games stay above 60 FPS even in busy scenes”
(preservar desempenho sob carga). Exemplos de `nao`: “This laptop has an
Ethernet port” (fato isolado); “I love it” (elogio sem característica);
“This video is sponsored” (fora do produto).

Preencha `aspect_ref` com um ou mais aspectos, separados por vírgula:
`bateria`, `termico`, `tela`, `portas`, `teclado`, `audio`, `camera`, `preco`,
`construcao`, `desempenho`, ou `outro`. Uma frase pode ter dois aspectos.
Escreva em `review_notes` a razão de casos ambíguos. Não consulte
`amostra_origem.csv` durante a anotação: ele contém método, tópico e pontuação.

## Uso e análise

`amostra_cega.csv` é a única planilha destinada ao anotador. Não preencha
automaticamente as três colunas de decisão. O `review_id` faz a ligação com o
arquivo de origem depois da revisão. Um segundo anotador deve julgar 50 itens
de modo independente para estimar concordância; discordâncias são discutidas
após os dois julgamentos originais ficarem salvos.

Os resultados de precisão@50 para `topico`, `lexical` e `topico_lexical` e a
taxa entre as 100 aleatórias só devem ser calculados quando todos esses itens
tiverem `sim`, `nao` ou `incerto`. O pool seleciona itens por várias regras e
não fornece, sozinho, recall no corpus inteiro.
O score de tópico (peso STM × peso NMF) é uma heurística de ordenação, não uma
probabilidade calibrada de requisito. A amostra aleatória exclui os top-50 das
outras listas; sua taxa é um controle desse restante do corpus. Os intervalos
de Wilson produzidos pelo script são exploratórios e não modelam a dependência
entre sentenças do mesmo vídeo.

Depois de salvar uma cópia preenchida da planilha, execute:

```powershell
& '.venv/Scripts/python.exe' '04-requirements/score_topic_requirement_review.py' --run '04-requirements/data/output/topicos_requisitos/20260928_v4' --annotations 'CAMINHO_DA_COPIA_PREENCHIDA.csv'
```

O comando grava `resultado_revisao.json` e
`confirmados_por_topico_na_amostra.csv` no diretório do run. A planilha
original e o arquivo de origem não são modificados.
