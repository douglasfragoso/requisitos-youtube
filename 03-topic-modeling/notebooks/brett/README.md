# BRETT na sentença: resultado do gate C0

O pacote R `NMFregress` 1.0.1 foi instalado localmente a partir do commit
`597c2e7a513665372736789398e165a0eb41dba3`. O script
[`check_brett_c0.py`](../../scripts/check_brett_c0.py) usa o run STM de sentença
`youtube_sent_20260921_190124`, executa dois ajustes piloto de 400 sentenças
com o pacote e mede a cobertura das âncoras no corpus completo.

| Recorte | T | Limite mínimo exato de sentenças sem âncora | Piloto no corpus completo |
| --- | ---: | ---: | ---: |
| Rodada A, 114.888 sentenças | 15 | 26,6% | 52,7% |
| Rodada B, tema 6, 4.291 sentenças | 6 | 31,9% | 44,8% |

O limite exato usa a soma das T maiores frequências documentais. Para qualquer
escolha de T palavras-âncora, a união dos documentos que as contêm não pode
ser maior que essa soma. Portanto, os dois limites ultrapassam o teto de 15%
do protocolo **sem depender das âncoras escolhidas pelo piloto**. Os ajustes
piloto validam a instalação e a integração, mas suas porcentagens são apenas
exploratórias. Na Rodada B, o mesmo limite já excede 15% em quatro dos oito
temas quando T=6 (temas 1, 6, 8 e 10).

O gate C0 falhou para as configurações propostas. Por isso a regressão e o
bootstrap do BRETT não foram executados no corpus completo. Um novo protocolo
precisaria definir previamente mais âncoras, mais contexto por sentença ou
outra unidade de texto antes de repetir o teste. Os resultados completos e os
modelos piloto ficam em `data/output/youtube_sent/brett/c0_20260927/`.

## Segunda rodada: janela de três sentenças

O protocolo `2026-09-27-v2-three-sentence-window` preserva cada uma das 114.888
sentenças centrais e acrescenta até uma vizinha anterior e uma posterior do
mesmo vídeo. A janela é construída antes de excluir as sentenças removidas
pelo STM, de modo que a adjacência original seja respeitada. O script
[`check_brett_context_v2.py`](../../scripts/check_brett_context_v2.py) executou
o `NMFregress` oficial com T=20 em três amostras de 1.000 janelas.

| Seed | Janelas sem âncora no corpus completo |
| ---: | ---: |
| 42 | 11,02% |
| 7 | 12,26% |
| 2026 | 11,63% |

O limite numérico de C0 (15%) foi alcançado nas três amostras. A inspeção das
âncoras e dos oito termos principais de cada tópico ainda encontra vários
tópicos de discurso genérico, com âncoras como `get`, `think`, `well` e `good`.
Portanto, a cobertura isolada não valida a interpretação de aspectos nem a
substituição do STM. A ausência de âncora também varia por categoria de
produto; as planilhas `coverage_by_category.csv` registram essa diferença.
Não houve regressão por categoria: a categoria é atribuída por vídeo, e o
bootstrap por sentença do pacote não incorpora esse agrupamento.

Artefatos: `data/output/youtube_sent/brett/c0_context_v2_20260927/`, incluindo
`c0_summary.json`, `topic_preview.csv` e `coverage_by_category.csv` em cada
seed. Uma ablação exploratória adicional excluiu 25 termos genéricos de um
piloto anterior à correção de adjacência: com T=20, 30 e 35, deixou 21,41%,
15,87% e 12,98% das janelas sem âncora. Essa ablação de uma seed não é o gate
pré-fixado e precisa ser repetida com a adjacência corrigida e outras seeds.
