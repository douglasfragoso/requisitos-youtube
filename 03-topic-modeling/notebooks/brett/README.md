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
