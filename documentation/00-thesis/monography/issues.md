# Revisão da tese — problemas encontrados e soluções propostas

Revisão feita em 2026-11-12 sobre o texto em `chapters/` (commit `6012591d`), confrontado com
as tabelas regeneradas (`tables/phase00_*.csv`, `tables/phase01_auth.csv`) e, onde o texto
descreve o experimento, com o código em `software/nn/src/experiments/thesis/`.

**Como ler.** Cada item traz onde está, qual é o problema e o que proponho. A gravidade segue
esta escala:

| Gravidade | Significado |
|---|---|
| **A** | Uma banca pode reprovar ou exigir reexecução: a afirmação é falsa, ou o desenho não sustenta a conclusão. |
| **B** | Enfraquece a tese, mas se resolve com texto, uma análise extra ou uma ressalva. |
| **C** | Acabamento: consistência, redação, compilação. |

Quando escrevo "verificado", conferi no código ou nos dados. Quando escrevo "pela leitura",
é inferência a partir do texto e não foi medido.

## Resumo

| # | Gravidade | Problema | Esforço |
|---|---|---|---|
| 1 | A | Título e motivação falam de disfonia severa; nenhum sinal disfônico é testado | texto, ou experimento novo |
| 2 | A | A prosa de resultados ainda descreve a execução inválida de julho | reescrita |
| 3 | A | A "contribuição 2" (ordem `direct ≻ latency ≻ poisson`) não se repete na reexecução | reescrita + decisão |
| 4 | A | Questão 4: os modos de texto diferem também no tamanho do treino (80% contra ~45%) | reexecução barata |
| 5 | A | O extrator é escolhido na Fase 00 com todos os locutores, inclusive os de teste da Fase 01 | reexecução barata ou ressalva |
| 6 | A | Não há linha de base: nem classificador simples, nem comparação com a literatura | experimento pequeno |
| 7 | B | A vantagem da fusão sobre o EEG não está estabelecida; p-valores com pseudorreplicação | análise |
| 8 | B | Fase 00 sem incerteza: o vencedor ganha por 0,014 e σ = 0 | análise (bootstrap) |
| 9 | B | "Dependente de texto" não é a definição usual, e o texto define de dois jeitos | texto ou protocolo |
| 10 | B | Padronização decide tudo no EEG (0,43 → 0,25) e o texto não explica por quê | análise + texto |
| 11 | B | Eficiência energética motiva as redes de pulso e nunca é medida | medir ou retirar |
| 12 | B | Capítulo 7 contém histórico de depuração desatualizado e contraditório com o capítulo 8 | cortar |
| 13 | B | Revisão do estado da arte termina em 03/2024 e conclui com um protocolo de coleta que saiu do escopo | atualizar |
| 14 | B | Introdução curta demais: sem hipótese, justificativa nem contribuições | reescrita |
| 15 | B | Fusão precoce reamostra o EEG em 43×; a DTWPT de nível 4 dá bandas de 1,4 kHz na voz | ressalva ou ablação |
| 16 | B | Questão 3 continua sem resposta na forma em que foi formulada | experimento médio |
| 17 | C | Rótulo "LFCC (c1)" chama de cepstral o que não é | renomear |
| 18 | C | Contagens e números desatualizados espalhados (208, 300, 58, 1,4835…) | busca e troca |
| 19 | C | Jargão interno no texto ("Experimento 05", nomes de arquivos, "Wiki") | redação |
| 20 | C | Compilação: referência indefinida, entrada duplicada no `.bib`, figura pendente | correções pontuais |
| 21 | C | Terminologia oscilante (RNP / SNN / DSNN; EQM / MSE) | padronizar |

A ordem que recomendo para atacar: 4, 5 e 6 primeiro, porque mudam números; depois 2 e 3,
que reescrevem os resultados já com os números finais; depois 1 e 14, que reenquadram a tese.

---

## Gravidade A

### 1. A tese promete disfonia e não testa disfonia

**Onde.** Título ("Locutores Drasticamente Disfônicos"), resumo, `06-Introduction.tex`
(objetivos), `08-proposedApproach.tex` §base de sinais.

**Problema.** A base usada (`10.1117/12.2255697`) tem 15 voluntários saudáveis. Nenhum sinal
de voz é disfônico, real ou simulado. A pergunta do título — o EEG compensa a voz degradada?
— nunca é posta à prova. O que o experimento mede é outra coisa: se voz e EEG de fala
imaginada, em locutores saudáveis, autenticam sozinhos e em conjunto.

**Por que é grave.** É a primeira pergunta de qualquer banca, e o texto hoje não a antecipa.
A falha é silenciosa: o leitor só percebe ao chegar na descrição da base.

**Soluções, da mais forte para a mais barata.**
1. *Simular a degradação.* Degradar o áudio de teste (ruído, filtragem passa-baixa severa,
   remoção de vozeamento, ou uma voz sussurrada sintetizada) em níveis crescentes e traçar
   EER × nível de degradação para voz, EEG e fusão tardia. Se o EER da fusão cair menos que
   o da voz, a tese do título fica demonstrada em condição controlada. A Fase 01 inteira
   roda em 4 minutos no GridUnesp, então o custo é sobretudo de implementação.
2. *Reenquadrar.* Mudar título e resumo para o que foi medido (por exemplo, "Autenticação
   biométrica por fala fonada e fala imaginada…") e mover a disfonia para motivação e
   trabalhos futuros.
3. *No mínimo*, declarar a limitação explicitamente em Introdução e Limitações.

O resultado atual ajuda a opção 1: o EEG sozinho (EER 0,236) já supera a voz (0,294), o
que é o pré-requisito para o argumento "o EEG sustenta a autenticação quando a voz falha".

### 2. A prosa de resultados ainda é a da execução inválida

**Onde.** `04-abstract.tex` (resumo e abstract), `09-testsAndResults.tex` (parágrafo após a
nota da Fase 01, Questões 3 e 4), `10-conclusions.tex` (Síntese, Limitações, Trabalhos
futuros).

**Problema.** As notas `[PENDENTE]` dizem o que mudou, mas o texto corrido continua afirmando
EER 0,4534, "resultado negativo", "padronização piora o EER", "sem diferença entre os modos
de texto". Hoje a tese se contradiz parágrafo a parágrafo.

**Solução.** Reescrever, nesta ordem, usando os números das notas:
- Fase 01: melhor configuração EER 0,1823 / AUC 0,8936; padrão (i) mantido; (ii) removido;
  (iii) invertido.
- Questão 3: incluir a PCA; trocar a frase de β.
- Questão 4: ver item 4 antes de reescrever.
- Conclusões: retirar "resultado negativo"; trocar por "sinal claro, ainda insuficiente para
  uso prático".
- Resumo e abstract por último.

Faça isso **depois** dos itens 4, 5 e 6, para não reescrever duas vezes.

### 3. A contribuição do "piso de ruído" não se sustenta nos dados novos

**Onde.** `08-proposedApproach.tex`, parágrafo "Comparação entre as codificações temporais"
e os quatro seguintes; `10-conclusions.tex`, Contribuição 2 e Trabalho futuro 2.

**Problema (verificado nas tabelas).** O texto afirma que a reexecução "confirma a ordem
`direct ≻ latency ≻ poisson` em ambos os sinais", com a `poisson` nas três últimas posições
e α ≤ 0,003. Na reexecução de 2026-11-12:

| Codificação | Posições no EEG | Posições na voz | α (EEG) |
|---|---|---|---|
| `direct` | 5, 10, 12 | 8, 10, 12 | 0,09–0,21 |
| `latency` | 3, 4, 11 | 1, 5, 6 | 0,14–0,21 |
| `poisson` | **1**, 7, 8 | 7, 9, 11 | 0,16–0,22 |

A `poisson` é a melhor do EEG, a `direct` é a pior nos dois sinais, e o α da `poisson` não
é nulo. A ordem observada é quase o inverso da prevista. A execução de julho, que
"confirmava" a previsão, tinha `base` e `small` como a mesma rede.

**Por que é grave.** A conclusão lista isso como uma das três contribuições da tese.

**Solução.**
- Reescrever o trecho como o que ele passou a ser: uma previsão teórica (o cálculo de
  σ = 0,125 continua correto) que **os dados não confirmaram**. Isso é informativo: o ruído
  de entrada não é o fator dominante neste regime.
- Notar que as diferenças entre codificações são pequenas (médias de d_pen entre 1,873 e
  1,910) e da ordem do σ_d; a leitura segura é "sem ordem clara".
- Retirar a Contribuição 2 da lista, ou rebaixá-la a "análise do piso de ruído e seu teste".
- O teste em T = 64 perde a motivação original. Mantê-lo só se reformulado.

### 4. Questão 4: o modo de texto está confundido com o tamanho do treino

**Onde.** `08-proposedApproach.tex` (eixo "Modo de texto"), Questão 4 no capítulo 9.
Código: `make_text_split` em `ThesisDataset.cpp`.

**Problema (verificado no código).** Os dois modos não diferem só no conteúdo falado:

| | Treino | Teste |
|---|---|---|
| Dependente | 80% das amostras, sorteadas | os 20% restantes |
| Independente | amostras de 5 das 11 frases (~45%) | amostras das outras 6 frases (~55%) |

O modo dependente treina com quase o dobro de dados. A vantagem dele em todas as 10
combinações acima do acaso (0,004 a 0,066 de EER) pode vir disso, e não da dependência de
texto. O teste também tem tamanhos diferentes, o que muda a variância do EER.

**Falha silenciosa.** Nada no texto nem nas tabelas revela a diferença de tamanho.

**Solução.** Igualar os tamanhos e reexecutar (4 minutos de cluster):
- no modo dependente, subamostrar o treino para o mesmo número de amostras do independente,
  e o teste idem; ou
- no dependente, treinar e testar nas mesmas 5 frases (divididas por repetição), e no
  independente treinar em 5 e testar nas outras 6, com o mesmo número de amostras.

Só então responder à Questão 4. Se a reexecução não for feita, a resposta honesta é "o modo
dependente foi melhor, mas a comparação está confundida com o tamanho do treino".

### 5. O extrator é escolhido olhando os locutores de teste

**Onde.** Protocolo em duas fases (`08-proposedApproach.tex`); código:
`ThesisParaconsistent.cpp` (as classes são os 15 locutores, todas as 1974 amostras).

**Problema (verificado).** A Fase 00 calcula α e β com os rótulos de **todos** os locutores e
escolhe o extrator. A Fase 01 depois avalia esse extrator em validação cruzada sobre os
mesmos locutores. A escolha do extrator viu os locutores de teste: é um viés de seleção. Os
autoencoders e a PCA da Fase 00 também são ajustados sobre todas as amostras.

**Tamanho provável do efeito.** Pequeno: a escolha é entre extratores fixos, por um critério
que não treina classificador, e o vencedor é o mesmo nos dois sinais. Mas "provavelmente
pequeno" é raciocínio, não medição, e a tese afirma que o protocolo em duas fases "evita que
a seleção seja contaminada".

**Solução.**
- *Rigorosa:* refazer o ranking da Fase 00 dentro de cada dobra, só com os 12 locutores de
  treino, e verificar se o vencedor muda. As 184 configurações manuais são determinísticas
  e baratas. Se Haar + LFCC (c1) vencer em todas as dobras, basta uma frase e uma tabela.
- *Mínima:* declarar o viés em Limitações e não afirmar que a seleção é livre de
  contaminação.

### 6. Não há linha de base

**Onde.** Capítulo 9 inteiro.

**Problema.** O EER 0,1823 não tem com o que ser comparado:
- nenhum classificador simples sobre o mesmo vetor (cosseno direto sobre as características
  padronizadas, LDA, regressão logística, uma MLP não pulsante de mesmo tamanho);
- nenhuma comparação com trabalhos sobre a mesma base ou sobre autenticação por EEG, embora
  o capítulo 8 diga que as métricas foram escolhidas "garantindo a comparação com outros
  estudos".

Sem isso não se sabe quanto do resultado vem da DSNN. O ganho enorme da padronização
(item 10) sugere que boa parte vem das características.

**Solução.**
1. Acrescentar à Fase 01 duas linhas de base sobre o mesmo vetor e as mesmas dobras:
   cosseno sobre as características padronizadas, sem rede; e uma MLP 128-128-128 com ReLU.
   É a "ANN de controle" que o estudo preliminar citado na própria tese já usa.
2. Montar uma tabela com os EER publicados para autenticação por EEG e por fala imaginada,
   com número de sujeitos e protocolo, deixando claro o que é e o que não é comparável.

---

## Gravidade B

### 7. A vantagem da fusão não está estabelecida

**Problema.** A melhor configuração (fusão tardia, 0,1823) não difere significativamente de
9 das outras 11 configurações padronizadas de EEG ou fusão, incluindo o EEG sozinho
(0,2360). Além disso, os p-valores tratam as 15 dobras (3 repetições × 5) como
independentes, e elas reutilizam os mesmos dados. A referência do teste é a melhor
configuração escolhida nos mesmos dados.

**Solução.**
- Para a pergunta central (fusão × EEG × voz), usar um teste pareado: as três fontes têm as
  mesmas dobras quando modo de texto, esquema de validação e semente coincidem. O script de
  significância já aceita `--pairing paired` e `--reference <run_tag>`; fixar a referência
  de antemão (EEG sozinho, por exemplo).
- Contra a pseudorreplicação: fazer a média das 3 repetições por dobra e testar sobre 5
  valores, ou usar o teste t corrigido de Nadeau–Bengio. Declarar o que foi feito.
- Reportar intervalos de confiança do EER, e não só σ entre 3 repetições.
- Redigir a conclusão como "a fusão tardia teve o menor EER, sem diferença significativa
  para o EEG isolado".

### 8. A Fase 00 escolhe o vencedor sem medir incerteza

**Problema.** Na voz, Haar + LFCC (c1) vence com 1,4836, contra 1,4973 do segundo lugar: uma
diferença de 0,014. Os extratores manuais são determinísticos, então as três "repetições"
dão σ_d = 0,0000 (verificado) e a coluna σ_d passa a impressão de estabilidade perfeita. A
mesma voz Haar dá 1,48363 no cluster e 1,48355 localmente.

Há um segundo ponto: α é definido por mínimos e máximos intraclasse, portanto é dominado
por amostras extremas. O próprio texto reconhece isso ao discutir as codificações.

**Solução.**
- Estimar a incerteza do ranking por *bootstrap* sobre amostras (ou deixando um locutor de
  fora por vez) e reportar com que frequência cada configuração vence.
- Na tabela, trocar o σ_d das linhas manuais por "—" ou por essa incerteza, e explicar no
  texto que três repetições de um extrator determinístico são a mesma medida.
- Dizer com clareza que os primeiros colocados estão empatados na prática, se o bootstrap
  mostrar isso. A convergência dos dois sinais para Haar continua sendo um achado, mas "a
  wavelet mais simples superou 22 famílias Daubechies" precisa dessa ressalva.

### 9. "Dependente de texto" tem duas definições, e nenhuma é a usual

**Problema.** A Introdução define o cenário dependente como "a mesma frase é falada e
imaginada". O capítulo 8 define como "o mesmo enunciado aparece no treino e no teste". No
código (verificado), dependente significa apenas que todas as frases aparecem em treino e
teste; os pares de verificação combinam quaisquer duas amostras de teste, de frases
diferentes inclusive. Na literatura de verificação de locutor, "dependente de texto" quer
dizer que cadastro e teste usam a mesma frase.

**Solução.** Escolher uma:
- manter o protocolo e renomear o eixo para "frases vistas × frases não vistas no treino",
  com uma única definição na Introdução e no capítulo 8; ou
- implementar a definição usual: no modo dependente, formar pares só entre amostras da
  mesma frase. Isso se combina com a reexecução do item 4.

### 10. O efeito da padronização precisa de explicação

**Problema.** Sem padronização, EEG e fusões ficam no acaso (0,43–0,49); com ela, vão a
0,18–0,29. Na voz, quase não muda. É o maior efeito da Fase 01 e o texto atual o descreve
ao contrário. Um efeito desse tamanho costuma indicar escalas muito desiguais entre
características, o que no vetor fundido significa uma modalidade dominando a outra.

**Solução.**
- Mostrar a faixa de valores das características de EEG e de voz antes da padronização
  (uma tabela pequena ou um boxplot).
- Explicar o mecanismo: a entrada da DSNN alimenta neurônios LIF com limiar fixo; entradas
  em escala errada deixam a camada muda ou saturada. Medir a taxa de disparo da primeira
  camada com e sem padronização confirmaria isso.
- Tratar "sem padronização" como ablação, e não como metade do experimento. As conclusões
  devem se apoiar nas 16 configurações padronizadas.

### 11. Eficiência energética: prometida, não medida

**Onde.** Resumo ("motivadas pela eficiência energética"), `08-proposedApproach.tex`
§Desenvolvimento e Validação, apêndice de regularização. A seção "Resultados Esperados",
hoje comentada, prometia "avaliar se as alegações sobre o consumo energético … se confirmam".

**Problema.** Nenhuma medida de energia, taxa de disparo ou número de operações aparece nos
resultados.

**Solução.** O *framework* já tem um modelo de energia (o do estudo preliminar citado).
Reportar, para a melhor configuração, a taxa de disparo por camada e a energia estimada por
inferência, ao lado da MLP de controle do item 6. Se não for medir, reduzir a afirmação a
motivação da literatura, com citação, e listar a medição em trabalhos futuros.

### 12. O capítulo de revisão contém um diário de depuração, e ele está desatualizado

**Onde.** `07-bibliographicRevision.tex`: §"Taxa de aprendizado por grupo de parâmetros"
(a partir de "O defeito e sua descoberta"), §"Colapso do latente no autoencoder pulsante",
e o fim de §`sec:dPenalizado`.

**Problemas.**
- Afirma que os "300 *rankings* da Fase 00" foram produzidos antes da correção e que
  reexecutar "é uma decisão em aberto". O capítulo 8 diz que a Fase 00 "foi integralmente
  reexecutada". Os dois não podem estar certos; hoje vale o segundo, com 214 execuções.
- "Primeiro lugar entre as 58 configurações do EEG" e "as 208 combinações": as contagens
  atuais são 61 e 214.
- Descreve um defeito do artigo de congresso "ainda em rascunho", dizendo que sua tabela
  "favorece indevidamente a rede LSTM". Isso não é revisão bibliográfica, expõe um trabalho
  não publicado, e já não é verdade: os LSTM saíram do artigo.
- Cita "Wiki, `Core/O…`" e nomes de perfis como fonte.

**Solução.** No capítulo 7, manter só o conceito: por que parâmetros biofísicos pedem taxa
menor, a equação, e o que é colapso de latente. Mover o que for relevante da história dos
defeitos para um apêndice curto ("Defeitos encontrados e corrigidos durante o
desenvolvimento"), com datas e efeito sobre os resultados. Esse apêndice é um ponto forte
de transparência se for factual e atual. Apagar as "situações em aberto" já resolvidas e
toda menção ao artigo.

### 13. Estado da arte: janela encerrada em 2024 e conclusão fora do escopo

**Onde.** `07-bibliographicRevision.tex` §"Trabalhos mais recentes".

**Problemas.**
- A busca cobre 30/03/2019 a 30/03/2024. A defesa será em 2027.
- A conclusão da revisão diz que os estudos "forneceram ideias para a construção do
  protocolo para coleta de dados que será explicado no capítulo 8". A coleta própria saiu
  do escopo e o capítulo 8 não explica protocolo de coleta algum.
- A mesma conclusão recomenda filtro de rede elétrica, ICA, CAR e separação por bandas α/β/θ,
  e a tese não aplica nenhum deles nem explica por quê.
- A subseção se chama "Protocolo de coleta" e descreve o protocolo de busca.
- A busca não tem termos para autenticação por EEG, fusão multimodal nem redes de pulso,
  que são o núcleo da tese. Os trabalhos são descritos um a um, sem tabela comparativa.

**Solução.** Refazer a busca até a data de fechamento, com termos para "EEG biometrics /
authentication", "imagined speech", "multimodal fusion" e "spiking neural network".
Renomear a subseção para "Protocolo de busca". Fechar com uma tabela (trabalho, sinal, base,
sujeitos, método, métrica) e um parágrafo que situe a tese. Trocar a frase da coleta por
uma justificativa das escolhas de pré-processamento, inclusive as que não foram adotadas.

### 14. A Introdução não cumpre o papel de introdução

**Onde.** `06-Introduction.tex` (37 linhas).

**Problemas.**
- Abre com a descrição da base de dados, antes da motivação.
- Não tem hipótese, justificativa, nem lista de contribuições.
- O objetivo fala em "sinais cerebrais extraídos durante a fonação (*imagined speech*)":
  fala imaginada é justamente a que ocorre sem fonação.
- Afirma que o EEG "evita tais problemas" de discriminação das biometrias, sem citação e sem
  mencionar o custo prático (touca, eletrodos, tempo de preparo).
- Promete o cronograma (`sec:cronograma`), que foi comentado no capítulo 8.

**Solução.** Reestruturar em: contexto e problema → lacuna → hipótese → objetivos geral e
específicos → questões de pesquisa → contribuições → estrutura. Mover a base para o capítulo
8. Corrigir a frase da fonação. Retirar a menção ao cronograma.

### 15. Duas escolhas de pré-processamento que merecem ressalva

**Fusão precoce.** O EEG é reamostrado de 1024 Hz para 44100 Hz (43×) e concatenado à voz
antes da DTWPT. O próprio texto admite que a metade de EEG passa a ser quase toda
interpolada. Depois da reamostragem, o conteúdo do EEG ocupa menos de 3% da faixa de
frequência, ou seja, cabe inteiro na primeira das 16 sub-bandas. Pela leitura, a fusão
precoce está estruturalmente em desvantagem; o resultado (0,2182 contra 0,1823 da tardia)
é compatível com isso, mas não o prova.
*Solução:* declarar a limitação junto do resultado, ou trocar por uma fusão precoce que
normalize cada modalidade antes de concatenar.

**Resolução espectral.** Com DTWPT de nível 4, a voz a 44,1 kHz é dividida em 16 bandas de
cerca de 1,4 kHz. A região que mais distingue locutores (0–4 kHz) cabe em 3 bandas, e as
escalas Mel e Bark, construídas para resolução fina nos graves, não têm o que reagrupar.
Isso pode explicar por que a Questão 1 encontra efeito tão fraco (0,026 entre escalas).
*Solução:* dizer isso na resposta à Questão 1, ou repetir a Fase 00 da voz com nível 6 ou 7
(ou com a voz reamostrada para 16 kHz). As configurações manuais são baratas.

### 16. A Questão 3 continua sem resposta

**Problema.** A tese pergunta se autoencoders superam técnicas manuais e responde só pelo
critério paraconsistente. O texto já reconhece isso com honestidade. Mas agora há um motivo
a mais para fechar a lacuna: a Fase 01 funciona (EER 0,18), então a comparação através do
classificador passou a ser informativa.

**Solução.** Rodar a Fase 01 também com o melhor autoencoder e a melhor PCA de cada sinal,
nas configurações padronizadas. São poucas dezenas de processos de minutos. Isso também
valida o próprio critério paraconsistente: se o ranking da Fase 00 prevê o ranking de EER
da Fase 01, a Contribuição 3 ganha evidência; hoje ela é só uma proposta.

---

## Gravidade C

### 17. "LFCC (c1)" não é cepstral

As tabelas chamam de "LFCC (c1)" as energias em bandas lineares **sem** logaritmo e DCT. LFCC
significa coeficientes cepstrais. A Introdução usa o nome certo (*Linear-band energies*).
O vencedor da tese aparece, portanto, com o nome do que ele não é.
*Solução:* nas tabelas e no texto, usar "Linear (c1)" / "LFCC (c2)", "Mel (c1)" / "MFCC (c2)",
"Bark (c1)" / "BFCC (c2)". É uma mudança no script que gera os CSV.

### 18. Números e contagens desatualizados

| Onde | Está | Deveria ser |
|---|---|---|
| Cap. 7, `sec:dPenalizado` | "208 combinações", "58 configurações" | 214; 61 no EEG |
| Cap. 7, lr-scale e colapso | "300 rankings/resultados" | 214 (ou cortar, item 12) |
| Cap. 10, Síntese | "208 execuções" | 214 |
| Cap. 9, Q1 | 1,4835 / 1,4972 / 1,5099 | 1,4836 / 1,4973 / 1,5100 |
| Cap. 9, Q2 | melhor cepstral "1,6068" | 1,6069 |
| Cap. 8, nota da PCA | "ainda não foi executada" | executada; remover a nota |
| Cap. 8, Fase 00 | "os valores atuais estão nas Tabelas…" + ordem das codificações | ver item 3 |
| Cap. 9, intro das tabelas | "entre as duas famílias" | três, com a PCA |

Os demais números da Q2 conferem com as tabelas novas (médias 1,5247 / 1,6285 na voz;
1,7980 / 1,7921 no EEG).
*Solução de fundo:* gerar esses números como macros a partir dos CSV, como já se faz no
artigo, para que o texto não possa divergir das tabelas.

### 19. Jargão interno

"Experimento 05", "mesmo esquema do Experimento 04", `classifier.enabled`,
`thesis_phase00_rank.py`, `p00_ae_snn_direct_tiny_eeg`, "perfis `smoke`", "Wiki". O leitor
não conhece a numeração dos experimentos nem o repositório.
*Solução:* descrever o procedimento em palavras no corpo do texto e reunir nomes de scripts,
perfis e chaves de configuração num apêndice de reprodutibilidade, com o endereço do
repositório e o commit.

### 20. Compilação e arquivos

- `\autoref{sec:cronograma}` na Introdução aponta para uma seção comentada: sai "??" no PDF.
- `bibliography.bib` tem a entrada `10.1117/12.2255697` duas vezes (linhas 155 e 179); o
  BibTeX acusa erro e o `latexmk` só termina com `-f`.
- A figura das três codificações temporais está como TODO no capítulo 8.
- `furlan2026spiking` é `@unpublished` e aponta para caminhos do repositório; atualizar
  quando houver aceite, ou citar como "em revisão" sem os caminhos.
- `backup.tex` e `texput.log` estão no diretório da tese.
- `11-appendices.tex` tem cerca de 380 linhas comentadas de apêndices abandonados, e uma
  seção "Penalidade de Esparsidade" que o texto principal não usa (pela leitura: não achei
  referência a ela).
- As notas `[PENDENTE]` são 15 hoje; nenhuma pode chegar à versão de defesa.

### 21. Terminologia

- O resumo usa "Redes Neurais de Pulso (RNP)"; o corpo alterna RNP, SNN e DSNN.
- "EQM" e "MSE"; "fold", "dobra" e "partição"; "locutor", "sujeito" e "voluntário".
- `direct` é chamada de "codificação temporal" e, na mesma lista, de "linha de base
  não-temporal".
*Solução:* fixar um termo por conceito na lista de siglas e aplicar em todo o texto.

---

## O que está bom e deve ser preservado

- A separação em duas fases e o registro de proveniência (revisão, sementes, um arquivo por
  repetição) são mais rigorosos que o usual.
- A derivação de λ = 2 − √2 para o D penalizado é clara e verificável.
- O protocolo de verificação está bem explicado: locutores disjuntos, pares genuínos e
  impostores, por que as métricas de conjunto fechado são `NaN`.
- A padronização é ajustada só no treino de cada dobra (verificado no código).
- A honestidade sobre limitações (Questão 3, escalas cocleares no EEG) é um ponto forte;
  os itens acima pedem que ela se estenda aos pontos 1, 4 e 5.

---

## Situação das correções (2026-11-12)

**Achado novo, mais grave que todos os itens acima.** O carregador lia o EEG da linha
`original_row - 1`, supondo as linhas numeradas a partir de 1; elas começam em 0. Cada
tentativa de voz recebeu o EEG da tentativa **anterior**: 1833 de fala pronunciada, 141 de
fala imaginada, e só 193 de 1974 com a mesma palavra da voz. O erro foi silencioso: o locutor
era o certo e o formato também. Por isso todos os números de EEG e de fusão até aqui são nulos,
tanto os de julho quanto os da reexecução de 12/11 (incluindo o EER 0,1823 da fusão tardia).
Corrigido: o pareamento agora é pela identidade da tentativa, com verificação de modalidade,
e a condição do EEG virou fator (`pronounced` × `imagined`). A identidade é a leitura certa:
as tentativas com áudio são exatamente as 1974 pronunciadas e nenhuma das 7341 imaginadas, o
que um deslocamento de uma linha quebraria. Os carregadores compartilhados tinham o mesmo
deslocamento e foram corrigidos também.

| Item | Código | Falta |
|---|---|---|
| 1 disfonia | voz sussurrada por LPC, níveis 0,25–1,0, só sondas × cadastro e sondas (`p01_dysph_*`) | rodar; texto (é um modelo de voz soprosa, não de toda disfonia) |
| 2 prosa antiga | — | reescrever com os novos resultados |
| 3 piso de ruído | — | reavaliar com os novos resultados |
| 4, 9 modo de texto | as duas condições treinam nas mesmas 5 frases; teste em frases vistas × não vistas | rodar; texto com uma só definição |
| 5 seleção vê o teste | a Fase 00 pontua 15 subconjuntos sem um locutor e os 15 conjuntos de treino da Fase 01; `winner_stability.csv` | rodar; texto |
| 6 linha de base | `ann` (gêmea não pulsante da DSNN, mesmos pesos iniciais) e `cosine` (sem rede) | rodar; tabela de trabalhos publicados |
| 7 fusão / pseudorreplicação | teste t corrigido de Nadeau–Bengio, referência fixa, família declarada | rodar; texto |
| 8 incerteza da Fase 00 | σ das repetições vira "—" nos determinísticos; colunas `loso_sd` e `fold_wins` | rodar; texto |
| 10 padronização | — | texto, com os novos resultados |
| 11 energia | medida na passagem de teste, modelo de Lemaire 2022 com preços de Horowitz 2014, com verificações de sensibilidade | rodar; texto |
| 12–14, 18–21 | — | texto |
| 15 fusão precoce / nível 4 | — | texto (ressalva) |
| 16 Questão 3 | `p01_cmp-<família>_*`: o melhor de cada família pela mesma DSNN | rodar; texto |
| 17 "LFCC (c1)" | rótulos: Linear/Bark/Mel (c1), LFCC/BFCC/MFCC (c2) | regenerar tabelas |
| 20 compilação | — | texto |

Detalhes de cada mudança: `software/nn/.wiki/Experiments/Thesis.md`, seção "Protocol revision".

## Situação após a reexecução e a reescrita (2026-11-12, fim do dia)

**Segundo achado silencioso.** Os *autoencoders* e a PCA recebiam a forma de onda promediada
em 256 blocos (voz: 689 amostras por bloco, passa-baixas em ~32 Hz; EEG: 96 por bloco, ~5 Hz).
A entrada não continha informação de locutor, e todos ficavam com EER entre 0,48 e 0,52.
Corrigido: a entrada é o espectro de potência logarítmico de 256 valores (Welch), conferido
contra `scipy.signal.welch` e `torch.linalg.svd` em teste automatizado. As duas fases foram
reexecutadas (Fase 00: job 5323830; Fase 01: job 5323900).

**O texto não tem mais números digitados.** Todo número de resultado nos capítulos 4 (resumo),
6, 8, 9 e 10 é uma chave `\tn{...}` de `tables/thesis_numbers.tex`, gerado por
`software/nn/scripts/pipeline/thesis/thesis_build_text_numbers.py`. Chave ausente é erro de
compilação; a lista `CLAIMS` do gerador recusa a geração quando uma afirmação da prosa deixa
de valer nos dados.

| Item | Situação |
|---|---|
| 1 disfonia | rodado; texto reescrito com a ressalva (voz soprosa simulada; nada significativo) |
| 2 prosa antiga | reescrita (capítulos 9 e 10, resumo e *abstract*) |
| 3 piso de ruído | previsão não confirmada (latência < Poisson < direta); o texto diz isso |
| 4, 9 modo de texto | rodado; uma só definição; sem diferença significativa |
| 5, 8 seleção e incerteza | rodado; estabilidade 15/15 no EEG, 10/15 na voz; no texto |
| 6 linha de base | controles `ann` e `cosine` rodados; tabela de trabalhos publicados no cap. 7 (`tab:estadoDaArte`), com o que é e não é comparável |
| 7 fusão | não se distingue do EEG sozinho; no texto |
| 10 padronização | no texto, com a explicação de escala declarada como não testada |
| 11 energia | medida; a DSNN gasta 4,37 a 6,18 vezes a gêmea não pulsante |
| 16 Questão 3 | respondida: pelo classificador as três famílias não se distinguem |
| 17 rótulos | tabelas regeneradas |
| 18 números | resolvido pelo `\tn` |
| 20 compilação | compila sem referência indefinida (entrada duplicada do `.bib` removida) |
| 12 diário de depuração | cap. 7 mantém só os conceitos (lr por grupo, colapso do latente, $D_{\text{penalizado}}$); a história foi para o Apêndice A (`apend:defeitos`, oito defeitos, datados, com o que cada um invalidou); menção ao artigo removida |
| 13 estado da arte | busca atualizada até 12/11/2026 (termos de biometria por EEG, fala imaginada, fusão, redes de pulso, voz patológica), 13 referências novas conferidas na Crossref/arXiv; "Protocolo de busca"; tabela comparativa; "Implicações" com o pré-processamento não adotado (CAR, ICA, filtro de rede, ritmos) e por quê. Atualização refeita também no Athena; **ressalva:** não repetida no Web of Science |
| 14 Introdução | reestruturada: contexto e problema → lacuna e hipótese → objetivos geral e específicos → questões → contribuições → estrutura; base movida para o cap. 8; frase da fonação corrigida; custo prático do EEG citado; cap. 10 retoma as hipóteses |
| 19 jargão | nomes de scripts, caminhos e chaves saíram do corpo (caps. 7–9); Apêndice B (`apend:reprodutibilidade`) os reúne |
| 21 terminologia | padronizada: locutor (esta tese) / sujeito (trabalhos de EEG citados), fala pronunciada, dobra, rede neural de pulsos (RNP; DSNN, SNN-AE e ANN-AE como nomes de modelos), EQM |
| 15 fusão precoce / nível 4 | ressalva no capítulo 8 |

**Resolvido depois.** Vírgula decimal nas tabelas por `\dec{}` (os CSV continuam com ponto);
as cinco referências de método (Horowitz 2014, Lemaire 2022, Nadeau e Bengio 2003, Holm 1979,
Welch 1967) conferidas na Crossref/JSTOR.

**Achado novo (limitação, não defeito), medido.** O EEG da base já vem filtrado pelos autores (FIR
passa-faixa de 2 a 40 Hz, declarado no artigo da base; a medição confirma) (−58,5 dB em 50 Hz em relação a α; `scripts/analysis/thesis_eeg_band_power.py`):
98,1% da potência abaixo de 32 Hz, 1,9% entre 32 e 64 Hz, ~0 acima. Com a DTWPT de nível 4 a
1024 Hz (sub-bandas de 32 Hz), os ritmos δ–β ficam juntos na sub-banda 0 e as sub-bandas 2–15
quase não têm sinal; a interferência de 50 Hz não é questão nesta base. 2653 das 9315 tentativas
têm a marca de piscada e não são excluídas. Registrado no cap. 7 (tab:potenciaEeg) e no cap. 10.

**Atualização (fim do dia).** Item 13: busca repetida no Athena pela VPN (6 combinações, contagens
no cap. 7); 10 referências novas, entre elas a BIOMEX-DB (voz + EEG síncronos, identificação
bimodal 97,6%), que estreitou a lacuna para "verificação com locutores fora do treino e sob
degradação". O Web of Science não foi consultado: o acesso automatizado por sessão foi bloqueado
pela política de permissões desta sessão. Item 21: terminologia padronizada (seção "Terminologia"
na Introdução). LSTM-AE alinhado à entrada espectral (924deede).

---

## Inclusão da BIOMEX-DB: replicação confirmatória (decidido em 2026-11-13)

**Decisão.** Incluir a BIOMEX-DB (Moreno-Rodriguez et al., IEEE Access 9:111267, 2021; dados
em 10.17632/s7chktmb6x.1, CC BY 4.0) como **replicação confirmatória**: os vencedores da
Coretto (extrator, hiperparâmetros, tipo de fusão) são fixados antes e rodados **uma vez** na
BIOMEX-DB, sem nova seleção. Só entram voz, EEG pronunciado e fusão de pronunciado. O grid
completo na BIOMEX-DB, se for feito, é exploratório e rotulado assim.

**O que a BIOMEX-DB cobre, e o que não cobre.** Só fala pronunciada, em que o EEG e o áudio são
gravados juntos. **Não tem fala imaginada**, então não replica o resultado central da tese.

### Medido no voluntário F001 (7,5 MB, baixado em 2026-11-13)

| | Coretto (base atual) | BIOMEX-DB |
|---|---|---|
| Locutores | 15 | 51 (25 F) |
| Condições | imaginada (só EEG) e pronunciada (EEG e áudio) | **só pronunciada** (EEG e áudio) |
| Estímulos | 5 vogais e 6 comandos | dígitos 0–9 em espanhol, 140 por locutor (14 por dígito): sessão `01G10` = 10 séries de 10 dígitos, `01G04` = 10 séries de 4 |
| EEG | 6 canais, 1024 Hz, FIR 2–40 Hz dos autores | Emotiv Epoc, **14 canais (AF3 F7 F3 FC5 T7 P7 O1 O2 P8 T8 FC6 F4 F8 AF4), 128 Hz**; EDF (36 sinais: + COUNTER, INTERPOLATED, qualidade de contato, giroscópio, MARKER) e CSV; rejeita-faixa visível em 50/60 Hz (−33 dB re α), 40 Hz só −7 dB |
| Voz | 44,1 kHz | 16 kHz mono 16 bits, um WAV por série, `.LAB` por palavra em unidades de 100 ns |
| Download | — | 51 `.rar` em uma pasta `BIOMEX` do Mendeley, **8,1 GB** (quase tudo vídeo; os 12 sem vídeo têm ~8 MB) |

Arquivos de dados: API pública do Mendeley, `public-api/datasets/s7chktmb6x/folders/1` → pasta
`BIOMEX` → `files?folder_id=<id>&version=1` (a raiz da versão está vazia, por isso a página
mostra "Files" vazio e `size` 0).

**Pareamento voz–EEG.** O canal MARKER do EDF tem um código por dígito (1–9, e 10 = "cero"),
99 no início, 89 nos olhos abertos, 69 no descanso entre séries. A ordem dos marcadores bate
com a ordem dos dígitos no `.LAB` (série 1 do F001: 7 9 0 2 1 5 8 6 4 3 nos dois). O par é
pela ordem dentro da série, conferindo a identidade do dígito.

### Três cuidados

1. **As "duas sessões" não são dias diferentes** (F001: 12h29 e 12h36 do mesmo dia). A base não
   testa reconhecimento em outro dia; não apresentar isso como vantagem.
2. **Voz e EEG não compartilham relógio.** Dígitos a ~2,0 s no WAV e ~2,6 s no EEG; uma série
   dura 20 s no WAV e 26,3 s no EEG. A janela de EEG sai do marcador do EEG, nunca do tempo do
   WAV (a janela errada não dá erro, só fica deslocada).
3. **EEG durante a fala tem EMG.** F001, 2 s após cada marcador ÷ repouso de olhos abertos:
   1–4 Hz 11,8×, 4–8 Hz 7,3×, 8–13 Hz 4,1×, 13–30 Hz 2,3×, 30–64 Hz 2,3×; em 30–64 Hz os
   maiores são T7 4,4× e F7 3,8× (mais perto da mandíbula). Parte do aumento em baixa
   frequência pode ser movimento ocular de leitura. Vale também para a condição pronunciada
   da Coretto: o texto deve dizer que "EEG pronunciado" carrega atividade muscular.

### Trabalho

- Conversor Python BIOMEX → mesmo esquema SQLite da Coretto, em arquivo separado: 2 s de EEG
  após cada marcador (256 amostras × 14 canais), voz pelo `.LAB` do dígito; tabela explícita
  de rótulos (há erros de digitação, como `UN0`), rótulo desconhecido é erro.
- C++: taxa do EEG fixa em `kEegSampleRate = 1024.0` (`ThesisFeatureExtractionInternal.hpp`) e
  comentários 6×4096 → taxa, canais e duração declarados na configuração (sem padrão). A 128 Hz
  a DTWPT de nível 4 dá sub-bandas de 4 Hz (separa δ/θ/α/β; dizer no texto).
- Perfis da replicação no gerador, auditoria de perfis, testes do conversor.
- Download dos 8,1 GB só depois que o conversor passar nos testes com o F001; vídeo descartado.

### Implementado (2026-11-13)

- `scripts/data/biomex_download.py` (API do Mendeley, confere o SHA-256 de cada `.rar`, baixa em
  blocos de 4 MB por 8 conexões, porque uma conexão longa caía para ~85 KB/s; extrai só
  EEG e Voice, retoma onde parou) e `scripts/data/biomex_to_sqlite.py` (pareamento acima,
  filtro 2–40 Hz FIR de fase zero na sessão inteira, voz de 1 s a partir do início da palavra,
  tudo que não pareia com certeza é recusado). A janela de voz é recuada, o mínimo necessário,
  quando alcançaria a palavra seguinte ou o fim do WAV (F004 diz UNO e DOS com 0,526 s de
  intervalo; o WAV de F006 termina 0,996 s após a última palavra). O recuo de cada par fica em
  `pair.voice_shift_s`. Se não houver espaço sem invadir a palavra anterior, o voluntário é recusado.
  Encolher a janela de todos cortaria as palavras longas; descartar os pares seria uma seleção
  silenciosa. Testes: `scripts/testing/test_biomex_convert.py`
  (voluntário sintético com assinatura do dígito no EEG e na voz; na CI).
- `dataset.source` obrigatório (`coretto2017` | `biomex2021`); BIOMEX-DB com condição imaginada
  é recusada. As taxas de amostragem saem da base (`ThesisDatasetView`), não de constantes.
- 48 perfis em `profiles/biomex/` (`bx01_*`): DSNN, `ann` e `cosine` × voz, EEG pronunciado,
  fusão precoce e tardia de pronunciado × frases vistas/não vistas (24) e a disfonia simulada
  nas fontes que leem a voz (24). Nada é selecionado na BIOMEX-DB.
- Vencedores congelados antes de qualquer resultado da BIOMEX-DB:
  `results/thesis/biomex/coretto_phase00_winners_frozen.json` (voz e EEG pronunciado:
  Haar + bandas lineares c1, DTWPT nível 4).
- GridUnesp: `THESIS_PHASE=biomex sbatch --array=0-47 scripts/pipeline/thesis/thesis_gridunesp.sbatch`;
  o deploy envia `~/biomex.sqlite` quando ele existe.

### Mesmo formato nas duas bases (auditoria de 2026-11-13)

O classificador recebe, nas duas bases, uma linha de 96 números por amostra (192 na fusão
tardia), padronizada por z-score ajustado no treino de cada fold. A DTWPT nível 4 divide
0..Nyquist em 16 sub-bandas iguais, então cada coluna só mede a mesma faixa de Hz se a taxa for
a mesma. Na taxa nativa da BIOMEX (16 kHz / 128 Hz) isso não acontecia: a voz da Coretto tem 99,2%
da potência na sub-banda 0 (0–1378 Hz) e o EEG 97,2% (0–32 Hz), enquanto a BIOMEX se espalhava
por 9–10 sub-bandas de 500 Hz / 4 Hz. **Decisão (usuário): a BIOMEX é gravada nas taxas da
Coretto (44,1 kHz / 1024 Hz).** Depois disso, a BIOMEX fica com 89,0% (voz) e 96,8% (EEG) na
sub-banda 0, e a Coretto e seus vencedores congelados continuam válidos.

- Conversor: `resample_poly` do WAV inteiro (441/160) e da sessão de EEG filtrada inteira (×8),
  antes do corte. Teste: nada acima do Nyquist nativo (< −50 dB).
- Leitor (`load_paired`): recusa um arquivo BIOMEX em qualquer outra taxa.
- Teste `ThesisBiomexDataset.BothBasesReachTheClassifierInTheSameFormat`: as duas formas
  passam pelo vencedor congelado e precisam dar 96 colunas (192 na tardia), todas finitas.
- Conferido: voz em PCM float [−1, 1] nas duas; EEG em µV nas duas (RMS 15,3 vs 14,5); tensores
  `(N, 1)` e `(canais, amostras)` canal-major nas duas.
- Diferenças que restam (registradas no wiki, seção 11): duração (4 s vs 1 s de voz; 4 s vs 2 s
  de EEG), canais (6 vs 14, concatenados num só sinal antes da DTWPT) e a fração de zeros do
  preenchimento até 2^n. Todas são constantes por coluna dentro de uma base, e o z-score as
  remove. Duas são de conteúdo e não somem:
  - **O EEG da Coretto tem 19,5% da potência abaixo de 2 Hz**, apesar do filtro "2–40 Hz" dos
    autores. A BIOMEX, com o FIR do conversor, tem 0,4%. Igualar exigiria refiltrar a Coretto e
    refazer as Phases 00/01. **Aberto:** decidir se isso entra como limitação ou se refiltramos.
  - A voz da Coretto concentra 77,7% da potência abaixo de 300 Hz; a BIOMEX, 28,3%. É
    característica das gravações (sala, microfone).

**Achado lateral.** O passo da CI `test_thesis_phase_scripts.py` já falha no `main` (25 de 29):
os scripts de ranking e significância ganharam argumentos obrigatórios em 12/11 (13fafd0b) e o
teste ficou com a interface de 11/11. Não é efeito da BIOMEX-DB.
