# Questões e tarefas — meeting01

Os enunciados foram escritos em 2026-11-03, às 12:45, com o job pela metade. As respostas foram
escritas no mesmo dia, depois que o job terminou.

**Origem dos números.** Todos vêm da pré-validação no formato 7, o job 5315335 no GridUnesp:
- código `fd4ed177`, que gera a mesma árvore registrada pelo job, `f2e400eb+dirty/e30f46e2aa5c`;
- terminou em 2026-11-03 às 18:20, com as 8 tarefas encerradas com código de saída 0, em 11 h 52 min;
- usa só os folds de desenvolvimento (FSDD fold 0 e eegmmidb fold 0), que ficam fora de todo número
  principal do paper;
- duas sementes por família (42 e 43); "s42 / s43" nas tabelas indica uma semente e a outra;
- os resultados estão em `software/nn/results/meeting01_prevalidation/`, já com o `03_` e o `00_`
  rodados e o paper recompilado (7 páginas, sem erros).

**Os scripts de análise** citados nas respostas ficam no scratchpad desta sessão, não no
repositório:
- `/tmp/claude-1000/-home-ensismoebius-Repos-doutorado/2fd85a4d-d426-42f4-9905-727ed3d49827/scratchpad/mean_check.py`, para a Q2;
- `/tmp/claude-1000/-home-ensismoebius-Repos-doutorado/2fd85a4d-d426-42f4-9905-727ed3d49827/scratchpad/direct_once.py`, para a T2.

---

## Questões

### Q1. Como funciona e para que serve o "preditor da média"?

**Contexto.** A regra de parada do formato 7 (`training.early_stop_min_skill`) compara cada
treino com um preditor de referência:

- **O preditor.** Ele ignora a entrada e devolve sempre a mesma janela: a média, posição a
  posição, das janelas-alvo de treino.
- **L0.** É a perda desse preditor, calculada com a mesma MSE mascarada, em blocos, que o
  Trainer usa na validação (`input_independent_monitor_loss`,
  [lib/include/Meeting01Stopping.hpp](lib/include/Meeting01Stopping.hpp)).
- **Quando a paciência começa a contar.** Só quando a perda de validação chega a
  ≤ (1 − s)·L0, com s = 0,10. Antes disso o treino roda todas as 30 épocas.
- **No fim.** Mantém-se a rede da melhor época (`BestWeightsCallback`).

Valores medidos no job 5315335:

| | FSDD fold 0 | eegmmidb fold 0 |
|---|---|---|
| L0 (depende da semente, que sorteia as 150 janelas de validação) | 0,997 – 1,001 | 1,001 – 1,002 |
| Limiar para a paciência começar a contar, 0,90·L0 | 0,897 – 0,901 | 0,901 – 0,902 |

Por que s = 0,10: nos 216 treinos da pré-validação anterior (job 5314513), o desvio mais fundo de
um treino que nunca saiu do platô foi de 0,066 (GRU-AE, FSDD). O raciocínio está documentado no
cabeçalho de `Meeting01Stopping.hpp`. Os eventos `train_end` gravam `armed_epoch`, e
`armed_epoch = 0` significa que a paciência nunca começou a contar
([lib/include/Meeting01EventCallback.hpp:118](lib/include/Meeting01EventCallback.hpp#L118)).

**A resposta deve cobrir:**
- Por que L0 ≈ 1,0 neste experimento. As janelas passam por z-score uma a uma (`WindowZScore`).
- O que é "habilidade" (skill), ou seja, 1 − perda/L0, e por que "a paciência começa a contar"
  depende dela e não do número da época.
- Por que a regra antiga (paciência 5 desde a época 1) cortava treinos no platô. Exemplo
  documentado: 8 de 11 avaliações do LSTM-AE no EEG pararam nas épocas 8–16 com MSE de validação
  entre 0,98 e 1,00.
- O que o preditor da média **não** é: não é o baseline PCA nem o baseline de média que o
  `03_meeting01_pca_mean_baselines.py` reporta no paper. Explicar a diferença, se houver.

#### Resposta

**O problema.** Um número de erro sozinho não diz nada. Uma MSE de 0,90 é boa? Depende de quanto
erra quem nem olha para a entrada. O preditor da média é essa régua: o melhor resultado possível
sem usar a janela de entrada.

**Como funciona, com uma janela real.** Usei a janela de teste do eegmmidb fold 0 do sujeito 0,
gravação 0, janela 5.
- O z-score é feito janela a janela, então toda janela sai com média 0 e variância 1. Medido nesta
  janela: média 0,000 e variância 1,000.
- O preditor calcula, posição a posição, a média das 1000 janelas de treino. Como cada uma está
  centrada em 0, essa média fica quase zero nas 256 posições: entre −0,066 e +0,069, com valor
  absoluto típico de 0,021.
- Ele devolve **essa mesma janela** para qualquer entrada:

```
posição               1     2     3     4      5      6      7      8   ...  256
entrada x          1,39  0,93  0,49  0,00  −0,50  −0,65  −0,40  −0,10  ...
saída (sempre)    ≈0,00 ≈0,00 ≈0,00 ≈0,00  ≈0,00  ≈0,00  ≈0,00  ≈0,00  ...   (a mesma para TODA janela)
erro² por posição  1,93  0,86  0,24  0,00   0,25   0,42   0,16   0,01  ...
                   └──────────── média sobre as 256 posições ≈ variância da janela ≈ 1,0 ────────────┘
```

Nesta janela, o erro é 1,011 com a média calculada só no treino (o L0 da regra de parada) e 1,008
com a média de treino ∪ validação (a linha "Mean" do paper). É por isso que L0 ≈ 1,0 aqui: errar
sem olhar a entrada custa a variância da janela, e o z-score fixa essa variância em 1.

**Para que serve: dois usos do mesmo preditor.**

| Uso | Onde | Ajustado em | Valor medido |
|---|---|---|---|
| Régua da regra de parada (L0) | `input_independent_monitor_loss`, `Meeting01Stopping.hpp` | 1000 janelas de treino; erro na validação | L0 entre 0,997 e 1,002; a paciência começa a contar em 0,90·L0 |
| Linha de referência do paper ("Mean") | `03_meeting01_pca_mean_baselines.py` | treino ∪ validação, 1150 janelas | teste 1,0005 no EEG, 0,9955 no FSDD |

Isso responde o último item do enunciado: o "Mean" do `03_` **é** o mesmo preditor. A única
diferença é o conjunto de ajuste (1150 contra 1000 janelas). No teste, a média calculada só no
treino dá 1,0008 no EEG e 0,9956 no FSDD, praticamente o mesmo que o `03_`. No FSDD, o erro por
janela das duas versões tem correlação r = 0,999. O PCA, por outro lado, é outra coisa: ele usa a
entrada (projeção linear em k = 64 componentes no EEG, k = 16 no FSDD) e erra 0,183 no EEG.

**Habilidade (skill) = 1 − MSE/L0.** É a fração do erro do preditor da média que o modelo
conseguiu eliminar. No teste do EEG, com L0 ≈ 1,001:

| Modelo | MSE teste (média de 2 sementes) | Habilidade |
|---|---|---|
| PCA (k = 64) | 0,183 | 0,82 |
| ANN-AE | 0,291 | 0,71 |
| SNN-AE | 0,407 | 0,59 |
| GRU-AE | 0,688 | 0,31 |
| LSTM-AE | 0,997 | 0,00 |
| Transformer-AE | 0,997 | 0,00 |

**Por que a paciência depende da habilidade, e não do número da época.** O tempo que um treino
passa no platô varia muito de execução para execução. No job 5315335 houve treinos de busca do
LSTM-AE no FSDD em que a paciência só começou a contar na época 28 (melhor validação 0,894 na
época 28), e treinos do GRU-AE no FSDD em que isso aconteceu entre as épocas 20 e 27. Com a regra
antiga (paciência de 5 contando desde a época 1), todos esses treinos teriam sido cortados por
volta da época 6, ainda no platô, e registrados com MSE ≈ 1,0.

**Modo de falha: silencioso.** Sem a régua, um modelo no platô aparece na tabela com MSE 1,00,
como qualquer outro número. Parece "um modelo ruim", mas na verdade é "nenhum modelo". Foi isso
que o job 5314513 (formato 6) produziu: 3 dos 8 retreinos finais das famílias recorrentes
devolviam a média.

---

### Q2. O que significa "o Transformer-AE reproduz a média e ignora a entrada"?

**Contexto.** No job 5315335, a paciência do Transformer-AE nunca começou a contar em nenhum dos
55 treinos (28 no EEG, 27 no FSDD). Todos rodaram as 30 épocas, com `stop_reason = max_epochs`.

| | EEG | FSDD |
|---|---|---|
| Melhor perda de validação nos treinos de busca (mín / mediana / máx) | 0,925 / 0,995 / 1,000 | 0,908 / 0,961 / 1,000 |
| MSE de teste dos vencedores retreinados (seed 42 / 43) | 0,993 / 1,000 | 0,914 / 0,932 |
| Comparação: SNN-AE | 0,396 / 0,419 | 0,734 / 0,691 |
| Comparação: ANN-AE | 0,291 / 0,292 | 0,668 / 0,654 |
| L0 (Q1) | ≈ 1,00 | ≈ 1,00 |

Vencedores que o GA escolheu para o Transformer-AE:

| | EEG | FSDD |
|---|---|---|
| seed 42 | d_model 96, d_ff 135, 8 cabeças, 1 camada | d_model 27, d_ff 135, 1 cabeça, 1 camada |
| seed 43 | d_model 79, d_ff 105, 1 cabeça, 1 camada | d_model 24, d_ff 101, 4 cabeças, 3 camadas |

**Ressalva importante.** A frase é uma **inferência** a partir de MSE ≈ L0, não uma medição. Uma
MSE perto de L0 é compatível com uma saída constante (a média), mas também com uma saída que varia
pouco e mal. Para confirmar, seria preciso olhar as reconstruções. Por exemplo, a variância da
saída entre janelas diferentes, ou a correlação entre entrada e saída. O
`*_per_window_errors.csv` de cada família só traz o erro por janela, não a saída.

**A resposta deve cobrir:**
- O que é uma saída "independente da entrada" e por que ela tem MSE ≈ L0.
- Por que isso é um platô do qual o otimizador às vezes sai (como o GRU-AE e o SNN-AE fizeram) e
  às vezes não (o Transformer-AE, em 30 épocas).
- Que a regra nova afasta a explicação "foi cortado cedo demais": o Transformer-AE teve o
  orçamento inteiro e não aprendeu. Ela **não** afasta a explicação "o protocolo de treino não
  serve para essa arquitetura" (ver Q3 e T1).
- Como confirmar a inferência com a variância da saída. Isso exige gravar as reconstruções; hoje
  elas não são gravadas.

#### Resposta

**O problema.** A frase partia de uma única média (MSE ≈ L0). Uma média, porém, não diz se o
modelo é igual ao preditor da média em toda janela ou se acerta umas e erra outras. Por isso
medi janela a janela.

**A medição.** Para cada uma das 1500 janelas de teste, comparei o erro do modelo com o erro do
preditor da média **naquela mesma janela** (`mean_check.py`). Um modelo que ignora a entrada tem,
em toda janela, o mesmo erro que o preditor da média. Um modelo que usa a entrada erra menos em
pelo menos parte das janelas.

| Fold 0 (s42 / s43) | MSE teste | Preditor da média | Janelas com erro ≥ 10% menor que o da média | Diferença mediana por janela |
|---|---|---|---|---|
| **Transformer-AE, EEG** | 0,993 / 1,000 | 1,001 | **0 / 0** de 1500 | 0,008 / 0,005 |
| **LSTM-AE, EEG** | 0,994 / 1,000 | 1,001 | **0 / 0** de 1500 | 0,013 / 0,005 |
| Transformer-AE, FSDD | 0,914 / 0,932 | 0,996 | 12,7% / 8,3% | 0,018 / 0,012 |
| LSTM-AE, FSDD | 0,907 / 0,886 | 0,996 | 15,2% / 24,9% | 0,025 / 0,037 |
| GRU-AE, EEG | 0,649 / 0,728 | 1,001 | 87,7% / 77,0% | 0,32 / 0,24 |
| SNN-AE, EEG | 0,396 / 0,419 | 1,001 | 98,8% / 98,2% | 0,64 / 0,61 |
| ANN-AE, EEG | 0,291 / 0,292 | 1,001 | 100% / 100% | 0,75 / 0,75 |

A mesma janela da Q1 (EEG, sujeito 0, gravação 0, janela 5), vista por todos os modelos:

```
preditor da média  1,008  ████████████████████
LSTM-AE            1,009  ████████████████████   <- igual à média
Transformer-AE     0,990  ████████████████████   <- igual à média
GRU-AE             0,708  ██████████████
SNN-AE             0,376  ███████
ANN-AE             0,276  █████
PCA                0,121  ██
```

**O que a frase significa, agora medido.**
- **No EEG:** o Transformer-AE e o LSTM-AE têm, em **cada uma** das 1500 janelas, o mesmo erro do
  preditor da média, com diferença de cerca de 1%. Isso vale para qualquer janela, ou seja, a
  saída não depende da entrada.
- **No FSDD:** os dois ficam "quase" nesse ponto. Eliminam de 6% a 11% do erro, e só numa minoria
  das janelas (8% a 25%).

**Ressalva.** Ainda não é prova de que a saída seja *igual* à média. Ter o mesmo erro em toda
janela é exatamente o que uma saída independente da entrada produz. Uma saída diferente com o
mesmo erro nas 1500 janelas é possível em tese, mas implausível. A prova direta exigiria gravar as
reconstruções, e o binário não grava.

**Novo: o LSTM-AE no EEG também.** Os dois retreinos finais rodaram as 30 épocas sem que a
paciência começasse a contar (melhor validação 0,996 e 1,000). Ou seja, a regra nova não os
cortou. Na busca do GA, porém, a **mesma configuração** chegou a validação 0,577 (s42) e 0,572
(s43). ~~Portanto, sair do platô no LSTM-AE depende da sorte da inicialização e da ordem das
janelas~~ **Corrigido em 2026-11-03 (achado da T1):** a inicialização e a ordem das janelas são
as **mesmas** na busca e no retreino. O LSTM, o GRU e o Transformer inicializam com sementes
fixas no código, e o `train_ae` não passa a semente ao embaralhamento, que fica no padrão 42 (ver
"Outros achados", item 4). O que muda entre a busca e o retreino final são os **dados**:

| | Treina em | Monitora em |
|---|---|---|
| Avaliação do GA | as 1000 janelas de treino | a validação (150 janelas, locutor jackson) |
| Retreino final | (treino sem o monitor) ∪ validação | o monitor recortado do treino por gravação, com a semente da rodada |

Portanto, sair do platô no LSTM-AE depende de **quais janelas** entram no ajuste e no monitor, não
de sorte da inicialização nem da parada antecipada. O LSTM não tem LayerNorm, então warmup e Pre-LN
(Q3, T1) não são a explicação padrão para ele.

**Modo de falha: silencioso.** A tabela de custo do paper compara a energia do Transformer-AE
(de 2,6× a 10× a do SNN-AE) com a dos outros, mas é o custo de uma rede que não reconstrói nada.
O veredito "não aprende" vale para o par (arquitetura + protocolo comum de treino), não para a
arquitetura sozinha. É isso que a Q3 e a T1 tratam.

---

### Q3. O que seria treinar o Transformer-AE com warmup?

**Contexto do código.**
- **Posição da LayerNorm.** O `TransformerEncoderBlock` coloca a LayerNorm *depois* da soma
  residual (chamado "Post-LN"): `n1 = LayerNorm1(x + a)`, `out = LayerNorm2(n1 + f)`
  ([include/layers/attention/TransformerEncoderBlock.hpp:9-11](../../../include/layers/attention/TransformerEncoderBlock.hpp#L9-L11)).
- **Otimizador.** O meeting01 treina todas as famílias com Adam (β1 = 0,9, β2 = 0,999,
  ε = 1e-8) e learning rate **constante** de 1e-3, sem warmup nem scheduler. Verificado por grep
  em `lib/`: nenhum warmup.
- **Passos de otimização.** O lote tem 1 janela (`samples_per_batch = 1`,
  [lib/src/Meeting01Training.cpp:309](lib/src/Meeting01Training.cpp#L309)). Com 1000 janelas de
  treino, são ~1000 passos por época e ~30 000 em 30 épocas.
- **O que já existe na biblioteca.** Há um otimizador com warmup, o `ScheduleFreeAdamW` com
  `warmup_steps`, testado em
  [src/core/optimizers/tests/optimizer_parity_gtest.cpp:175](../../core/optimizers/tests/optimizer_parity_gtest.cpp#L175).
  O meeting01 não o usa.

**Contexto da literatura (verificado 2026-11-03).** Xiong et al., *On Layer Normalization in the
Transformer Architecture*, ICML 2020, PMLR 119:10524–10533 (arXiv 2002.04745):
- No Post-LN, os gradientes dos parâmetros perto da saída são grandes na inicialização. Com
  learning rate alto, o treino fica instável, e o warmup contorna isso.
- Com a LayerNorm *antes* da subcamada (chamado "Pre-LN"), os gradientes começam bem comportados,
  e o warmup pode ser removido.
- **Ressalva:** os experimentos deles são com Transformers profundos (tradução, BERT). Os nossos
  têm 1–3 camadas e d_model 24–96. A aplicabilidade ao nosso caso é uma hipótese, não um fato.

**A resposta deve cobrir:**
- O que é warmup: o learning rate começa perto de 0 e sobe até o valor nominal nos primeiros N
  passos. Mostrar formas comuns: rampa linear, e o esquema de Vaswani et al. 2017,
  lr = d_model^−0,5 · min(passo^−0,5, passo · warmup^−1,5), com 4000 passos de warmup, o que aqui
  daria ≈ 4 épocas.
- Por que o Post-LN precisa dele e o Pre-LN não (Xiong et al. 2020).
- A diferença entre "o Transformer não serve para esta tarefa" e "o protocolo comum (Adam 1e-3
  constante, lote 1, 30 épocas) não serve para o Transformer Post-LN". Um revisor vai cobrar
  essa distinção.
- Outras causas possíveis, não verificadas, a não descartar: lote de 1 janela e learning rate
  compartilhado entre as famílias.

#### Resposta

**O problema.** Uma rede recém-inicializada ainda não tem uma direção boa de descida. Quando o
learning rate cheio é aplicado já no passo 1, os primeiros passos podem ser grandes demais. Xiong
et al. (2020) mostram que, no Transformer Post-LN, os gradientes perto da saída são justamente os
maiores na inicialização. Warmup é o remédio: começar com learning rate pequeno e subir aos poucos.

**Com os números do nosso protocolo.** São 1000 janelas por época com lote de 1, ou seja, 1000
passos por época e 30 000 passos em 30 épocas. Um warmup linear de 4000 passos (o número de
Vaswani et al., 2017) até 1e-3 seria lr(passo) = 1e-3 · min(1, passo / 4000):

```
lr
1e-3 |                    •───────────────────────────────────────  (constante até o passo 30 000)
     |               •
5e-4 |          •                      hoje: 1e-3 desde o passo 1
     |     •
1e-4 | •
   0 +----+----+----+----+----------------------------------------
     0  1000 2000 3000 4000                              passo
     (época 0 ... época 4)
```

No passo 400 o learning rate seria 1e-4, no passo 2000 seria 5e-4 e, a partir do passo 4000 (fim
da época 4), 1e-3. Depois disso, o resto do treino fica igual ao de hoje.

**Post-LN e Pre-LN lado a lado.** São as duas saídas que a literatura dá para o mesmo problema:

| | Post-LN (o nosso `TransformerEncoderBlock`) | Pre-LN |
|---|---|---|
| Fórmula de cada subcamada F | out = LN(x + F(x)) | out = x + F(LN(x)) |
| Caminho residual | passa pela LayerNorm em todo bloco | identidade pura do começo ao fim |
| Gradientes na inicialização (Xiong et al. 2020) | grandes perto da saída | bem comportados |
| Precisa de warmup? | sim (Vaswani et al. 2017 usaram 4000 passos) | Xiong et al. treinam sem warmup e obtêm resultados comparáveis |

**Distinção que um revisor vai cobrar.**

| Afirmação | O que sustenta | Situação |
|---|---|---|
| "O protocolo comum (Adam 1e-3 constante, lote 1, 30 épocas) não treina o nosso Transformer Post-LN" | Q2: 55 treinos, nenhum saiu do platô | medida |
| "O Transformer-AE não serve para esta tarefa" | exigiria que ele falhasse também com warmup ou Pre-LN | **não testada** (T1) |

**Modo de falha: silencioso.**
- Xiong et al. descrevem treino instável ou pior sem warmup. O nosso sintoma é outro: um platô.
  A perda nunca explode, não aparece NaN, e o treino termina por `max_epochs`.
- Por isso, uma rede que não aprende por falta de warmup fica igual a uma arquitetura incapaz de
  aprender a tarefa. Só a T1 separa os dois casos.
- Que o mecanismo de Xiong et al. valha para Transformers de 1 a 3 camadas como os nossos é uma
  hipótese.

---

## Tarefas

### T1. Teste local do Transformer-AE com warmup e com Pre-LN (FSDD fold 0, sem cluster)

**Por quê.** O paper justifica tirar o Transformer-AE da grade pela pré-validação. Hoje esse
resultado ("não aprende") tem uma explicação alternativa óbvia (Q3): Post-LN, sem warmup, com
learning rate de 1e-3. Se um revisor apontar isso e não tivermos testado, a justificativa cai.

**Desenho**, fixado antes de rodar para não ajustar pelo resultado:

| Braço | LayerNorm | Learning rate | Papel |
|---|---|---|---|
| A (controle) | Post-LN (atual) | 1e-3 constante | Precisa reproduzir o platô; se não reproduzir, o teste local não representa o cluster |
| B | Post-LN | Warmup definido a priori (ex.: rampa linear 0 → 1e-3 em 4000 passos ≈ 4 épocas, depois constante) | Testa a hipótese do warmup |
| C | Pre-LN | 1e-3 constante | Testa a hipótese de Xiong et al. sem warmup |

- **Configurações:** os dois vencedores do FSDD no job 5315335.
  - (d_model 27, d_ff 135, 1 cabeça, 1 camada)
  - (d_model 24, d_ff 101, 4 cabeças, 3 camadas). Esta é a mais profunda, onde o efeito do
    Post-LN deve aparecer mais.
- **Sementes:** 42 e 43, as mesmas do job.
- **Igual ao job:**
  - FSDD fold 0;
  - limites de 1000 janelas de treino, 150 de validação e 1500 de teste;
  - 30 épocas;
  - a regra de parada do formato 7, com restauração da melhor época;
  - seleção só pela validação.
- **Critério de sucesso, definido a priori:** a paciência começa a contar (validação ≤ 0,90·L0)
  **e** a MSE de teste fica claramente abaixo de L0. Reportar também a distância até o GRU-AE e
  até o ANN-AE no mesmo fold.

**Consequências:**
- **Nenhum braço aprende:** o paper pode dizer que warmup e Pre-LN foram testados, e a remoção
  fica mais sólida.
- **B ou C aprende:** a justificativa da remoção muda. Seria preciso refazer a pré-validação do
  Transformer-AE com a correção, e a decisão de mantê-lo ou não na grade volta para o autor.

**O que precisa ser construído:**
- Opção de warmup no treino, ou um scheduler no Adam, e/ou uma flag Pre-LN no
  `TransformerEncoderBlock`, com o gtest de gradiente por diferenças finitas da variante nova.
- O campo de configuração correspondente e a validação dele.
- `profile_audit_gtest`, se algum perfil de produção mudar.
- Atualização do `.wiki/Core/` e do `.wiki/References.md`, que a regra do wiki exige para
  mudança de camada ou de otimizador.
- O padrão tem que reproduzir o comportamento atual byte a byte (sem mudar o `results_format`)
  enquanto a opção estiver desligada.

**Alternativa mais rápida:** o gêmeo PyTorch (`nn.TransformerEncoderLayer(norm_first=True)` e um
`LambdaLR` para o warmup). Ele responde a pergunta "a arquitetura aprende com warmup ou Pre-LN?",
mas não testa a implementação C++, e o paper se apoia na implementação C++. Serve como triagem,
não como evidência final.

**Custo esperado:** no cluster, os treinos do Transformer-AE no FSDD levaram de 1 a 18 min cada.
São 3 braços × 2 configurações × 2 sementes = 12 treinos. Localmente deve ficar na ordem de poucas
horas na CPU (backend xtensor; nada de OpenCL). Isso é uma estimativa, não uma medição. Comandos
acima de 600 s precisam de `# CONFIRMED`.

#### Resultado (2026-11-04): rodado localmente, 18 treinos distintos

**Como foi rodado.**
- Seis processos locais (CPU, xtensor, 1 thread cada; 5 a 15 min cada), um por (configuração,
  braço). Cada um é o perfil do transformer-ae do job 5315335 com três mudanças: o GA fixado no
  genoma (população 1, 0 gerações, `winner_seeds` 1), só *direct* e só FSDD, e a chave do braço.
- Perfis: `results/meeting01_t1/profiles/`. Resultados: `results/meeting01_t1/<cfg>_<braço>/`.
  Análise: `t1_report.py` no scratchpad, com a mesma medida janela a janela da Q2.
- Cada processo faz 4 treinos: uma avaliação do GA e um retreino final por semente. As duas
  avaliações do GA saíram **idênticas**, porque a semente não muda a inicialização nem a ordem das
  janelas nas famílias sequenciais ("Outros achados", item 4). Sobram 3 treinos distintos por
  (configuração, braço), que diferem só nos dados de ajuste/monitor e compartilham a mesma
  inicialização.

**Resultado.** L0 (preditor da média) = 1,001 na validação e 0,996 no teste. Arma = a validação
chegou a ≤ 0,90·L0. "Janelas" = fração das 1500 de teste com erro ≥ 10% menor que o da média.

| Config | Braço | Treinos que armaram (de 3) | MSE teste s42 / s43 | Teste ÷ L0 | Janelas |
|---|---|---|---|---|---|
| c1 (27, 135, 1 cabeça, 1 camada) | A controle | 0 | 0,931 / 0,931 | 0,935 | 8% |
| c1 | **B warmup** | **2** (finais: época 26 e 11; a do GA não) | **0,900 / 0,901** | **0,904** | **18–22%** |
| c1 | C Pre-LN | 0 | 0,932 / 0,946 | 0,94–0,95 | 8% |
| c2 (24, 101, 4 cabeças, 3 camadas) | A controle | 0 | 0,931 / 0,931 | 0,935 | 8% |
| c2 | B warmup | 0 | 0,931 / 0,933 | 0,935 | 8% |
| c2 | C Pre-LN | 0 | 0,932 / 0,941 | 0,94 | 8% |

As outras famílias, no mesmo fold (job 5315335, teste, média de 2 sementes): GRU-AE 0,857,
SNN-AE 0,713, ANN-AE 0,661, PCA 0,649.

```
MSE de teste, FSDD fold 0 (barra a partir de 0,6; cada █ = 0,01)
preditor da média  0.996  ████████████████████████████████████████
Transformer A e C  0.931  █████████████████████████████████
Transformer c1-B   0.900  ██████████████████████████████
GRU-AE             0.857  ██████████████████████████
SNN-AE             0.713  ███████████
ANN-AE             0.661  ██████
PCA                0.649  █████
```

**Leitura.**
1. **Pre-LN não muda nada.** 0 de 6 treinos armaram, e o teste é igual ao do controle nas duas
   configurações.
2. **O warmup ajuda só a configuração rasa, e pouco.** Em c1 (1 camada), os dois retreinos finais
   armaram e o teste caiu de 0,931 para 0,900. A avaliação do GA, com os mesmos pesos iniciais e
   outros dados, não armou. A melhor época foi a 29 e a 30: o treino ainda descia quando bateu o
   limite de 30 épocas.
3. **Na configuração profunda (c2, 3 camadas), o warmup não fez nada.** É o contrário do que a
   hipótese de Xiong et al. prevê: o Post-LN profundo deveria ser o que mais precisa de warmup.
   Então "Post-LN sem warmup" não explica o platô do nosso Transformer-AE.
4. **Nenhum braço chega perto das outras famílias.** O melhor (0,900) fica acima do GRU-AE (0,857)
   e muito acima do ANN-AE (0,661).

**O critério a priori.** Ele pedia "arma **e** MSE de teste claramente abaixo de L0". c1-B arma,
mas o teste fica 9,6% abaixo de L0, exatamente a margem de armar. **"Claramente" não foi
quantificado antes de rodar; admito a falha no desenho.** Por isso não declaro sucesso nem fracasso
pelo critério. A minha leitura, feita depois de ver os dados, é que não é "claramente".

**Controle (braço A).** Ele reproduz o platô do job: não arma, teste ÷ L0 = 0,935 e 8% das janelas
(no job, 8–13%). Os números não são idênticos. No c2, s43 deu 0,9313 contra 0,9316 no job. No c1,
s42 deu 0,9309 contra 0,9135. A causa provável, **não verificada**, é a máquina diferente: o BLAS do
cluster e o local somam em outra ordem, e num platô a trajetória diverge. O código do Post-LN não
mudou de comportamento: os testes de gradiente continuam passando e o caminho `norm_first = false`
é o de antes.

**Limites.** Só FSDD e só o fold 0. Só os dois vencedores. Teto de 30 épocas. Os 3 treinos de cada
braço têm a mesma inicialização (item 4), então a variância de inicialização não foi amostrada.
No EEG, onde o Transformer-AE ficou na média em toda janela, não foi testado.

**Consequência para o paper (decisão do autor).** A remoção do Transformer-AE pode citar que warmup
e Pre-LN foram testados no fold de desenvolvimento: Pre-LN não muda nada; warmup deixa a
configuração rasa sair do platô por pouco (0,90 contra 1,00 da média), longe do GRU-AE (0,86) e do
ANN-AE (0,66).

---

### T2. Apresentar as duas contagens de energia (primeira camada T vezes × uma vez)

**Contexto: como está hoje.** Na codificação *direct*, a mesma janela analógica entra em cada um
dos T passos. O modelo de energia percorre a rede como ela é executada:
- a camada `Linear` recebe T·B linhas (`rows = in->rows()`,
  [lib/src/Meeting01Energy.cpp:205](lib/src/Meeting01Energy.cpp#L205));
- então `add_dense_fc` / `add_matmul` (linhas 112–131) cobram T vezes as MACs, as leituras de
  peso e as leituras de entrada da primeira camada.

Um acelerador poderia calcular a corrente I = W·x + b **uma vez**, guardá-la e só lê-la a cada
passo. As duas contagens se defendem:

| Contagem | Pergunta que responde | Situação |
|---|---|---|
| "Como executado" (atual) | Quanto custa a rede do jeito que o nosso código a roda? | Implementada; é a do paper |
| "Primeira camada uma vez" | Quanto custaria num hardware que aproveita a entrada constante? | Não implementada |

**Quem é afetado.** Só os vencedores com codificação *direct*. No job 5315335:
- **FSDD seed 42:** SNN-AE *direct* com T = 28, larguras [59, 16]. **É afetado.**
- **FSDD seed 43:** Poisson com T = 19. Não é afetado.
- **EEG:** os dois vencedores são Poisson. Não são afetados.
- **Grade principal:** qualquer vencedor *direct* que aparecer será afetado.

| FSDD fold 0, seed 42 | MSE teste | E com memória | só aritmética |
|---|---|---|---|
| SNN-AE *direct*, T = 28 | 0,734 | 12 321 nJ | 2 038 nJ |
| ANN-AE, larguras [59, 16] | 0,668 | 854 nJ | 148 nJ |
| Razão SNN ÷ ANN | | 14,4× | 13,8× |

**Conta de envelope** (raciocinada, **não medida**: a T2 existe justamente para medir):
- A aritmética da primeira camada por passo é 256 × 59 = 15 104 MACs × 4,6 pJ ≈ 69,5 nJ.
- Conferência com um número medido: a aritmética do ANN-AE, 147,64 nJ, é exatamente
  32 096 MACs × 4,6 pJ (256·59 + 59·16 + 16·59 + 59·256).
- Calcular a primeira camada uma vez tira ≈ 27 × 69,5 ≈ 1 876 nJ. Sobram ≈ 162 nJ contra 148 nJ
  do ANN: **≈ 1,1× em vez de 13,8×**.
- Na energia com memória também saem 27 repetições da leitura de 15 163 pesos e bias e de 256
  entradas. O tamanho dessa queda só a conta de verdade dá.
- **Consequência:** para vencedores *direct*, a parte "o ANN-AE gasta menos" da conclusão **não
  está garantida** pela contagem alternativa. A parte "erra menos" se mantém: 0,734 contra 0,668
  neste caso.

**O que fazer:**
1. Implementar a contagem "uma vez":
   - Melhor opção: um segundo campo no ledger C++ (ex.: `energy_nj_direct_once`), com os mesmos
     preços de Horowitz e a mesma interpolação de tamanho de SRAM. Exige rodar de novo.
   - Opção sem rodar de novo: recálculo pós-hoc em Python a partir dos manifests (larguras, T,
     codificação, janela de 256 amostras). Antes de calcular o caso "uma vez", o script **tem que
     reproduzir o número C++ atual** no caso "T vezes"; sem isso ficamos com duas implementações
     que podem divergir sem ninguém notar.
2. Contar o custo da corrente guardada: 59 escritas uma vez e 59 leituras por passo, no tamanho
   da memória correspondente.
3. Pôr as duas contagens na tabela de energia do paper (`paper_loso_energy.tex`, gerada por
   `write_energy_tex` em `scripts/pipeline/meeting01/02_meeting01_build_loso_paper_data.py`) e no
   texto de robustez do veredito (`_energy_robustness` no `00_`).
4. Decisão do autor: qual das duas é a principal e qual fica como sensibilidade. A regra "Energy
   first" do paper usa a principal.
5. Verificar na literatura como outros trabalhos contam a primeira camada sob codificação
   *direct* (ex.: Dampfhoffer et al. 2023, IEEE TETCI). Não citar nada sem ter conferido.

#### Resultado (2026-11-03): medido, sem rodar nada de novo

**Método.** O recálculo foi feito em Python, depois do job (`direct_once.py`).
1. **Validação primeiro.** O script reproduz as funções de energia do C++ que entram na conta:
   `sram_access_pj`, `EnergyLedger::access` e `add_dense_fc`. Antes de qualquer cálculo, conferi
   que ele reproduz o `energy_nj` do binário nas **4 linhas do ANN-AE**, que é toda densa e passa
   pelo mesmo `add_dense_fc`. Conferi também as variantes com 8 KB e 1 MB. A diferença ficou
   abaixo de 1e-3 nJ, por exemplo 854,4853 contra 854,4853 nJ.
2. **O que muda na contagem "uma vez".** Tira-se o custo de (T − 1) cópias da primeira camada
   (MACs, leituras de entrada e de pesos, bias). Soma-se o que guardar a corrente custa: 59
   escritas uma vez e T × 59 leituras.

Só o vencedor *direct* é afetado: FSDD, seed 42, T = 28, primeira camada 256 → 59.

| FSDD s42 | Como executado | Primeira camada uma vez | ANN-AE | Razão SNN ÷ ANN |
|---|---|---|---|---|
| E com memória (principal) | 12 321 nJ | **1 304 nJ** | 854 nJ | 14,4× → **1,53×** |
| tudo em SRAM de 8 KB | 6 955 nJ | 931 nJ | 481 nJ | 14,5× → 1,93× |
| tudo em SRAM de 32 KB | 11 865 nJ | 1 695 nJ | 814 nJ | 14,6× → 2,08× |
| tudo em SRAM de 1 MB | 51 148 nJ | 7 810 nJ | 3 475 nJ | 14,7× → 2,25× |
| só aritmética | 2 038 nJ | 162 nJ | 148 nJ | 13,8× → **1,10×** |
| MSE teste | 0,734 | 0,734 | 0,668 | o erro não muda |

Na média de família (FSDD, 2 sementes; a s43 é Poisson e não muda):

| Contagem | SNN-AE com memória | Razão | SNN-AE só aritmética | Razão |
|---|---|---|---|---|
| Como executado | 8 098 nJ | 12,2× | 1 081 nJ | 9,2× |
| Primeira camada uma vez | 2 590 nJ | 3,9× | 143 nJ | 1,2× |

No EEG nada muda: os dois vencedores são Poisson (9,4× com memória, 1,6× só aritmética).

**O que isso significa.**
1. **A conclusão se mantém nos dois eixos.** Mesmo na contagem mais favorável ao SNN, o ANN-AE
   gasta menos que o SNN-AE nos 4 vencedores e em todas as precificações (a menor razão é 1,10×,
   só aritmética). Ele também erra menos em todos.
2. **A magnitude não se mantém para vencedores *direct*.** O vencedor *direct* cai de 14× para
   1,1–2,3×. Na contagem "uma vez", ele vira o SNN-AE **mais barato**: 1 304 nJ contra 3 875 nJ
   do Poisson com T = 19. Isso pesa na regra "Energy first" do paper.
3. **O custo do SNN-AE agora está na codificação.** O que encarece é a codificação Poisson com T
   alto (T = 19 a 31, com 36% a 41% dos pares neurônio-passo disparando), não o
   fato de ser uma SNN.

**Decisões pendentes do autor.**
- **Qual contagem é a principal no paper.** Recomendação: a "primeira camada uma vez" como
  principal e a "como executado" como sensibilidade.
  - Motivo: o paper conclui que o SNN-AE gasta mais. A conclusão fica mais à prova de revisor
    quando é tirada com a contagem mais favorável ao SNN.
  - Além disso, o modelo de energia já supõe um acelerador com SRAM própria (Lemaire et al. 2022),
    e um acelerador reaproveitaria uma corrente constante.
- **O objetivo de custo do GA.** O `estimate_snn_macs` também conta a primeira camada T vezes.
  Isso só afeta o desempate e a frente de Pareto, não o vencedor, que é escolhido pela menor MSE
  de validação. Mudar esse objetivo mudaria a busca e exigiria rodar de novo.
- **A implementação.** Opções: um segundo campo no ledger C++, o que exige rodar de novo, ou este
  recálculo pós-hoc incorporado ao `02_`/`00_`, com a validação contra o ANN-AE mantida como
  autoteste.
- **O item 5 acima continua em aberto:** ver como a literatura conta a primeira camada sob
  codificação *direct*.

---

## Outros achados ao analisar o job 5315335

1. **Defeito de registro em `*_events.jsonl`. Silencioso; afeta só esse log, não os resultados.**
   - Os eventos `config_begin` das avaliações do GA saem sem metadados: `config_id` vazio e
     patience −1.
   - Depois do primeiro retreino final, eles passam a carregar os metadados **daquele retreino**:
     `role = *_final`, o `config_id` dele e os hiperparâmetros dele. Exemplo: no Transformer-AE do
     FSDD, 12 eventos aparecem como `transformer-ae_direct_seed42_run1`
     (d_model 27, d_ff 135, 1 cabeça, 1 camada), mas só um deles é o retreino final. Os outros 11
     são avaliações da busca da semente 43, com outras configurações.
   - O relatório `00_` não usa esses campos (lê `final_fit` dos manifests), então as tabelas do
     paper estão corretas.
   - Quem ler o log, o monitor ou uma análise futura vai atribuir treinos à configuração errada.
     Corrigir exige mexer no `Meeting01EventCallback`.
   - **Corrigido no formato 8 (2026-11-03, autor: "3 - Sim").** `take_pending_context()` esvazia o
     contexto depois de usá-lo (um contexto, um treino). As avaliações do GA gravam o próprio
     contexto (`<modelo>_ga_<genoma>_seed<s>`, papel `ga_eval`). Um treino sem contexto, com o log
     aberto, lança exceção. Verificado na T1: cada `train_end` traz o `config_id` certo.
2. **O PCA continua batendo todos os autoencoders.** No teste: EEG 0,183 contra 0,291 do ANN-AE e
   0,407 do SNN-AE; FSDD 0,649 contra 0,661 do ANN-AE e 0,713 do SNN-AE. Já era assim no formato 6;
   o veredito gerado diz "none of the three reached the test error of PCA on either development
   fold".
3. **Formato 6 contra formato 7.** A regra nova melhorou o GRU-AE no FSDD (0,929 → 0,857) e o
   Transformer-AE no FSDD (0,998 → 0,923). Não mudou o LSTM-AE no EEG (1,000 → 0,997) nem o
   Transformer-AE no EEG (0,997 → 0,997). O custo de tempo subiu, por exemplo LSTM-AE FSDD de
   3,6 h para 11,9 h, porque um treino que não sai do platô agora roda as 30 épocas.
4. **Nas famílias sequenciais, a "semente" não muda a inicialização nem a ordem das janelas.
   Silencioso. Achado durante a T1 (2026-11-03).**
   - **Sintoma.** Na T1, a avaliação do GA da mesma configuração deu resultado idêntico, até o
     último dígito, com as sementes 42 e 43 (melhor validação 0,9611603617668152 na época 23, nas
     duas).
   - **Causa, lida no código.**
     - O LSTM-AE, o GRU-AE e o Transformer-AE inicializam com sementes fixas no código (por
       exemplo `init_linear(*embed_, 4100u)` no `TransformerAutoencoder.cpp`, `42u + offset` no
       LSTM e no GRU).
     - O `train_ae` não passa a semente da rodada ao `TrainerConfig::sampler_shuffle_seed`, que
       fica no padrão 42.
     - Com codificação *direct*, nada mais usa a semente.
     - O SNN-AE e o ANN-AE recebem a semente na inicialização (`proto.seed = run_seed`). O
       embaralhamento também é fixo para eles, porque nenhum caminho do meeting01 define
       `sampler_shuffle_seed`.
   - **O que a semente ainda muda nas famílias sequenciais:** a amostragem de genomas do GA e o
     monitor recortado no retreino final. Os retreinos finais das duas sementes diferem só por
     isso.

   | | SNN-AE / ANN-AE | LSTM / GRU / Transformer |
   |---|---|---|
   | Inicialização varia com a semente | sim | **não** |
   | Ordem das janelas varia com a semente | **não** | **não** |
   | Recorte do monitor no retreino final | sim | sim |
   | Amostragem do GA | sim | sim |

   - **Consequência.** "2 sementes" na pré-validação das famílias sequenciais não amostra a
     variância de inicialização. A frase do perfil ("enough to see whether a family's result is one
     seed's luck") vale só em parte para elas. Na grade (SNN-AE + ANN-AE), a inicialização varia,
     mas a ordem das janelas é a mesma nas 5 sementes.
   - **Correção proposta, aguardando o autor.** Passar a semente da rodada à inicialização dos três
     modelos sequenciais e ao `sampler_shuffle_seed` de todas as famílias. Isso muda os resultados,
     então só vale junto com a nova pré-validação no formato 8.
