# Meeting01 — proposições de posicionamento do paper

Registro da discussão de 2026-10-29 a 2026-11-01 sobre como posicionar a revisão da submissão 71.
Objetivo do autor desde o início: **comparar um autoencoder SNN com baseline(s)**.

> Status das evidências: tudo abaixo vem de **um fold do FSDD (fold 0)**, GA mínimo e 2 a 3
> execuções por célula. É direção, não resultado publicável. EEG ainda não foi rodado.

---

## 1. O problema que motivou esta nota

Uma comparação "SNN contra LSTM/GRU/Transformer" não diz **o que** fez o SNN ganhar ou perder:
os quatro modelos diferem em tudo ao mesmo tempo (recorrência, atenção, comprimento de sequência,
spikes). Dois achados recentes mostraram o risco:

| Achado | Consequência |
|---|---|
| Os baselines recorrentes liam a janela repetida 16 vezes (S = 512 quadros) e **nunca treinavam** (MSE ~1,0 = prever a média) | Uma vitória do SNN sobre eles não provaria nada. Corrigido com `model.recurrent_time_steps = 1` (results_format 4). |
| O controle pareado `ann-ae` (mesma rede do SNN, LeakyReLU no lugar do LIF) **vence o SNN com folga** | A pergunta honesta passa a ser "quanto o spike custa e o que devolve", não "o SNN é melhor". |

### A rede comparada (corrigida a partir de `AutoencoderBuilders.hpp`)

```
SNN-AE   janela(256) x T passos -> Linear 64 -> LIF   -> Linear 16 -> Linear 64 -> LIF   -> Linear 256   (perda em cada um dos T passos)
ann-ae   janela(256) x 1 passo  -> Linear 64 -> leaky -> Linear 16 -> Linear 64 -> leaky -> Linear 256
                                                          ^ gargalo ANALÓGICO nos dois (sem neurônio após Linear 16)
```

Só as duas camadas ocultas de 64 disparam. Gargalo e saída são `Linear` analógicos nos dois braços.

### Evidência atual (FSDD fold 0, MSE de teste, 1,0 = prever a média, média de 2 execuções)

| Janelas de treino | SNN-AE (C++) | ann-ae (C++) | GRU-AE (C++) | PCA k=16 | MLP-AE (PyTorch) |
|---|---|---|---|---|---|
| 200 | 0,880 | **0,706** | 0,952 | 0,590 | 0,691 |
| 1000 | 0,827 | **0,662** | 0,867 | 0,649 | 0,658 |
| 3000 | 0,783 | **0,599** | não rodado | 0,657 | 0,638 |

- O SNN perde ~0,15–0,17 de MSE para a mesma rede sem spikes.
- `time_steps` 4 / 16 / 32 (1000 janelas): 0,793/0,770, 0,797/0,776, 0,797/0,776 — **plano**. Simular
  mais tempo não fecha a diferença.
- Mais janelas de treino ajudam todos os modelos: `loso_max_train_windows` (200 na produção) é a
  maior alavanca medida até agora.
- O Transformer-AE fica em 1,000 em todos os tamanhos (não investigado).
- O GA do SNN prefere limiar baixo: os melhores vencedores têm `v_th = 0,6, α = 0,83` (val ≈ 0,64–0,66)
  e `v_th = 1,3, α = 0,91`.

---

## 2. Lacunas na literatura

Busca feita em 2026-10-29; **não exaustiva**, e só resumos foram lidos. A redação segura no paper é
"até onde sabemos".

### O que já existe

| Trabalho | O que comparou | Resultado | Limite para nós |
|---|---|---|---|
| [Spiking Autoencoders With Temporal Coding (Frontiers in Neuroscience, 2021)](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2021.712667/full) | AE espiguado × AE convencional, MNIST e Fashion-MNIST | Perdas parecidas | Baseline com outras ativações: **não pareado** |
| [Progressive Tandem Learning (arXiv 2007.01204)](https://arxiv.org/pdf/2007.01204) | AE espiguado × ANN pré-treinado, MNIST | MSE 0,00662 × 0,00667 | O ANN é o ponto de partida da conversão |
| [Deep spiking AE regularizado (arXiv 2109.11045)](https://arxiv.org/abs/2109.11045) | Conv espiguado × AE e VAE, MNIST | Comparável, sem vantagem sob ruído | Imagens |
| [Fully Spiking VAE (AAAI 2022)](https://arxiv.org/pdf/2110.00375) | VAE espiguado × VAE ANN de mesma arquitetura, MNIST, CIFAR10 e outros | SNN melhor nos números dos autores | Imagens; código do ANN pareado é público |
| [Sparse spiking AE (Neuromorphic Computing and Engineering, 2024)](https://iopscience.iop.org/article/10.1088/2634-4386/ad5c97) | Reconstrução, denoising, classificação | — | **Conteúdo não lido** |
| [AE-SNN com backprop por spikes (arXiv 1906.08861)](https://arxiv.org/pdf/1906.08861) | Spiking AE × ANN, imagens; áudio convertido em imagem | Perdas próximas em MNIST | Áudio só como conversão para imagem |

Referências de apoio: PCA é o ótimo do autoencoder linear ([arXiv 2302.11337](https://arxiv.org/abs/2302.11337));
Hinton & Salakhutdinov (Science, 2006).

### O que não encontrei

1. SNN-AE × ANN **pareado** em **reconstrução de fala curta** ou de **EEG**.
2. Essa comparação sob **leave-one-subject-out aninhado**, com **estatística no nível da gravação**.
3. SNN-AE × LSTM/GRU/Transformer-AE em reconstrução, com os baselines comprovadamente treinados.

### O que a literatura diz sobre o desenho

- O ANN pareado é **prática estabelecida** (FSVAE publica o seu).
- Os trabalhos relatam o SNN "comparável" ao ANN, não melhor (exceto o FSVAE, nos números dos
  próprios autores). Aqui ainda não chegamos à paridade.

---

## 3. Quatro posicionamentos possíveis

| Posicionamento | De onde vem o ineditismo | Risco | Depende de |
|---|---|---|---|
| **A. Benchmark rigoroso: SNN × ANN pareado** | Protocolo sem vazamento, quatro bases (fala + EEG), estatística por gravação, baselines verificados | Revisor pode achar que "o SNN perdeu" não é contribuição | Todos os folds e bases rodados |
| **B. Trade-off qualidade × custo** | Fronteira de Pareto MSE × custo estimado (operações, esparsidade de spikes) | Energia é estimativa analítica, não medida em hardware | Modelo de custo declarado para ANN e SNN (MACs × ACs) |
| **C. Método: fechar a diferença do SNN** | O mais forte, se funcionar | Hoje é pesquisa em aberto; pode não fechar | Diagnóstico das causas (seção 6) |
| **D. Seleção de arquitetura por ranking paraconsistente** | Peça própria do repositório (`d_penalized`) | Fora do escopo desta comparação; dilui a mensagem | Decisão do autor |

### Pré-requisito de B: as métricas de custo atuais não servem (verificado no código em 2026-11-01)

| Métrica gravada | O que o código calcula (`Meeting01Evaluation.cpp`, `Meeting01Metrics.cpp`) | Problema |
|---|---|---|
| `spike_rate` | fração dos valores da **reconstrução de saída** maiores que 0 (`recon > 0`) | A saída é analógica: isso é ≈ 0,5 para qualquer modelo. **Não conta spikes** das camadas LIF. Falha silenciosa. |
| `energy` | `spike_rate × n_valores + 10 × macs` | Soma grandezas sem unidade comum; não é um modelo de energia. |
| `macs` (SNN) | MACs densos da rede **× T** | Não dá crédito à esparsidade: com T = 16, o SNN "custa" 16× o `ann-ae`. |

Para B é preciso um modelo de custo declarado: contar spikes por camada oculta, custo de entrada
por AC (acumulação, só quando há spike) para o SNN e MAC para o ANN, com custos por operação de
uma referência explícita (ex.: Horowitz, ISSCC 2014, 45 nm), e dizer que é estimativa.
A primeira camada do SNN recebe entrada analógica (`direct`), então ela é MAC, não AC.

---

## 4. Recomendação: A + B, com um diagnóstico curto de C

1. **A é a base e a melhor defesa.** A submissão 71 foi rejeitada por vazamento. Um paper cuja
   contribuição é "comparamos sem vazamento, com baselines que de fato treinaram e com o controle
   pareado" responde diretamente à revisão. Os dois achados acima mostram por que isso é necessário.
2. **B torna um resultado negativo informativo.** Se o spike custa ~0,15 de MSE, o leitor quer saber
   o que ganha em troca: operações por inferência, taxa de disparo. O modelo de custo tem de ser
   declarado (ex.: MAC para o ANN, AC por spike para o SNN), sem prometer medição em hardware.
3. **Um diagnóstico curto de C explica o porquê.** Se alguma variação fechar a diferença, vira
   contribuição de método; se não fechar, entra como análise. Em andamento (seção 6).

Se a diferença sumir ou encolher muito no EEG, o posicionamento muda para "o SNN é competitivo onde
o sinal é temporalmente rico" — por isso o EEG precisa rodar antes de fixar o texto.

---

## 5. O que eu evitaria

- **Chamar de ineditismo de método** um resultado que só mostra o ANN vencendo.
- **Comparar contra baselines mais fracos que o ANN pareado** — expõe o paper à mesma crítica dos
  baselines que não treinavam.
- **Orçamento de busca desigual sem declarar.** Hoje o `ann-ae` **não é buscado** (copia as larguras
  do vencedor do SNN), enquanto o SNN busca larguras, limiar e α. Isso favorece o SNN; se o ANN vence
  mesmo assim, a conclusão é conservadora — mas o texto tem de dizer isso.
- **Alegar energia sem modelo de custo** explícito e sem dizer que é estimativa.
- **Escrever "primeiro a..."** sem uma revisão dirigida (Scholar, IEEE Xplore, arXiv; termos como
  *spiking autoencoder EEG reconstruction*, *spiking autoencoder speech reconstruction*).
- **Tirar conclusões de um fold.** Os números acima são de FSDD fold 0.

---

## 6. Próximos passos registrados (2026-11-01)

| Passo | Estado |
|---|---|
| Isolar a causa da diferença SNN × ann-ae em fala, uma variável por vez (`software/nn/scripts/analysis/snn_gap_ablation_torch.py`) | feito (seção 7) |
| Testar o número de janelas de treino (500 / 2000 / 5000 / 10000, além de 200 / 1000 / 3000) | feito (seção 8) |
| Baixar eegmmidb | feito (1526 EDF, 3,4 GB; sinal 0 = canal `Fc5.` nos 1526 arquivos); Siena (20,3 GB) não cabe no disco (22 GB livres) |
| Implementar no C++ as três mudanças de C (média temporal, decoder analógico, T = 4) | feito, `results_format` 5 (seção 8) |
| Corrigir `spike_rate` / `energy` | feito, `results_format` 5 (seção 8) |
| Rodar o diagnóstico em pelo menos uma base de EEG | feito no eegmmidb (seção 8) |

Variáveis isoladas no diagnóstico (cada variante muda **uma** coisa em relação ao gêmeo do SNN):

| Variante | O que muda | Hipótese testada |
|---|---|---|
| `snn` | nada (gêmeo PyTorch do SNN C++) | valida o gêmeo contra o C++ |
| `snn_rate` | perda na **média temporal** da saída | a perda por passo sobre códigos binários limita a resolução |
| `enc_lif_only` | só o encoder dispara | o spike custa mais no decoder |
| `dec_lif_only` | só o decoder dispara | o spike custa mais no encoder |
| `snn_integr` | saída passa por integrador não espiguado | falta uma leitura que acumule no tempo |
| `snn_soft` | reset suave (`v -= v_th`) | o reset a zero descarta informação |
| `ann` | nenhum neurônio dispara | o controle |

---

## 7. Resultado do diagnóstico em fala (2026-11-01)

Gêmeo PyTorch do SNN-AE C++, FSDD fold 0, 1000 janelas de treino, 1500 de teste, `v_th = 0,6`,
`α = 0,83` (o vencedor mais frequente do GA), 3 sementes por linha, mesma regra de parada do C++.
**Validação do gêmeo:** val do SNN 0,646–0,656 (C++ com os mesmos `v_th`/`α`: 0,644–0,662);
`ann` 0,667 no teste (C++ `ann-ae`: 0,662). Dispersão entre sementes ≈ 0,005–0,01.

### A pergunta, com números

O SNN erra 0,780; a mesma rede sem spike erra 0,667. Diferença: **0,113**. De onde ela vem?

```
0,667  ann (LeakyReLU)                          ┐
0,685  ann_sigmoid  (limitada, gradiente suave) │  faixa limitada custa pouco: +0,018
0,743  ann_clip     (recorte duro em [0,1])     │  saturação dura custa:      +0,076   ← sem nenhum spike
0,734  snn_rate     (spikes, perda na média)    ┘  ≈ ann_clip: o spike binário não acrescenta quase nada
0,780  snn          (spikes, perda por passo)      perda por passo custa:      +0,046
```

### Tabela completa (MSE de teste, média de 3 sementes)

| Grupo | Variante | MSE | Leitura |
|---|---|---|---|
| referência | `snn` (T = 16) | 0,780 | — |
| referência | `ann` | 0,667 | — |
| perda / leitura | `snn` avaliado na média temporal, **sem retreinar** | 0,741 | a saída por passo oscila em torno de uma média melhor |
| perda / leitura | `snn_rate` (treina na média) | 0,734 | fecha 41% |
| perda / leitura | `snn_integr` (integrador na saída) | 0,771 | 8% |
| onde dispara | `dec_lif_only` (encoder analógico) | 0,759 | 19% |
| onde dispara | `enc_lif_only` (decoder analógico) | 0,737 | 38% |
| onde dispara | `enc_lif_only_rate` | 0,721 | 52%, melhor SNN medido |
| passos | `snn` T = 64 | 0,779 | T não importa com perda por passo |
| passos | `snn_rate` T = 4 / 16 / 32 / 64 | 0,742 / 0,734 / 0,735 / 0,735 | **nem com perda por taxa** |
| neurônio | `snn_soft` (reset suave) | 0,799 | piora |
| neurônio | `v_th` 0,3 / 0,6 / 1,0 | 0,790 / 0,780 / 0,789 | efeito pequeno |
| neurônio | surrogate α 0,5 / 2 (C++) / 8, com `snn_rate` | 0,754 / 0,734 / 0,744 | o valor do C++ já é o melhor |
| ANN limitado | `ann_relu` | 0,665 | igual ao `ann` |
| ANN limitado | `ann_sigmoid` | 0,685 | +0,018 |
| ANN limitado | `ann_clip` = clamp(x, 0, 1) | 0,743 | +0,076, sem spikes |
| ANN limitado | `ann_clipste` (clamp, gradiente identidade) | 0,841 | gradiente que não corresponde à função piora; não separa faixa × gradiente morto |

### O que isso diz (medido × raciocinado)

- **Medido:** a perda por passo custa ~0,04–0,05; T (4 a 64), largura do surrogate, limiar e tipo de
  reset **não** são a causa; um ANN com recorte duro em [0, 1] e **sem spikes** reproduz o SNN com
  perda por taxa (0,743 × 0,734).
- **Raciocinado, não provado:** com entrada `direct`, o LIF do encoder recebe a mesma corrente I a
  cada passo, e um LIF com reset a zero vira uma função de transferência recortada: com
  I ≤ `v_th(1−β)` nunca dispara (morto), com I > `v_th` dispara em todo passo (saturado, taxa = 1).
  Para `v_th = 0,6`, `β = 0,83` a faixa útil é só (0,10; 0,60]. Isso explicaria por que o recorte
  duro imita o SNN e por que T não ajuda. (O LIF do decoder recebe corrente que varia no tempo, então
  o argumento vale estritamente só para o encoder.)
- **Não medido:** EEG; outros folds; a mesma ablação no C++ (exigiria mudar a perda e as specs
  renderizadas pelo GA — mudança de método, decisão do autor).

### Consequência para o posicionamento B (custo): estimativa, não medição

Custo por janela com o desenho atual (`direct`, gargalo analógico): a primeira camada recebe entrada
analógica e a terceira recebe o gargalo analógico, então as duas são **MAC a cada passo**; só as
camadas alimentadas por spikes viram AC. Com MAC = 4,6 pJ e AC = 0,9 pJ (Horowitz, ISSCC 2014,
45 nm, fp32) e as taxas de disparo medidas (≈ 0,35–0,43, densas):

| Modelo | MACs | ACs | Energia estimada | × ANN |
|---|---|---|---|---|
| ANN | 34 816 | 0 | 160 nJ | 1,0 |
| SNN T = 1 (qualidade não medida) | 17 408 | ~6 800 | 86 nJ | 0,5 |
| SNN T = 4 | 69 632 | ~24 400 | 342 nJ | 2,1 |
| SNN T = 16 (atual) | 278 528 | ~109 000 | 1 379 nJ | **8,6** |
| SNN T = 64 | 1 114 112 | ~446 000 | 5 526 nJ | 34,5 |

**Com o desenho atual, o SNN perde nos dois eixos**: reconstrói pior *e* custa ~8,6× mais. O
posicionamento B só se sustenta se a arquitetura mudar (T baixo, entrada e gargalo espiguados, taxa
de disparo esparsa), o que já é o posicionamento C. Como T não melhora a qualidade (tabela acima),
reduzir T é ganho de custo sem perda medida — o primeiro candidato.

### Recomendação revista

A continua sendo a base. B, como está, vira um resultado negativo claro ("o SNN direto é dominado")
— honesto, mas fraco sozinho. O diagnóstico aponta três mudanças baratas e testáveis para C, todas
mudança de método (decisão do autor): (1) perda na média temporal; (2) decoder analógico;
(3) T = 4 ou menos. Antes de decidir, repetir o diagnóstico em EEG (eegmmidb), onde o resultado pode
mudar.

---

## 8. As mudanças implementadas e o diagnóstico em EEG (2026-11-01, `results_format` 5)

### 8.1 Mais janelas não fecham a diferença (FSDD fold 0, C++, média de 2 execuções)

| Janelas de treino | 200 | 500 | 1000 | 2000 | 3000 | 5000 | 10000 |
|---|---|---|---|---|---|---|---|
| SNN-AE (desenho antigo) | 0,880 | 0,851 | 0,827 | 0,791 | 0,783 | 0,782 | 0,764 |
| ann-ae | 0,706 | 0,680 | 0,662 | 0,630 | 0,599 | 0,584 | 0,544 |
| diferença | 0,174 | 0,171 | 0,165 | 0,161 | 0,184 | 0,198 | **0,220** |

A partir de 3000 janelas o ann-ae continua melhorando e o SNN-AE quase para: a diferença **cresce**
com os dados. Dado não é o que falta ao SNN.

### 8.2 O que mudou no C++ (as três mudanças de C)

```
antes   janela x T=16 -> Linear 64 -> LIF -> Linear 16 -> Linear 64 -> LIF   -> Linear 256   perda nas 16 linhas
depois  janela x T=4  -> Linear 64 -> LIF -> Linear 16 -> média em T
                                                        -> Linear 64 -> leaky -> Linear 256   perda em 1 linha
ann-ae  janela x 1    -> Linear 64 -> leaky -> Linear 16 -> Linear 64 -> leaky -> Linear 256
```

| Campo do perfil | Antes (padrão) | Produção | Por quê |
|---|---|---|---|
| `model.snn_time_pool` | `none` | `latent` | a perda passa a cobrar a média, que é o que um código de taxa entrega; no latente, o decoder roda uma vez |
| `model.snn_decoder_neuron` | `lif` | `analog` | o LIF do decoder só acrescentava erro; `latent` exige isso |
| `model.time_steps` | 16 | 4 | no FSDD, T não comprou qualidade (abaixo); ver 8.5 para o EEG |
| `model.initializer` | (cada construtor o seu) | `xavier` | **achado novo**, ver 8.4 |

Onde fazer a média foi medido antes de implementar (gêmeo, FSDD, 3 sementes): média no latente e
média na saída dão o mesmo (0,724 × 0,724 em T = 4), e no latente o decoder custa T vezes menos.
Os padrões antigos continuam reproduzindo o `results_format` 4, por teste.

### 8.3 As métricas de custo corrigidas

`spike_rate` agora é spikes ÷ pares (neurônio, passo) das camadas LIF ocultas, e NaN ("não se
aplica") para modelos sem spike. `energy` deu lugar a `macs` + `acs` (contados do forward de cada
janela) e `arith_energy_nj` = 4,6 pJ × MAC + 0,9 pJ × AC. Conferi os valores na fonte primária
(Horowitz, ISSCC 2014, Fig. 1.1.9: soma float 32 bits 0,9 pJ, multiplicação 3,7 pJ); é a mesma
convenção do apêndice do Spikformer (ICLR 2023). A mesma figura dá 20 pJ por acesso a uma cache de
32 KB: a estimativa é **só aritmética**, não a energia de um chip. O `C_raw` do artigo passou a ser
`macs + acs`, como o texto do artigo já definia (o código não fazia isso).

Exemplo do desenho novo (larguras 64/16, T = 4, entrada `direct`, taxa 0,39): ≈ 83 200 MACs +
≈ 1 700 ACs ≈ **384 nJ**, contra 160 nJ do ann-ae (2,4×). A primeira camada domina: uma janela
`direct` é analógica e passa densa a cada um dos 4 passos.

### 8.4 Achado novo: o controle não era só "outro neurônio"

O construtor ANN do C++ inicializa com **Xavier** (√(6/(fan_in+fan_out))) e o SNN com **Kaiming**
(√(6/fan_in)). O ann-ae diferia do SNN-AE no neurônio **e** na inicialização. No FSDD isso não
aparecia; no EEG, sim (gêmeo, 3 sementes, MSE de teste):

| | Kaiming | Xavier |
|---|---|---|
| ANN, eegmmidb | 0,489 | **0,293** (C++ ann-ae com Xavier: 0,305) |
| SNN-AE novo (T = 4), eegmmidb | 0,484 | **0,449** |
| ANN, FSDD | 0,667 | 0,654 |
| SNN-AE novo (T = 4), FSDD | 0,724 | 0,729 |

Até a convergência (teto de 200 épocas, 1 semente) a ANN com Kaiming para em 0,479 e com Xavier em
0,291: não é só velocidade. A conclusão depende de qual inicialização os dois braços compartilham.
No EEG, com o desenho novo:

| Inicialização dos dois braços | SNN-AE novo | ANN | Diferença |
|---|---|---|---|
| Kaiming | 0,484 | 0,489 | ≈ 0 |
| Xavier | 0,449 | 0,293 | 0,156 |

Com Kaiming nos dois, o SNN parece empatar só porque a Kaiming atrapalha a ANN. **Correção:**
`model.initializer` vale para os dois braços, e `validate()` recusa um perfil com ann-ae sem ele.
Escolhi `xavier` porque é a melhor ou igual para **cada** braço nas duas bases. Assim, a comparação
com a mesma inicialização é também a do melhor de cada um. (O desenho antigo é insensível a isso:
0,592 × 0,589. Por isso a combinação desigual do C++, SNN com Kaiming e ANN com Xavier, deu a mesma
diferença de ~0,30 que Xavier nos dois.)

### 8.5 Diagnóstico em EEG (eegmmidb fold 0, gêmeo PyTorch, 3 sementes, vencedor do GA em EEG: larguras 65/64, `v_th` 0,96, α 0,965)

Gêmeo validado: `snn` 0,592 contra 0,593 do C++ (média de 2 execuções).

| Variante | Kaiming | Xavier |
|---|---|---|
| `ann` (controle) | 0,489 | **0,293** |
| `ann_sigmoid` | 0,308 | — |
| `ann_clip` (recorte, sem spike) | 0,547 | — |
| `snn` (desenho antigo, T = 16) | 0,592 | 0,589 |
| `snn_rate` (média na saída, decoder LIF, T = 16) | 0,454 | — |
| `dec_lif_only` (só o decoder dispara) | 0,606 | — |
| novo: média no latente, decoder analógico, T = 4 | 0,484 | 0,449 |
| idem, média na saída, T = 4 | — | 0,446 |
| idem, média no latente, T = 16 | 0,457 | 0,396 |
| idem, média na saída, T = 16 | 0,403 | 0,392 |

O que isso diz:

- **As três mudanças valem também em EEG, e mais:** com Xavier nos dois braços, a diferença
  SNN × ANN cai de 0,296 (0,589 × 0,293) para 0,156 em T = 4.
- **O padrão do FSDD se repete:** o recorte sem spike (`ann_clip` 0,547) fica perto do SNN antigo,
  e a sigmoide (0,308) perto da ANN. A saturação dura continua sendo a principal explicação.
- **Diferente do FSDD, T importa no EEG:** T = 16 é ~0,05 melhor que T = 4 (0,396 × 0,449), ao custo
  de ~4× em aritmética. "T ≤ 4" veio só do FSDD. Ver a decisão pendente abaixo.
- Média no latente e na saída continuam empatadas (0,449 × 0,446), então a escolha pelo latente
  se sustenta.
- **Ressalva:** no EEG quase tudo para no teto de 30 épocas ainda melhorando; o número é "depois de
  30 épocas". Com teto de 200 épocas, a mudança foi pequena (SNN Kaiming T = 4: 0,484 → 0,465).

### 8.6 As decisões foram tomadas olhando o teste do fold 0 — conferidas na validação

Os números acima são MSE de **teste** do fold 0 (FSDD e eegmmidb). Escolher o desenho por eles é
seleção no teste. Conferi cada decisão no MSE de **validação** (os falantes/sujeitos internos, o
critério legítimo), e todas saem iguais:

| Decisão | Validação (gêmeo, média de 3 sementes) |
|---|---|
| Xavier × Kaiming, ANN, EEG | 0,272 × 0,470 |
| Xavier × Kaiming, SNN-AE novo T = 4, EEG / FSDD | 0,425 × 0,465 / 0,594 × 0,598 |
| média no latente × na saída, T = 4, EEG / FSDD | 0,425 × 0,424 / 0,598 × 0,592 |
| T = 4 / 8 / 16, EEG (Xavier) | 0,425 / 0,394 / 0,372 |
| T = 1 / 2 / 4 / 16, FSDD (Kaiming) | 0,601 / 0,606 / 0,598 / 0,599 |

Mesmo assim, o fold 0 dessas duas bases foi olhado. **O rigoroso é tratá-lo como fold de
desenvolvimento**: declará-lo no artigo e reportar os resultados principais sem ele (ou com ele à
parte). Decisão sua.

### 8.7 Conferência no binário C++ (fold 0, 1000 janelas, GA local pop 4 × 2 gerações, 2 execuções)

O gêmeo PyTorch roda um genoma fixo; o artigo sai do C++, em que o GA escolhe o genoma por
(fold, execução). Por isso o perfil de formato 5 foi rodado de ponta a ponta. A tabela traz o MSE de
teste (média de 2 execuções) e a energia aritmética (§8.3):

| Base | Init (os dois braços) | SNN-AE | ann-ae | Diferença | Energia SNN / ann |
|---|---|---|---|---|---|
| FSDD | Kaiming | 0,786 | 0,731 | 0,055 | 485 / 339 nJ |
| FSDD | **Xavier** | 0,768 | 0,652 | 0,116 | 180 / 292 nJ |
| eegmmidb | Kaiming | 0,543 | 0,550 | ~0 | 546 / 268 nJ |
| eegmmidb | **Xavier** | 0,532 | 0,302 | 0,230 | 263 / 191 nJ |

Desenho antigo (formato 4) nas mesmas janelas: FSDD 0,827 × 0,662; eegmmidb 0,593 × 0,305.

- **As mudanças valem no C++:** o SNN melhora ~0,06 nas duas bases, e a diferença com Xavier cai de
  0,165 para 0,116 (FSDD) e de 0,29 para 0,23 (EEG).
- **Com Kaiming, o C++ repete a armadilha do §8.4:** no EEG os braços só parecem empatados porque o
  controle está prejudicado.
- **Ruído de seleção:** no EEG com Xavier, a execução 1 escolheu `direct`/`dense` (0,476) e a 2
  escolheu `recurrent` (0,589). Com 8 avaliações por execução, a arquitetura vencedora é quase sorte.
  Quem dá sentido à seleção é o orçamento do GridUnesp; estas 2 execuções não são o resultado.
- **A energia depende do vencedor:** no FSDD com Xavier, o SNN gasta menos que o ann-ae (180 × 292 nJ)
  porque as entradas binárias (`poisson`, `recurrent`) fazem a 1ª camada custar ACs. Um vencedor
  `direct` paga a 1ª camada densa a cada um dos 4 passos.
- **No FSDD, o ann-ae empata com o PCA:** k = 16 dá 0,649 contra 0,652 do ann-ae, e o SNN-AE (0,768)
  perde para os dois.
- **O pipeline Python inteiro (03_ → 02_ → 04_) lê o formato 5.** O 02_ recusava o `ann-ae`, porque só
  conhecia as famílias buscadas, e foi corrigido: o ann-ae virou família "não buscada", com o
  manifesto conferido e entrada nas tabelas como "ANN-AE (matched)". A falha era ruidosa: travaria o
  build do artigo, não o falsificaria.

### 8.8 Decisões abertas (suas)

1. **T no EEG:** 4 (produção, escolhido no FSDD) ou 16 (~0,05 melhor no EEG, na validação e no
   teste, com ~4× a aritmética). A opção rigorosa é decidir pela validação, por exemplo com T como gene
   do GA ∈ {4, 8, 16}, e nunca pelo teste.
2. **Fold 0 de FSDD e eegmmidb:** declará-lo fold de desenvolvimento (§8.6) e reportar sem ele, ou
   sinalizá-lo.
3. **ann-ae na produção:** colocá-lo em `baselines` do `meeting01-loso.json`. Hoje ele só existe nos
   perfis de validação.
4. **Teto de 30 épocas:** limita o EEG (quase tudo para no teto ainda melhorando). Opções: subir o teto
   ou deixar a paciência decidir.
5. **Objetivo de custo do GA:** hoje é a estimativa analítica, cega aos spikes. Pode passar a usar a
   energia aritmética medida.
6. **Commit:** nada foi commitado.
