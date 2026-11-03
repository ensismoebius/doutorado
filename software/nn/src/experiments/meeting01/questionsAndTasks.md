# Questões e tarefas — meeting01

Situação em 2026-11-03, 12:45. Os números vêm da pré-validação no formato 7, rodando no GridUnesp
como job 5315335 (revisão `f2e400eb+dirty/e30f46e2aa5c`). Ela usa só os folds de desenvolvimento
(FSDD fold 0 e eegmmidb fold 0), que ficam fora de todo número principal do paper. Das 8 tarefas,
4 tinham terminado: SNN-AE (com o controle ANN-AE) e Transformer-AE, nos dois datasets. GRU-AE e
LSTM-AE ainda estavam rodando. Os números dessas duas famílias são parciais e podem mudar quando o
job terminar.

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
