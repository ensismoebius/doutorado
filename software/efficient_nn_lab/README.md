# Efficient Neural Networks Lab

Laboratório visual desktop, complementar aos slides em Beamer da palestra
["As fronteiras da arquitetura de redes neurais: BitNets e redes de
pulso"](../../documentation/08-lectures/fronteiras-bitnets-redes-pulso/),
para ilustrar de forma interativa e passo a passo:

- forward e backward tradicionais (regra da cadeia, gradiente descendente),
  com um exemplo real convergindo para um alvo, iteração por iteração;
- quantização escalar e a família BitNet (b1.58);
- o problema de treinar através de uma função não-diferenciável, o gráfico
  da derivada real vs. a derivada que o STE de fato usa, e o
  *Straight-Through Estimator* (STE);
- o neurônio *Leaky Integrate-and-Fire* (LIF) e redes neurais de pulso (SNN);
- o *surrogate gradient* usado para treinar SNNs;
- uma comparação lado a lado entre RNA convencional, BitNet e SNN.

Implementa a especificação em [`ESPECIFICACAO_DLVL.md`](ESPECIFICACAO_DLVL.md), com os desvios
deliberados listados em [Desvios deliberados da especificação](#desvios-deliberados-da-especificação).

Este software **não substitui** os slides — é um laboratório visual que o
palestrante abre ao vivo, no momento correspondente do slide, para tornar
concreto o que a equação ou o diagrama estático não mostra.

## Instalação

Requer Python 3.11+. Recomenda-se um ambiente virtual:

```bash
cd software/efficient_nn_lab
python3 -m venv --system-site-packages .venv
./.venv/bin/pip install -e .
```

(`--system-site-packages` reaproveita PySide6/matplotlib/numpy já
instalados no sistema, se houver; caso contrário `pip install -e .` os
instala normalmente dentro do venv.)

## Execução

Mais simples — funciona a partir de qualquer diretório, cria o venv na
primeira vez se necessário:

```bash
./run.sh
```

Equivalente manual (precisa estar dentro de `software/efficient_nn_lab/`,
não em `software/` — esse diretório tem uma subpasta chamada
`efficient_nn_lab`, que sombreia o pacote instalado se o comando for
rodado um nível acima):

```bash
python -m efficient_nn_lab
```

ou:

```bash
python src/efficient_nn_lab/main.py
```

Em ambos os casos o pacote precisa estar instalado (`pip install -e .`)
para que os imports absolutos (`efficient_nn_lab....`) resolvam.

## Atalhos de teclado (modo palestra)

| Tecla | Ação |
|---|---|
| `Espaço` | play / pause |
| `->` | próximo passo |
| `<-` | passo anterior |
| `R` | reset |
| `N` | próxima demo (ver abaixo) |
| `Esc` | voltar ao menu |

### Navegação entre demos: **Próxima demo ▸**

Botão no alto da janela, ao lado dos botões de modo. Vai para o **próximo
item da seção atual**; se o item atual for o último da seção, vai para o
**primeiro item da seção seguinte**; e do último item de todos volta ao
primeiro (a volta é intencional — um botão morto no fim do roteiro só
apareceria ao vivo, no meio da palestra). Atalho: `N`.

Existe por causa do **Modo palestra**: esse modo esconde a árvore lateral,
que era o único jeito de trocar de demonstração — ou seja, o modo feito
para apresentar era o único em que não se conseguia avançar sem sair dele.
A ordem seguida é exatamente a da árvore (`_demo_order` deriva da mesma
estrutura que constrói a árvore, para as duas não divergirem), e a seleção
da árvore acompanha o botão, de modo que sair do modo palestra não revela
uma barra lateral apontando para outra demo.

`N`, e não `->`/`PageDown`: `->` já é "próximo passo *dentro* da demo", e
apresentadores remotos mandam `PageDown` para passar slide.

### Loop rápido (só em `SNN → Codificação Poisson (imagem)`)

Botão extra na fileira de transporte, visível apenas nas demos que o
declaram (`DemoModule.supports_fast_loop`). Roda todos os passos em
sequência, **sem a espera de 1,1s por passo** e voltando ao início ao
terminar: ~1,2s por volta a ~25 quadros/s, contra ~33s de uma passagem do
`Play` normal. Pausa com um segundo clique, com `Pause` ou com `Reset`.

Existe porque nessa demo **a cadência é o conteúdo**: um único passo de
tempo Poisson é indistinguível de ruído, e o rosto só emerge quando os
quadros passam rápido o bastante para o olho integrar os disparos.
Percorrer 30 checkpoints com 1,1s de espera cada nunca mostra isso. Nas
outras demos o botão não aparece — cada passo ali tem texto para ler, e
passar voando por eles seria ruído.

O intervalo do loop é igual ao do tick normal (40ms) de propósito: o
redesenho dessa demo custava ~39ms (p90 medido na máquina da palestra),
então um intervalo menor só enfileiraria redesenhos que o widget não
consegue atender. O painel da estimativa (original e reconstrução lado a
lado) deixou cada redesenho ~14% mais caro (medido offscreen, mesma
máquina, antes e depois); se o loop engasgar no projetor, é esse o custo.

O botão **Modo palestra** simplifica a barra lateral; o botão **Modo
professor** revela equações e o estado numérico completo de cada passo.

## Animação: checkpoints e tweens

Nenhuma demonstração pula direto de uma imagem para outra. Cada uma define
poucos **checkpoints** — os passos didáticos nomeados, o que "Passo X/Y" e
os botões Anterior/Próximo contam — e o motor (`core/demo.py`,
`core/animation.py`) preenche dezenas de **quadros intermediários**
interpolados entre cada par de checkpoints. Tanto "Próximo" quanto "Play"
sempre *animam* essa transição fina (nunca saltam instantaneamente); Play
ainda pausa por um instante em cada checkpoint para dar tempo de ler a
explicação antes de seguir para o próximo.

Cada widget desenha o **mesmo diagrama persistente** em todos os quadros
de uma demonstração — caixas e setas em posições fixas, inclusive um
"esqueleto" fantasma de tudo o que ainda vai aparecer, visível desde o
primeiro quadro. O que muda de um quadro para o outro são apenas campos
contínuos (opacidade de revelação, fração preenchida de uma seta, o valor
mostrado dentro de uma caixa) — nunca a própria composição da cena. A
única exceção deliberada é uma troca de cena genuína (por exemplo, da
curva de quantização para o diagrama de blocos do STE, em
`bitnet/demos/backward.py`): nesse caso o corte é instantâneo
(`steps=0`), porque misturar duas figuras estruturalmente diferentes seria
pior do que um corte limpo.

## Estrutura

```text
src/efficient_nn_lab/
├── main.py, __main__.py      ponto de entrada (CLI --demo <slug>)
├── app/
│   ├── main_window.py        janela principal, roteamento demo -> widget
│   ├── math_render.py        pseudo-LaTeX das demos -> mathtext (equações)
│   └── theme.py              cores e stylesheet (mesma paleta dos slides)
├── core/
│   ├── demo.py               contrato DemoModule + Frame (ver abaixo)
│   ├── animation.py          StepPlayer (QTimer, play/pause/step)
│   ├── state.py              modo palestra / modo professor
│   └── math_utils.py         seed determinística, interpolação (tweens)
├── backprop/
│   ├── activation.py         sigmoide + derivada
│   └── demos/                4 demonstrações (ver tabela abaixo)
│       ├── traditional_gd.py     forward/backward clássicos + convergência
│       ├── multilayer_network.py rede 3-2-2-1, um neurônio de cada vez
│       ├── matrix_algebra.py     a mesma rede como vetores/matrizes
│       └── chain_rule_layers.py  rede 1-1-1: um bloco, um fator da cadeia
├── bitnet/
│   ├── quantization.py       Q(w) ternário didático
│   ├── ste.py                Straight-Through Estimator (numpy)
│   ├── ste_torch_reference.py versão PyTorch de referência (opcional)
│   ├── linear.py             neurônio linear mínimo + perda
│   └── demos/                4 demonstrações (ver tabela abaixo)
├── snn/
│   ├── lif.py                neurônio LIF (integração de Euler)
│   ├── surrogate.py          função de disparo + gradiente substituto
│   ├── encoding.py           sinal sintético + spike por cruzamento de nível + ruído estrutural
│   ├── tdbn.py                normalização dependente de limiar (tdBN)
│   ├── rate_reg.py            regularização de taxa de disparo
│   ├── normalization.py       z-score por característica/janela + risco de vazamento
│   └── demos/                11 demonstrações (ver tabela abaixo)
├── paraconsistent/
│   ├── metrics.py             alpha/beta -> plano paraconsistente -> D_truth/D_penalized
│   ├── ga_synthetic.py        populações sintéticas + dominância restrita de Deb (NSGA-II)
│   └── demos/                3 demonstrações (ver tabela abaixo)
├── comparison/
│   ├── ann_bitnet_snn.py     comparação lado a lado
│   ├── autoencoder_synthetic.py  janelas sintéticas + PCA real + constantes do Meeting01
│   └── autoencoders.py       comparação SNN-AE x LSTM-AE x GRU-AE x Transformer-AE
├── resources/                imagem usada pela codificação Poisson
└── widgets/
    ├── signal_view.py        sinal/corrente + potencial + raster de spikes
    ├── weight_view.py        reta numérica, escada de quantização, curvas
    ├── neuron_view.py        diagramas de blocos, matrizes e tabelas
    ├── controls.py           Reset/Step/Play/Pause/velocidade/sliders
    └── _mpl_perf.py          limpeza rápida de eixos (custo por quadro)
```

### O contrato `DemoModule` (`core/demo.py`)

Cada demonstração pré-computa, de forma determinística, a sequência
completa de `Frame`s a partir dos parâmetros atuais (sem aleatoriedade —
ver `core/math_utils.seed_everything`). Passo adiante/atrás apenas move um
ponteiro nessa lista; isso torna "Anterior" e "Próximo" triviais e
simétricos, e mantém a matemática 100% testável sem depender do Qt.

```python
class DemoModule(ABC):
    title: str
    description: str
    def initialize(self) -> None: ...
    def reset(self) -> None: ...
    def play(self) -> None: ...
    def pause(self) -> None: ...
    def step_forward(self) -> None: ...
    def step_backward(self) -> None: ...
    current_step: int
    total_steps: int
```

Novas demonstrações só precisam implementar `_build_frames()` — o resto
(navegação, play/pause, contrato) é herdado.

## Demonstrações

| Módulo | Pergunta única respondida | Fixo/configurável |
|---|---|---|
| Backprop → Forward e backward clássicos | Como o forward/backward funcionam sem quantização, e o exemplo converge de fato? | `target`, taxa de aprendizado |
| Backprop → Rede 3-2-2-1 | Como o forward/backward funcionam numa rede de verdade (3 entradas, duas camadas ocultas de 2, 1 saída; a entrada não conta como camada)? Um neurônio de cada vez, com entradas, saída, pesos e equação de cada um — e **cada neurônio com seu próprio gráfico** de sigmoide/derivada (5 gráficos, sempre visíveis, atualizando independentemente conforme o forward/backward avança). | `target`, taxa de aprendizado |
| Backprop → A rede como matrizes | Em que sentido a rede inteira é **só** multiplicação de matrizes — inclusive o backward? Liga cada célula de `W` à seta correspondente do grafo (`W[i,j]` *é* aquela seta), faz o forward `z = Wx` termo a termo, o backward pelos mesmos pesos transpostos (`Wᵀ`), e fecha conferindo a regra da cadeia de um peso contra `grad_W1[H1,x1]`. Um passo por operação escalar, sem agrupar nada. | `target` |
| Backprop → Camadas e a regra da cadeia | De onde sai cada fator da regra da cadeia? Cada camada aparece como **dois blocos** (operação linear `w·entrada + b` e ativação `σ`) e embaixo de cada bloco fica a sua derivada local, na mesma coluna: um bloco, um fator. O produto acumulado δ desce a fila da direita para a esquerda, e cada parâmetro (`w1`, `b1`, `w2`, `b2`) pega o δ da sua camada e multiplica pela sua própria derivada local — inclusive os vieses, cuja derivada local vale 1, que é de onde sai `∂L/∂b = δ`. Fecha com os cinco fatores de `∂L/∂w1` em fila, multiplicados um por vez. | `target` |
| BitNet → Quantização | O que significa quantizar um peso? | `w`, `tau` |
| BitNet → Forward | O que acontece no forward, e quão longe do alvo? | `x1,x2,w1,w2,target` |
| BitNet → Backward → STE | Por que o backward é problemático (com o gráfico da derivada real vs. a do STE), e como o STE resolve? O gradiente que chega ao quantizador já inclui o fator x (`∂L/∂Q(w) = ∂L/∂y · x`), o mesmo número do exemplo guiado. | `w`, `target`, `tau` |
| BitNet → Exemplo guiado | Sequência fixa "Do peso real ao BitNet" (10 passos) | fixo |
| SNN → Sinal e spikes | O que é um spike? | nível de disparo |
| SNN → Codificação Poisson | A informação pode estar na *probabilidade* de disparo, não só no instante exato? | `max_rate` |
| SNN → Codificação Poisson (imagem) | Como fica a esparsidade num caso real, pixel a pixel — e por que só a soma de vários passos reconstrói a imagem? A estimativa (spikes até t ÷ (t+1) ÷ taxa máxima) aparece ao lado do original, com o erro médio caindo a cada passo. | `max_rate` |
| SNN → LIF | Como um neurônio LIF integra, dispara e reseta — e, quando a corrente desliga, como o vazamento traz o potencial de volta ao repouso? | `tau, R, V_th`, amplitude |
| SNN → Surrogate gradient | Como se treina através de uma função em degrau? A sigmoide suave vai de 0 a 1 e fica mais íngreme com `k`; o gradiente substituto é a inclinação exata dela (área 1). | `k` |
| SNN → time_steps x delta_t | Por que confundir "quantos quadros" com "quanto dura um quadro" quebra o treino, e por que a ORDEM das linhas no tensor `(T*B,F)` importa? | `n_samples` |
| SNN → Ruído estrutural da codificação | "Direta > latência > Poisson" em reconstrução é uma diferença real de informação, ou só o chão de ruído de cada codificação? | `time_steps` |
| SNN → Perda incompatível com a codificação | Por que a perda precisa medir o lugar onde a codificação guarda o valor? Com a perda errada o treino reporta "perfeito" sem corrigir nada; com a certa, uma unidade que nunca dispara trava sem gradiente — duas falhas silenciosas, e a validação que as torna barulhentas. | fixo |
| SNN → Normalização dependente de limiar (tdBN) | Como tdBN reescala a corrente de entrada para um espalhamento proporcional ao limiar `V_th`, em vez de variância unitária? | `V_th` |
| SNN → Regularização de taxa de disparo | A penalidade olha a taxa MÉDIA de cada camada: como ela empurra de volta à faixa uma camada quase morta e uma em rajada, por que dentro da faixa não faz nada — e por que uma camada que mistura uma unidade morta e uma em rajada passa despercebida? | `lambda`, `r_min` |
| SNN → Normalização: por característica x por janela | Por que áudio normaliza cada característica com estatísticas ajustadas uma vez no treino e EEG recalcula a cada janela — e por que só o caminho ajustado pode vazar dados de teste? | fixo |
| Paraconsistente → Plano paraconsistente | Dado o quão compacta (`alpha`) e sobreposta (`beta`) cada classe é, onde cai o ponto no plano paraconsistente, e quão perto de "Verdade" ele está? | `alpha`, `beta` |
| Paraconsistente → D_truth x D_penalized | Por que um extrator "morto" (sempre a mesma saída) passa à frente de extratores fracos porém reais na distância ingênua até "Verdade", e como `D_penalized` fecha essa brecha? | fixo |
| Paraconsistente → Busca genética de arquiteturas (extensão) | Como uma busca NSGA-II escolhe arquiteturas quando qualidade (`D_penalized`) e custo brigam — dominância, fronteira de Pareto, teto de latência pela regra de Deb, e por que o resultado é um cardápio e não um vencedor? Fora da monografia: é o experimento `paraconsistentGA` do software/nn, com dados sintéticos. | `latency_ceiling` |
| Comparação → ANN x BitNet x SNN | Em que ANN, BitNet e SNN diferem? | fixo |
| Comparação → Autoencoders (SNN x LSTM x GRU x Transformer) | Como o Meeting01 compara quatro famílias de autoencoder sem trapacear: mesmo gargalo (o tamanho do latente decide o placar), mesmo alvo (o bug B2), referências média e PCA? PCA de verdade sobre janelas sintéticas; as quatro famílias não recebem números inventados. | `latent` |

### A tabela da comparação se dimensiona sozinha

Essa é a única demo feita **só de texto**, e por isso é a única em que o
corpo da letra não é um número fixo no código. O canvas do matplotlib
dentro do Qt mantém o dpi e **cresce em polegadas** junto com a janela —
então um `fontsize=8` calibrado num canvas de 900x400 vira 8pt dentro de
uma figura de ~1000pt de altura quando o app está no projetor: encolhe,
relativamente, exatamente quando precisa ser legível (era a queixa
registrada no antigo `FIXME.md` da raiz do repositório, removido no commit
933133a3).

O tamanho agora vem da geometria (`NeuronView._cmp_geometry`): o menor
entre o que a **altura** permite (todas as linhas de texto mais os
espaços) e o que a **largura de uma coluna** permite (a célula já quebrada
em no máximo duas linhas). A altura que a letra não conseguiu usar vira
espaço entre as linhas, para a tabela preencher a caixa em vez de se
amontoar em cima. Medido no tamanho real da janela: ~22pt por célula, ~3x
o anterior. Os invariantes estão em `tests/test_comparison_table_layout.py`
(cresce com o canvas, nada se sobrepõe, preenche a caixa).

## Precisão científica (ESPECIFICACAO_DLVL.md #32)

- A quantização usada nas demos (`bitnet/quantization.py`) é um limiar fixo
  e simétrico — uma simplificação didática, **não** a quantização absmean
  real de BitNet b1.58 (que usa uma escala γ = média(|W|) por tensor).
- O STE aqui é a ideia central do estimador (forward quantizado, backward
  identidade), não uma cópia da implementação oficial. Só `dQ/dw` vira 1:
  os outros fatores da cadeia continuam, então com `y = x · Q(w)` o
  gradiente entregue ao peso real é `∂L/∂y · x`.
- O gradiente substituto da SNN é uma sigmoide rápida comum na literatura
  didática, análoga em espírito ao STE — não a mesma técnica, nem a mesma
  fórmula de nenhum artigo específico. O par é normalizado: a sigmoide
  `0,5 + 0,5·kx/(1 + k|x|)` vai de 0 a 1 como o degrau, e o substituto é a
  derivada exata dela, `(k/2)/(1 + k|x|)²`, com área 1. Muitas bibliotecas
  usam a mesma forma com pico 1, que difere só pelo fator constante `2/k`.
- A comparação ANN x BitNet x SNN compara peso com peso e ativação com
  ativação: os pesos de uma SNN são contínuos; o que é binário nela é a
  ativação (spikes). A BitNet b1.58 usa ativações de 8 bits.
- Nenhuma tela afirma que BitNet e SNN são equivalentes, nem que a
  eficiência energética é garantida pela arquitetura por si só — ver a
  tela **Comparação → Advertência sobre eficiência**.

## Referências

Ver a tela **Referências** dentro da aplicação. Resumo:

- Wang, H. et al. *BitNet: Scaling 1-bit Transformers for Large Language
  Models*. arXiv:2310.11453, 2023.
- Ma, S. et al. *The Era of 1-bit LLMs: All Large Language Models are in
  1.58 Bits*. arXiv:2402.17764, 2024.
- Neftci, E. O.; Mostafa, H.; Zenke, F. *Surrogate Gradient Learning in
  Spiking Neural Networks*. IEEE Signal Processing Magazine, v. 36, n. 6,
  p. 51-63, 2019. DOI: 10.1109/MSP.2019.2931595.
- Zheng, H.; Wu, Y.; Deng, L.; Hu, Y.; Li, G. *Going Deeper With
  Directly-Trained Larger Spiking Neural Networks*. Proceedings of the AAAI
  Conference on Artificial Intelligence, v. 35, n. 12, p. 11062-11070,
  2021. DOI: 10.1609/aaai.v35i12.17320. (tdBN)
- Deb, K.; Pratap, A.; Agarwal, S.; Meyarivan, T. *A Fast and Elitist
  Multiobjective Genetic Algorithm: NSGA-II*. IEEE Transactions on
  Evolutionary Computation, v. 6, n. 2, p. 182-197, 2002.
  DOI: 10.1109/4235.996017. (NSGA-II e dominância com restrições)
- Eckart, C.; Young, G. *The Approximation of One Matrix by Another of
  Lower Rank*. Psychometrika, v. 1, n. 3, p. 211-218, 1936.
  DOI: 10.1007/BF02288367. (PCA como melhor compressor linear)

As seis referências foram verificadas por resolução de DOI/busca antes da
inclusão (metadados conferidos, não apenas lembrados). As três últimas
entraram porque o texto das demos atribui afirmações a elas.

## Testes

```bash
./.venv/bin/pip install -e .[dev]
QT_QPA_PLATFORM=offscreen ./.venv/bin/python -m pytest
```

`QT_QPA_PLATFORM=offscreen` permite rodar os testes que exercitam os
widgets Qt (`tests/test_widgets_render.py`) sem display — útil em CI ou
sessões remotas. `tests/conftest.py` já define isso por padrão.

Cobertura:

- `test_backprop.py` — forward/backward clássicos: números do passo único
  batem com a conta manual, e a sequência de gradiente descendente converge
  de fato (distância ao alvo cai monotonicamente até < 0,05), sem saltos
  instantâneos entre iterações; o passo a passo e o gráfico contam o mesmo
  número de atualizações. Também as demos matricial, 3-2-2-1 e da regra da
  cadeia (um número novo por passo, cadeia = gradiente matricial).
- `test_bitnet.py` — quantização, STE (incluindo o gráfico da derivada real
  morfando na constante que o STE usa), neurônio linear, perda, e os
  números exatos dos exemplos da especificação (y=2, loss=2, w: 0,80→0,84).
  O gradiente do STE (`∂L/∂w = -4`) é o mesmo na demo do backward e no
  exemplo guiado, e todo texto que depende da posição de w segue o slider.
- `test_snn.py` — integração LIF (nunca dispara sem corrente, dispara e
  reseta com corrente suficiente, nunca excede o limiar, o vazamento
  aparece quando a corrente desliga), par sigmoide/substituto (degrau suave
  de 0 a 1, área 1, derivada exata), codificação por cruzamento de nível,
  chãos de ruído conferidos por Monte Carlo, e o ponto cego da
  regularização pela média da camada.
- `test_surrogate_example_markers.py`, `test_math_render.py`,
  `test_comparison_table_layout.py`, `test_widgets_no_clipping.py`,
  `test_chain_layout.py`, `test_main_window.py` — exemplo numérico do
  gradiente substituto, equações tipografáveis, legibilidade da tabela,
  nada desenhado fora do canvas, layout da regra da cadeia e navegação.
- `test_demo_interface.py` — contrato genérico: toda demonstração começa
  no passo 0, não ultrapassa os limites, é determinística ao resetar.
- `test_widgets_render.py` — todo frame de toda demonstração renderiza sem
  lançar exceção no widget para o qual é roteado.

## Desvios deliberados da especificação

- **Duração das animações (#27, 3 a 15 s).** As demos de backprop são
  passo a passo por decisão didática: `Backprop → Forward e backward
  clássicos` repete o ciclo de 9 passos até convergir (fixado em
  `test_pipeline_cycle_repeats_until_close_enough_to_target`), e as demos
  matricial e da regra da cadeia revelam um número por passo. No modo
  automático elas passam bem de 15 s.
- **Tabela da comparação (#21).** A linha "Pesos" diz que os pesos da SNN
  são contínuos em qualquer precisão, como a especificação. A linha
  "Potencial de eficiência" virou a advertência final, porque a própria
  especificação proíbe tratar eficiência como propriedade da arquitetura.
  "Domínio temporal" diz "normalmente ausente" para ANN e BitNet
  feedforward, em vez de "não explícito".
- **Contagem de camadas.** A entrada não conta como camada (não tem pesos
  nem ativação), como no slide `fundamentosArquitetura.tex`. Por isso a
  demo antes chamada "Rede de 4 camadas" é `Backprop → Rede 3-2-2-1`.

## Escopo negativo (ESPECIFICACAO_DLVL.md #37)

Não implementado nesta versão, deliberadamente: treinamento de modelos
grandes, benchmark energético real, CUDA obrigatório, hardware
neuromórfico, LLMs completos, treinamento distribuído, quantização de
modelos externos, inferência de LLM real.

## Referência opcional em PyTorch

`bitnet/ste_torch_reference.py` reproduz o STE com autograd real (não é
usado pela interface gráfica). Requer o extra opcional:

```bash
./.venv/bin/pip install -e .[torch-reference]
./.venv/bin/python -m efficient_nn_lab.bitnet.ste_torch_reference
```
