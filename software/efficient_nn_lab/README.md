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

Implementa a especificação em [`ESPECIFICACAO_DLVL.md`](../../ESPECIFICACAO_DLVL.md).

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
| `Esc` | voltar ao menu |

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
├── main.py, __main__.py      ponto de entrada
├── app/
│   ├── main_window.py        janela principal, roteamento demo -> widget
│   └── theme.py               cores e stylesheet (mesma paleta dos slides)
├── core/
│   ├── demo.py                 contrato DemoModule + Frame (ver abaixo)
│   ├── animation.py            StepPlayer (QTimer, play/pause/step)
│   ├── state.py                 modo palestra / modo professor
│   └── math_utils.py           seed determinística, interpolação
├── backprop/
│   └── demos/
│       └── traditional_gd.py   forward/backward clássicos + convergência
├── bitnet/
│   ├── quantization.py         Q(w) ternário didático
│   ├── ste.py                    Straight-Through Estimator (numpy)
│   ├── ste_torch_reference.py  versão PyTorch de referência (opcional)
│   ├── linear.py                 neurônio linear mínimo + perda
│   └── demos/                    4 demonstrações (ver tabela abaixo)
├── snn/
│   ├── lif.py                    neurônio LIF (integração de Euler)
│   ├── surrogate.py             função de disparo + gradiente substituto
│   ├── encoding.py               sinal sintético + spike por limiar direto
│   └── demos/                    3 demonstrações
├── comparison/
│   └── ann_bitnet_snn.py        comparação lado a lado
└── widgets/
    ├── signal_view.py           sinal/corrente + potencial + raster de spikes
    ├── weight_view.py            reta numérica, escada de quantização, curvas
    ├── neuron_view.py            diagramas de blocos e tabelas
    └── controls.py                 Reset/Step/Play/Pause/velocidade/sliders
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
| Backprop → Rede de 4 camadas | Como o forward/backward funcionam numa rede de verdade (3→2→2→1)? Um neurônio de cada vez, com a curva sigmoide/derivada, entradas, saída, pesos e equação de cada um. | `target`, taxa de aprendizado |
| BitNet → Quantização | O que significa quantizar um peso? | `w`, `tau` |
| BitNet → Forward | O que acontece no forward, e quão longe do alvo? | `x1,x2,w1,w2,target` |
| BitNet → Backward → STE | Por que o backward é problemático (com o gráfico da derivada real vs. a do STE), e como o STE resolve? | `tau` |
| BitNet → Exemplo guiado | Sequência fixa "Do peso real ao BitNet" (10 passos) | fixo |
| SNN → Sinal e spikes | O que é um spike? | nível de disparo |
| SNN → LIF | Como um neurônio LIF integra, dispara e reseta? | `tau, R, V_th`, amplitude |
| SNN → Surrogate gradient | Como se treina através de uma função em degrau? | `k` |
| Comparação | Em que ANN, BitNet e SNN diferem? | fixo |

## Precisão científica (ESPECIFICACAO_DLVL.md #32)

- A quantização usada nas demos (`bitnet/quantization.py`) é um limiar fixo
  e simétrico — uma simplificação didática, **não** a quantização absmean
  real de BitNet b1.58 (que usa uma escala γ = média(|W|) por tensor).
- O STE aqui é a ideia central do estimador (forward quantizado, backward
  identidade), não uma cópia da implementação oficial.
- O gradiente substituto da SNN é uma sigmoide rápida comum na literatura
  didática, análoga em espírito ao STE — não a mesma técnica, nem a mesma
  fórmula de nenhum artigo específico.
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

Todas as três referências foram verificadas por resolução de DOI/busca
antes da inclusão (metadados conferidos, não apenas lembrados).

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
  de fato (distância ao alvo cai monotonicamente até < 0,1), sem saltos
  instantâneos entre iterações.
- `test_bitnet.py` — quantização, STE (incluindo o gráfico da derivada real
  morfando na constante que o STE usa), neurônio linear, perda, e os
  números exatos dos exemplos da especificação (y=2, loss=2, w: 0,80→0,84).
- `test_snn.py` — integração LIF (nunca dispara sem corrente, dispara e
  reseta com corrente suficiente, nunca excede o limiar), forma do
  gradiente substituto, codificação por limiar direto.
- `test_demo_interface.py` — contrato genérico: toda demonstração começa
  no passo 0, não ultrapassa os limites, é determinística ao resetar.
- `test_widgets_render.py` — todo frame de toda demonstração renderiza sem
  lançar exceção no widget para o qual é roteado.

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
