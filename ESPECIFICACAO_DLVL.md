# Especificação — Laboratório Visual de BitNet e SNN

**Arquivo:** `ESPECIFICACAO_DLVL.md`  
**Objetivo:** fornecer a um agente de programação uma especificação executável, modular e didática para implementar um software de demonstração visual de redes neurais eficientes, inicialmente com BitNet e Spiking Neural Networks (SNNs).

---

## 1. Visão geral

Criar uma aplicação desktop interativa para uso em palestra universitária de aproximadamente uma hora sobre **BitNet, SNNs, Deep Learning eficiente e eficiência energética**.

A aplicação não substitui os slides em LaTeX/Beamer. Ela deve funcionar como um **laboratório visual complementar**: durante a apresentação, o palestrante abre a demonstração correspondente ao conceito explicado no slide.

O público-alvo é composto por estudantes universitários de Computação. Portanto:

- priorizar intuição visual;
- evitar excesso de parâmetros simultâneos;
- mostrar equações somente quando elas ajudam a explicação;
- usar exemplos numéricos pequenos;
- permitir execução passo a passo;
- usar animações curtas e determinísticas;
- separar claramente treinamento, forward e backward;
- não afirmar que BitNet e SNN são equivalentes.

---

# 2. Escolha tecnológica

## 2.1 Tecnologia principal

Usar **Python**.

### Justificativa

Python é preferível ao JavaFX para este projeto porque oferece um ecossistema mais adequado para:

- computação científica;
- NumPy;
- PyTorch;
- Matplotlib;
- animações;
- visualização de sinais;
- prototipagem rápida;
- futura integração com modelos reais.

A aplicação deve, entretanto, ser arquitetada de maneira independente da biblioteca gráfica específica.

## 2.2 Stack sugerida

- Python >= 3.11
- NumPy
- PySide6 para interface desktop
- Matplotlib para gráficos científicos e animações
- PyTorch apenas no módulo opcional de treinamento real
- pytest para testes
- dataclasses e typing da biblioteca padrão

Evitar dependências desnecessárias.

---

# 3. Arquitetura

Estrutura sugerida:

```text
efficient_nn_lab/
├── pyproject.toml
├── README.md
├── src/
│   └── efficient_nn_lab/
│       ├── main.py
│       ├── app/
│       │   ├── main_window.py
│       │   └── theme.py
│       ├── core/
│       │   ├── demo.py
│       │   ├── state.py
│       │   ├── animation.py
│       │   └── math_utils.py
│       ├── bitnet/
│       │   ├── quantization.py
│       │   ├── ste.py
│       │   ├── linear.py
│       │   └── demos/
│       │       ├── scalar_quantization.py
│       │       ├── forward.py
│       │       └── backward.py
│       ├── snn/
│       │   ├── lif.py
│       │   ├── surrogate.py
│       │   ├── encoding.py
│       │   └── demos/
│       │       ├── spike_generation.py
│       │       ├── lif_dynamics.py
│       │       └── surrogate_gradient.py
│       ├── comparison/
│       │   └── ann_bitnet_snn.py
│       └── widgets/
│           ├── neuron_view.py
│           ├── signal_view.py
│           ├── weight_view.py
│           └── controls.py
└── tests/
```

---

# 4. Interface comum de demonstrações

Cada demonstração deve implementar uma interface conceitual:

```python
class DemoModule:
    def initialize(self) -> None: ...
    def reset(self) -> None: ...
    def play(self) -> None: ...
    def pause(self) -> None: ...
    def step_forward(self) -> None: ...
    def step_backward(self) -> None: ...
```

Também fornecer:

```python
title: str
description: str
current_step: int
total_steps: int
```

A interface gráfica deve tratar qualquer demonstração por essa interface, permitindo adicionar novos módulos sem alterar o núcleo da aplicação.

---

# 5. Princípios didáticos

Cada animação deve responder a uma única pergunta.

Exemplos:

1. O que significa quantizar um peso?
2. Por que BitNet usa valores discretos?
3. O que acontece no forward?
4. Por que o backward é problemático?
5. Como o Straight-Through Estimator resolve o problema?
6. O que é um spike?
7. Como um neurônio LIF gera spikes?
8. Como uma SNN transmite informação?
9. Como o gradiente substituto funciona?
10. Em que BitNet e SNN diferem?

Não criar uma animação que tente responder todas essas perguntas simultaneamente.

---

# 6. Módulo BitNet

## 6.1 Objetivo

Mostrar intuitivamente a ideia de pesos de baixa precisão, especialmente a família **BitNet b1.58**, na qual os pesos quantizados utilizados pela operação são ternários:

\[
W_q \in \{-1,0,+1\}.
\]

Importante: explicar que isso não significa que todo o estado interno usado durante o treinamento seja armazenado exclusivamente nesses três valores. O treinamento mantém parâmetros de maior precisão para atualização.

---

# 7. Demonstração 1 — Quantização escalar

Usar um único peso.

Exemplo inicial:

\[
w=0.80.
\]

Mostrar visualmente:

```text
peso real
   |
   v
  0.80
   |
quantização
   |
   v
 +1
```

Animar o valor deslizando para o nível quantizado.

Testar exemplos:

```text
-0.80 -> -1
-0.20 ->  0
 0.20 ->  0
 0.80 -> +1
```

A interface deve permitir alterar `w` com um slider.

---

# 8. Quantização ternária

Para fins didáticos, usar uma função explícita de quantização com escala.

Uma forma simplificada:

\[
\hat W = \alpha Q(W),
\]

onde

\[
Q(w)\in\{-1,0,+1\}.
\]

Para a demonstração, usar limiares configuráveis.

Por exemplo:

\[
Q(w)=
\begin{cases}
-1,&w<-\tau\\
0,&-\tau\le w\le\tau\\
+1,&w>\tau
\end{cases}
\]

com \(\tau=0.5\) como valor inicial.

**Importante:** esta função é uma visualização didática. Não apresentá-la como a especificação completa de todas as variantes de BitNet.

---

# 9. Demonstração 2 — Forward

Usar um neurônio linear mínimo.

Entrada:

\[
x_1=2,\qquad x_2=3.
\]

Pesos reais:

\[
w_1=0.8,\qquad w_2=0.2.
\]

Após quantização:

\[
\hat w_1=+1,\qquad \hat w_2=0.
\]

Forward:

\[
y=x_1\hat w_1+x_2\hat w_2
\]

\[
y=2(1)+3(0)=2.
\]

Mostrar graficamente:

```text
x1 = 2 ----> [ +1 ] ----\
                         \
                          > soma ---> y = 2
                         /
x2 = 3 ----> [  0 ] ----/
```

Animar cada multiplicação.

---

# 10. Demonstração 3 — erro

Definir alvo:

\[
t=4.
\]

Saída:

\[
y=2.
\]

Erro quadrático:

\[
L=\frac12(y-t)^2.
\]

Portanto:

\[
L=\frac12(2-4)^2=2.
\]

Mostrar:

```text
target = 4
output = 2

       diferença
           |
           v
          -2
           |
           v
         loss = 2
```

---

# 11. Demonstração 4 — problema do backward

Mostrar a função de quantização como uma escada.

Eixo x:

```text
peso real
```

Eixo y:

```text
peso quantizado
```

Mostrar os três níveis:

```text
 +1 ───────────────
        |
  0 ────┼──────────
        |
 -1 ───────────────
```

Explicar visualmente que a derivada da função discreta é inadequada para backpropagation convencional.

Não afirmar simplesmente que “a derivada é sempre zero” sem contexto: destacar que a quantização possui regiões constantes e pontos de descontinuidade, tornando o gradiente convencional inútil ou inadequado para otimização direta.

---

# 12. Demonstração 5 — Straight-Through Estimator

Mostrar dois caminhos:

```text
FORWARD

peso real
   |
quantização
   |
peso ternário
   |
operação
```

e:

```text
BACKWARD

loss
 |
gradiente
 |
STE
 |
peso real
```

Representação visual:

```text
             FORWARD
w_real -----------------> Q(w) ----------> y
  ^                         |
  |                         |
  |                         v
  +------- STE <--------- loss
             BACKWARD
```

Mensagem central:

> O forward utiliza o valor quantizado; o backward utiliza uma aproximação da derivada para permitir que o gradiente alcance os parâmetros de alta precisão.

---

# 13. Exemplo mínimo de STE em Python

Implementar uma função educacional:

```python
import torch

def ternary_quantize(w, threshold=0.5):
    q = torch.zeros_like(w)
    q = torch.where(w > threshold, torch.ones_like(w), q)
    q = torch.where(w < -threshold, -torch.ones_like(w), q)
    return q

def ste_quantize(w, threshold=0.5):
    q = ternary_quantize(w, threshold)
    return w + (q - w).detach()
```

Explicar:

```python
w + (q - w).detach()
```

produz:

- forward: `q`;
- backward: gradiente em relação a `w` aproximadamente como identidade.

Isso é uma implementação didática do STE, não uma reprodução integral de uma implementação oficial de BitNet.

---

# 14. Exemplo escalar de treinamento

Usar:

```python
import torch

w = torch.tensor(0.8, requires_grad=True)

target = torch.tensor(4.0)
x = torch.tensor(2.0)

q = ste_quantize(w)
y = x * q

loss = 0.5 * (y - target) ** 2

loss.backward()

print("w real:", w.item())
print("w quantizado:", q.item())
print("y:", y.item())
print("loss:", loss.item())
print("grad:", w.grad.item())
```

A aplicação deve apresentar esses valores em tempo real.

---

# 15. Demonstração visual do “peso oculto”

Usar uma metáfora visual:

```text
        parâmetro contínuo
              |
              v
        [   0.80   ]
              |
              | quantização
              v
        [    +1    ]
              |
              v
        operação neural
```

No backward:

```text
loss
 |
 v
gradiente
 |
 v
STE
 |
 v
[ 0.80 ]  ---> [ 0.84 ]
```

Mostrar claramente que `0.84` continua sendo o parâmetro real, mesmo que o forward ainda produza `+1`.

---

# 16. Módulo SNN

## 16.1 Objetivo

Introduzir SNNs sem assumir conhecimento prévio de neurociência.

Sequência:

1. sinal contínuo;
2. integração;
3. limiar;
4. spike;
5. reset;
6. repetição no tempo.

---

# 17. Demonstração SNN 1 — sinal e spikes

Mostrar um sinal de entrada:

\[
x(t)
\]

e um trem de spikes:

\[
s(t).
\]

Usar duas áreas gráficas:

```text
Entrada

 amplitude
    |
    |       /\
    |  /\  /  \__
    |_/  \/       \____
    +--------------------> tempo


Spikes

  1 |     |   | |
    |     |   | |
  0 +-----+---+-+--------> tempo
```

Animar o sinal da esquerda para a direita.

---

# 18. Demonstração SNN 2 — neurônio LIF

Usar o modelo:

\[
\tau\frac{dV}{dt}=-(V-V_{\mathrm{rest}})+RI(t).
\]

Quando:

\[
V(t)\ge V_{\mathrm{th}},
\]

gerar spike e resetar:

\[
V\leftarrow V_{\mathrm{reset}}.
\]

Não mostrar a equação diferencial como elemento central inicialmente.

Primeiro mostrar a animação:

```text
potencial
   |
Vth|--------------------
   |          /\
   |       __/  \__
   |    __/
   |___/
   +--------------------> tempo
              spike
```

---

# 19. Demonstração SNN 3 — integração temporal

Mostrar pequenos incrementos:

```text
t0: V = 0.20
t1: V = 0.35
t2: V = 0.55
t3: V = 0.82
t4: V = 1.05 -> SPIKE
t5: V = reset
```

Cada passo deve ser executável com botão `Step`.

---

# 20. Demonstração SNN 4 — surrogate gradient

Mostrar o problema:

```text
spike
  ^
  |       ______
  |      |
  |______|
         threshold
```

Depois mostrar uma aproximação suave apenas para o backward:

```text
gradiente
  ^
  |        /\
  |       /  \
  |______/    \____
          threshold
```

Explicar:

> O neurônio continua produzindo spikes discretos no forward. Durante o treinamento, uma função diferenciável aproximada é usada para calcular o gradiente.

Destacar a analogia conceitual com STE, mas também mostrar que os problemas matemáticos e as funções utilizadas não são idênticos.

---

# 21. Comparação BitNet × SNN

Criar um módulo comparativo.

| Característica | ANN convencional | BitNet | SNN |
|---|---|---|---|
| Pesos | FP32/BF16 etc. | baixa precisão, p.ex. ternários | pode usar várias precisões |
| Ativação | contínua | geralmente quantizada conforme arquitetura | spikes |
| Domínio temporal | normalmente não explícito | normalmente não explícito | explícito |
| Backprop direto | sim | usa técnicas de quantização/STE | usa surrogate gradients |
| Potencial de eficiência | alto | alto | alto, especialmente em hardware neuromórfico |
| Operação principal | MAC | operações de baixa precisão | eventos/spikes |

Não apresentar eficiência energética como garantida apenas pela arquitetura. O ganho depende de hardware, implementação, memória, largura de banda, sparsidade, algoritmo e workload.

---

# 22. Comparação visual de operações

Mostrar:

```text
ANN

x1*w1 + x2*w2 + x3*w3 + ...


BitNet

x1*(−1/0/+1)
+ x2*(−1/0/+1)
+ ...


SNN

spike ──> evento
spike ──> evento
         |
         v
       LIF
         |
       spike
```

O objetivo é destacar a diferença operacional, não produzir uma estimativa quantitativa universal de energia.

---

# 23. Demonstração integrada

Criar uma cena final contendo três neurônios:

```text
┌────────────┐
│ ANN        │
│ FP weights │
└────────────┘

┌────────────┐
│ BitNet     │
│ -1,0,+1    │
└────────────┘

┌────────────┐
│ SNN        │
│ spikes     │
└────────────┘
```

Executar a mesma entrada conceitual nos três.

Mostrar:

- representação;
- operação;
- saída;
- treinamento;
- tipo de gradiente.

---

# 24. Interface

Layout principal:

```text
┌──────────────────────────────────────────────┐
│ Efficient Neural Networks Lab                │
├───────────────┬──────────────────────────────┤
│ DEMONSTRAÇÕES │                              │
│               │                              │
│ BitNet        │       ÁREA DE ANIMAÇÃO       │
│  Quantização  │                              │
│  Forward      │                              │
│  Backward     │                              │
│  STE          │                              │
│               │                              │
│ SNN           │                              │
│  Spikes       │                              │
│  LIF          │                              │
│  Surrogate    │                              │
│               │                              │
│ Comparação    │                              │
│               │                              │
├───────────────┴──────────────────────────────┤
│ [Reset] [Step] [Play] [Pause]               │
│ Velocidade: ─────●────                       │
└──────────────────────────────────────────────┘
```

---

# 25. Controles

Todos os módulos devem possuir:

- `Reset`
- `Step`
- `Play`
- `Pause`
- controle de velocidade
- controle de parâmetros relevantes
- botão `Show equation`
- botão `Show explanation`

Os parâmetros devem aparecer somente quando forem relevantes à demonstração.

---

# 26. Cores e semântica visual

Não depender exclusivamente de cor.

Usar também:

- forma;
- espessura;
- posição;
- rótulos;
- setas;
- animação.

Sugestão semântica:

- peso real: representação contínua;
- peso quantizado: bloco discreto;
- spike: linha vertical;
- gradiente: seta;
- loss: destaque textual;
- operação: bloco matemático.

Garantir contraste suficiente.

---

# 27. Animações

As animações devem ser:

- determinísticas;
- curtas;
- pausáveis;
- reiniciáveis;
- executáveis passo a passo.

Evitar animações decorativas.

Cada animação deve ter entre aproximadamente 3 e 15 segundos no modo automático.

---

# 28. Modo palestra

Implementar um modo simplificado:

```text
[◀ Anterior] [Próximo ▶]
```

O palestrante deve conseguir controlar a demonstração sem teclado especializado.

Atalhos:

```text
Space = play/pause
Right = próximo passo
Left = passo anterior
R = reset
Esc = sair da demonstração
```

---

# 29. Modo professor

Adicionar opcionalmente:

- valores numéricos;
- equações;
- gradientes;
- parâmetros internos;
- estado completo.

O modo padrão deve esconder detalhes avançados.

---

# 30. Exemplo didático obrigatório

O software deve conter uma sequência automática chamada:

## “Do peso real ao BitNet”

Passos:

### Passo 1

```text
w = 0.80
```

### Passo 2

```text
Q(w) = +1
```

### Passo 3

```text
x = 2
```

### Passo 4

```text
y = 2 × 1 = 2
```

### Passo 5

```text
target = 4
```

### Passo 6

```text
loss = 2
```

### Passo 7

Mostrar gradiente.

### Passo 8

Mostrar STE.

### Passo 9

Atualizar:

```text
w = 0.80 -> 0.84
```

### Passo 10

Mostrar:

```text
Q(0.84) = +1
```

A mensagem didática é:

> O parâmetro contínuo mudou, mas a representação usada no forward pode continuar igual.

---

# 31. Exemplo didático obrigatório para SNN

Sequência:

### Passo 1

Entrada:

```text
I(t)
```

### Passo 2

Integração:

```text
V(t) aumenta
```

### Passo 3

Limiar:

```text
V >= Vth
```

### Passo 4

Spike:

```text
|
```

### Passo 5

Reset:

```text
V -> Vreset
```

### Passo 6

Repetir.

---

# 32. Precisão científica

O software deve diferenciar explicitamente:

- exemplo didático;
- equação simplificada;
- implementação representativa;
- implementação fiel a uma publicação.

Nunca apresentar uma simplificação como sendo a definição oficial de BitNet.

Especialmente:

- BitNet b1.58 deve ser descrita de acordo com a literatura correspondente;
- STE deve ser apresentado como estimador de gradiente;
- surrogate gradient em SNN deve ser apresentado como técnica de treinamento;
- consumo energético deve ser tratado como resultado dependente do hardware e do workload.

---

# 33. Evidência

O agente deve implementar uma tela “References” contendo referências bibliográficas.

Referência central para BitNet:

> Wang, H. et al. “BitNet: Scaling 1-bit Transformers for Large Language Models.” arXiv, 2023.

Para BitNet b1.58:

> Ma, S. et al. “The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits.” arXiv, 2024.

Para surrogate gradients/SNN:

> Neftci, E. O., Mostafa, H., Zenke, F. “Surrogate Gradient Learning in Spiking Neural Networks.” IEEE Signal Processing Magazine, 2019.

O agente deve verificar metadados bibliográficos antes de colocar DOI, URL ou detalhes adicionais.

---

# 34. Testes

Criar testes unitários para:

## BitNet

- quantização negativa;
- zero;
- quantização positiva;
- limiar;
- STE;
- forward;
- loss.

## SNN

- integração;
- threshold;
- spike;
- reset;
- surrogate gradient.

Exemplo:

```python
def test_ternary_quantization():
    assert ternary_quantize(torch.tensor(-0.8)).item() == -1
    assert ternary_quantize(torch.tensor(0.0)).item() == 0
    assert ternary_quantize(torch.tensor(0.8)).item() == 1
```

---

# 35. Reprodutibilidade

Fixar seeds quando houver aleatoriedade:

```python
import random
import numpy as np
import torch

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
```

As demonstrações principais devem funcionar sem aleatoriedade.

---

# 36. Performance

A aplicação deve ser capaz de executar as demonstrações em computador convencional sem GPU.

Prioridades:

1. responsividade da interface;
2. animação fluida;
3. baixa latência entre passos;
4. simplicidade.

Não utilizar treinamento de LLM real na demonstração.

---

# 37. Escopo negativo

Não implementar inicialmente:

- treinamento de modelos grandes;
- benchmark energético real;
- CUDA obrigatório;
- suporte a hardware neuromórfico;
- modelos de linguagem completos;
- distributed training;
- quantização de modelos externos;
- inferência de LLM real.

Esses recursos podem ser adicionados futuramente.

---

# 38. Extensibilidade

A arquitetura deve permitir futuramente:

```text
Efficient Neural Networks Lab
│
├── ANN
├── Quantization
├── BitNet
├── SNN
├── Binary Neural Networks
├── Ternary Neural Networks
├── Knowledge Distillation
├── Pruning
├── Low-rank models
└── Neuromorphic Computing
```

---

# 39. Integração com futuras SNNs

O módulo SNN deve ser preparado para futuramente suportar:

- LIF;
- IF;
- AdEx;
- Hodgkin-Huxley simplificado;
- Poisson encoding;
- rate coding;
- temporal coding;
- TTFS;
- surrogate gradients.

Não implementar todos inicialmente.

---

# 40. Requisitos de UX

Um estudante deve conseguir entender a demonstração sem ler documentação externa.

Cada módulo deve possuir:

```text
O que está acontecendo?
Por que isso é necessário?
Qual é a ideia principal?
```

Exemplo:

> **STE:** a quantização é discreta no forward, então usamos uma aproximação para permitir que o gradiente seja propagado durante o treinamento.

---

# 41. Roteiro da palestra

Organizar os módulos para acompanhar aproximadamente:

```text
00–10 min  Motivação: eficiência
10–20 min  Quantização
20–30 min  BitNet
30–40 min  Backpropagation + STE
40–50 min  SNN + LIF + surrogate gradient
50–57 min  BitNet × SNN
57–60 min  Conclusões
```

O software deve permitir navegar nessa ordem, mas não obrigar o palestrante a segui-la.

---

# 42. Relação com os slides LaTeX

O software deve utilizar nomes compatíveis com os títulos dos slides.

Exemplo:

```text
Slide:
"Quantização ternária"

Software:
BitNet → Quantização
```

```text
Slide:
"Straight-Through Estimator"

Software:
BitNet → Backward → STE
```

```text
Slide:
"Leaky Integrate-and-Fire"

Software:
SNN → LIF
```

Isso reduz o custo cognitivo durante a apresentação.

---

# 43. Critérios de aceitação

O projeto será considerado funcional quando:

- [ ] aplicação iniciar com um comando simples;
- [ ] menu principal apresentar BitNet, SNN e comparação;
- [ ] quantização ternária puder ser animada;
- [ ] forward puder ser executado passo a passo;
- [ ] loss puder ser visualizada;
- [ ] STE puder ser visualizado;
- [ ] peso real puder ser atualizado;
- [ ] LIF puder gerar spikes;
- [ ] threshold e reset forem visualizados;
- [ ] surrogate gradient puder ser comparado ao spike;
- [ ] BitNet e SNN puderem ser comparados;
- [ ] todas as animações tiverem Reset/Play/Pause/Step;
- [ ] modo palestra estiver funcional;
- [ ] documentação explicar cada demonstração;
- [ ] testes unitários passarem.

---

# 44. Princípio central de implementação

Não transformar o software em uma simulação excessivamente complexa.

A prioridade é:

```text
clareza > realismo
```

e:

```text
compreensão > quantidade de parâmetros
```

A aplicação deve tornar visível aquilo que normalmente fica escondido em uma equação ou em um algoritmo de treinamento.

---

# 45. Entregáveis

O agente de programação deve produzir:

1. código-fonte completo;
2. `README.md`;
3. `pyproject.toml`;
4. testes;
5. demonstrações BitNet;
6. demonstrações SNN;
7. módulo comparativo;
8. documentação das equações;
9. referências bibliográficas;
10. instruções de execução;
11. modo palestra;
12. screenshots ou GIFs opcionais para documentação.

---

# 46. Comando esperado

Preferencialmente:

```bash
python -m efficient_nn_lab
```

ou:

```bash
python src/efficient_nn_lab/main.py
```

Documentar claramente o método escolhido.

---

# 47. Regra final para o agente

Antes de implementar cada módulo:

1. identificar o conceito científico;
2. escrever a equação mínima necessária;
3. definir o estado interno;
4. definir o que será animado;
5. definir os controles;
6. implementar a lógica matemática;
7. implementar a visualização;
8. criar teste;
9. validar numericamente;
10. verificar se a animação continua compreensível sem explicação oral.

A implementação deve favorecer uma sequência didática de pequenos experimentos, e não uma única simulação monolítica.

**Resultado esperado:** um laboratório visual desktop, executável localmente, que permita explicar de forma intuitiva e tecnicamente correta como quantização, BitNet, backpropagation com STE, spikes, neurônios LIF e surrogate gradients funcionam, além de mostrar as diferenças conceituais entre BitNet e SNNs no contexto de Deep Learning eficiente.
