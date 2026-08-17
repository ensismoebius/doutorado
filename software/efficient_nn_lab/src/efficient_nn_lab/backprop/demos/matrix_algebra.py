"""Demonstração — a rede é, no fundo, multiplicação de matrizes.

As outras demos de backprop desenham a rede como caixas e setas e caminham
neurônio por neurônio. Isso ensina a mecânica, mas esconde uma coisa que
todo framework (PyTorch, TensorFlow) explora: **cada camada inteira é uma
única multiplicação matriz-vetor**, e o backward é a mesma multiplicação
com a matriz transposta. Quem só viu a versão "um neurônio de cada vez"
costuma achar que são dois assuntos diferentes -- grafo de um lado, álgebra
linear do outro.

Esta demo põe os dois lados na mesma tela e os liga elemento por elemento:

* o **grafo** fica à esquerda (2 entradas -> 2 neurônios ocultos -> 1
  saída, tudo sigmoide, sem viés);
* o **quadro de álgebra** fica à direita, com as mesmas quantidades como
  vetores e matrizes de verdade, com números dentro;
* destacar uma **célula** da matriz destaca, no mesmo instante, a **seta**
  correspondente do grafo -- porque `W[i, j]` *é* a seta que vai da entrada
  `j` para o neurônio `i`. Esse é o mapeamento que a demo existe para
  tornar óbvio (linha = neurônio de destino, coluna = de onde vem).

A caminhada é passo a passo, no nível do termo individual: o primeiro
produto escalar é montado um termo por vez (0,80·0,90, depois
-0,40·0,40), com a soma parcial contando na tela, antes de a célula de
`z` finalmente aparecer. Só depois de a mecânica estar visível é que os
passos seguintes tratam uma linha inteira de uma vez.

Fecha com a regra da cadeia aplicada a *esta* rede: o gradiente de um peso
concreto (`w11`, a seta x1 -> H1) montado fator por fator, com o produto
parcial atualizando a cada passo, terminando na conferência contra o valor
que o backward matricial já havia calculado -- os dois caminhos dão o mesmo
número, o que é exatamente a afirmação que a regra da cadeia faz.

Rede pequena de propósito: com 2x2 e 1x2 cabe cada número na tela, e cada
produto escalar tem só dois termos, então o passo a passo é curto o
bastante para ser assistido inteiro.

Pesos fixos e determinísticos (ESPECIFICACAO_DLVL.md #35): nada de
inicialização aleatória.
"""

from __future__ import annotations

import numpy as np

from efficient_nn_lab.backprop.activation import sigmoid, sigmoid_derivative
from efficient_nn_lab.bitnet.linear import loss_gradient_wrt_y, squared_error_loss
from efficient_nn_lab.core.demo import DemoModule, Frame, build_sequence

#: 2 entradas -> 2 ocultos (H1, H2) -> 1 saída (O). Escolhidos para que todos
#: os números intermediários saiam distintos e legíveis com duas casas: nenhum
#: produto escalar dá zero por acidente, nenhuma sigmoide satura, e -- para a
#: fase da regra da cadeia -- NENHUM fator vale 1,00. Um fator 1,00 deixaria o
#: produto parcial idêntico antes e depois daquele passo, e o passo pareceria
#: não ter feito nada (foi o que aconteceu com x1 = 1,00).
_X = np.array([0.9, 0.4])
_W1 = np.array([[0.8, -0.4], [-0.5, 0.9]])  # linha = H1/H2, coluna = x1/x2
_W2 = np.array([[1.2, -0.9]])  # linha = O, coluna = H1/H2

_ZERO_2 = np.zeros(2)
_ZERO_22 = np.zeros((2, 2))
_ZERO_12 = np.zeros((1, 2))
_N_CHAIN_FACTORS = 5


def _copy_state(state: dict[str, object]) -> dict[str, object]:
    """Snapshot of a frame's state, with arrays copied.

    The builder below mutates one running ``state`` dict instead of
    respelling forty fields per checkpoint. Every Frame therefore needs its
    *own* copy of the mutable arrays, or all frames would end up sharing
    (and showing) the final reveal state.
    """
    return {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in state.items()}


class MatrixAlgebraDemo(DemoModule):
    title = "Backprop -> A rede como matrizes"
    slug = "backprop.matrix"
    description = (
        "A mesma rede, dos dois jeitos ao mesmo tempo: grafo à esquerda, vetores e "
        "matrizes à direita. Destacar uma célula de W acende a seta correspondente do "
        "grafo -- W[i,j] é a seta da entrada j para o neurônio i. O forward é W · entrada "
        "(montado termo por termo, com a soma parcial contando na tela), o backward é a "
        "mesma matriz transposta, e a regra da cadeia fecha conferindo o gradiente de um "
        "peso concreto contra o que o backward matricial já calculou."
    )

    def __init__(self) -> None:
        self.target = 0.9
        super().__init__()

    def parameters(self) -> dict[str, dict[str, object]]:
        return {
            "target": {"label": "Alvo (0-1)", "min": 0.1, "max": 0.95, "step": 0.05, "value": self.target},
        }

    # -- the actual math: one forward pass, one full backward pass ---------
    def _forward_backward(self) -> dict[str, object]:
        x, target = _X, self.target

        z1 = _W1 @ x
        y1 = sigmoid(z1)
        z2 = float((_W2 @ y1)[0])
        y2 = float(sigmoid(z2))
        loss = squared_error_loss(y2, target)

        sp1 = sigmoid_derivative(z1)
        sp2 = float(sigmoid_derivative(z2))

        grad_y2 = loss_gradient_wrt_y(y2, target)
        grad_z2 = grad_y2 * sp2
        grad_w2 = np.outer(np.array([grad_z2]), y1)  # 1x2, mesma forma de W2
        grad_y1 = _W2.T @ np.array([grad_z2])
        grad_z1 = grad_y1 * sp1
        grad_w1 = np.outer(grad_z1, x)  # 2x2, mesma forma de W1

        return {
            "x": x, "w1": _W1, "w2": _W2, "target": target,
            "z1": z1, "y1": y1, "z2": z2, "y2": y2, "loss": loss,
            "sp1": sp1, "sp2": sp2,
            "gy2": grad_y2, "gz2": grad_z2, "gw2": grad_w2,
            "gy1": grad_y1, "gz1": grad_z1, "gw1": grad_w1,
        }

    def _build_frames(self) -> list[Frame]:
        c = self._forward_backward()
        x, w1, w2 = c["x"], c["w1"], c["w2"]
        z1, y1, z2, y2 = c["z1"], c["y1"], c["z2"], c["y2"]
        sp1, sp2 = c["sp1"], c["sp2"]
        gy2, gz2, gw2 = c["gy2"], c["gz2"], c["gw2"]
        gy1, gz1, gw1 = c["gy1"], c["gz1"], c["gw1"]
        target, loss = c["target"], c["loss"]

        # The five factors of dL/dw11 walked one at a time in the closing
        # phase. H1 feeds only O (single output neuron), so this weight has
        # exactly ONE path to the loss and the chain rule is a plain product
        # -- no sum over paths. (The 4-layer demo covers the multi-path case.)
        chain_names = ("dL/dy_O", "σ'(z_O)", "w_H1O", "σ'(z_H1)", "x1")
        chain_values = (float(gy2), float(sp2), float(w2[0, 0]), float(sp1[0]), float(x[0]))
        chain_partials = tuple(float(np.prod(chain_values[: i + 1])) for i in range(len(chain_values)))
        assert len(chain_values) == _N_CHAIN_FACTORS

        state: dict[str, object] = {
            "kind": "matrix_algebra",
            # -- the numbers themselves (constant across every frame) -----
            "x": x, "w1": w1, "w2": w2,
            "z1": z1, "y1": y1, "z2": z2, "y2": y2, "target": target, "loss": loss,
            "sp1": sp1, "sp2": sp2,
            "gy2": gy2, "gz2": gz2, "gw2": gw2, "gy1": gy1, "gz1": gz1, "gw1": gw1,
            "chain_names": chain_names, "chain_values": chain_values,
            # -- reveals: 0 = not computed yet, 1 = on screen. Per-cell, so
            # a matrix can fill in one entry at a time (numpy arrays are
            # interpolated element-wise by core/math_utils.tween_values).
            "rv_graph": 0.0,
            "rv_x": _ZERO_2.copy(), "rv_w1": _ZERO_22.copy(), "rv_w2": _ZERO_12.copy(),
            "rv_z1": _ZERO_2.copy(), "rv_y1": _ZERO_2.copy(),
            "rv_z2": 0.0, "rv_y2": 0.0, "rv_target": 0.0, "rv_loss": 0.0,
            "rv_gy2": 0.0, "rv_gz2": 0.0, "rv_gw2": _ZERO_12.copy(),
            "rv_gy1": _ZERO_2.copy(), "rv_gz1": _ZERO_2.copy(), "rv_gw1": _ZERO_22.copy(),
            "rv_chain": np.zeros(_N_CHAIN_FACTORS),
            # -- highlights: the mapping made mechanical. hl_w1[i, j] drives
            # BOTH the glow on that matrix cell AND the glow on the graph
            # edge x_j -> H_i, because they are the same weight.
            "hl_w1": _ZERO_22.copy(), "hl_w2": _ZERO_12.copy(),
            "hl_x": _ZERO_2.copy(), "hl_y1": _ZERO_2.copy(), "hl_out": 0.0,
            # -- which operation the right-hand board is showing -----------
            "focus": "l1",
            "board_title": "Camada 1:  z = W1 · x",
            # -- the worked arithmetic under everything, plus the running
            # partial sum (a real number, so it visibly counts as it moves)
            "work_text": "",
            "accum": 0.0, "rv_accum": 0.0,
            "chain_product": 0.0, "rv_chain_product": 0.0, "rv_check": 0.0,
        }

        frames: list[Frame] = []

        def snap(label: str, explanation: str, equation: str = "", **updates: object) -> None:
            for key, value in updates.items():
                if key not in state:
                    # No silent fallbacks (CLAUDE.md): a mistyped field name
                    # would otherwise quietly add a field no renderer reads,
                    # and the step would just not animate.
                    raise KeyError(f"unknown frame field {key!r}")
                state[key] = value
            frames.append(Frame(label, _copy_state(state), explanation, equation))

        def num(value: float, decimals: int = 2) -> str:
            return f"{value:+.{decimals}f}"

        # ============ fase 1: o mapeamento grafo <-> álgebra ==============
        snap(
            "A rede",
            "Duas entradas, dois neurônios ocultos (H1, H2), uma saída (O), tudo sigmoide e "
            "sem viés. À direita, o mesmo desenho esperando em forma de matriz: a caixa de W1 "
            "tem uma linha por neurônio de destino e uma coluna por entrada. Ainda vazia -- "
            "vamos preenchê-la ligando cada célula à sua seta.",
            equation="y = σ(W · entrada)",
            rv_graph=1.0,
            work_text="A rede tem 6 pesos: 4 da entrada para a camada oculta, 2 da oculta para a saída.\nCada peso é uma seta no grafo E uma célula numa matriz. São a mesma coisa.",
        )
        snap(
            "As entradas viram um vetor",
            f"x1 = {x[0]:.2f} e x2 = {x[1]:.2f} deixam de ser dois nós soltos e passam a ser um "
            "vetor coluna: um valor por linha, na mesma ordem dos nós de entrada. É só uma "
            "mudança de notação -- os números são idênticos.",
            equation="x = [x_1, x_2]",
            rv_x=np.ones(2), hl_x=np.ones(2),
            work_text=f"x = coluna [{x[0]:+.2f}; {x[1]:+.2f}]  <->  os dois nós de entrada do grafo,\nna mesma ordem de cima para baixo.",
        )
        snap(
            "Linha 1 de W1 = as setas que chegam em H1",
            "As duas setas que entram em H1 acabaram de acender no grafo, e com elas a linha 1 "
            f"da matriz: w11 = {w1[0, 0]:.2f} (de x1) e w12 = {w1[0, 1]:.2f} (de x2). Uma linha "
            "da matriz é o conjunto de pesos que chegam em UM neurônio.",
            equation="\\text{linha 1 de W1} = [w_11, w_12]",
            rv_w1=np.array([[1.0, 1.0], [0.0, 0.0]]),
            hl_w1=np.array([[1.0, 1.0], [0.0, 0.0]]), hl_x=_ZERO_2.copy(),
            work_text=f"w11 = {num(w1[0, 0])}  = seta x1 -> H1\nw12 = {num(w1[0, 1])}  = seta x2 -> H1\nlinha 1 da matriz = tudo que chega em H1.",
        )
        snap(
            "Linha 2 = as setas que chegam em H2",
            "Mesma regra para o segundo neurônio: as setas que entram em H2 são a linha 2. "
            "Agora W1 está completa e é uma matriz 2x2 -- 2 neurônios de destino (linhas) por 2 "
            "entradas (colunas).",
            equation="W1 = [[w_11, w_12], [w_21, w_22]]",
            rv_w1=np.ones((2, 2)),
            hl_w1=np.array([[0.0, 0.0], [1.0, 1.0]]),
            work_text=f"w21 = {num(w1[1, 0])}  = seta x1 -> H2\nw22 = {num(w1[1, 1])}  = seta x2 -> H2\nW1 é 2x2: linha = neurônio de destino, coluna = de onde vem.",
        )
        snap(
            "A camada 2 é uma matriz de uma linha",
            "Só existe um neurônio de saída, então W2 tem uma linha só -- e duas colunas, uma "
            "por neurônio oculto. Uma \"matriz 1x2\" parece exagero para dois números, mas é a "
            "mesma regra: a forma da matriz é ditada pela forma da rede.",
            equation="W2 = [[w_H1O, w_H2O]]",
            rv_w2=np.ones((1, 2)),
            hl_w1=_ZERO_22.copy(), hl_w2=np.ones((1, 2)),
            work_text=f"W2 = [w_H1O  w_H2O] = [{num(w2[0, 0])}  {num(w2[0, 1])}]   (1 linha = 1 saída,\n2 colunas = os dois neurônios ocultos que o alimentam)",
        )

        # ============ fase 2: forward, termo por termo ====================
        term1 = float(w1[0, 0] * x[0])
        term2 = float(w1[0, 1] * x[1])
        snap(
            "Forward, primeiro termo",
            "Começa o produto escalar da linha 1 com o vetor de entrada. Primeiro termo: a "
            f"célula w11 vezes a entrada x1 -- {w1[0, 0]:.2f}·{x[0]:.2f} = {term1:+.2f}. A soma "
            "parcial aparece embaixo; a célula de z ainda não, porque o produto escalar não "
            "terminou.",
            equation="z_H1 = w_11·x_1 + w_12·x_2",
            focus="l1", board_title="Camada 1:  z = W1 · x",
            hl_w1=np.array([[1.0, 0.0], [0.0, 0.0]]), hl_w2=_ZERO_12.copy(),
            hl_x=np.array([1.0, 0.0]),
            accum=term1, rv_accum=1.0,
            work_text=(
                "z_H1 = w11·x1 + w12·x2\n"
                f"     = ({num(w1[0, 0])})·({num(x[0])})  = {num(term1)}"
            ),
        )
        snap(
            "Forward, segundo termo",
            f"Segundo (e último) termo da linha: {w1[0, 1]:.2f}·{x[1]:.2f} = {term2:+.2f}. A soma "
            f"parcial anda de {term1:+.2f} para {z1[0]:+.2f} -- é ela que está terminando de virar "
            "o z de H1. Repare que a linha da matriz e o vetor de entrada acendem em par: linha "
            "vezes coluna.",
            equation="z_H1 = w_11·x_1 + w_12·x_2",
            hl_w1=np.array([[0.0, 1.0], [0.0, 0.0]]), hl_x=np.array([0.0, 1.0]),
            accum=float(z1[0]),
            work_text=(
                "z_H1 = w11·x1 + w12·x2\n"
                f"     = ({num(w1[0, 0])})·({num(x[0])})  = {num(term1)}\n"
                f"     + ({num(w1[0, 1])})·({num(x[1])})  = {num(term2)}\n"
                "       ----------------------\n"
                f"     = {num(float(z1[0]))}"
            ),
        )
        snap(
            "z de H1 pronto",
            f"O produto escalar fechou: z_H1 = {z1[0]:+.2f}. Só agora a primeira célula do vetor "
            "z existe. Uma linha da matriz produziu exatamente uma célula do resultado -- é essa "
            "a correspondência que faz o formato bater (2 linhas entram, 2 valores saem).",
            equation="z = W1 · x",
            rv_z1=np.array([1.0, 0.0]),
            hl_w1=np.array([[1.0, 1.0], [0.0, 0.0]]), hl_x=np.ones(2),
            work_text=f"z_H1 = {num(float(z1[0]))}   <- linha 1 de W1 vezes o vetor x\n\nAinda falta z_H2: mesma conta, linha 2.",
        )
        snap(
            "A sigmoide age célula por célula",
            f"z_H1 = {z1[0]:+.2f} atravessa a sigmoide e vira y_H1 = {y1[0]:.2f}. Diferente da "
            "multiplicação, a ativação não mistura células nenhuma: ela é aplicada em cada "
            "posição do vetor separadamente (elemento a elemento).",
            equation="y = σ(z)",
            rv_y1=np.array([1.0, 0.0]),
            hl_w1=_ZERO_22.copy(), hl_x=_ZERO_2.copy(), hl_y1=np.array([1.0, 0.0]),
            rv_accum=0.0,
            work_text=f"y_H1 = σ({num(float(z1[0]))}) = {y1[0]:.4f}\n\nσ é aplicada célula por célula -- não é multiplicação de matriz.",
        )
        snap(
            "H2: mesma conta, outra linha",
            "Agora a linha 2, de uma vez -- a mecânica termo a termo já está vista. Os pesos "
            f"mudam ({w1[1, 0]:.2f} e {w1[1, 1]:.2f}), o vetor de entrada é o mesmo, e o "
            f"resultado cai na segunda célula: z_H2 = {z1[1]:+.2f}, y_H2 = {y1[1]:.2f}.",
            equation="z_H2 = w_21·x_1 + w_22·x_2",
            rv_z1=np.ones(2), rv_y1=np.ones(2),
            hl_w1=np.array([[0.0, 0.0], [1.0, 1.0]]), hl_x=np.ones(2),
            hl_y1=np.array([0.0, 1.0]),
            accum=float(z1[1]), rv_accum=1.0,
            work_text=f"z_H2 = {num(w1[1, 0])} · {num(x[0])} + {num(w1[1, 1])} · {num(x[1])} = {num(float(z1[1]))}\ny_H2 = σ({num(float(z1[1]))}) = {y1[1]:.4f}\n\nA camada 1 inteira: z = W1 · x  (uma multiplicação, dois neurônios).",
        )
        snap(
            "Camada 2: a mesma operação, entrada diferente",
            "A camada de saída não vê x nenhum -- vê o vetor y que a camada 1 produziu. Fora "
            "isso é a mesma multiplicação: uma linha de pesos vezes um vetor de entrada, "
            f"resultando em um escalar z_O = {z2:+.4f}. Empilhar camadas é encadear matrizes.",
            equation="z_O = w_H1O·y_H1 + w_H2O·y_H2",
            focus="l2", board_title="Camada 2:  z = W2 · y",
            hl_w1=_ZERO_22.copy(), hl_x=_ZERO_2.copy(),
            hl_w2=np.ones((1, 2)), hl_y1=np.ones(2),
            rv_z2=1.0,
            accum=z2,
            work_text=f"z_O = {num(w2[0, 0])} · {y1[0]:.4f} + {num(w2[0, 1])} · {y1[1]:.4f}\n    = {num(z2, 4)}\n\nA saída da camada 1 é a entrada da camada 2.",
        )
        snap(
            "A saída da rede",
            f"Última sigmoide: y_O = σ({z2:+.4f}) = {y2:.4f}. A rede inteira, do começo ao fim, "
            "foram duas multiplicações matriz-vetor e duas ativações elemento a elemento. Nada "
            "além disso.",
            equation="y_O = σ(z_O)",
            rv_y2=1.0, hl_out=1.0, hl_w2=_ZERO_12.copy(), hl_y1=_ZERO_2.copy(),
            rv_accum=0.0,
            work_text=f"y_O = σ({num(z2, 4)}) = {y2:.4f}\n\nRede inteira = W1 · x -> σ -> W2 · y -> σ.  Duas matrizes, duas sigmoides.",
        )
        snap(
            "O erro também é só uma conta",
            f"O alvo é {target:.2f} e a rede respondeu {y2:.4f}. A função de erro comprime essa "
            f"diferença num único número: L = ½(y_O - alvo)² = {loss:.4f}. É esse escalar que o "
            "backward vai derivar -- e derivar um escalar em relação a matrizes é o que gera os "
            "gradientes.",
            equation="L = 1/2 (y_O - alvo)^2",
            focus="loss", board_title="Erro:  L = ½(y_O - alvo)²",
            rv_target=1.0, rv_loss=1.0,
            work_text=f"L = ½ · ({y2:.4f} - {target:.2f})²\n  = ½ · ({y2 - target:+.4f})²\n  = {loss:.4f}",
        )

        # ============ fase 3: backward, matriz por matriz =================
        snap(
            "Backward começa no erro",
            "O backward pergunta: se y_O subisse um pouquinho, L subiria ou desceria? Para o "
            f"erro quadrático a resposta é direta: dL/dy_O = y_O - alvo = {gy2:+.4f}. Negativo "
            "porque a rede está ABAIXO do alvo -- aumentar y_O reduziria o erro.",
            equation="dL/dy = y_O - alvo",
            rv_gy2=1.0,
            work_text=f"dL/dy_O = {y2:.4f} - {target:.2f} = {gy2:+.4f}\n\nSinal negativo = a saída precisa CRESCER para o erro cair.",
        )
        snap(
            "Atravessar a sigmoide da saída",
            "Antes de chegar aos pesos, o gradiente precisa passar pela ativação. Isso é uma "
            f"multiplicação pela derivada local: σ'(z_O) = {sp2:.4f}, então "
            f"dL/dz_O = {gy2:+.4f}·{sp2:.4f} = {gz2:+.4f}. Elemento a elemento, igual no forward.",
            equation="dL/dz = dL/dy · σ'(z_O)",
            rv_gz2=1.0, hl_out=1.0,
            work_text=f"σ'(z_O) = y_O·(1-y_O) = {y2:.4f}·{1 - y2:.4f} = {sp2:.4f}\ndL/dz_O = {gy2:+.4f} · {sp2:.4f} = {gz2:+.4f}",
        )
        snap(
            "Gradiente dos pesos da camada 2",
            "Cada peso de W2 tem um gradiente próprio, e a regra é a mesma para todos: o "
            "gradiente que chegou no neurônio de destino vezes o valor que entrou por aquele "
            "peso. Isso é um produto externo -- e sai com exatamente a forma de W2 (1x2), como "
            "tem de ser para poder atualizar os pesos.",
            equation="grad_W2 = dL/dz_O \\otimes y",
            focus="gw2", board_title="Gradiente de W2:  dL/dz_O ⊗ y",
            rv_gw2=np.ones((1, 2)), hl_y1=np.ones(2), hl_out=0.0,
            work_text=f"grad_W2[O,H1] = {gz2:+.4f} · y_H1({y1[0]:.4f}) = {gw2[0, 0]:+.5f}\ngrad_W2[O,H2] = {gz2:+.4f} · y_H2({y1[1]:.4f}) = {gw2[0, 1]:+.5f}\n\nForma do gradiente = forma da matriz. Sempre.",
        )
        snap(
            "Voltar pela matriz transposta",
            "Para continuar descendo, o gradiente tem de atravessar W2 na direção contrária. No "
            "forward, W2 levava 2 valores (ocultos) para 1 (saída); agora precisamos levar 1 "
            "gradiente de volta para 2. A matriz que faz isso é a MESMA, transposta: o que era "
            "linha vira coluna.",
            equation="dL/dy = W2^T · dL/dz_O",
            focus="w2t", board_title="Voltando:  dL/dy = W2ᵀ · dL/dz_O",
            rv_gy1=np.ones(2), hl_w2=np.ones((1, 2)), hl_y1=_ZERO_2.copy(),
            work_text=f"W2 é 1x2  ->  W2ᵀ é 2x1\ndL/dy_H1 = {num(w2[0, 0])} · {gz2:+.4f} = {gy1[0]:+.5f}\ndL/dy_H2 = {num(w2[0, 1])} · {gz2:+.4f} = {gy1[1]:+.5f}\n\nMesmos pesos do forward, direção oposta.",
        )
        snap(
            "Atravessar as sigmoides da camada 1",
            "Cada neurônio oculto tem a sua própria derivada local, então essa etapa é outra "
            f"multiplicação elemento a elemento: σ'(z_H1) = {sp1[0]:.4f} e "
            f"σ'(z_H2) = {sp1[1]:.4f}. Nenhuma mistura entre células -- H1 só multiplica pelo "
            "seu, H2 pelo dele.",
            equation="dL/dz = dL/dy \\odot σ'(z)",
            focus="gz1", board_title="Camada 1:  dL/dz = dL/dy ⊙ σ'(z)",
            rv_gz1=np.ones(2), hl_w2=_ZERO_12.copy(), hl_y1=np.ones(2),
            work_text=f"dL/dz_H1 = {gy1[0]:+.5f} · {sp1[0]:.4f} = {gz1[0]:+.5f}\ndL/dz_H2 = {gy1[1]:+.5f} · {sp1[1]:.4f} = {gz1[1]:+.5f}",
        )
        snap(
            "Gradiente dos pesos da camada 1",
            "Mesma regra do outro produto externo, agora contra o vetor de entrada x -- e o "
            "resultado tem a forma 2x2 de W1. Fim do backward: todos os 6 pesos da rede têm "
            "gradiente, e cada um foi obtido com multiplicação de matrizes, não com uma fórmula "
            "especial por peso.",
            equation="grad_W1 = dL/dz \\otimes x",
            focus="gw1", board_title="Gradiente de W1:  dL/dz ⊗ x",
            rv_gw1=np.ones((2, 2)), hl_y1=_ZERO_2.copy(), hl_x=np.ones(2),
            work_text=f"grad_W1[H1,x1] = {gz1[0]:+.5f} · {num(x[0])} = {gw1[0, 0]:+.5f}\ngrad_W1[H1,x2] = {gz1[0]:+.5f} · {num(x[1])} = {gw1[0, 1]:+.5f}\ngrad_W1[H2,x1] = {gz1[1]:+.5f} · {num(x[0])} = {gw1[1, 0]:+.5f}\ngrad_W1[H2,x2] = {gz1[1]:+.5f} · {num(x[1])} = {gw1[1, 1]:+.5f}",
        )

        # ============ fase 4: a regra da cadeia, fator por fator ==========
        chain_intro = (
            "Onde estava a regra da cadeia nessa conta toda? Estava diluída nas matrizes. "
            "Vamos desenrolá-la para UM peso concreto: w11, a seta x1 -> H1 (acesa no grafo). "
            "Ela pergunta quanto L muda quando w11 muda, e a resposta é o produto das "
            "derivadas locais de cada elo do caminho."
        )
        snap(
            "A regra da cadeia, elo por elo",
            chain_intro,
            equation="dL/dw_11 = dL/dy_O · σ'(z_O) · w_H1O · σ'(z_H1) · x_1",
            focus="chain", board_title="Regra da cadeia para w11 (x1 → H1)",
            hl_x=np.array([1.0, 0.0]), hl_w1=np.array([[1.0, 0.0], [0.0, 0.0]]),
            hl_w2=np.array([[1.0, 0.0]]), hl_y1=np.array([1.0, 0.0]), hl_out=1.0,
            work_text="Caminho de w11 até L:  x1 -> H1 -> O -> L\nH1 alimenta só um neurônio de saída, então existe UM caminho -- a regra\nda cadeia é um produto puro, sem soma de caminhos.",
        )
        for i, (name, value) in enumerate(zip(chain_names, chain_values)):
            reveal = np.zeros(_N_CHAIN_FACTORS)
            reveal[: i + 1] = 1.0
            factor_lines = "\n".join(
                f"  {'  ' if k else ''}{'· ' if k else ''}{chain_names[k]:<12} = {chain_values[k]:+.4f}"
                for k in range(i + 1)
            )
            snap(
                f"Fator {i + 1}/{_N_CHAIN_FACTORS}: {name}",
                f"Entra o fator {name} = {value:+.4f}. O produto parcial vai de "
                f"{(chain_partials[i - 1] if i else 1.0):+.5f} para {chain_partials[i]:+.5f}. Cada "
                "fator é a derivada de um único elo do caminho -- é o encadeamento deles que "
                "atravessa a rede inteira.",
                equation="dL/dw_11 = dL/dy_O · σ'(z_O) · w_H1O · σ'(z_H1) · x_1",
                rv_chain=reveal,
                chain_product=chain_partials[i], rv_chain_product=1.0,
                work_text=f"{factor_lines}\n\nproduto parcial = {chain_partials[i]:+.5f}",
            )
        snap(
            "O mesmo número, pelos dois caminhos",
            f"O produto da cadeia deu {chain_partials[-1]:+.5f}. O backward matricial, que nunca "
            f"escreveu essa cadeia, já havia calculado grad_W1[H1,x1] = {gw1[0, 0]:+.5f}. São o "
            "mesmo número porque são a mesma conta: a multiplicação de matrizes é a regra da "
            "cadeia feita por atacado, todos os pesos de uma vez.",
            equation="dL/dw_11 = grad_W1[H1, x_1]",
            rv_check=1.0,
            work_text=(
                f"regra da cadeia, fator por fator : {chain_partials[-1]:+.8f}\n"
                f"backward matricial (grad_W1)     : {gw1[0, 0]:+.8f}\n"
                f"diferença                        : {abs(chain_partials[-1] - float(gw1[0, 0])):.1e}\n\n"
                "Não é coincidência: multiplicar matrizes É aplicar a regra da cadeia em lote."
            ),
        )

        return build_sequence(frames, steps=6)
