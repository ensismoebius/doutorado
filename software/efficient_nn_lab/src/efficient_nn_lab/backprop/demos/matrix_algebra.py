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
  saída, tudo sigmoide, sem viés), com o valor de cada neurônio dentro do
  próprio nó;
* o **quadro de álgebra** fica à direita, com as mesmas quantidades como
  vetores e matrizes de verdade, com números dentro;
* destacar uma **célula** da matriz destaca, no mesmo instante, a **seta**
  correspondente do grafo -- porque `W[i, j]` *é* a seta que vai da entrada
  `j` para o neurônio `i`. Esse é o mapeamento que a demo existe para
  tornar óbvio (linha = neurônio de destino, coluna = de onde vem).

**Um passo = uma operação escalar, sem exceção.** Nenhum passo revela dois
números de uma vez: cada termo de cada produto escalar, cada soma, cada
sigmoide, cada derivada local e cada célula de cada gradiente têm o seu
próprio passo. Em particular a **ativação de cada neurônio é um passo
próprio** (H1, H2 e O têm cada um o seu σ), e no backward a derivada local
σ'(z) é revelada antes de ser usada, em vez de aparecer já multiplicada.
Isso é verificado por teste (`test_matrix_demo_reveals_at_most_one_new_value_per_step`),
porque a tentação de agrupar "os dois neurônios da camada" num passo só é
exatamente o que faz o aluno perder a conta.

Fecha com a regra da cadeia aplicada a *esta* rede: o gradiente de um peso
concreto (`w11`, a seta x1 -> H1) montado fator por fator, com o produto
parcial atualizando a cada passo, terminando na conferência contra o valor
que o backward matricial já havia calculado -- os dois caminhos dão o mesmo
número, o que é exatamente a afirmação que a regra da cadeia faz.

Rede pequena de propósito: com 2x2 e 1x2 cabe cada número na tela em corpo
grande, e cada produto escalar tem só dois termos, então o passo a passo
completo é curto o bastante para ser assistido inteiro.

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

#: Every field that reveals ONE number on screen. The "one operation per
#: step" rule is stated in terms of this list, and enforced over it by
#: tests/test_backprop.py -- keep the two in sync when adding a quantity.
VALUE_REVEAL_FIELDS = (
    "rv_x", "rv_w1", "rv_w2", "rv_z1", "rv_y1", "rv_z2", "rv_y2",
    "rv_target", "rv_diff", "rv_loss",
    "rv_gy2", "rv_sp2", "rv_gz2", "rv_gw2", "rv_gy1", "rv_sp1", "rv_gz1", "rv_gw1",
    "rv_chain",
)


def _copy_state(state: dict[str, object]) -> dict[str, object]:
    """Snapshot of a frame's state, with arrays copied.

    The builder below mutates one running ``state`` dict instead of
    respelling fifty fields per checkpoint. Every Frame therefore needs its
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
        "grafo -- W[i,j] é a seta da entrada j para o neurônio i. Um passo por operação "
        "escalar, sem agrupar nada: cada termo, cada soma, cada sigmoide, cada derivada "
        "local e cada célula de gradiente aparecem no seu próprio passo. Fecha com a "
        "regra da cadeia conferindo o gradiente de um peso contra o backward matricial."
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
        diff = y2 - target

        # The five factors of ∂L/∂w11 walked one at a time in the closing
        # phase. H1 feeds only O (single output neuron), so this weight has
        # exactly ONE path to the loss and the chain rule is a plain product
        # -- no sum over paths. (The 4-layer demo covers the multi-path case.)
        chain_names = ("∂L/∂y_O", "σ'(z_O)", "w_H1O", "σ'(z_H1)", "x1")
        chain_values = (float(gy2), float(sp2), float(w2[0, 0]), float(sp1[0]), float(x[0]))
        chain_partials = tuple(float(np.prod(chain_values[: i + 1])) for i in range(len(chain_values)))
        assert len(chain_values) == _N_CHAIN_FACTORS

        state: dict[str, object] = {
            "kind": "matrix_algebra",
            # -- the numbers themselves (constant across every frame) -----
            "x": x, "w1": w1, "w2": w2,
            "z1": z1, "y1": y1, "z2": z2, "y2": y2,
            "target": target, "diff": diff, "loss": loss,
            "sp1": sp1, "sp2": sp2,
            "gy2": gy2, "gz2": gz2, "gw2": gw2, "gy1": gy1, "gz1": gz1, "gw1": gw1,
            "chain_names": chain_names, "chain_values": chain_values,
            # -- reveals: 0 = not computed yet, 1 = on screen. Per-cell, so
            # a matrix fills in one entry at a time (numpy arrays are
            # interpolated element-wise by core/math_utils.tween_values).
            "rv_graph": 0.0,
            "rv_x": _ZERO_2.copy(), "rv_w1": _ZERO_22.copy(), "rv_w2": _ZERO_12.copy(),
            "rv_z1": _ZERO_2.copy(), "rv_y1": _ZERO_2.copy(),
            "rv_z2": 0.0, "rv_y2": 0.0,
            "rv_target": 0.0, "rv_diff": 0.0, "rv_loss": 0.0,
            "rv_gy2": 0.0, "rv_sp2": 0.0, "rv_gz2": 0.0, "rv_gw2": _ZERO_12.copy(),
            "rv_gy1": _ZERO_2.copy(), "rv_sp1": _ZERO_2.copy(), "rv_gz1": _ZERO_2.copy(),
            "rv_gw1": _ZERO_22.copy(),
            "rv_chain": np.zeros(_N_CHAIN_FACTORS),
            "rv_check": 0.0,
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
            "chain_product": 0.0, "rv_chain_product": 0.0,
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

        def n2(value: float) -> str:
            return f"{float(value):+.2f}"

        def n4(value: float) -> str:
            return f"{float(value):+.4f}"

        cell = lambda *pairs: np.array(pairs, dtype=float)  # noqa: E731 - tiny local alias

        # ================ fase 1: o mapeamento grafo <-> álgebra ==========
        snap(
            "A rede",
            "Duas entradas, dois neurônios ocultos (H1, H2), uma saída (O), tudo sigmoide e "
            "sem viés. À direita, o mesmo desenho esperando em forma de matriz: W1 tem uma "
            "linha por neurônio de destino e uma coluna por entrada. Ainda vazia -- cada "
            "célula vai entrar ligada à sua seta.",
            equation="y = σ(W · entrada)",
            rv_graph=1.0,
            work_text="6 pesos: 4 da entrada para a camada oculta, 2 da oculta para a saída.\nCada peso é uma seta no grafo E uma célula numa matriz. São a mesma coisa.",
        )
        snap(
            "x1 vira a primeira célula",
            f"O nó de entrada x1 = {x[0]:.2f} passa a ser a primeira linha de um vetor coluna. "
            "Só notação: o número é o mesmo, o lugar é que virou uma posição fixa.",
            equation="x = [x_1, x_2]",
            rv_x=cell(1.0, 0.0), hl_x=cell(1.0, 0.0),
            work_text=f"x1 = {n2(x[0])}  ->  célula 1 do vetor x",
        )
        snap(
            "x2 vira a segunda célula",
            f"E x2 = {x[1]:.2f} é a segunda. O vetor de entrada está completo: uma coluna com "
            "um valor por nó de entrada, de cima para baixo na mesma ordem do grafo.",
            equation="x = [x_1, x_2]",
            rv_x=np.ones(2), hl_x=cell(0.0, 1.0),
            work_text=f"x2 = {n2(x[1])}  ->  célula 2 do vetor x\nx = coluna [{n2(x[0])}; {n2(x[1])}]",
        )
        for (i, j), name, src, dest in (
            ((0, 0), "w11", "x1", "H1"), ((0, 1), "w12", "x2", "H1"),
            ((1, 0), "w21", "x1", "H2"), ((1, 1), "w22", "x2", "H2"),
        ):
            reveal = np.asarray(state["rv_w1"]).copy()
            reveal[i, j] = 1.0
            spot = _ZERO_22.copy()
            spot[i, j] = 1.0
            snap(
                f"{name}: a seta {src} → {dest}",
                f"A seta {src} → {dest} acendeu no grafo, e com ela a célula da linha {i + 1} "
                f"(o neurônio de destino, {dest}) e coluna {j + 1} (de onde vem, {src}). "
                f"Vale {w1[i, j]:.2f}. Linha = destino, coluna = origem -- essa é a regra "
                "inteira.",
                equation=f"W1[{dest}, {src}] = {name}",
                rv_w1=reveal, hl_w1=spot, hl_x=_ZERO_2.copy(),
                work_text=f"{name} = {n2(w1[i, j])}   =   seta {src} -> {dest}\nlinha {i + 1} = {dest} (destino)   coluna {j + 1} = {src} (origem)",
            )
        for j, (name, src) in enumerate((("w_H1O", "H1"), ("w_H2O", "H2"))):
            reveal = np.asarray(state["rv_w2"]).copy()
            reveal[0, j] = 1.0
            spot = _ZERO_12.copy()
            spot[0, j] = 1.0
            snap(
                f"{name}: a seta {src} → O",
                f"A camada 2 tem um único neurônio de destino, então W2 tem uma linha só. "
                f"Esta célula é a seta {src} → O e vale {w2[0, j]:.2f}. Uma matriz 1x2 parece "
                "exagero para dois números, mas a forma é ditada pela forma da rede.",
                equation="W2 = [[w_H1O, w_H2O]]",
                rv_w2=reveal, hl_w2=spot, hl_w1=_ZERO_22.copy(),
                work_text=f"{name} = {n2(w2[0, j])}   =   seta {src} -> O\nW2 é 1x2: 1 linha (a saída) por 2 colunas (os ocultos).",
            )

        # ================ fase 2: forward, uma operação por passo =========
        def forward_neuron_steps(
            neuron: str, row: int, weights, inputs, input_names: tuple[str, str],
            z_value: float, y_value: float, focus: str, reveal_z, reveal_y,
            hl_input_field: str,
        ) -> None:
            """Four steps for one neuron: term, term, sum, activation."""
            running = 0.0
            term_lines = []
            for k in range(2):
                term = float(weights[k] * inputs[k])
                running += term
                spot_w = _ZERO_22.copy() if focus == "l1" else _ZERO_12.copy()
                spot_w[row if focus == "l1" else 0, k] = 1.0
                spot_in = _ZERO_2.copy()
                spot_in[k] = 1.0
                term_lines.append(
                    f"     {'=' if k == 0 else '+'} ({n2(weights[k])})·({n2(inputs[k])}) = {n2(term)}"
                )
                ordinal = "primeiro" if k == 0 else "segundo"
                snap(
                    f"{neuron}: {ordinal} termo",
                    f"{'Começa' if k == 0 else 'Fecha'} o produto escalar da linha de "
                    f"{neuron} com o vetor de entrada: {weights[k]:.2f}·{inputs[k]:.2f} = "
                    f"{term:+.2f}. A soma parcial embaixo "
                    f"{'aparece' if k == 0 else f'anda para {running:+.2f}'}; a célula de z "
                    "ainda não, porque o produto escalar não acabou.",
                    equation=f"z_{neuron} = w·{input_names[0]} + w·{input_names[1]}",
                    focus=focus,
                    **{
                        "hl_w1" if focus == "l1" else "hl_w2": spot_w,
                        hl_input_field: spot_in,
                    },
                    **({"hl_w2": _ZERO_12.copy()} if focus == "l1" else {"hl_w1": _ZERO_22.copy()}),
                    accum=running, rv_accum=1.0,
                    work_text=f"z_{neuron} = w·{input_names[0]} + w·{input_names[1]}\n" + "\n".join(term_lines),
                )
            snap(
                f"{neuron}: z pronto",
                f"O produto escalar fechou: z_{neuron} = {z_value:+.3f}. Só agora a célula de z "
                f"existe. Uma linha da matriz produziu exatamente uma célula do resultado -- é "
                "por isso que as formas batem.",
                equation="z = W · entrada",
                **{("rv_z1" if focus == "l1" else "rv_z2"): reveal_z},
                accum=z_value,
                work_text=f"z_{neuron} = {n4(z_value)}\n(linha de {neuron} vezes o vetor de entrada)",
            )
            snap(
                f"{neuron}: a sigmoide",
                f"Agora a ativação, que é um passo à parte: z_{neuron} = {z_value:+.3f} entra na "
                f"sigmoide e sai y_{neuron} = {y_value:.4f}. Diferente da multiplicação, σ não "
                "mistura células nenhuma -- age em cada posição do vetor separadamente.",
                equation="y = σ(z)",
                **{("rv_y1" if focus == "l1" else "rv_y2"): reveal_y},
                hl_w1=_ZERO_22.copy(), hl_w2=_ZERO_12.copy(), hl_x=_ZERO_2.copy(),
                hl_y1=(cell(1.0, 0.0) if (focus == "l1" and row == 0) else
                       cell(0.0, 1.0) if focus == "l1" else _ZERO_2.copy()),
                hl_out=1.0 if focus == "l2" else 0.0,
                rv_accum=0.0,
                work_text=f"y_{neuron} = σ({n4(z_value)}) = {y_value:.4f}\nσ age célula por célula -- não é multiplicação de matriz.",
            )

        forward_neuron_steps(
            "H1", 0, w1[0], x, ("x1", "x2"), float(z1[0]), float(y1[0]),
            "l1", cell(1.0, 0.0), cell(1.0, 0.0), "hl_x",
        )
        forward_neuron_steps(
            "H2", 1, w1[1], x, ("x1", "x2"), float(z1[1]), float(y1[1]),
            "l1", np.ones(2), np.ones(2), "hl_x",
        )
        state["board_title"] = "Camada 2:  z = W2 · y"
        forward_neuron_steps(
            "O", 0, w2[0], y1, ("y_H1", "y_H2"), z2, y2,
            "l2", 1.0, 1.0, "hl_y1",
        )

        # ================ fase 3: o erro ==================================
        snap(
            "O alvo",
            f"A rede respondeu {y2:.4f}. O alvo é {target:.2f} -- um número que vem dos dados, "
            "não da rede. Sozinho ele ainda não é erro nenhum; é só a referência.",
            equation="alvo",
            focus="loss", board_title="Erro:  L = ½(y_O - alvo)²",
            rv_target=1.0, hl_out=0.0, hl_y1=_ZERO_2.copy(),
            work_text=f"y_O  = {y2:.4f}   (o que a rede deu)\nalvo = {target:.2f}     (o que se queria)",
        )
        snap(
            "A diferença",
            f"A diferença y_O - alvo = {diff:+.4f} é negativa: a rede está ABAIXO do alvo. O "
            "sinal importa, porque é ele que vai dizer para que lado os pesos devem andar.",
            equation="y_O - alvo",
            rv_diff=1.0,
            work_text=f"y_O - alvo = {y2:.4f} - {target:.2f} = {diff:+.4f}\nnegativo => a saída precisa CRESCER",
        )
        snap(
            "A perda",
            f"Elevar ao quadrado e dividir por dois transforma a diferença num único número "
            f"positivo: L = {loss:.4f}. É esse escalar que o backward vai derivar em relação a "
            "cada peso -- e derivar um escalar em relação a matrizes é o que produz gradientes "
            "com a forma das matrizes.",
            equation="L = 1/2 (y_O - alvo)^2",
            rv_loss=1.0,
            work_text=f"L = ½ · ({diff:+.4f})²\n  = {loss:.4f}",
        )

        # ================ fase 4: backward, uma operação por passo ========
        snap(
            "∂L/∂y_O",
            "O backward pergunta: se y_O subisse um pouquinho, L subiria ou desceria? Para o "
            f"erro quadrático a derivada é a própria diferença: ∂L/∂y_O = {gy2:+.4f}.",
            equation="∂L/∂y = y_O - alvo",
            focus="gz2", board_title="Saída:  ∂L/∂z_O = ∂L/∂y_O · σ'(z_O)",
            rv_gy2=1.0,
            work_text=f"∂L/∂y_O = y_O - alvo = {gy2:+.4f}",
        )
        snap(
            "σ'(z_O): a derivada da ativação",
            "Antes de tocar em peso nenhum, o gradiente tem de atravessar a sigmoide da saída "
            f"-- e para isso precisa da derivada local dela: σ'(z_O) = y_O·(1-y_O) = {sp2:.4f}. "
            "Este passo só calcula essa derivada; ainda não multiplicou nada.",
            equation="σ'(z) = y·(1 - y)",
            rv_sp2=1.0, hl_out=1.0,
            work_text=f"σ'(z_O) = y_O·(1 - y_O)\n        = {y2:.4f} · {1 - y2:.4f}\n        = {sp2:.4f}",
        )
        snap(
            "∂L/∂z_O",
            f"Agora sim a travessia: {gy2:+.4f} · {sp2:.4f} = {gz2:+.4f}. Este é o gradiente do "
            "lado de dentro da ativação, o ponto de partida para todos os pesos da camada 2.",
            equation="∂L/∂z = ∂L/∂y · σ'(z_O)",
            rv_gz2=1.0, hl_out=0.0,
            work_text=f"∂L/∂z_O = {gy2:+.4f} · {sp2:.4f} = {gz2:+.4f}",
        )
        for j, src in enumerate(("H1", "H2")):
            reveal = np.asarray(state["rv_gw2"]).copy()
            reveal[0, j] = 1.0
            spot = _ZERO_2.copy()
            spot[j] = 1.0
            snap(
                f"grad_W2[O,{src}]",
                f"O gradiente de um peso é sempre o mesmo produto: o gradiente que chegou no "
                f"neurônio de destino vezes o valor que entrou por aquele peso. Aqui, "
                f"{gz2:+.4f} · y_{src}({y1[j]:.4f}) = {gw2[0, j]:+.5f}.",
                equation="grad_W2 = ∂L/∂z_O \\otimes y",
                focus="gw2", board_title="Gradiente de W2:  ∂L/∂z_O ⊗ y",
                rv_gw2=reveal, hl_y1=spot,
                work_text=f"grad_W2[O,{src}] = ∂L/∂z_O · y_{src}\n              = {gz2:+.4f} · {y1[j]:.4f}\n              = {gw2[0, j]:+.5f}",
            )
        for j, dest in enumerate(("H1", "H2")):
            reveal = np.asarray(state["rv_gy1"]).copy()
            reveal[j] = 1.0
            spot = _ZERO_12.copy()
            spot[0, j] = 1.0
            snap(
                f"∂L/∂y_{dest}: voltando pela transposta",
                "Para continuar descendo, o gradiente atravessa W2 na direção contrária. No "
                "forward W2 levava 2 valores para 1; agora 1 gradiente volta para 2. A matriz "
                f"que faz isso é a MESMA, transposta: {w2[0, j]:.2f} · {gz2:+.4f} = "
                f"{gy1[j]:+.5f}.",
                equation="∂L/∂y = W2^T · ∂L/∂z_O",
                focus="w2t", board_title="Voltando:  ∂L/∂y = W2ᵀ · ∂L/∂z_O",
                rv_gy1=reveal, hl_w2=spot, hl_y1=_ZERO_2.copy(),
                work_text=f"∂L/∂y_{dest} = w_{dest}O · ∂L/∂z_O\n         = {n2(w2[0, j])} · {gz2:+.4f}\n         = {gy1[j]:+.5f}",
            )
        for j, dest in enumerate(("H1", "H2")):
            reveal_sp = np.asarray(state["rv_sp1"]).copy()
            reveal_sp[j] = 1.0
            spot = _ZERO_2.copy()
            spot[j] = 1.0
            snap(
                f"σ'(z_{dest})",
                f"Cada neurônio oculto tem a SUA derivada local, calculada do próprio y: "
                f"σ'(z_{dest}) = {y1[j]:.4f}·{1 - y1[j]:.4f} = {sp1[j]:.4f}. Passo separado, de "
                "novo, para deixar claro que essa derivada é uma quantidade por conta própria.",
                equation="σ'(z) = y·(1 - y)",
                focus="gz1", board_title="Camada 1:  ∂L/∂z = ∂L/∂y ⊙ σ'(z)",
                rv_sp1=reveal_sp, hl_y1=spot, hl_w2=_ZERO_12.copy(),
                work_text=f"σ'(z_{dest}) = {y1[j]:.4f} · {1 - y1[j]:.4f} = {sp1[j]:.4f}",
            )
            reveal_gz = np.asarray(state["rv_gz1"]).copy()
            reveal_gz[j] = 1.0
            snap(
                f"∂L/∂z_{dest}",
                f"E a travessia da ativação de {dest}: {gy1[j]:+.5f} · {sp1[j]:.4f} = "
                f"{gz1[j]:+.5f}. Multiplicação elemento a elemento -- {dest} só usa o seu "
                "próprio σ', não o do vizinho.",
                equation="∂L/∂z = ∂L/∂y \\odot σ'(z)",
                rv_gz1=reveal_gz,
                work_text=f"∂L/∂z_{dest} = {gy1[j]:+.5f} · {sp1[j]:.4f} = {gz1[j]:+.5f}",
            )
        for (i, j), dest, src in (
            ((0, 0), "H1", "x1"), ((0, 1), "H1", "x2"),
            ((1, 0), "H2", "x1"), ((1, 1), "H2", "x2"),
        ):
            reveal = np.asarray(state["rv_gw1"]).copy()
            reveal[i, j] = 1.0
            spot = _ZERO_2.copy()
            spot[j] = 1.0
            spot_z = _ZERO_2.copy()
            spot_z[i] = 1.0
            snap(
                f"grad_W1[{dest},{src}]",
                f"Mesma regra, agora contra o vetor de entrada: {gz1[i]:+.5f} · {src}"
                f"({x[j]:.2f}) = {gw1[i, j]:+.5f}. Uma célula por passo; quando as quatro "
                "estiverem lá, o gradiente terá exatamente a forma 2x2 de W1.",
                equation="grad_W1 = ∂L/∂z \\otimes x",
                focus="gw1", board_title="Gradiente de W1:  ∂L/∂z ⊗ x",
                rv_gw1=reveal, hl_x=spot, hl_y1=spot_z, hl_w2=_ZERO_12.copy(),
                work_text=f"grad_W1[{dest},{src}] = ∂L/∂z_{dest} · {src}\n                 = {gz1[i]:+.5f} · {n2(x[j])}\n                 = {gw1[i, j]:+.5f}",
            )

        # ================ fase 5: a regra da cadeia, fator por fator ======
        snap(
            "A regra da cadeia, elo por elo",
            "Onde estava a regra da cadeia nessa conta toda? Diluída nas matrizes. Vamos "
            "desenrolá-la para UM peso concreto: w11, a seta x1 → H1, acesa no grafo. Ela "
            "pergunta quanto L muda quando w11 muda, e a resposta é o produto das derivadas "
            "locais de cada elo do caminho.",
            equation="∂L/∂w_11 = ∂L/∂y_O · σ'(z_O) · w_H1O · σ'(z_H1) · x_1",
            focus="chain", board_title="Regra da cadeia para w11 (x1 → H1)",
            hl_x=cell(1.0, 0.0), hl_w1=cell((1.0, 0.0), (0.0, 0.0)),
            hl_w2=np.array([[1.0, 0.0]]), hl_y1=cell(1.0, 0.0), hl_out=1.0,
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
                f"{(chain_partials[i - 1] if i else 1.0):+.5f} para {chain_partials[i]:+.5f}. "
                "Cada fator é a derivada de um único elo do caminho -- é o encadeamento deles "
                "que atravessa a rede inteira.",
                equation="∂L/∂w_11 = ∂L/∂y_O · σ'(z_O) · w_H1O · σ'(z_H1) · x_1",
                rv_chain=reveal,
                chain_product=chain_partials[i], rv_chain_product=1.0,
                work_text=f"{factor_lines}\n\nproduto parcial = {chain_partials[i]:+.5f}",
            )
        snap(
            "O mesmo número, pelos dois caminhos",
            f"O produto da cadeia deu {chain_partials[-1]:+.5f}. O backward matricial, que nunca "
            f"escreveu essa cadeia, já havia posto grad_W1[H1,x1] = {gw1[0, 0]:+.5f}. São o "
            "mesmo número porque são a mesma conta: multiplicar matrizes é a regra da cadeia "
            "feita por atacado, todos os pesos de uma vez.",
            equation="∂L/∂w_11 = grad_W1[H1, x_1]",
            rv_check=1.0,
            work_text=(
                f"regra da cadeia, fator por fator : {chain_partials[-1]:+.8f}\n"
                f"backward matricial (grad_W1)     : {float(gw1[0, 0]):+.8f}\n"
                f"diferença                        : {abs(chain_partials[-1] - float(gw1[0, 0])):.1e}"
            ),
        )

        return build_sequence(frames, steps=6)
