"""Demonstração — Backpropagation numa rede de 4 camadas.

O neurônio único de traditional_gd.py mostra a mecânica (forward, ativação,
regra da cadeia) num caso mínimo. Esta demonstração é o mesmo mecanismo
aplicado numa rede de verdade: entrada (tamanho 3) -> camada 1 (2
neurônios) -> camada 2 (2 neurônios) -> saída (1 neurônio), todos com
ativação sigmoide, sem viés (mesma simplificação didática do neurônio
único).

A caminhada é sempre um neurônio de cada vez, na ordem topológica correta:
forward da entrada até a saída, depois backward da saída até a entrada —
exatamente a ordem em que os valores ficam disponíveis para cada cálculo.
Em cada passo, o neurônio "ativo" ganha um destaque no diagrama, o painel
de detalhe mostra a equação e os números daquele neurônio especificamente,
e o gráfico da sigmoide ao lado atualiza para mostrar o ponto, a tangente
e a direção do gradiente daquele neurônio -- literalmente "o gráfico da
sigmoide de cada neurônio", um de cada vez, não cinco sobrepostos.

Pesos fixos e determinísticos (ESPECIFICACAO_DLVL.md #35): nada de
inicialização aleatória.
"""

from __future__ import annotations

import numpy as np

from efficient_nn_lab.backprop.activation import sigmoid, sigmoid_derivative
from efficient_nn_lab.bitnet.linear import loss_gradient_wrt_y, squared_error_loss
from efficient_nn_lab.bitnet.ste import sgd_update
from efficient_nn_lab.core.demo import DemoModule, Frame, build_sequence

_X = np.array([1.0, 0.5, -0.5])
_W1 = np.array([[0.4, -0.3, 0.6], [-0.2, 0.5, 0.1]])  # layer 1 (A, B): 2x3
_W2 = np.array([[0.7, -0.4], [0.3, 0.6]])  # layer 2 (C, D): 2x2
_W3 = np.array([[0.5, -0.6]])  # output (O): 1x2


class MultilayerNetworkDemo(DemoModule):
    title = "Backprop -> Rede de 4 camadas"
    slug = "backprop.mlp"
    description = (
        "O mesmo forward/backward do neurônio único, agora numa rede de verdade: "
        "entrada (3) -> camada 1 (2) -> camada 2 (2) -> saída (1), toda sigmoide. "
        "Um neurônio de cada vez, na ordem em que os valores realmente ficam prontos."
    )

    def __init__(self) -> None:
        self.target = 0.8
        self.learning_rate = 5.0
        super().__init__()

    def parameters(self) -> dict[str, dict[str, object]]:
        return {
            "target": {"label": "Alvo (0-1)", "min": 0.1, "max": 0.95, "step": 0.05, "value": self.target},
            "learning_rate": {"label": "Taxa de aprendizado", "min": 1.0, "max": 10.0, "step": 1.0, "value": self.learning_rate},
        }

    # -- the actual math: one forward pass, one full backward pass ---------
    def _forward_backward(self) -> dict[str, object]:
        x, target, lr = _X, self.target, self.learning_rate

        z1 = _W1 @ x
        y1 = sigmoid(z1)
        z2 = _W2 @ y1
        y2 = sigmoid(z2)
        zO = float((_W3 @ y2)[0])
        yO = float(sigmoid(zO))
        loss = squared_error_loss(yO, target)

        grad_yO = loss_gradient_wrt_y(yO, target)
        grad_zO = grad_yO * sigmoid_derivative(zO)
        grad_y2 = _W3[0] * grad_zO
        grad_z2 = grad_y2 * sigmoid_derivative(z2)
        grad_y1 = _W2.T @ grad_z2
        grad_z1 = grad_y1 * sigmoid_derivative(z1)

        grad_W3 = grad_zO * y2
        grad_W2 = np.outer(grad_z2, y1)
        grad_W1 = np.outer(grad_z1, x)

        return {
            "x": x, "z1": z1, "y1": y1, "z2": z2, "y2": y2, "zO": zO, "yO": yO, "loss": loss,
            "grad_yO": grad_yO, "grad_zO": grad_zO, "grad_y2": grad_y2, "grad_z2": grad_z2,
            "grad_y1": grad_y1, "grad_z1": grad_z1,
            "w1": _W1, "w2": _W2, "w3": _W3,
            "grad_W1": grad_W1, "grad_W2": grad_W2, "grad_W3": grad_W3,
            "w1_updated": sgd_update(_W1, grad_W1, lr),
            "w2_updated": sgd_update(_W2, grad_W2, lr),
            "w3_updated": sgd_update(_W3, grad_W3, lr),
        }

    def _build_frames(self) -> list[Frame]:
        c = self._forward_backward()

        base = {
            "kind": "mlp_network",
            "x": c["x"], "target": self.target, "loss": c["loss"],
            "w1": c["w1"], "w2": c["w2"], "w3": c["w3"],
            "z1": c["z1"], "y1": c["y1"], "z2": c["z2"], "y2": c["y2"], "zO": c["zO"], "yO": c["yO"],
            "grad_z1": c["grad_z1"], "grad_z2": c["grad_z2"], "grad_zO": c["grad_zO"],
            "w1_updated": c["w1_updated"], "w2_updated": c["w2_updated"], "w3_updated": c["w3_updated"],
            # per-neuron forward reveal
            "fwd_l1a": 0.0, "fwd_l1b": 0.0, "fwd_l2c": 0.0, "fwd_l2d": 0.0, "fwd_o": 0.0,
            "loss_reveal": 0.0,
            # per-neuron backward reveal
            "bwd_o": 0.0, "bwd_l2d": 0.0, "bwd_l2c": 0.0, "bwd_l1b": 0.0, "bwd_l1a": 0.0,
            "update_reveal": 0.0,
            # "spotlight": which single neuron the sigmoid inset + detail
            # panel currently focus on, and that neuron's own numbers.
            "active": "", "active_z": 0.0, "active_y": 0.5, "active_slope": 0.25, "active_grad_z": 0.0,
            "active_detail": "",
        }

        def frame(label: str, explanation: str, equation: str = "", **overrides) -> Frame:
            values = dict(base)
            values.update(overrides)
            return Frame(label, values, explanation, equation)

        def fwd_detail(name: str, z: float, y: float, weights, inputs, input_names: list[str]) -> str:
            terms = " ".join(
                (f"{w:.2f}·{n}" if i == 0 else f"{'+ ' if w >= 0 else '- '}{abs(w):.2f}·{n}")
                for i, (w, n) in enumerate(zip(weights, input_names))
            )
            return f"{name}: z = {terms}\n     = {z:.3f}\ny = σ(z) = {y:.3f}"

        def bwd_detail(name: str, grad_y: float, grad_z: float, slope: float) -> str:
            return (
                f"{name}: dL/dy = {grad_y:.3f}\n"
                f"dL/dz = dL/dy·σ'(z)\n"
                f"     = {grad_y:.3f}·{slope:.2f} = {grad_z:.3f}"
            )

        z1, y1, z2, y2, zO, yO = c["z1"], c["y1"], c["z2"], c["y2"], c["zO"], c["yO"]
        gy2, gz2, gy1, gz1 = c["grad_y2"], c["grad_z2"], c["grad_y1"], c["grad_z1"]
        s1 = sigmoid_derivative(z1)
        s2 = sigmoid_derivative(z2)
        sO = sigmoid_derivative(zO)

        checkpoints = [
            frame(
                "A rede",
                "Quatro camadas: 3 entradas, 2 neurônios na camada 1, 2 na camada 2, 1 na "
                "saída -- todas com ativação sigmoide, sem viés. Vamos calcular o forward "
                "neurônio por neurônio, na ordem em que cada valor fica disponível.",
                equation="y = σ(W · entrada)  em cada camada",
            ),
            frame(
                "Forward: L1-A", "Primeiro neurônio da camada 1: combina as três entradas.",
                equation="z = w1·x1 + w2·x2 + w3·x3;  y = σ(z)",
                fwd_l1a=1.0, active="L1-A", active_z=float(z1[0]), active_y=float(y1[0]), active_slope=float(s1[0]),
                active_detail=fwd_detail("L1-A", z1[0], y1[0], c["w1"][0], c["x"], ["x1", "x2", "x3"]),
            ),
            frame(
                "Forward: L1-B", "Segundo neurônio da camada 1, mesmas três entradas, pesos diferentes.",
                equation="z = w1·x1 + w2·x2 + w3·x3;  y = σ(z)",
                fwd_l1a=1.0, fwd_l1b=1.0, active="L1-B", active_z=float(z1[1]), active_y=float(y1[1]), active_slope=float(s1[1]),
                active_detail=fwd_detail("L1-B", z1[1], y1[1], c["w1"][1], c["x"], ["x1", "x2", "x3"]),
            ),
            frame(
                "Forward: L2-C", "Camada 2 já não vê a entrada -- vê as saídas da camada 1.",
                equation="z = w1·y_A + w2·y_B;  y = σ(z)",
                fwd_l1a=1.0, fwd_l1b=1.0, fwd_l2c=1.0, active="L2-C", active_z=float(z2[0]), active_y=float(y2[0]), active_slope=float(s2[0]),
                active_detail=fwd_detail("L2-C", z2[0], y2[0], c["w2"][0], y1, ["y_A", "y_B"]),
            ),
            frame(
                "Forward: L2-D", "Segundo neurônio da camada 2.",
                equation="z = w1·y_A + w2·y_B;  y = σ(z)",
                fwd_l1a=1.0, fwd_l1b=1.0, fwd_l2c=1.0, fwd_l2d=1.0, active="L2-D",
                active_z=float(z2[1]), active_y=float(y2[1]), active_slope=float(s2[1]),
                active_detail=fwd_detail("L2-D", z2[1], y2[1], c["w2"][1], y1, ["y_A", "y_B"]),
            ),
            frame(
                "Forward: Saída", "O neurônio de saída combina as duas saídas da camada 2.",
                equation="z = w1·y_C + w2·y_D;  y = σ(z)",
                fwd_l1a=1.0, fwd_l1b=1.0, fwd_l2c=1.0, fwd_l2d=1.0, fwd_o=1.0, active="Saída",
                active_z=float(zO), active_y=float(yO), active_slope=float(sO),
                active_detail=fwd_detail("Saída", zO, yO, c["w3"][0], y2, ["y_C", "y_D"]),
            ),
            frame(
                "A perda", f"A saída da rede é {yO:.3f}; o alvo é {self.target:g}. A perda resume essa "
                "distância num único número.",
                equation="L = 1/2 (y - target)^2",
                fwd_l1a=1.0, fwd_l1b=1.0, fwd_l2c=1.0, fwd_l2d=1.0, fwd_o=1.0, loss_reveal=1.0,
                active="Saída", active_z=float(zO), active_y=float(yO), active_slope=float(sO),
            ),
            frame(
                "Backward: Saída", "O backward começa onde o forward terminou: dL/dy na saída, depois "
                "dL/dz atravessando a sigmoide daquele neurônio.",
                equation="dL/dy = y - target;  dL/dz = dL/dy · σ'(z)",
                fwd_l1a=1.0, fwd_l1b=1.0, fwd_l2c=1.0, fwd_l2d=1.0, fwd_o=1.0, loss_reveal=1.0, bwd_o=1.0,
                active="Saída", active_z=float(zO), active_y=float(yO), active_slope=float(sO), active_grad_z=float(c["grad_zO"]),
                active_detail=bwd_detail("Saída", c["grad_yO"], c["grad_zO"], sO),
            ),
            frame(
                "Backward: L2-D", "O gradiente volta para L2-D através do peso que liga L2-D à saída.",
                equation="dL/dy_D = w_D→O · dL/dz_O;  dL/dz_D = dL/dy_D · σ'(z_D)",
                fwd_l1a=1.0, fwd_l1b=1.0, fwd_l2c=1.0, fwd_l2d=1.0, fwd_o=1.0, loss_reveal=1.0,
                bwd_o=1.0, bwd_l2d=1.0, active="L2-D", active_z=float(z2[1]), active_y=float(y2[1]),
                active_slope=float(s2[1]), active_grad_z=float(gz2[1]),
                active_detail=bwd_detail("L2-D", gy2[1], gz2[1], s2[1]),
            ),
            frame(
                "Backward: L2-C", "Mesma ideia para L2-C, pelo peso que liga L2-C à saída.",
                equation="dL/dy_C = w_C→O · dL/dz_O;  dL/dz_C = dL/dy_C · σ'(z_C)",
                fwd_l1a=1.0, fwd_l1b=1.0, fwd_l2c=1.0, fwd_l2d=1.0, fwd_o=1.0, loss_reveal=1.0,
                bwd_o=1.0, bwd_l2d=1.0, bwd_l2c=1.0, active="L2-C", active_z=float(z2[0]), active_y=float(y2[0]),
                active_slope=float(s2[0]), active_grad_z=float(gz2[0]),
                active_detail=bwd_detail("L2-C", gy2[0], gz2[0], s2[0]),
            ),
            frame(
                "Backward: L1-B", "L1-B alimenta os dois neurônios da camada 2, então recebe gradiente "
                "de ambos -- soma das duas contribuições.",
                equation="dL/dy_B = w_B→C·dL/dz_C + w_B→D·dL/dz_D;  dL/dz_B = dL/dy_B · σ'(z_B)",
                fwd_l1a=1.0, fwd_l1b=1.0, fwd_l2c=1.0, fwd_l2d=1.0, fwd_o=1.0, loss_reveal=1.0,
                bwd_o=1.0, bwd_l2d=1.0, bwd_l2c=1.0, bwd_l1b=1.0, active="L1-B",
                active_z=float(z1[1]), active_y=float(y1[1]), active_slope=float(s1[1]), active_grad_z=float(gz1[1]),
                active_detail=bwd_detail("L1-B", gy1[1], gz1[1], s1[1]),
            ),
            frame(
                "Backward: L1-A", "E o mesmo para L1-A -- último neurônio, gradiente também somado das "
                "duas saídas da camada 2 que ele alimenta.",
                equation="dL/dy_A = w_A→C·dL/dz_C + w_A→D·dL/dz_D;  dL/dz_A = dL/dy_A · σ'(z_A)",
                fwd_l1a=1.0, fwd_l1b=1.0, fwd_l2c=1.0, fwd_l2d=1.0, fwd_o=1.0, loss_reveal=1.0,
                bwd_o=1.0, bwd_l2d=1.0, bwd_l2c=1.0, bwd_l1b=1.0, bwd_l1a=1.0, active="L1-A",
                active_z=float(z1[0]), active_y=float(y1[0]), active_slope=float(s1[0]), active_grad_z=float(gz1[0]),
                active_detail=bwd_detail("L1-A", gy1[0], gz1[0], s1[0]),
            ),
            frame(
                "Atualizar todos os pesos", "Cada peso da rede anda um passo contra o próprio gradiente -- "
                "dL/dw = dL/dz do neurônio de destino vezes o valor que entrou por aquele peso.",
                equation="w <- w - taxa · dL/dw",
                fwd_l1a=1.0, fwd_l1b=1.0, fwd_l2c=1.0, fwd_l2d=1.0, fwd_o=1.0, loss_reveal=1.0,
                bwd_o=1.0, bwd_l2d=1.0, bwd_l2c=1.0, bwd_l1b=1.0, bwd_l1a=1.0, update_reveal=1.0,
                active="", active_z=float(z1[0]), active_y=float(y1[0]), active_slope=float(s1[0]),
            ),
        ]
        return build_sequence(checkpoints, steps=7)
