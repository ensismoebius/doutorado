"""Demonstração SNN 4 — Surrogate gradient (ESPECIFICACAO_DLVL.md #20).

Single question answered: how is a spiking neuron trained if its spike
function's derivative is useless? Mirrors the STE story from the BitNet
module, but the text is careful to say the two are analogous, not
identical (ESPECIFICACAO_DLVL.md #20's closing note).

One persistent two-panel scene (forward function on top, "gradient used
in the backward pass" on the bottom). The centerpiece is the transition
from "A derivada real" into "O gradiente substituto": instead of the
gradient curve fading in all at once, it is *traced* left to right, with
the exact-antiderivative sigmoid curve traced alongside it at the same
pace — at every instant the sweep's leading edge shows, side by side, the
sigmoid's local slope and the gradient curve's height there, so the
answer to "where does this bump come from" is watched forming, point by
point, rather than stated as a formula.
"""

from __future__ import annotations

import numpy as np

from efficient_nn_lab.core.demo import DemoModule, Frame, build_sequence
from efficient_nn_lab.snn.surrogate import fast_sigmoid, fast_sigmoid_surrogate, heaviside, heaviside_derivative

_X = np.linspace(-2.0, 2.0, 400)


class SurrogateGradientDemo(DemoModule):
    title = "SNN -> Surrogate gradient"
    description = "O spike no forward continua discreto; apenas o backward usa uma aproximação suave."

    def __init__(self) -> None:
        self.k = 5.0
        super().__init__()

    def parameters(self) -> dict[str, dict[str, object]]:
        return {"k": {"label": "Inclinação (k)", "min": 1.0, "max": 15.0, "step": 0.5, "value": self.k}}

    def _build_frames(self) -> list[Frame]:
        spike = heaviside(_X)
        true_derivative = heaviside_derivative(_X)
        surrogate = fast_sigmoid_surrogate(_X, self.k)
        sigmoid = fast_sigmoid(_X, self.k)

        base = {
            "kind": "surrogate_curve",
            "x": _X,
            "spike": spike,
            "sigmoid": sigmoid,
            "sigmoid_reveal": 0.0,
            "true_derivative": true_derivative,
            "surrogate": surrogate,
            "draw_reveal": 0.0,
            "bottom_reveal": 0.0,
            "overlay_reveal": 0.0,
        }

        def frame(label: str, explanation: str, equation: str = "", **overrides) -> Frame:
            values = dict(base)
            values.update(overrides)
            return Frame(label, values, explanation, equation)

        checkpoints = [
            frame(
                "A função de disparo (forward)",
                "No forward, o neurônio sempre usa esta função em degrau: dispara ou não dispara.",
                equation="S(v) = 1 se v >= v_th; 0 caso contrario",
            ),
            frame(
                "A sigmoide suave por trás do gradiente substituto",
                "O degrau em si não muda. Mas a curva em S mostrada aqui é a antiderivada exata do "
                "gradiente substituto que será usado no backward — ou seja, a inclinação dessa "
                "sigmoide em cada ponto é, por construção, exatamente a curva de gradiente que vem a seguir.",
                equation="sigmoide(v) = 0,5 + (v - v_th) / (1 + k|v - v_th|)",
                sigmoid_reveal=1.0,
            ),
            frame(
                "A derivada real",
                "A derivada real do degrau é zero em quase todo ponto — inútil para descida de gradiente.",
                equation="dS/dv = 0 (quase todo ponto)",
                bottom_reveal=1.0,
                sigmoid_reveal=1.0,
            ),
            frame(
                "O gradiente substituto",
                "Observe o traço se formando da esquerda para a direita: a sigmoide e o gradiente são "
                "desenhados juntos, no mesmo ritmo, e a altura do gradiente em cada ponto é exatamente "
                "a inclinação da sigmoide naquele mesmo ponto — é literalmente de onde o gradiente vem.",
                equation="dS/dv ~= 1 / (1 + k|v - v_th|)^2",
                bottom_reveal=1.0,
                sigmoid_reveal=1.0,
                draw_reveal=1.0,
            ),
            frame(
                "Os dois juntos",
                "Forward continua discreto (spike/não-spike); só o backward usa a curva suave. "
                "Análogo ao STE do BitNet, mas com uma função diferente — não é a mesma técnica.",
                bottom_reveal=1.0,
                overlay_reveal=1.0,
                sigmoid_reveal=1.0,
                draw_reveal=1.0,
            ),
        ]
        return build_sequence(checkpoints, steps=[15, 0, 40, 10])
