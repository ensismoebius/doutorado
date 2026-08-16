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

One concrete worked example runs through every scene (same pattern as
bitnet/demos/backward.py's): v_th = 1.0, example v = 1.2 (0.2 above the
threshold) -> S(v) = 1, sigmoid = 0.60, real derivative = 0, surrogate
gradient = 0.25. The numbers come out of the real surrogate/Heaviside
code, never hand-typed into the f-strings.
"""

from __future__ import annotations

import numpy as np

from efficient_nn_lab.core.demo import DemoModule, Frame, build_sequence
from efficient_nn_lab.snn.surrogate import fast_sigmoid, fast_sigmoid_surrogate, heaviside, heaviside_derivative

_X = np.linspace(-2.0, 2.0, 400)


class SurrogateGradientDemo(DemoModule):
    title = "SNN -> Surrogate gradient"
    slug = "snn.surrogate"
    description = "O spike no forward continua discreto; apenas o backward usa uma aproximação suave."

    def __init__(self) -> None:
        self.k = 5.0
        self.v_th = 1.0
        super().__init__()

    def parameters(self) -> dict[str, dict[str, object]]:
        return {
            "k": {"label": "Inclinação (k)", "min": 1.0, "max": 15.0, "step": 0.5, "value": self.k},
            "v_th": {"label": "Limiar (v_th)", "min": 0.0, "max": 2.0, "step": 0.1, "value": self.v_th},
        }

    def _build_frames(self) -> list[Frame]:
        spike = heaviside(_X)
        true_derivative = heaviside_derivative(_X)
        surrogate = fast_sigmoid_surrogate(_X, self.k)
        sigmoid = fast_sigmoid(_X, self.k)

        # The concrete membrane value every scene references, sitting just
        # above the threshold (v - v_th = 0.2 -> v = v_th + 0.2).
        v_example = self.v_th + 0.2
        vmt_example = v_example - self.v_th
        example_surrogate = float(fast_sigmoid_surrogate(np.array([vmt_example]), self.k)[0])
        example_sigmoid = float(fast_sigmoid(np.array([vmt_example]), self.k)[0])
        example_spike = float(heaviside(np.array([vmt_example]))[0])

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
            "k": self.k,
            "v_th": self.v_th,
            "example_v": v_example,
            "example_vmt": vmt_example,
            "example_surrogate": example_surrogate,
            "example_sigmoid": example_sigmoid,
            "example_spike": example_spike,
        }

        def frame(label: str, explanation: str, equation: str = "", **overrides) -> Frame:
            values = dict(base)
            values.update(overrides)
            return Frame(label, values, explanation, equation)

        checkpoints = [
            frame(
                "A função de disparo (forward)",
                f"No forward, o neurônio sempre usa esta função em degrau: dispara ou não dispara. "
                f"Com v_th = {self.v_th:g}, ele dispara (S = 1) quando v >= {self.v_th:g} e fica "
                f"em silêncio (S = 0) abaixo disso. No nosso exemplo, v = {v_example:g} >= {self.v_th:g}: "
                f"dispara, S = {example_spike:g}.",
                equation="S(v) = 1 se v >= v_th; 0 caso contrario",
            ),
            frame(
                "A sigmoide suave por trás do gradiente substituto",
                f"O degrau em si não muda. Mas a curva em S mostrada aqui é a antiderivada exata do "
                f"gradiente substituto que será usado no backward — ou seja, a inclinação dessa "
                f"sigmoide em cada ponto é, por construção, exatamente a curva de gradiente que vem "
                f"a seguir. No nosso exemplo, v = {v_example:g} (v - v_th = {vmt_example:g}): a "
                f"sigmoide vale {example_sigmoid:.2f} e a sua inclinação ali é {example_surrogate:.2f}.",
                equation="sigmoide(v) = 0,5 + (v - v_th) / (1 + k|v - v_th|)",
                sigmoid_reveal=1.0,
            ),
            frame(
                "A derivada real",
                f"A derivada real do degrau é zero em quase todo ponto — inútil para descida de "
                f"gradiente. Em v = {v_example:g} (v - v_th = {vmt_example:g}), dS/dv = 0, "
                f"exatamente como no STE do BitNet: o gradiente não tem por onde passar.",
                equation="dS/dv = 0 (quase todo ponto)",
                bottom_reveal=1.0,
                sigmoid_reveal=1.0,
            ),
            frame(
                "O gradiente substituto",
                f"Observe o traço se formando da esquerda para a direita: a sigmoide e o gradiente são "
                f"desenhados juntos, no mesmo ritmo, e a altura do gradiente em cada ponto é exatamente "
                f"a inclinação da sigmoide naquele mesmo ponto — é literalmente de onde o gradiente vem. "
                f"Em v = {v_example:g}, com k = {self.k:g}, o gradiente substituto vale "
                f"{example_surrogate:.2f} (em vez do {0.0:.0f} real).",
                equation="dS/dv ~= 1 / (1 + k|v - v_th|)^2",
                bottom_reveal=1.0,
                sigmoid_reveal=1.0,
                draw_reveal=1.0,
            ),
            frame(
                "Os dois juntos",
                f"Forward continua discreto (spike/não-spike: em v = {v_example:g}, S = "
                f"{example_spike:g}); só o backward usa a curva suave (gradiente substituto = "
                f"{example_surrogate:.2f} no mesmo ponto). "
                f"Análogo ao STE do BitNet, mas com uma função diferente — não é a mesma técnica.",
                bottom_reveal=1.0,
                overlay_reveal=1.0,
                sigmoid_reveal=1.0,
                draw_reveal=1.0,
            ),
        ]
        return build_sequence(checkpoints, steps=[8, 0, 20, 5])
