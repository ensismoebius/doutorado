"""Demonstração 4 + 5 — O problema do backward e o STE.

(ESPECIFICACAO_DLVL.md #11, #12.)

Single question answered, in two parts: why is the quantization step a
problem for ordinary backpropagation, and how does the Straight-Through
Estimator route a gradient through it anyway?

Two persistent scenes, not five disconnected pictures: the staircase
curve fades in its "why this breaks backprop" annotation rather than
being redrawn, and the forward/backward path diagram is one fixed layout
where the backward arrows fade and grow in on top of the (already
visible) forward path — nothing is ever wiped and replaced.
"""

from __future__ import annotations

import numpy as np

from efficient_nn_lab.core.demo import DemoModule, Frame, build_sequence
from efficient_nn_lab.bitnet.quantization import DEFAULT_THRESHOLD, staircase

#: Sample points for the staircase curve shown to the widget.
_CURVE_W = np.linspace(-1.5, 1.5, 400)


class BackwardSTEDemo(DemoModule):
    title = "BitNet -> Backward -> STE"
    slug = "bitnet.ste"
    description = (
        "A funcao de quantizacao e uma escada: constante em quase toda "
        "parte, descontinua em dois pontos. O STE contorna o problema "
        "usando um caminho diferente no forward e no backward."
    )

    def __init__(self) -> None:
        self.threshold = DEFAULT_THRESHOLD
        super().__init__()

    def parameters(self) -> dict[str, dict[str, object]]:
        return {
            "threshold": {
                "label": "Limiar (tau)",
                "min": 0.05,
                "max": 1.0,
                "step": 0.05,
                "value": self.threshold,
            }
        }

    def _build_frames(self) -> list[Frame]:
        curve = staircase(_CURVE_W, self.threshold)

        def staircase_frame(label: str, explanation: str, annotate: float, equation: str = "") -> Frame:
            return Frame(
                label,
                {
                    "kind": "staircase",
                    "w": _CURVE_W,
                    "q": curve,
                    "threshold": self.threshold,
                    "annotate_reveal": annotate,
                },
                explanation,
                equation,
            )

        def path_frame(label: str, explanation: str, fwd: float, bwd: float, joined: float, equation: str = "") -> Frame:
            return Frame(
                label,
                {"kind": "ste_pipeline", "fwd_reveal": fwd, "bwd_reveal": bwd, "joined_reveal": joined},
                explanation,
                equation,
            )

        real_derivative = np.zeros_like(_CURVE_W)
        ste_derivative = np.ones_like(_CURVE_W)

        def derivative_frame(
            label: str, explanation: str, curve: np.ndarray, reveal: float, overlay: float, equation: str = ""
        ) -> Frame:
            return Frame(
                label,
                {
                    "kind": "quant_derivative",
                    "w": _CURVE_W,
                    "threshold": self.threshold,
                    "curve": curve,
                    "reveal": reveal,
                    "overlay_reveal": overlay,
                },
                explanation,
                equation,
            )

        checkpoints = [
            staircase_frame(
                "A função em degrau",
                "Q(w) tem três regiões planas (derivada zero) separadas por dois saltos "
                "(derivada indefinida) — nenhum dos dois serve para descida de gradiente.",
                annotate=0.0,
                equation="Q(w) = +1 se w > tau; -1 se w < -tau; 0 caso contrario.",
            ),
            staircase_frame(
                "Por que isso quebra a retropropagação",
                "Retropropagação padrão multiplica gradientes pela derivada local. Aqui essa "
                "derivada é zero (região plana) ou indefinida (salto): o gradiente não tem por onde passar.",
                annotate=1.0,
            ),
            derivative_frame(
                "A derivada real, em gráfico",
                "Plotando dQ/dw diretamente: uma reta achatada em zero, do início ao fim — "
                "não há inclinação nenhuma para seguir. Nos dois pontos de salto (w = ±tau) "
                "a derivada nem sequer existe.",
                curve=real_derivative,
                reveal=1.0,
                overlay=0.0,
                equation="dQ/dw = 0 (quase toda parte); indefinida em w = +-tau",
            ),
            derivative_frame(
                "O gradiente que o STE usa de verdade",
                "O STE substitui essa reta zerada por outra: a derivada da função identidade, "
                "que vale 1 em todo lugar. É uma troca deliberada, não uma aproximação da "
                "derivada real — repare que a curva não fica parecida com Q(w) em nenhum ponto.",
                curve=ste_derivative,
                reveal=1.0,
                overlay=1.0,
                equation="dQ/dw substituida por 1 (derivada da identidade)",
            ),
            path_frame(
                "Caminho do forward",
                "No forward, o peso real w passa pela quantização e o valor "
                "ternário resultante é o que participa da operação.",
                fwd=1.0,
                bwd=0.0,
                joined=0.0,
            ),
            path_frame(
                "Caminho do backward (STE)",
                "No backward, o STE ignora a derivada real da quantização e "
                "deixa o gradiente da perda passar como se a quantização "
                "fosse a função identidade.",
                fwd=1.0,
                bwd=1.0,
                joined=0.0,
                equation="dL/dw ~= dL/dQ(w)  (dQ/dw substituido por 1)",
            ),
            path_frame(
                "Os dois caminhos juntos",
                "Forward usa Q(w); backward finge que Q(w) = w. "
                "É essa assimetria deliberada que faz o STE funcionar.",
                fwd=1.0,
                bwd=1.0,
                joined=1.0,
            ),
        ]
        # same-kind gaps tween smoothly; kind changes (staircase -> derivative
        # graph -> block diagram) are deliberate cuts, not blends of
        # unrelated pictures. The derivative graph's own gap (real -> STE)
        # tweens: watching zero morph into a flat 1 is the whole point.
        return build_sequence(checkpoints, steps=[14, 0, 20, 0, 16, 12])
