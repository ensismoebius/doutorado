"""Demonstração 4 + 5 — O problema do backward e o STE.

(ESPECIFICACAO_DLVL.md #11, #12.)

Single question answered, in two parts: why is the quantization step a
problem for ordinary backpropagation, and how does the Straight-Through
Estimator route a gradient through it anyway?

One concrete worked example runs through all three scenes so every number
shown is the *same* number: w = 0.65, tau = 0.5 -> Q(w) = +1, y = x*Q(w) =
2, L = 2, dL/dQ(w) = -2, dL/dw = 0 (real derivative) vs -2 (STE). The
values come out of the real quantization/STE/linear code (bitnet/linear.py,
bitnet/ste.py), never hand-typed into the f-strings.

Two persistent scenes, not five disconnected pictures: the staircase
curve fades in its "why this breaks backprop" annotation rather than
being redrawn, and the forward/backward path diagram is one fixed layout
where the backward arrows fade and grow in on top of the (already
visible) forward path — nothing is ever wiped and replaced.
"""

from __future__ import annotations

import numpy as np

from efficient_nn_lab.core.demo import DemoModule, Frame, build_sequence
from efficient_nn_lab.bitnet.linear import (
    loss_gradient_wrt_y,
    quantized_forward,
    squared_error_loss,
)
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
        self.x = 2.0
        self.w = 0.65
        self.target = 4.0
        self.threshold = DEFAULT_THRESHOLD
        super().__init__()

    def parameters(self) -> dict[str, dict[str, object]]:
        return {
            "w": {
                "label": "Peso real (w)",
                "min": -1.2,
                "max": 1.2,
                "step": 0.05,
                "value": self.w,
            },
            "target": {
                "label": "Alvo (target)",
                "min": -10.0,
                "max": 10.0,
                "step": 0.5,
                "value": self.target,
            },
            "threshold": {
                "label": "Limiar (tau)",
                "min": 0.05,
                "max": 1.0,
                "step": 0.05,
                "value": self.threshold,
            },
        }

    def _build_frames(self) -> list[Frame]:
        # The one worked example every scene references. All numbers below
        # come from the real library code so the explanation text and the
        # widget boxes can never drift apart from what quantization/STE
        # actually compute.
        result = quantized_forward((self.x,), (self.w,), self.threshold)
        w_quant = result.w_quant[0]
        y = result.y
        loss = squared_error_loss(y, self.target)
        upstream_grad = loss_gradient_wrt_y(y, self.target)  # dL/dQ(w)
        dq_dw_real = 0.0  # true local derivative of Q on a flat region
        dq_dw_ste = 1.0  # what STE substitutes: derivative of the identity
        dl_dw_real = upstream_grad * dq_dw_real
        dl_dw_ste = upstream_grad * dq_dw_ste
        tau = self.threshold

        curve = staircase(_CURVE_W, tau)

        def staircase_frame(label: str, explanation: str, annotate: float, equation: str = "") -> Frame:
            return Frame(
                label,
                {
                    "kind": "staircase",
                    "w": _CURVE_W,
                    "q": curve,
                    "threshold": tau,
                    "example_w": self.w,
                    "annotate_reveal": annotate,
                },
                explanation,
                equation,
            )

        def path_values(fwd: float, bwd: float, joined: float) -> dict[str, object]:
            return {
                "kind": "ste_pipeline",
                "fwd_reveal": fwd,
                "bwd_reveal": bwd,
                "joined_reveal": joined,
                "w": self.w,
                "w_quant": w_quant,
                "x": self.x,
                "y": y,
                "loss": loss,
                "upstream_grad": upstream_grad,
                "dq_dw_real": dq_dw_real,
                "dq_dw_ste": dq_dw_ste,
                "dl_dw_real": dl_dw_real,
                "dl_dw_ste": dl_dw_ste,
                "threshold": tau,
                "example_w": self.w,
            }

        def path_frame(label: str, explanation: str, fwd: float, bwd: float, joined: float, equation: str = "") -> Frame:
            return Frame(
                label,
                path_values(fwd, bwd, joined),
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
                    "threshold": tau,
                    "example_w": self.w,
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
                f"Q(w) tem três regiões planas (derivada zero) separadas por dois saltos "
                f"(derivada indefinida). No nosso exemplo, w = {self.w:g} cai acima de "
                f"τ = {tau:g}, então Q(w) = {w_quant:+d} — e nessa região a derivada local é zero.",
                annotate=0.0,
                equation="Q(w) = +1 se w > tau; -1 se w < -tau; 0 caso contrario.",
            ),
            staircase_frame(
                "Por que isso quebra a retropropagação",
                f"Para w = {self.w:g} a derivada local de Q é dQ/dw = {dq_dw_real:g}. A "
                f"retropropagação padrão multiplica o gradiente por ela: dL/dw = dL/dQ(w) · "
                f"dQ/dw = {upstream_grad:g} · {dq_dw_real:g} = {dl_dw_real:g}. O gradiente "
                f"morre aqui.",
                annotate=1.0,
            ),
            derivative_frame(
                "A derivada real, em gráfico",
                f"Plotando dQ/dw diretamente: uma reta achatada em zero, do início ao fim. "
                f"Para w = {self.w:g}, dQ/dw = {dq_dw_real:g} — não há inclinação nenhuma para "
                f"seguir. Nos dois pontos de salto (w = ±τ = ±{tau:g}) a derivada nem sequer existe.",
                curve=real_derivative,
                reveal=1.0,
                overlay=0.0,
                equation="dQ/dw = 0 (quase toda parte); indefinida em w = +-tau",
            ),
            derivative_frame(
                "O gradiente que o STE usa de verdade",
                f"O STE substitui essa reta zerada por outra: a derivada da função identidade, "
                f"que vale 1 em todo lugar. Em w = {self.w:g}, o STE usa dQ/dw = {dq_dw_ste:g} "
                f"no lugar de {dq_dw_real:g}. É uma troca deliberada, não uma aproximação da "
                f"derivada real — repare que a curva não fica parecida com Q(w) em nenhum ponto.",
                curve=ste_derivative,
                reveal=1.0,
                overlay=1.0,
                equation="dQ/dw substituida por 1 (derivada da identidade)",
            ),
            path_frame(
                "Caminho do forward",
                f"Peso real w = {self.w:g} passa pela quantização: como {self.w:g} > τ = {tau:g}, "
                f"Q(w) = {w_quant:+d}. Esse valor ternário é o que participa da operação: "
                f"y = x · Q(w) = {self.x:g} · {w_quant:+d} = {y:g}.",
                fwd=1.0,
                bwd=0.0,
                joined=0.0,
            ),
            path_frame(
                "Caminho do backward (STE)",
                f"Da perda chega o gradiente dL/dQ(w) = {upstream_grad:g}. O STE ignora a derivada "
                f"real (dQ/dw = {dq_dw_real:g}) e usa a da identidade: dL/dw ≈ dL/dQ(w) · 1 = "
                f"{dl_dw_ste:g}. Sem o STE, esse gradiente seria {dl_dw_real:g}.",
                fwd=1.0,
                bwd=1.0,
                joined=0.0,
                equation="dL/dw ~= dL/dQ(w)  (dQ/dw substituido por 1)",
            ),
            path_frame(
                "Os dois caminhos juntos",
                f"Forward usa Q(w) = {w_quant:+d} (o valor real era w = {self.w:g}); backward finge "
                f"que Q(w) = w e entrega dL/dw = {dl_dw_ste:g} ao peso real. É essa assimetria "
                f"deliberada que faz o STE funcionar.",
                fwd=1.0,
                bwd=1.0,
                joined=1.0,
            ),
        ]
        # same-kind gaps tween smoothly; kind changes (staircase -> derivative
        # graph -> block diagram) are deliberate cuts, not blends of
        # unrelated pictures. The derivative graph's own gap (real -> STE)
        # tweens: watching zero morph into a flat 1 is the whole point.
        return build_sequence(checkpoints, steps=[7, 0, 10, 0, 8, 6])
