"""Demonstração 4 + 5 — O problema do backward e o STE.

(ESPECIFICACAO_DLVL.md #11, #12.)

Single question answered, in two parts: why is the quantization step a
problem for ordinary backpropagation, and how does the Straight-Through
Estimator route a gradient through it anyway?

One concrete worked example runs through all three scenes so every number
shown is the *same* number: x = 2, w = 0.65, tau = 0.5 -> Q(w) = +1,
y = x*Q(w) = 2, L = 2, ∂L/∂y = y - target = -2, and -- because y = x*Q(w) --
∂L/∂Q(w) = ∂L/∂y * x = -4. The real chain then multiplies by dQ/dw = 0
(∂L/∂w = 0); the STE multiplies by 1 instead (∂L/∂w = -4, the same number
the guided "Do peso real ao BitNet" sequence computes). The values come out
of the real quantization/STE/linear code (bitnet/linear.py, bitnet/ste.py),
never hand-typed into the f-strings, and every sentence that depends on
where w sits relative to tau is chosen from the actual position -- the
sliders can move w into the dead zone or onto a jump.

Two persistent scenes, not five disconnected pictures: the staircase
curve fades in its "why this breaks backprop" annotation rather than
being redrawn, and the forward/backward path diagram is one fixed layout
where the backward arrows fade and grow in on top of the (already
visible) forward path — nothing is ever wiped and replaced.
"""

from __future__ import annotations

import math

import numpy as np

from efficient_nn_lab.bitnet.linear import (
    loss_gradient_wrt_y,
    quantized_forward,
    squared_error_loss,
)
from efficient_nn_lab.bitnet.quantization import DEFAULT_THRESHOLD, format_level, staircase
from efficient_nn_lab.bitnet.ste import ste_backward
from efficient_nn_lab.core.demo import DemoModule, Frame, build_sequence, slider

#: Sample points for the staircase curve shown to the widget.
_CURVE_W = np.linspace(-1.5, 1.5, 400)


def _position(w: float, tau: float) -> tuple[str, bool]:
    """Where w sits on the staircase, in words, and whether that is a jump.

    The slider steps (0.05) let w land exactly on +-tau, where Q jumps and
    the derivative does not exist -- the text must say so instead of
    claiming a flat region.
    """
    if math.isclose(abs(w), tau, abs_tol=1e-9):
        return f"cai exatamente no salto $w = {'+' if w > 0 else '-'}τ$", True
    if w > tau:
        return f"cai acima de $τ = {tau:g}$", False
    if w < -tau:
        return f"cai abaixo de $-τ = {-tau:g}$", False
    return f"cai dentro da zona morta $[-{tau:g}, {tau:g}]$", False


class BackwardSTEDemo(DemoModule):
    title = "BitNet -> Backward -> STE"
    slug = "bitnet.ste"
    description = (
        "A função de quantização é uma escada: constante em quase toda "
        "parte, descontínua em dois pontos. O STE contorna o problema "
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
            "w": slider("Peso real (w)", -1.2, 1.2, 0.05, self.w),
            "target": slider("Alvo (target)", -10.0, 10.0, 0.5, self.target),
            "threshold": slider("Limiar (tau)", 0.05, 1.0, 0.05, self.threshold),
        }

    def _build_frames(self) -> list[Frame]:
        # The one worked example every scene references. All numbers below
        # come from the real library code so the explanation text and the
        # widget boxes can never drift apart from what quantization/STE
        # actually compute.
        result = quantized_forward((self.x,), (self.w,), self.threshold)
        w_quant = result.w_quant[0]
        q_txt = format_level(w_quant)
        y = result.y
        loss = squared_error_loss(y, self.target)
        grad_y = loss_gradient_wrt_y(y, self.target)  # ∂L/∂y = y - target
        # y = x * Q(w), so the gradient that reaches the quantizer's OUTPUT
        # still has to pass through the multiplication by x first.
        upstream_grad = grad_y * self.x  # ∂L/∂Q(w) = ∂L/∂y * ∂y/∂Q(w)
        dq_dw_real = 0.0  # true local derivative of Q (0 on a flat region; autograd also yields 0 at a jump)
        dq_dw_ste = 1.0  # what STE substitutes: derivative of the identity
        # "+ 0.0" turns IEEE -0.0 (e.g. -4 * 0) into 0.0, so the screen never
        # prints a "-0" gradient.
        dl_dw_real = upstream_grad * dq_dw_real + 0.0
        dl_dw_ste = ste_backward(upstream_grad) * dq_dw_ste
        tau = self.threshold
        where, at_jump = _position(self.w, tau)

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
                "grad_y": grad_y,
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

        if at_jump:
            local_derivative = "exatamente no salto, onde a derivada nem existe"
            breaks = (
                f"Em $w = {self.w:g}$, no salto, $dQ/dw$ não existe (o autograd devolve 0: a comparação "
                "com τ não tem gradiente)."
            )
            plotted = f"Em $w = {self.w:g}$, exatamente no salto, a derivada nem sequer existe."
        else:
            local_derivative = "e nessa região a derivada local é zero"
            breaks = f"Em $w = {self.w:g}$, $dQ/dw = {dq_dw_real:g}$."
            plotted = (
                f"Para $w = {self.w:g}$, $dQ/dw = {dq_dw_real:g}$ — não há inclinação nenhuma para seguir. "
                f"Nos dois pontos de salto ($w = ±τ = ±{tau:g}$) a derivada nem sequer existe."
            )

        checkpoints = [
            staircase_frame(
                "A função em degrau",
                f"Q(w) tem três regiões planas (derivada zero) separadas por dois saltos "
                f"(derivada indefinida). No nosso exemplo, $w = {self.w:g}$ {where}, então "
                f"$Q(w) = {q_txt}$ — {local_derivative}.",
                annotate=0.0,
                equation="Q(w) = +1 \\text{ se: } w > tau; -1 \\text{ se: } w < -tau; 0 \\text{ caso contrário}.",
            ),
            staircase_frame(
                "Por que isso quebra a retropropagação",
                f"{breaks} A backprop multiplica por ela o gradiente que chega a Q(w), "
                f"$∂L/∂Q(w) = ∂L/∂y · x = {upstream_grad:g}$: "
                f"$∂L/∂w = {upstream_grad:g} · {dq_dw_real:g} = {dl_dw_real:g}$. O gradiente morre aqui.",
                annotate=1.0,
            ),
            derivative_frame(
                "A derivada real, em gráfico",
                f"Plotando $dQ/dw$ diretamente: uma reta achatada em zero, do início ao fim. {plotted}",
                curve=real_derivative,
                reveal=1.0,
                overlay=0.0,
                equation="dQ/dw = 0 (quase toda parte); indefinida em w = +-tau",
            ),
            derivative_frame(
                "O gradiente que o STE usa de verdade",
                f"O STE substitui essa reta zerada por outra: a derivada da função identidade, "
                f"que vale 1 em todo lugar. Em $w = {self.w:g}$, o STE usa "
                f"$dQ/dw = {dq_dw_ste:g}$ no lugar de $dQ/dw = {dq_dw_real:g}$. É uma troca "
                f"deliberada: no backward o STE finge que $Q(w) = w$ — que é o que a escada parece "
                f"vista de longe, uma rampa em degraus. A derivada real, zero em cada degrau, nunca "
                f"enxerga essa tendência de subida; a identidade enxerga.",
                curve=ste_derivative,
                reveal=1.0,
                overlay=1.0,
                equation="dQ/dw substituída por 1 (derivada da identidade)",
            ),
            path_frame(
                "Caminho do forward",
                f"Peso real $w = {self.w:g}$ passa pela quantização: ele {where}, então "
                f"$Q(w) = {q_txt}$. Esse valor ternário é o que participa da operação: "
                f"$y = x · Q(w) = {self.x:g} · {q_txt} = {y:g}$.",
                fwd=1.0,
                bwd=0.0,
                joined=0.0,
            ),
            path_frame(
                "Caminho do backward (STE)",
                f"Como y = x · Q(w), o gradiente chega a Q(w) já multiplicado por x: "
                f"$∂L/∂Q(w) = ∂L/∂y · x = {grad_y:g} · {self.x:g} = {upstream_grad:g}$. O STE troca dQ/dw = 0 por 1: "
                f"$∂L/∂w ≈ {dl_dw_ste:g}$. Sem o STE, seria {dl_dw_real:g}.",
                fwd=1.0,
                bwd=1.0,
                joined=0.0,
                equation="∂L/∂w ≈ ∂L/∂Q(w) = ∂L/∂y · x  (dQ/dw substituído por 1)",
            ),
            path_frame(
                "Os dois caminhos juntos",
                f"Forward usa $Q(w) = {q_txt}$ (o valor real era $w = {self.w:g}$); backward "
                f"finge que $Q(w) = w$ e entrega $∂L/∂w = {dl_dw_ste:g}$ ao peso real. É essa "
                f"assimetria deliberada que faz o STE funcionar.",
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
