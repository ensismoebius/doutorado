"""Demonstração 2 + 3 — Forward e erro (ESPECIFICACAO_DLVL.md #9, #10).

Single question answered: what happens in the forward pass, and how far is
the result from the target? Reproduces the spec's worked example exactly
for the default parameters: x=(2,3), w=(0.8,0.2) -> what=(+1,0) -> y=2,
target=4 -> loss=2.

Everything the diagram can ever show (inputs, both weight values, both
products, y, target, diff, loss) is present in every single frame's
values — nothing is ever `None` and swapped in abruptly. What changes
between checkpoints is a set of continuous 0..1 "reveal" fields (opacity /
arrow fill fraction) and, once per weight, the weight *value itself*
morphing from real to quantized. NeuronView renders the exact same
persistent picture every frame; only these continuous fields move, which
is what core.demo.build_sequence's tweening turns into smooth, connected
motion instead of a slideshow of unrelated pictures.

Quantizing w1 and w2 are now two separate checkpoints (not one "both at
once" step), each paired with a number-line panel that marks where that
weight actually sits relative to +-tau -- so *why* Q(w1)=+1 and Q(w2)=0
is something the viewer sees geometrically, not just a formula and a
final number.
"""

from __future__ import annotations

from efficient_nn_lab.core.demo import DemoModule, Frame, build_sequence
from efficient_nn_lab.bitnet.linear import (
    loss_gradient_wrt_y,
    quantized_forward,
    squared_error_loss,
)
from efficient_nn_lab.bitnet.quantization import DEFAULT_THRESHOLD


def _base_values(result, target: float, diff: float, loss: float, grad: float, threshold: float) -> dict[str, object]:
    """Every field the pipeline diagram can ever draw, all-zero reveal."""
    return {
        "kind": "forward_pipeline",
        "x": result.x,
        "w_real": result.w_real,
        "w_quant": result.w_quant,
        "threshold": threshold,
        "quant1_reveal": 0.0,
        "quant2_reveal": 0.0,
        "arrow1_fill": 0.0,
        "arrow2_fill": 0.0,
        "product1": result.products[0],
        "product1_reveal": 0.0,
        "product2": result.products[1],
        "product2_reveal": 0.0,
        "highlight1": 0.0,
        "highlight2": 0.0,
        "sum_reveal": 0.0,
        "y": result.y,
        "y_reveal": 0.0,
        "target": target,
        "target_reveal": 0.0,
        "diff": diff,
        "diff_reveal": 0.0,
        "grad_y": grad,
        "loss": loss,
        "loss_reveal": 0.0,
    }


class ForwardLossDemo(DemoModule):
    title = "BitNet -> Forward"
    slug = "bitnet.forward"
    description = (
        "Um neuronio linear com dois pesos quantizados: entradas fluem "
        "pelos pesos ternarios ate a saida, que e comparada a um alvo."
    )

    def __init__(self) -> None:
        self.x1 = 2.0
        self.x2 = 3.0
        self.w1 = 0.8
        self.w2 = 0.2
        self.threshold = DEFAULT_THRESHOLD
        self.target = 4.0
        super().__init__()

    def parameters(self) -> dict[str, dict[str, object]]:
        return {
            "x1": {"label": "Entrada x1", "min": -5.0, "max": 5.0, "step": 0.5, "value": self.x1},
            "x2": {"label": "Entrada x2", "min": -5.0, "max": 5.0, "step": 0.5, "value": self.x2},
            "w1": {"label": "Peso real w1", "min": -1.2, "max": 1.2, "step": 0.05, "value": self.w1},
            "w2": {"label": "Peso real w2", "min": -1.2, "max": 1.2, "step": 0.05, "value": self.w2},
            "target": {"label": "Alvo (target)", "min": -10.0, "max": 10.0, "step": 0.5, "value": self.target},
        }

    def _build_frames(self) -> list[Frame]:
        result = quantized_forward((self.x1, self.x2), (self.w1, self.w2), self.threshold)
        w1q, w2q = result.w_quant
        p1, p2 = result.products
        diff = result.y - self.target
        loss = squared_error_loss(result.y, self.target)
        grad = loss_gradient_wrt_y(result.y, self.target)
        tau = self.threshold

        def reason(w: float, wq: int) -> str:
            if wq > 0:
                return f"{w:g} > tau ({tau:g}), então Q(w) = +1"
            if wq < 0:
                return f"{w:g} < -tau (-{tau:g}), então Q(w) = -1"
            return f"-tau <= {w:g} <= tau, então Q(w) = 0"

        def frame(label: str, explanation: str, equation: str = "", **overrides) -> Frame:
            values = _base_values(result, self.target, diff, loss, grad, tau)
            values.update(overrides)
            return Frame(label, values, explanation, equation)

        checkpoints = [
            frame(
                "Entradas e pesos reais",
                f"x1={self.x1:g}, x2={self.x2:g}; pesos reais w1={self.w1:g}, w2={self.w2:g}. "
                "Nenhum peso ainda foi quantizado -- é a próxima etapa, uma de cada vez.",
            ),
            frame(
                "Quantizar w1",
                f"w1 = {self.w1:g}: {reason(self.w1, w1q)}. No número-linha ao lado, w1 cai "
                "fora da faixa cinza (a 'zona morta' entre -tau e tau) — por isso não vira zero.",
                equation="Q(w) = +1 se w > tau; -1 se w < -tau; 0 caso contrario.",
                quant1_reveal=1.0,
            ),
            frame(
                "Quantizar w2",
                f"w2 = {self.w2:g}: {reason(self.w2, w2q)}. Desta vez w2 cai dentro da faixa "
                "cinza — a zona morta existe exatamente para isso: pesos pequenos colapsam a zero.",
                equation="Q(w) = +1 se w > tau; -1 se w < -tau; 0 caso contrario.",
                quant1_reveal=1.0,
                quant2_reveal=1.0,
            ),
            frame(
                "Multiplicação 1",
                f"x1 . Q(w1) = {self.x1:g} . {w1q:+d} = {p1:g}.",
                equation="produto_1 = x1 . Q(w1)",
                quant1_reveal=1.0,
                quant2_reveal=1.0,
                arrow1_fill=1.0,
                product1_reveal=1.0,
                highlight1=1.0,
            ),
            frame(
                "Multiplicação 2",
                f"x2 . Q(w2) = {self.x2:g} . {w2q:+d} = {p2:g}. Como Q(w2) = 0, este produto "
                "é sempre zero, não importa quanto valha x2 -- o segundo peso não contribui em nada para y.",
                equation="produto_2 = x2 . Q(w2)",
                quant1_reveal=1.0,
                quant2_reveal=1.0,
                arrow1_fill=1.0,
                product1_reveal=1.0,
                arrow2_fill=1.0,
                product2_reveal=1.0,
                highlight2=1.0,
            ),
            frame(
                "Soma (saída y)",
                f"y = {p1:g} + {p2:g} = {result.y:g}.",
                equation="y = sum_i x_i . Q(w_i)",
                quant1_reveal=1.0,
                quant2_reveal=1.0,
                arrow1_fill=1.0,
                product1_reveal=1.0,
                arrow2_fill=1.0,
                product2_reveal=1.0,
                sum_reveal=1.0,
                y_reveal=1.0,
            ),
            frame(
                "Alvo",
                f"O alvo desta amostra é target = {self.target:g}.",
                quant1_reveal=1.0,
                quant2_reveal=1.0,
                arrow1_fill=1.0,
                product1_reveal=1.0,
                arrow2_fill=1.0,
                product2_reveal=1.0,
                sum_reveal=1.0,
                y_reveal=1.0,
                target_reveal=1.0,
            ),
            frame(
                "Diferença",
                f"diferença = y - target = {result.y:g} - {self.target:g} = {diff:g}.",
                quant1_reveal=1.0,
                quant2_reveal=1.0,
                arrow1_fill=1.0,
                product1_reveal=1.0,
                arrow2_fill=1.0,
                product2_reveal=1.0,
                sum_reveal=1.0,
                y_reveal=1.0,
                target_reveal=1.0,
                diff_reveal=1.0,
            ),
            frame(
                "Loss",
                f"L = 1/2 (y - target)^2 = 1/2 * ({diff:g})^2 = {loss:g}.",
                equation="L = 1/2 (y - target)^2",
                quant1_reveal=1.0,
                quant2_reveal=1.0,
                arrow1_fill=1.0,
                product1_reveal=1.0,
                arrow2_fill=1.0,
                product2_reveal=1.0,
                sum_reveal=1.0,
                y_reveal=1.0,
                target_reveal=1.0,
                diff_reveal=1.0,
                loss_reveal=1.0,
            ),
        ]
        return build_sequence(checkpoints)
