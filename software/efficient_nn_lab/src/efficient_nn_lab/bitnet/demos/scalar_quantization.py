"""Demonstração 1 — Quantização escalar (ESPECIFICACAO_DLVL.md #7).

Single question answered: what does it mean to quantize one weight?
A real-valued w slides visually, continuously, to its ternary level Q(w).
Two checkpoints ("peso real", "resultado quantizado"); the slide itself is
the animated transition between them (core.demo.build_sequence), not a
row of separate slideshow frames the user has to click through one by one.
"""

from __future__ import annotations

from efficient_nn_lab.core.demo import DemoModule, Frame, build_sequence
from efficient_nn_lab.bitnet.quantization import DEFAULT_THRESHOLD, ternary_quantize

_SLIDE_TWEEN_STEPS = 12


class ScalarQuantizationDemo(DemoModule):
    title = "BitNet -> Quantizacao"
    slug = "bitnet.quant"
    description = (
        "Um unico peso real w e quantizado para um dos tres niveis "
        "{-1, 0, +1}. Ajuste w e o limiar para ver a fronteira de decisao."
    )

    def __init__(self) -> None:
        self.w = 0.80
        self.threshold = DEFAULT_THRESHOLD
        super().__init__()

    def parameters(self) -> dict[str, dict[str, object]]:
        return {
            "w": {"label": "Peso real (w)", "min": -1.2, "max": 1.2, "step": 0.05, "value": self.w},
            "threshold": {
                "label": "Limiar (tau)",
                "min": 0.05,
                "max": 1.0,
                "step": 0.05,
                "value": self.threshold,
            },
        }

    def _build_frames(self) -> list[Frame]:
        q = ternary_quantize(self.w, self.threshold)
        in_dead_zone = abs(self.w) <= self.threshold

        real = Frame(
            label="Peso real",
            values={"kind": "scalar_quantization", "w_display": self.w, "w_real": self.w, "w_quant": None, "threshold": self.threshold, "revealed": False},
            explanation=f"$w = {self.w:.2f}$. Este é o parâmetro em precisão plena, antes de qualquer quantização.",
        )
        result = Frame(
            label="Resultado quantizado",
            values={"kind": "scalar_quantization", "w_display": float(q), "w_real": self.w, "w_quant": q, "threshold": self.threshold, "revealed": True},
            explanation=(
                f"$Q({self.w:.2f}) = {q:+d}$. "
                f"{'Dentro' if in_dead_zone else 'Fora'} da zona morta "
                f"$[-{self.threshold:.2f}, {self.threshold:.2f}]$."
            ),
            equation="Q(w) = +1 \\text{ se: } w > tau; -1 \\text{ se: } w < -tau; 0 \\text{ caso contrario}.",
        )
        return build_sequence([real, result], steps=_SLIDE_TWEEN_STEPS)
