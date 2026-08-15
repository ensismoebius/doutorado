"""Minimal linear neuron used by the BitNet forward/loss demos.

Deliberately tiny (two inputs, two weights) so every intermediate number
fits on screen at once — see ESPECIFICACAO_DLVL.md #9's worked example,
which this module reproduces exactly for the default parameters
(x1=2, x2=3, w1=0.8, w2=0.2 -> what1=+1, what2=0 -> y=2).
"""

from __future__ import annotations

from dataclasses import dataclass

from efficient_nn_lab.bitnet.quantization import ternary_quantize


@dataclass
class ForwardResult:
    x: tuple[float, ...]
    w_real: tuple[float, ...]
    w_quant: tuple[int, ...]
    products: tuple[float, ...]
    y: float


def quantized_forward(
    x: tuple[float, ...], w_real: tuple[float, ...], threshold: float = 0.5
) -> ForwardResult:
    """y = sum(x_i * Q(w_i)) — the BitNet-style forward pass for this toy neuron."""
    if len(x) != len(w_real):
        raise ValueError("x and w_real must have the same length")
    w_quant = tuple(ternary_quantize(w, threshold) for w in w_real)
    products = tuple(xi * wi for xi, wi in zip(x, w_quant))
    y = sum(products)
    return ForwardResult(x=x, w_real=w_real, w_quant=w_quant, products=products, y=y)


def squared_error_loss(y: float, target: float) -> float:
    """L = 1/2 (y - target)^2, matching ESPECIFICACAO_DLVL.md #10."""
    return 0.5 * (y - target) ** 2


def loss_gradient_wrt_y(y: float, target: float) -> float:
    """dL/dy = (y - target), the input to the STE backward step."""
    return y - target
