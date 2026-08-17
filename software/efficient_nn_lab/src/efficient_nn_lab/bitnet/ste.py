"""Straight-Through Estimator (STE) — the numpy-only, GUI-facing version.

The GUI never depends on PyTorch (ESPECIFICACAO_DLVL.md #2.1: torch is only
for the *optional* reference module, bitnet/ste_torch_reference.py). Here
the STE is expressed directly as what it *means*, not as an autograd trick:

* forward: y = Q(w)               -- the quantized weight is actually used.
* backward: ∂L/∂w_real ~= ∂L/∂y   -- the gradient is passed through the
  quantization step unchanged, as if it had been the identity function.

This module exposes exactly those two pieces so a demo can show them
separately (ESPECIFICACAO_DLVL.md #12).
"""

from __future__ import annotations

from efficient_nn_lab.bitnet.quantization import ternary_quantize


def ste_forward(w_real: float, threshold: float = 0.5) -> int:
    """The value actually used in the forward pass: the quantized weight."""
    return ternary_quantize(w_real, threshold)


def ste_backward(grad_wrt_quantized: float) -> float:
    """Gradient handed back to the real-valued weight.

    STE's defining move: the (undefined/zero) derivative of Q is replaced
    by 1, so this is just the identity — the gradient that arrived at the
    quantized weight is forwarded unchanged to the real-valued one.
    """
    return grad_wrt_quantized


def sgd_update(w_real: float, grad: float, learning_rate: float) -> float:
    """One plain gradient-descent step on the real-valued (shadow) weight."""
    return w_real - learning_rate * grad
