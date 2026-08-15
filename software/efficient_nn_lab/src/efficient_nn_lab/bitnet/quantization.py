"""Scalar/ternary quantization used by the BitNet demos.

This is a *didactic* simplification (ESPECIFICACAO_DLVL.md #8, #32): a
fixed-threshold ternary quantizer, not a reproduction of BitNet b1.58's
absmean quantization (which uses a tensor-wide scale gamma = mean(|W|)
rather than a hand-set threshold). It is enough to make the forward/
backward story concrete with one weight at a time; it must never be
presented as "the" BitNet quantization function.
"""

from __future__ import annotations

import numpy as np

#: Default decision threshold for the didactic quantizer (matches
#: ESPECIFICACAO_DLVL.md #8's tau = 0.5 example).
DEFAULT_THRESHOLD = 0.5


def ternary_quantize(w: float, threshold: float = DEFAULT_THRESHOLD) -> int:
    """Q(w) in {-1, 0, +1} with a symmetric dead-zone of width 2*threshold."""
    if w > threshold:
        return 1
    if w < -threshold:
        return -1
    return 0


def ternary_quantize_np(w: np.ndarray, threshold: float = DEFAULT_THRESHOLD) -> np.ndarray:
    """Vectorized version of :func:`ternary_quantize`."""
    q = np.zeros_like(w, dtype=float)
    q = np.where(w > threshold, 1.0, q)
    q = np.where(w < -threshold, -1.0, q)
    return q


def staircase(w_values: np.ndarray, threshold: float = DEFAULT_THRESHOLD) -> np.ndarray:
    """Q(w) evaluated over an array — the "staircase" curve for plotting.

    Used by the backward-problem demo to draw the quantization function
    and make its flat regions / discontinuities visible (see
    ESPECIFICACAO_DLVL.md #11).
    """
    return ternary_quantize_np(w_values, threshold)
