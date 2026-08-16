"""Spike function (forward) and surrogate gradient (backward-only).

(ESPECIFICACAO_DLVL.md #20.) The forward spike function is a genuine
Heaviside step — the neuron really does either spike or not, nothing about
that changes during training. The "fast sigmoid" surrogate used here for
the backward pass is a common, simple choice (in the spirit of Neftci,
Mostafa & Zenke, 2019) — a specific, didactic pick, not a reproduction of
any single paper's exact formula. It is never used in the forward pass.
"""

from __future__ import annotations

import numpy as np


def heaviside(v_minus_th: np.ndarray) -> np.ndarray:
    """The real forward spike function: 1 if v >= v_th else 0."""
    return (np.asarray(v_minus_th) >= 0).astype(float)


def heaviside_derivative(v_minus_th: np.ndarray) -> np.ndarray:
    """The true derivative of the step: zero almost everywhere.

    Shown only to make the problem visible — this is *not* what is used
    for training.
    """
    return np.zeros_like(np.asarray(v_minus_th, dtype=float))


def fast_sigmoid_surrogate(v_minus_th: np.ndarray, k: float = 5.0) -> np.ndarray:
    """Smooth stand-in for the derivative, used only in the backward pass.

    d/dx [ x / (1 + k|x|) ] = 1 / (1 + k|x|)^2 — peaks at the threshold and
    decays away from it, unlike the true derivative above.
    """
    x = np.asarray(v_minus_th, dtype=float)
    return 1.0 / (1.0 + k * np.abs(x)) ** 2


def fast_sigmoid(v_minus_th: np.ndarray, k: float = 5.0) -> np.ndarray:
    """The smooth S-curve that `fast_sigmoid_surrogate` is the slope of.

    x / (1 + k|x|), shifted to sit at 0.5 at the threshold, is the exact
    antiderivative of `fast_sigmoid_surrogate` above (not just a
    similarly-shaped sigmoid picked separately) — differentiate it and the
    k-dependent factors cancel exactly, so the curve drawn here and the
    surrogate-gradient curve are honestly a function/derivative pair, not
    two independently-chosen formulas that merely look alike.
    """
    x = np.asarray(v_minus_th, dtype=float)
    return 0.5 + x / (1.0 + k * np.abs(x))
