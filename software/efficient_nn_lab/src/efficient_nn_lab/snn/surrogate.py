"""Spike function (forward) and surrogate gradient (backward-only).

(ESPECIFICACAO_DLVL.md #20.) The forward spike function is a genuine
Heaviside step — the neuron really does either spike or not, nothing about
that changes during training. The "fast sigmoid" surrogate used here for
the backward pass is a common, simple choice (in the spirit of Neftci,
Mostafa & Zenke, 2019) — a specific, didactic pick, not a reproduction of
any single paper's exact formula. It is never used in the forward pass.

The pair is normalized so it honestly stands in for the step:

* ``fast_sigmoid`` = 0.5 + 0.5·kx/(1 + k|x|) runs from 0 to 1 like the step
  itself and gets STEEPER as k grows (k -> inf gives the step back);
* ``fast_sigmoid_surrogate`` = (k/2)/(1 + k|x|)^2 is its exact derivative,
  so its area is 1 — the height of the step — and it gets TALLER and
  NARROWER as k grows (k -> inf gives the Dirac delta, the step's
  "derivative").

Many libraries (snnTorch's ``fast_sigmoid``, SuperSpike) use the same shape
with peak 1, ``1/(1 + k|x|)^2``. That is this surrogate times 2/k: a
constant factor, absorbed by the learning rate, that changes the step size
but never the direction of the gradient. The earlier peak-1 pairing here
drew 0.5 + x/(1 + k|x|) as "the smooth step", which only spans 0.5 ± 1/k --
it got FLATTER as k grew, the opposite of what the k slider teaches.
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

    d/dx [ 0.5 + 0.5·kx / (1 + k|x|) ] = (k/2) / (1 + k|x|)^2 — peaks at the
    threshold (height k/2) and decays away from it, unlike the true
    derivative above. Its integral over the whole line is exactly 1, the
    height of the step it replaces.
    """
    x = np.asarray(v_minus_th, dtype=float)
    return 0.5 * k / (1.0 + k * np.abs(x)) ** 2


def fast_sigmoid(v_minus_th: np.ndarray, k: float = 5.0) -> np.ndarray:
    """The smooth step that `fast_sigmoid_surrogate` is the slope of.

    0.5 + 0.5·kx / (1 + k|x|) goes from 0 (far below threshold) through 0.5
    (at threshold) to 1 (far above) -- the same range as the Heaviside step
    it is drawn over -- and approaches that step as k grows. It is the exact
    antiderivative of `fast_sigmoid_surrogate` (not a similarly-shaped curve
    picked separately), so the two plotted curves are honestly a
    function/derivative pair.
    """
    x = np.asarray(v_minus_th, dtype=float)
    return 0.5 + 0.5 * k * x / (1.0 + k * np.abs(x))
