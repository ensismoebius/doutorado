"""The logistic sigmoid — the activation function traditional_gd.py's toy
neuron is missing without it: a linear neuron (y = w*x) has no nonlinearity
to speak of, so there is no activation curve, no saturation, and no
"where on the curve am I" story to tell. Adding sigma(z) is what makes the
chain rule in the backward pass actually have three links instead of two:
∂L/∂w = ∂L/∂y * dy/dz * dz/dw.
"""

from __future__ import annotations

import numpy as np


def sigmoid(z: np.ndarray | float) -> np.ndarray | float:
    """sigma(z) = 1 / (1 + e^-z)."""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivative(z: np.ndarray | float) -> np.ndarray | float:
    """dsigma/dz = sigma(z) * (1 - sigma(z))."""
    s = sigmoid(z)
    return s * (1.0 - s)
