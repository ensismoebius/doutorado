"""Input normalization: per-feature z-score (fit once on train, e.g.
audio/CMVN-style) vs per-window z-score (recomputed for every window, e.g.
EEG) -- and the data-leakage hazard of fitting stats on test data too
(software/nn thesis chapter 07, sec:normalizacaoEntrada).
"""

from __future__ import annotations

import numpy as np


def zscore(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    return (np.asarray(x, dtype=float) - mean) / std


def fit_zscore(x: np.ndarray) -> tuple[float, float]:
    """Fit (mean, std) on ``x`` alone -- the correct, train-only path."""
    x = np.asarray(x, dtype=float)
    return float(x.mean()), float(x.std())


def leaky_fit_zscore(train: np.ndarray, test: np.ndarray) -> tuple[float, float]:
    """The data-leakage hazard: fitting stats on train+test combined,
    instead of train alone (the thesis's own "wrong path" diagram,
    images/normalizationFitTransform.tex).
    """
    combined = np.concatenate([np.asarray(train, dtype=float), np.asarray(test, dtype=float)])
    return float(combined.mean()), float(combined.std())
