"""Métricas paraconsistentes (Gc, Gct) para avaliação de separabilidade."""

from __future__ import annotations

import numpy as np


def mu_lambda_from_probabilities(
    probs: np.ndarray, y_true: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Calcula μ e λ a partir das probabilidades de classe.

    μ: probabilidade da classe correta.
    λ: maior probabilidade entre classes incorretas.
    """
    n, c = probs.shape
    mu = probs[np.arange(n), y_true]

    mask = np.ones((n, c), dtype=bool)
    mask[np.arange(n), y_true] = False
    wrong = np.where(mask, probs, -np.inf)
    lam = np.max(wrong, axis=1)
    lam = np.where(np.isfinite(lam), lam, 0.0)
    return mu.astype(np.float32), lam.astype(np.float32)


def certainty_and_contradiction(
    mu: np.ndarray, lam: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    gc = mu - lam
    gct = mu + lam - 1.0
    return gc.astype(np.float32), gct.astype(np.float32)


def summarize_paraconsistent(gc: np.ndarray, gct: np.ndarray) -> dict[str, float]:
    return {
        "gc_mean": float(np.mean(gc)),
        "gc_std": float(np.std(gc)),
        "gct_mean": float(np.mean(gct)),
        "gct_std": float(np.std(gct)),
    }
