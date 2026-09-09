"""Dimensionality reduction for the latent-space explorer (FIXME §19).

Every projection here is tagged ``Origin.PROJECTED`` by the caller — these are
*not* measured or deterministically-recomputed pipeline values, they are a
lossy 2-D/3-D view of a higher-dimensional latent vector, and the GUI must say
so (FIXME §33).

Only PCA and t-SNE are provided; both ship with scikit-learn, which is already a
dependency. UMAP is deliberately not added (FIXME §19 "only if dependency is
justified").
"""

from __future__ import annotations

import numpy as np


def _as_matrix(vectors) -> np.ndarray:
    m = np.asarray(vectors, dtype=float)
    if m.ndim != 2:
        raise ValueError(f"projection input must be 2-D (n_samples, n_features), got {m.shape}")
    return m


def pca(vectors, n_components: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(coords, explained_variance_ratio)``.

    ``coords`` is ``(n_samples, n_components)``. Falls back to fewer components
    than requested only when the data rank is lower, and reports that through
    the ratio array length.
    """
    from sklearn.decomposition import PCA

    m = _as_matrix(vectors)
    k = int(min(n_components, m.shape[0], m.shape[1]))
    if k < 1:
        raise ValueError("need at least one sample and one feature to project")
    if np.allclose(m, m[0]):  # no variance — PCA would divide by zero
        return np.zeros((m.shape[0], n_components)), np.zeros(0)
    model = PCA(n_components=k, svd_solver="full")
    coords = model.fit_transform(m)
    if coords.shape[1] < n_components:  # pad so callers can always index n_components
        pad = np.zeros((coords.shape[0], n_components - coords.shape[1]))
        coords = np.hstack([coords, pad])
    return coords, np.asarray(model.explained_variance_ratio_, dtype=float)


def tsne(vectors, n_components: int = 2, *, perplexity: float | None = None,
         seed: int = 0) -> np.ndarray:
    """t-SNE embedding. ``perplexity`` defaults to ``min(30, n_samples/3)``."""
    from sklearn.manifold import TSNE

    m = _as_matrix(vectors)
    n = m.shape[0]
    if n < 4:
        raise ValueError(f"t-SNE needs >= 4 samples, got {n}")
    perp = float(perplexity) if perplexity else max(5.0, min(30.0, (n - 1) / 3.0))
    perp = min(perp, (n - 1) / 3.0)
    model = TSNE(n_components=n_components, perplexity=perp, init="pca",
                 random_state=seed, learning_rate="auto")
    return np.asarray(model.fit_transform(m), dtype=float)
