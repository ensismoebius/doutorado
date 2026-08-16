"""Signal generation and the simplest possible signal-to-spike mapping.

(ESPECIFICACAO_DLVL.md #17.) This module intentionally does *not* use the
LIF neuron: the very first SNN demo should build the "spike = event"
intuition before membrane dynamics are introduced, so spikes here come
from direct level-crossing on a synthetic waveform, not from integration.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from efficient_nn_lab.core.math_utils import SEED


def synthetic_signal(n_steps: int = 60) -> np.ndarray:
    """A deterministic bump-shaped waveform, not unlike a spoken syllable's envelope."""
    t = np.linspace(0.0, 4.0 * np.pi, n_steps)
    return np.clip(np.sin(t) * np.exp(-((t - 6.0) ** 2) / 18.0) * 3.0, -1.0, 1.0)


def direct_threshold_spikes(signal: np.ndarray, level: float = 0.4) -> np.ndarray:
    """Spike whenever the signal crosses ``level`` from below (rising edge only)."""
    signal = np.asarray(signal, dtype=float)
    above = signal >= level
    rising_edge = np.zeros_like(signal)
    rising_edge[1:] = np.logical_and(above[1:], ~above[:-1]).astype(float)
    rising_edge[0] = float(above[0])
    return rising_edge


def spike_probability(signal: np.ndarray, max_rate: float = 0.9) -> np.ndarray:
    """Map signal intensity to a per-step spike probability (rate coding).

    Only the non-negative part of the signal carries intensity here — the
    same convention `direct_threshold_spikes` uses implicitly by comparing
    against a positive level. Intensity 0 -> probability 0; intensity 1 ->
    probability `max_rate` (kept below 1 so even a maximally-intense input
    still looks like a *rate*, not a spike on every single step).
    """
    intensity = np.clip(np.asarray(signal, dtype=float), 0.0, 1.0)
    return intensity * max_rate


def poisson_spikes(signal: np.ndarray, max_rate: float = 0.9, seed: int = SEED) -> np.ndarray:
    """Rate (Poisson) coding: an independent coin flip per time-step.

    Unlike `direct_threshold_spikes`, where the same intensity always
    produces the same outcome, here the probability of a spike is
    proportional to intensity but the outcome itself is a draw — two
    identical-looking steps in the signal can differ in whether they spike.
    The draw uses a fixed, explicit seed (ESPECIFICACAO_DLVL.md #35) so
    replaying the same demo always reproduces the exact same spike train,
    rather than a different one on every launch.
    """
    prob = spike_probability(signal, max_rate)
    rng = np.random.RandomState(seed)
    draws = rng.random_sample(prob.shape)
    return (draws < prob).astype(float)


def load_grayscale_image(path: str | Path, size: tuple[int, int] = (32, 32)) -> np.ndarray:
    """Load an image file as a small grayscale intensity grid in [0, 1].

    Nearest-neighbor downsampling (index picking, not averaging) keeps
    this to plain numpy — no new dependency beyond matplotlib's own image
    reader, which every install already has (it needs Pillow for JPEG).
    """
    import matplotlib.image as mpimg

    img = mpimg.imread(str(path)).astype(float)
    gray = img @ np.array([0.299, 0.587, 0.114]) if img.ndim == 3 else img
    if gray.max() > 1.0:
        gray = gray / 255.0
    rows = np.linspace(0, gray.shape[0] - 1, size[0]).astype(int)
    cols = np.linspace(0, gray.shape[1] - 1, size[1]).astype(int)
    return gray[np.ix_(rows, cols)]


def poisson_spike_frames(
    intensity: np.ndarray, n_steps: int, max_rate: float = 0.9, seed: int = SEED
) -> np.ndarray:
    """`poisson_spikes`, generalized to an N-D intensity grid over time.

    Every element of `intensity` (e.g. every pixel of an image) becomes
    its own independent Poisson-coded neuron: same probability rule, one
    fresh coin flip per element per time-step. Returns an array shaped
    ``(n_steps, *intensity.shape)``.
    """
    prob = spike_probability(intensity, max_rate)
    rng = np.random.RandomState(seed)
    draws = rng.random_sample((n_steps,) + prob.shape)
    return (draws < prob[None, ...]).astype(float)
