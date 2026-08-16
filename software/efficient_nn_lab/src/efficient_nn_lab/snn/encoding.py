"""Signal generation and the simplest possible signal-to-spike mapping.

(ESPECIFICACAO_DLVL.md #17.) This module intentionally does *not* use the
LIF neuron: the very first SNN demo should build the "spike = event"
intuition before membrane dynamics are introduced, so spikes here come
from direct level-crossing on a synthetic waveform, not from integration.
"""

from __future__ import annotations

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
