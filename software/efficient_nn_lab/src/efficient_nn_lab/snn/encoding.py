"""Signal generation and the simplest possible signal-to-spike mapping.

(ESPECIFICACAO_DLVL.md #17.) This module intentionally does *not* use the
LIF neuron: the very first SNN demo should build the "spike = event"
intuition before membrane dynamics are introduced, so spikes here come
from direct level-crossing on a synthetic waveform, not from integration.
"""

from __future__ import annotations

import numpy as np


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
