"""Data for comparison.autoencoders: synthetic windows, a REAL PCA, and
constants quoted from software/nn's Meeting01 experiment.

What is computed here is real math on synthetic signals: speech-like
windows of 256 samples, z-scored per window exactly as every Meeting01
loader does (`include/utility/WindowZScore.hpp`: one mean/std over the whole
window), a PCA actually fitted on the training windows (never on the window
it is scored on) -- the linear reference
`scripts/pipeline/meeting01/03_meeting01_pca_mean_baselines.py` fits per
fold, there on train + val, the same non-test data the families' final fit
sees -- and the mean-frame reference.

What is NOT here, on purpose: any reconstruction or error for the four
trained families (SNN-AE, LSTM-AE, GRU-AE, Transformer-AE). They are not
trained by this app, and an invented ranking shown in a lecture becomes a
"result" nobody can trace. The demo leaves their bars empty and says why.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from efficient_nn_lab.core.math_utils import SEED

WINDOW = 256
#: latent_dim is fixed per dataset in profiles/meeting01-loso.json (never a
#: GA gene), so every family is compared at the same compression ratio:
LATENT_AUDIO = 16  # fsdd, audiomnist: 256:16 = 16:1
LATENT_EEG = 64  # eegmmidb, siena: 256:64 = 4:1

FAMILIES = ("SNN-AE", "LSTM-AE", "GRU-AE", "Transformer-AE")

#: Mean wall-clock per epoch, MEASURED 2026-09-23 at production settings
#: (window 256, T = 16, 200 train / 150 val windows, batch 1, CPU; SNN 64->32,
#: LSTM/GRU hidden 64, Transformer d_model 64 / 4 heads / 2 layers / d_ff 128)
#: -- quoted from meeting01-loso.json's `_total_runs_breakdown`, not re-measured.
EPOCH_SECONDS = {"SNN-AE": 0.5, "Transformer-AE": 83.75, "LSTM-AE": 97.2, "GRU-AE": 136.0}

#: MSE of a model that learns nothing (predicts the mean) when the target is
#: the ENCODED tensor instead of the window -- Meeting01's bug B2
#: (.wiki/Experiments/Meeting01.md, "B2 in numbers"), quoted verbatim.
#: Measured with the encoder of that time (git 20b47e9f^), which used the
#: window's 256 rows as the time axis and set a spike only where a sample's
#: own latency frame equalled its row index: about 1 spike per 256
#: positions, variance ~0.0039 ("~99.6% zeros"). Today's encoder uses
#: T = 16 frames (one spike in 16, variance 1/16 * 15/16 ~ 0.059) -- see
#: LATENCY_TRIVIAL_MSE_T16 -- so the 0.0038 must not be read as a T = 16 number.
TRIVIAL_MSE_BY_TARGET = {"direta": 1.000, "Poisson": 0.247, "latência": 0.0038}
OLD_LATENCY_TIME_AXIS = WINDOW  # the pre-fix encoder's time axis: the 256 window samples
LATENCY_TRIVIAL_MSE_T16 = (1 / 16) * (1 - 1 / 16)  # one spike per feature in T = 16 frames
B2_FREE_WIN = 261  # the wiki's own ratio (from unrounded values)


def zscore_window(window: np.ndarray) -> np.ndarray:
    """Whole-window z-score: one mean and one (population) std for all samples."""
    return (window - window.mean()) / window.std()


def synthetic_window(rng: np.random.RandomState) -> np.ndarray:
    """A short voiced sound: a fundamental of 2-7 cycles per window (about
    60-220 Hz for 256 samples at 8 kHz, a speaking pitch) with 8 harmonics
    decaying roughly as 1/h, under a smooth amplitude envelope, plus a little
    noise, then z-scored. With only 2-3 harmonics a 16-component PCA would
    capture almost everything and the latent size would seem not to matter."""
    t = np.arange(WINDOW) / WINDOW
    f0 = rng.uniform(2.0, 7.0)
    w = np.zeros(WINDOW)
    for harmonic in range(1, 9):
        amplitude = rng.uniform(0.5, 1.0) / harmonic
        w += amplitude * np.sin(2.0 * np.pi * harmonic * f0 * t + rng.uniform(0.0, 2.0 * np.pi))
    center, width = rng.uniform(0.3, 0.7), rng.uniform(0.15, 0.35)
    w *= np.exp(-((t - center) ** 2) / (2.0 * width**2))
    w += rng.normal(0.0, 0.08, WINDOW)
    return zscore_window(w)


def window_set(n: int, seed: int = SEED) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return np.array([synthetic_window(rng) for _ in range(n)])


@dataclass(frozen=True)
class Pca:
    mean: np.ndarray  # (WINDOW,) the mean-frame reference
    components: np.ndarray  # (r, WINDOW) principal directions, most significant first


def fit_pca(train: np.ndarray) -> Pca:
    """Fitted on the training windows only, as the real reference is."""
    mean = train.mean(axis=0)
    _, _, vt = np.linalg.svd(train - mean, full_matrices=False)
    return Pca(mean=mean, components=vt)


def pca_reconstruct(pca: Pca, window: np.ndarray, k: int) -> np.ndarray:
    """mu + (x - mu) V_k V_k^T: keep k coordinates, rebuild 256 samples."""
    if not 1 <= k <= len(pca.components):
        raise ValueError(f"k must be in [1, {len(pca.components)}], got {k}")
    v = pca.components[:k]
    return pca.mean + (window - pca.mean) @ v.T @ v


def mse(target: np.ndarray, reconstruction: np.ndarray) -> float:
    return float(np.mean((target - reconstruction) ** 2))
