"""Audio -> a small fixed-length feature vector, pure numpy.

Vowels are told apart mainly by their formants -- peaks in the spectral
envelope -- not by loudness or pitch. A log-energy filterbank over the FFT
spectrum is the simplest feature that keeps that shape information: one
number per frequency band, low where there's no energy, high at a formant.
No librosa/scipy dependency: just `numpy.fft` plus a hand-built triangular
filterbank, the same "no new heavy dependency" choice this app already
makes everywhere else in `snn/`.
"""

from __future__ import annotations

import numpy as np

#: Melodic pitch is irrelevant to which vowel was spoken; linear (not mel)
#: spacing is used here on purpose, since the formant frequencies that DO
#: separate vowels (roughly 200-3500 Hz for a single speaker) are not
#: bunched at the low end the way pitch perception is -- mel spacing would
#: spend most of its bins on a range vowels don't need.
def _triangular_filterbank(n_fft_bins: int, sample_rate: int, n_bins: int, fmin: float, fmax: float) -> np.ndarray:
    """Returns a ``(n_bins, n_fft_bins)`` matrix of triangular weights."""
    edges = np.linspace(fmin, fmax, n_bins + 2)
    freqs = np.linspace(0.0, sample_rate / 2.0, n_fft_bins)
    bank = np.zeros((n_bins, n_fft_bins))
    for i in range(n_bins):
        lo, mid, hi = edges[i], edges[i + 1], edges[i + 2]
        rising = (freqs - lo) / max(mid - lo, 1e-9)
        falling = (hi - freqs) / max(hi - mid, 1e-9)
        bank[i] = np.clip(np.minimum(rising, falling), 0.0, None)
    return bank


def log_energy_bins(
    samples: np.ndarray, sample_rate: int, n_bins: int = 20, fmin: float = 80.0, fmax: float = 4000.0
) -> np.ndarray:
    """One log-energy value per frequency band, for a single audio window.

    ``samples`` is one short window (e.g. ~400ms); a Hann window tapers its
    edges before the FFT so the window's own hard edges don't themselves
    look like high-frequency energy (spectral leakage). Energy is `log1p`'d
    because loudness differences compress logarithmically (same reason
    decibels exist) -- without it, a single loud frame would dominate the
    z-scoring in `vowel_model.py` and flatten every quieter one to near-zero.
    """
    samples = np.asarray(samples, dtype=float)
    windowed = samples * np.hanning(len(samples))
    spectrum = np.abs(np.fft.rfft(windowed)) ** 2
    bank = _triangular_filterbank(len(spectrum), sample_rate, n_bins, fmin, fmax)
    energy = bank @ spectrum
    return np.log1p(energy)
