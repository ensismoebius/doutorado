#!/usr/bin/env python3
"""Ground truth for the thesis autoencoder/PCA input: scipy for the spectrum, PyTorch for PCA.

What is checked. The autoencoders and PCA of the thesis read, per signal, 256 values:

    signal --Welch power spectrum--> 256 bins --log10--> --min-max--> [0, 1]^256
              (thesis::spectral_input, ThesisFeatureExtraction.hpp)

and PCA projects those vectors on their first k principal axes (thesis::fit_pca /
project_pca, ThesisPca.hpp). This script computes both with external libraries and writes
the signals and the answers to a fixture; thesis_spectrum_parity_gtest feeds the same
signals to the C++ and compares.

    quantity          reference                                    C++
    power spectrum    scipy.signal.welch                           log_power_spectrum
    PCA projection    torch.linalg.svd of the centered rows        fit_pca + project_pca

The Welch call spelled out, because every argument is a way to disagree silently:
    window=np.hanning(512)   the SYMMETRIC Hann the C++ uses (scipy's default "hann" is the
                             periodic one: different values, a different spectrum);
    nperseg=512, noverlap=256, detrend=False (scipy's default removes each frame's mean);
    return_onesided=False    the one-sided estimate doubles every bin except DC and Nyquist,
                             so bin 256 would be off by a factor of 2 against the others;
    bins 1..256              DC dropped.
The scale of the spectrum is free: a constant factor is a constant shift after log10, and
min-max removes it.

The signals are synthetic (the dataset is not redistributable and is 2.3 GB): sums of
sines over broadband noise, in three lengths, one of them not a multiple of the hop so the
"samples after the last whole frame are not read" rule is exercised.

Run (developer step; needs numpy, scipy and torch):
    ~/.venv/bin/python software/nn/scripts/testing/gen_thesis_spectrum_refs.py

Output: software/nn/src/experiments/thesis/tests/fixtures/thesis_spectrum_refs.npz
"""
import os

import numpy as np
import scipy
import scipy.signal
import torch

OUT = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "src", "experiments", "thesis",
    "tests", "fixtures", "thesis_spectrum_refs.npz"))

FRAME, HOP, BINS = 512, 256, 256
PCA_K = 4


def spectral_input(signal: np.ndarray) -> np.ndarray:
    _, power = scipy.signal.welch(
        signal, window=np.hanning(FRAME), nperseg=FRAME, noverlap=FRAME - HOP,
        detrend=False, return_onesided=False, scaling="spectrum")
    log_power = np.log10(power[1:BINS + 1])
    return (log_power - log_power.min()) / (log_power.max() - log_power.min())


def make_signal(rng: np.random.Generator, n: int, freqs, noise: float) -> np.ndarray:
    t = np.arange(n)
    sig = sum(a * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi)) for f, a in freqs)
    # float32 on purpose: the dataset's samples are float32, and the C++ reads them as such.
    return (sig + noise * rng.standard_normal(n)).astype(np.float32)


def main() -> None:
    rng = np.random.default_rng(20261112)
    out = {}
    # (length, [(cycles/sample, amplitude)], noise sd)
    cases = [
        (512, [(0.05, 1.0)], 0.05),                                # exactly one frame
        (4096, [(0.03, 1.0), (0.21, 0.3)], 0.02),                  # 15 frames
        (5000, [(0.11, 0.5), (0.40, 0.5), (0.47, 0.1)], 0.10),     # 18 frames, 136 unread
    ]
    for i, (n, freqs, noise) in enumerate(cases):
        sig = make_signal(rng, n, freqs, noise)
        out[f"signal_{i}"] = sig
        out[f"input_{i}"] = spectral_input(sig.astype(np.float64)).astype(np.float32)
    out["n_cases"] = np.array([len(cases)], dtype=np.float32)

    # PCA: 12 rows of spectral input (3 "speakers" x 4 takes), projected on k = 4 axes.
    rows = []
    for speaker in range(3):
        for _take in range(4):
            sig = make_signal(rng, 2048, [(0.04 + 0.09 * speaker, 1.0), (0.33, 0.2)], 0.05)
            out[f"pca_signal_{len(rows)}"] = sig
            rows.append(spectral_input(sig.astype(np.float64)).astype(np.float32))
    x = torch.tensor(np.stack(rows), dtype=torch.float64)
    centered = x - x.mean(dim=0, keepdim=True)
    _, s, vh = torch.linalg.svd(centered, full_matrices=False)
    out["pca_rows"] = np.stack(rows)
    # A principal axis is defined up to its sign; the test compares up to that sign.
    out["pca_projection"] = (centered @ vh[:PCA_K].T).numpy().astype(np.float32)
    out["pca_explained_variance"] = (s[:PCA_K] ** 2 / (x.shape[0] - 1)).numpy().astype(np.float32)

    np.savez(OUT, **out)
    print(f"numpy {np.__version__}, scipy {scipy.__version__}, torch {torch.__version__}")
    print(f"wrote {len(out)} arrays -> {OUT}")


if __name__ == "__main__":
    main()
