#!/usr/bin/env python3
"""Does the autoencoders'/PCA's input carry the speaker? scipy + PyTorch reference.

The question. Phase 01 gave EER 0.48-0.52 (chance) for every autoencoder and PCA extractor
while the handcrafted one gave 0.25-0.32. Either those extractors are poor, or what they
are fed holds no speaker information. This script answers it without any of the C++: it
rebuilds the extractors' INPUT from the database and asks a linear classifier to name the
speaker from it.

    input (256 values per trial)         how
    block-averaged waveform ("old")      the signal averaged in 256 contiguous blocks,
                                         min-max: the input until 2026-11-12
    log power spectrum ("new")           scipy.signal.welch (symmetric Hann, 512/256, no
                                         detrend, bins 1..256), log10, min-max:
                                         thesis::spectral_input

The signals are the ones the C++ reads (ThesisFeatureExtractionInternal.hpp): voice with
pre-emphasis 0.97; EEG as the six channels one after the other; early fusion as voice
followed by EEG. Trials: the 1974 pronounced ones (the only ones with audio).

The classifier is deliberately the weakest that can show the information is there: per
fold, standardize on the training rows, optionally project on the first k principal axes
(torch.linalg.svd of the training rows), then ridge regression on one-hot speakers
(torch.linalg.solve), argmax. Stratified 5 folds, seed 42. Chance is 1/15 = 0.067.

Run (needs numpy, scipy, torch; reads ~/database.sqlite, about one minute):
    ~/.venv/bin/python software/nn/scripts/analysis/thesis_ae_input_reference.py --db ~/database.sqlite
"""
from __future__ import annotations

import argparse
import os
import sqlite3

import numpy as np
import scipy
import scipy.signal
import torch

BINS, FRAME, HOP = 256, 512, 256


def block_average(sig: np.ndarray) -> np.ndarray:
    edges = (np.arange(BINS + 1) * len(sig)) // BINS
    v = np.add.reduceat(sig, edges[:-1]) / np.diff(edges)
    return (v - v.min()) / (v.max() - v.min())


def spectral_input(sig: np.ndarray) -> np.ndarray:
    _, power = scipy.signal.welch(sig, window=np.hanning(FRAME), nperseg=FRAME,
                                  noverlap=FRAME - HOP, detrend=False,
                                  return_onesided=False, scaling="spectrum")
    lp = np.log10(power[1:BINS + 1])
    return (lp - lp.min()) / (lp.max() - lp.min())


def accuracy(x: np.ndarray, y: np.ndarray, folds: np.ndarray, k: int) -> tuple[float, float]:
    xt = torch.tensor(x, dtype=torch.float64)
    classes = np.unique(y)
    onehot = torch.tensor((y[:, None] == classes[None]).astype(np.float64))
    accs = []
    for f in range(5):
        tr, te = torch.tensor(folds != f), torch.tensor(folds == f)
        mu, sd = xt[tr].mean(0), xt[tr].std(0, unbiased=False) + 1e-12
        a, b = (xt[tr] - mu) / sd, (xt[te] - mu) / sd
        if k:
            _, _, vh = torch.linalg.svd(a, full_matrices=False)
            a, b = a @ vh[:k].T, b @ vh[:k].T
        a1 = torch.cat([a, torch.ones(len(a), 1, dtype=torch.float64)], 1)
        b1 = torch.cat([b, torch.ones(len(b), 1, dtype=torch.float64)], 1)
        w = torch.linalg.solve(a1.T @ a1 + torch.eye(a1.shape[1], dtype=torch.float64),
                               a1.T @ onehot[tr])
        pred = classes[(b1 @ w).argmax(1).numpy()]
        accs.append(float((pred == y[folds == f]).mean()))
    return float(np.mean(accs)), float(np.std(accs))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--db", required=True, help="the dataset's SQLite file")
    args = ap.parse_args()

    db = sqlite3.connect(os.path.expanduser(args.db))
    rows = db.execute(
        """select t.subject_id, a.samples, e.F3, e.F4, e.C3, e.C4, e.P3, e.P4
           from trial t join audio_samples a on a.trial_id = t.id
                        join eeg_samples e on e.trial_id = t.id
                        join modalities m on m.id = t.modality_id
           where m.name = 'Pronounced' order by t.id""").fetchall()

    y, feats = [], {f"{s} {v}": [] for s in ("voice", "eeg", "early") for v in ("old", "new")}
    for sid, audio, *channels in rows:
        y.append(sid)
        x = np.frombuffer(audio, dtype=np.float32).astype(np.float64)
        voice = np.append(x[0], x[1:] - 0.97 * x[:-1])
        eeg = np.concatenate([np.frombuffer(c, dtype=np.float32) for c in channels]).astype(np.float64)
        for name, sig in (("voice", voice), ("eeg", eeg), ("early", np.concatenate([voice, eeg]))):
            feats[f"{name} old"].append(block_average(sig))
            feats[f"{name} new"].append(spectral_input(sig))
    y = np.array(y)

    rng = np.random.default_rng(42)
    folds = np.empty(len(y), int)
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        folds[idx] = np.arange(len(idx)) % 5

    print(f"numpy {np.__version__}, scipy {scipy.__version__}, torch {torch.__version__}")
    print(f"{len(y)} trials, {len(np.unique(y))} speakers, chance {1 / len(np.unique(y)):.3f}\n")
    print(f"{'input':12s} {'PCA 8':>14s} {'PCA 32':>14s} {'all 256':>14s}")
    for name, v in feats.items():
        x = np.array(v)
        cells = ["%.3f ± %.3f" % accuracy(x, y, folds, k) for k in (8, 32, 0)]
        print(f"{name:12s} {cells[0]:>14s} {cells[1]:>14s} {cells[2]:>14s}")


if __name__ == "__main__":
    main()
