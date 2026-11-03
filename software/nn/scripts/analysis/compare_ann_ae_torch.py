#!/usr/bin/env python3
"""Matched ANN autoencoder on the windows a meeting01 run saved: the baseline that differs from the
SNN-AE only in the neuron (leaky ReLU instead of LIF).

Why this exists: LSTM/GRU/Transformer differ from the SNN in everything at once, so a gap between
them cannot be attributed to spiking. The matched ANN has the SNN's layer widths
(`linear:H:leaky` / `linear:Z:identity` / `linear:H:leaky` / `linear:output:identity`, a window of
256 values in and out, bottleneck Z = the dataset's latent_dim), so only the neuron changes.

It also fits PCA with k = Z on train + val and scores it on the same test windows, so the table
this prints has the MLP-AE, PCA and the mean on one scale (masked MSE, 1.0 = predicting the mean).

Input: the *_target_{train,val,test}_windows.npy + *_meta.csv a meeting01 run writes next to its
results (valid_length gives the mask). Not the C++ trainer: batch 1, Adam, early stopping on val
with patience, the best-val weights restored.

Usage:
    software/nn/.venv/bin/python software/nn/scripts/analysis/compare_ann_ae_torch.py \
        --results DIR --prefix meeting01_loso_fsdd_fold0 --latent 16
"""
from __future__ import annotations

import argparse
import copy
import csv
import os

import numpy as np
import torch
import torch.nn as nn


def load(results: str, prefix: str, split: str):
    w = np.load(os.path.join(results, f"{prefix}_target_{split}_windows.npy")).astype(np.float32)
    with open(os.path.join(results, f"{prefix}_target_{split}_windows_meta.csv")) as f:
        valid = np.array([int(r["valid_length"]) for r in csv.DictReader(f)])
    mask = (np.arange(w.shape[1])[None, :] < valid[:, None]).astype(np.float32)
    return torch.from_numpy(w), torch.from_numpy(mask)


def masked_mse_rows(recon, x, m):
    """Per-window sum(mask * diff^2) / sum(mask): the loss the C++ trainer minimises."""
    return (m * (recon - x) ** 2).sum(dim=1) / m.sum(dim=1)


def make_mlp(window: int, hidden: int, latent: int, slope: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(window, hidden), nn.LeakyReLU(slope),
        nn.Linear(hidden, latent),
        nn.Linear(latent, hidden), nn.LeakyReLU(slope),
        nn.Linear(hidden, window))


def evaluate(model, x, m) -> float:
    with torch.no_grad():
        return float(masked_mse_rows(model(x), x, m).mean())


def train(args, tr, va, seed: int):
    torch.manual_seed(seed)
    model = make_mlp(tr[0].shape[1], args.hidden, args.latent, args.slope)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    rng = np.random.default_rng(seed)
    best, best_state, bad, ep_stop = 9.0, None, 0, args.epochs
    for ep in range(1, args.epochs + 1):
        for i in rng.permutation(len(tr[0])):
            x, m = tr[0][i:i + 1], tr[1][i:i + 1]
            opt.zero_grad()
            masked_mse_rows(model(x), x, m).mean().backward()
            opt.step()
        v = evaluate(model, *va)
        if v < best - 1e-6:
            best, best_state, bad = v, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= args.patience:
                ep_stop = ep
                break
    model.load_state_dict(best_state)
    return model, best, ep_stop


def pca_test_mse(train_x, val_x, test, k: int) -> float:
    """PCA on train+val (centred, unmasked rows as the pipeline's 03_ script does), scored with
    the test mask. A reference, not the pipeline's own numbers."""
    fit = torch.cat([train_x, val_x]).numpy().astype(np.float64)
    mu = fit.mean(axis=0)
    _, _, vt = np.linalg.svd(fit - mu, full_matrices=False)
    comps = vt[:k]
    x, m = test
    xc = x.numpy().astype(np.float64) - mu
    recon = torch.from_numpy(((xc @ comps.T) @ comps + mu).astype(np.float32))
    return float(masked_mse_rows(recon, x, m).mean())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    p.add_argument("--prefix", required=True)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--latent", type=int, default=16)
    p.add_argument("--slope", type=float, default=0.01, help="LeakyReLU negative slope")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--seeds", default="41,42,43")
    p.add_argument("--threads", type=int, default=1)
    args = p.parse_args()
    torch.set_num_threads(args.threads)

    tr, va, te = (load(args.results, args.prefix, s) for s in ("train", "val", "test"))
    print(f"windows: train={len(tr[0])} val={len(va[0])} test={len(te[0])}  "
          f"MLP-AE {tr[0].shape[1]}-{args.hidden}-{args.latent}-{args.hidden}-{tr[0].shape[1]}  "
          f"params={sum(p.numel() for p in make_mlp(tr[0].shape[1], args.hidden, args.latent, args.slope).parameters())}")
    print(f"mean-of-train reference (predict train mean): "
          f"{float(masked_mse_rows(tr[0].mean(0, keepdim=True).expand_as(te[0]), *te).mean()):.4f}")
    print(f"PCA k={args.latent} on train+val:             {pca_test_mse(tr[0], va[0], te, args.latent):.4f}")
    for seed in [int(s) for s in args.seeds.split(",")]:
        model, best_val, ep_stop = train(args, tr, va, seed)
        print(f"MLP-AE seed={seed}  stopped@{ep_stop}  best_val={best_val:.4f}  "
              f"test={evaluate(model, *te):.4f}", flush=True)


if __name__ == "__main__":
    main()
