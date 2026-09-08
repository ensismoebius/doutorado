#!/usr/bin/env python3
"""03_guayaquil_pca_mean_baselines.py — linear reference baselines for the nested-LOSO
comparison.

For each fold f and encoding e the guayaquil binary dumps, in the exact framed-encoded
representation the trained autoencoders reconstruct:

    results/guayaquil/<tag>_fold<f>_<e>_train_windows.npy      (N_train, F)
    results/guayaquil/<tag>_fold<f>_<e>_test_windows.npy       (N_test,  F)
    results/guayaquil/<tag>_fold<f>_<e>_test_windows_meta.csv  (speaker_id,recording_id,window_id,source_window_index)

This script fits two references on the TRAIN matrix only and scores them on TEST:

    mean : x_hat = mean_train                         (the mean-frame null baseline)
    pca  : x_hat = mu + (x - mu) Vk Vk^T,  k = latent (linear compression to the AE bottleneck)

Per-window mse/mae are appended to <tag>_fold<f>_per_window_errors.csv with
model in {mean, pca}, split=test, seed=0, run_id=0 — the same schema the C++ writes, so
the significance script pairs them against the trained models by window_id.

Fitted separately per fold and per encoding; never on pooled data. The dumped encoding
is the seed-0 realization (poisson is stochastic); the linear references are reported as
one representative realization. direct and latency are deterministic.

Usage:
    python scripts/pipeline/guayaquil/03_guayaquil_pca_mean_baselines.py \\
        --results-dir results/guayaquil --run-tag article_loso --latent 32
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import re
import sys

import numpy as np

_STEM_RE = re.compile(r"^(?P<tag>.+)_fold(?P<fold>\d+)_(?P<enc>[a-z]+)_train_windows\.npy$")

PW_HEADER = [
    "model", "encoding", "architecture", "v_th", "alpha", "run_id", "seed",
    "cv_fold", "split", "speaker_id", "recording_id", "window_id",
    "source_window_index", "mse", "mae",
]


def _pca_reconstruct(train: np.ndarray, test: np.ndarray, k: int) -> np.ndarray:
    mu = train.mean(axis=0, keepdims=True)
    xc = train - mu
    # economy SVD: rows of Vt are the principal directions, most significant first.
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    k = max(1, min(k, vt.shape[0]))
    vk = vt[:k]                      # (k, F)
    proj = (test - mu) @ vk.T @ vk   # (N_test, F)
    return mu + proj


def _per_window_errors(target: np.ndarray, recon: np.ndarray):
    diff = target - recon
    return (diff ** 2).mean(axis=1), np.abs(diff).mean(axis=1)


def _rows_for(model: str, fold: int, enc: str, meta, mse, mae):
    for m, mse_v, mae_v in zip(meta, mse, mae):
        yield {
            "model": model, "encoding": enc, "architecture": model,
            "v_th": 0.0, "alpha": 0.0, "run_id": 0, "seed": 0, "cv_fold": fold,
            "split": "test", "speaker_id": m["speaker_id"],
            "recording_id": m["recording_id"], "window_id": m["window_id"],
            "source_window_index": m["source_window_index"],
            "mse": f"{mse_v:.8f}", "mae": f"{mae_v:.8f}",
        }


def _append_rows(csv_path: pathlib.Path, rows: list[dict]):
    need_header = not csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PW_HEADER)
        if need_header:
            w.writeheader()
        for r in rows:
            w.writerow(r)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=pathlib.Path, default=pathlib.Path("results/guayaquil"))
    ap.add_argument("--run-tag", default="article_loso")
    ap.add_argument("--latent", type=int, default=32, help="PCA components (= AE bottleneck)")
    args = ap.parse_args(argv)

    rdir: pathlib.Path = args.results_dir
    train_files = sorted(rdir.glob(f"{args.run_tag}_fold*_*_train_windows.npy"))
    if not train_files:
        print(f"[pca-mean] no dump files under {rdir} for tag {args.run_tag}", file=sys.stderr)
        return 1

    n_done = 0
    for tf in train_files:
        m = _STEM_RE.match(tf.name)
        if not m:
            continue
        fold, enc = int(m["fold"]), m["enc"]
        stem = tf.name[: -len("_train_windows.npy")]
        test_f = rdir / f"{stem}_test_windows.npy"
        meta_f = rdir / f"{stem}_test_windows_meta.csv"
        if not (test_f.exists() and meta_f.exists()):
            print(f"[pca-mean] skipping {stem}: missing test/meta", file=sys.stderr)
            continue

        train = np.load(tf).astype(np.float64)
        test = np.load(test_f).astype(np.float64)
        with meta_f.open() as f:
            meta = [
                {k: int(v) for k, v in row.items()}
                for row in csv.DictReader(f)
            ]
        if len(meta) != test.shape[0]:
            print(f"[pca-mean] {stem}: meta rows {len(meta)} != test rows {test.shape[0]}",
                  file=sys.stderr)
            return 2

        pw_csv = rdir / f"{args.run_tag}_fold{fold}_per_window_errors.csv"

        mu = train.mean(axis=0, keepdims=True)
        mean_mse, mean_mae = _per_window_errors(test, np.repeat(mu, test.shape[0], axis=0))
        _append_rows(pw_csv, list(_rows_for("mean", fold, enc, meta, mean_mse, mean_mae)))

        pca_recon = _pca_reconstruct(train, test, args.latent)
        pca_mse, pca_mae = _per_window_errors(test, pca_recon)
        _append_rows(pw_csv, list(_rows_for("pca", fold, enc, meta, pca_mse, pca_mae)))

        n_done += 1
        print(f"[pca-mean] fold {fold} {enc}: mean MSE {mean_mse.mean():.5f}  "
              f"pca(k={args.latent}) MSE {pca_mse.mean():.5f}")

    print(f"[pca-mean] wrote references for {n_done} (fold, encoding) pairs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
