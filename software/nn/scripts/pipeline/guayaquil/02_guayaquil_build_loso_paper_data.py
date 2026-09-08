#!/usr/bin/env python3
"""02_guayaquil_build_loso_paper_data.py — aggregate the nested-LOSO run into
paper-ready tables.

Inputs (per fold f, written by the guayaquil binary and 03_guayaquil_pca_mean_baselines.py):
    results/guayaquil/<tag>_fold<f>_comparative_metrics.csv   (split = val | test rows)
    results/guayaquil/<tag>_fold<f>_per_window_errors.csv     (every model incl. pca/mean)

Only  split == "test"  rows feed the headline tables. Reconstruction error is reported as
mean +/- sample standard deviation across the FIVE seeds (seeds establish optimization /
reproducibility robustness around the estimate, not independent samples — see the paper's
Statistical methods and 04_guayaquil_significance_tests.py for the inferential analysis).

Model label map:
    lstm-ae         -> LSTM-AE
    gru-ae          -> GRU-AE
    transformer-ae  -> Transformer-AE
    snn-ae + arch   -> SNN-{dense|conv1d|recurrent}
    pca             -> PCA        (linear reference)
    mean            -> Mean       (mean-frame null baseline)

Outputs (--data-dir):
    paper_loso_summary.csv        model; mse mean/std; mae mean/std; r2; param_count; macs;
                                  train_ms; infer_ms   (';' delimited, pgfplots-friendly)
    paper_loso_by_encoding.csv    model; encoding; mse mean/std; mae mean/std
    paper_loso_mse_plot.csv       model; y (mse mean); y_err (mse std)   (per model, error bars)

Usage:
    python scripts/pipeline/guayaquil/02_guayaquil_build_loso_paper_data.py \\
        --results-dir results/guayaquil --run-tag article_loso --data-dir <paper-data-dir>
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys
from collections import defaultdict

import numpy as np

ARCH_LABEL = {"dense": "SNN-dense", "conv1d": "SNN-conv1d", "recurrent": "SNN-recurrent"}
MODEL_LABEL = {
    "lstm-ae": "LSTM-AE",
    "gru-ae": "GRU-AE",
    "transformer-ae": "Transformer-AE",
    "pca": "PCA",
    "mean": "Mean",
}
MODEL_ORDER = ["Mean", "PCA", "LSTM-AE", "GRU-AE", "Transformer-AE",
               "SNN-dense", "SNN-conv1d", "SNN-recurrent"]


def _label(model: str, architecture: str) -> str:
    if model == "snn-ae":
        return ARCH_LABEL.get(architecture, f"SNN-{architecture}")
    return MODEL_LABEL.get(model, model)


def _mean_std(per_seed_means: list[float]) -> tuple[float, float]:
    a = np.asarray(per_seed_means, dtype=float)
    if a.size == 0:
        return (float("nan"), float("nan"))
    return (float(a.mean()), float(a.std(ddof=1)) if a.size > 1 else 0.0)


def load_per_window(results_dir: pathlib.Path, run_tag: str) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(results_dir.glob(f"{run_tag}_fold*_per_window_errors.csv")):
        with path.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("split") != "test":
                    continue
                rows.append({
                    "label": _label(r["model"], r.get("architecture", "")),
                    "encoding": r["encoding"],
                    "seed": int(r["seed"]),
                    "mse": float(r["mse"]),
                    "mae": float(r["mae"]),
                })
    return rows


def load_comparative(results_dir: pathlib.Path, run_tag: str) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(results_dir.glob(f"{run_tag}_fold*_comparative_metrics.csv")):
        with path.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("split") != "test":
                    continue
                rows.append({
                    "label": _label(r["model"], r.get("architecture", "")),
                    "param_count": float(r["param_count"]),
                    "macs": float(r["macs"]),
                    "train_ms": float(r["train_ms"]),
                    "infer_ms": float(r["infer_ms"]),
                })
    return rows


def _seed_means(rows: list[dict], keyer, metric: str) -> dict:
    """metric mean per (key, seed), then collected per key as a list over seeds."""
    by_key_seed = defaultdict(list)
    for r in rows:
        by_key_seed[(keyer(r), r["seed"])].append(r[metric])
    per_key = defaultdict(list)
    for (key, _seed), vals in by_key_seed.items():
        per_key[key].append(float(np.mean(vals)))
    return per_key


def _sorted_labels(labels) -> list[str]:
    known = [m for m in MODEL_ORDER if m in labels]
    extra = sorted(l for l in labels if l not in MODEL_ORDER)
    return known + extra


def write_summary(pw_rows, cmp_rows, data_dir: pathlib.Path):
    mse = _seed_means(pw_rows, lambda r: r["label"], "mse")
    mae = _seed_means(pw_rows, lambda r: r["label"], "mae")

    cmp_by_label = defaultdict(lambda: defaultdict(list))
    for r in cmp_rows:
        for k in ("param_count", "macs", "train_ms", "infer_ms"):
            cmp_by_label[r["label"]][k].append(r[k])

    out = data_dir / "paper_loso_summary.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["model", "mse_mean", "mse_std", "mae_mean", "mae_std",
                    "param_count", "macs", "train_ms", "infer_ms"])
        for label in _sorted_labels(mse.keys()):
            mm, ms = _mean_std(mse[label])
            am, as_ = _mean_std(mae[label])
            c = cmp_by_label.get(label, {})

            def cell(key: str, prec: int) -> str:
                # pca / mean are analytic references — no entry in comparative_metrics.
                return format(float(np.mean(c[key])), f".{prec}f") if c.get(key) else ""

            w.writerow([label, f"{mm:.6f}", f"{ms:.6f}", f"{am:.6f}", f"{as_:.6f}",
                        cell("param_count", 0), cell("macs", 0),
                        cell("train_ms", 1), cell("infer_ms", 2)])
    return out


def write_by_encoding(pw_rows, data_dir: pathlib.Path):
    mse = _seed_means(pw_rows, lambda r: (r["label"], r["encoding"]), "mse")
    mae = _seed_means(pw_rows, lambda r: (r["label"], r["encoding"]), "mae")
    out = data_dir / "paper_loso_by_encoding.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["model", "encoding", "mse_mean", "mse_std", "mae_mean", "mae_std"])
        for key in sorted(mse.keys()):
            mm, ms = _mean_std(mse[key])
            am, as_ = _mean_std(mae[key])
            w.writerow([key[0], key[1], f"{mm:.6f}", f"{ms:.6f}", f"{am:.6f}", f"{as_:.6f}"])
    return out


def write_mse_plot(pw_rows, data_dir: pathlib.Path):
    mse = _seed_means(pw_rows, lambda r: r["label"], "mse")
    out = data_dir / "paper_loso_mse_plot.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "y", "y_err"])
        for label in _sorted_labels(mse.keys()):
            mm, ms = _mean_std(mse[label])
            w.writerow([label, f"{mm:.6f}", f"{ms:.6f}"])
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=pathlib.Path, default=pathlib.Path("results/guayaquil"))
    ap.add_argument("--run-tag", default="article_loso")
    ap.add_argument(
        "--data-dir",
        type=pathlib.Path,
        default=pathlib.Path(
            "/home/ensismoebius/Repos/doutorado/documentation/07-articlesProduced/"
            "conference71070Guaiaquil/data"),
    )
    args = ap.parse_args(argv)
    args.data_dir.mkdir(parents=True, exist_ok=True)

    pw_rows = load_per_window(args.results_dir, args.run_tag)
    if not pw_rows:
        print(f"[loso-data] no test-split per-window rows under {args.results_dir}", file=sys.stderr)
        return 1
    cmp_rows = load_comparative(args.results_dir, args.run_tag)

    written = [
        write_summary(pw_rows, cmp_rows, args.data_dir),
        write_by_encoding(pw_rows, args.data_dir),
        write_mse_plot(pw_rows, args.data_dir),
    ]
    print("[loso-data] wrote " + ", ".join(p.name for p in written) + f" to {args.data_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
