#!/usr/bin/env python3
"""Aggregate per-run-tag CSVs into the `paper_*.csv` files referenced by paper.tex.

Per-run-tag inputs (written by `write_latex_exports` in `ComparativeOutput.cpp`):
  {run_tag}_summary_by_model.csv
  {run_tag}_recon_by_encoding.csv
  {run_tag}_efficiency_by_encoding.csv
  {run_tag}_mse_plot.csv
  {run_tag}_sweep_alpha.csv
  {run_tag}_profile_manifest.csv

Outputs (consumed by paper.tex via \\pgfplotstabletypeset / \\pgfplotstableread):
  paper_summary_by_model.csv
  paper_recon_by_encoding.csv
  paper_eff_by_encoding.csv
  paper_mse_plot.csv
  paper_sweep_alpha.csv
  paper_profiles.csv
  paper_backend_comparison.csv  (placeholder if no backend run yet)

Usage:
  python3 scripts/aggregate_paper_csvs.py [DATA_DIR]

DATA_DIR defaults to the article's data/ directory.
"""

import csv
import pathlib
import sys
from collections import OrderedDict

DEFAULT_DIR = pathlib.Path(
    "/home/ensismoebius/Repos/doutorado/documentation/"
    "07-articlesProduced/conference71070Guaiaquil/data"
)


def read_rows(path: pathlib.Path):
    if not path.is_file():
        return [], []
    with path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return reader.fieldnames or [], rows


def write_rows(path: pathlib.Path, fieldnames, rows):
    if not fieldnames:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    print(f"wrote {path.name} ({len(rows)} rows)")


def concat_csvs(data_dir: pathlib.Path, suffix: str, out_path: pathlib.Path,
                drop_cols=None):
    drop_cols = set(drop_cols or [])
    fieldnames = None
    all_rows = []
    for path in sorted(data_dir.glob(f"article_*_{suffix}")):
        cols, rows = read_rows(path)
        kept = [c for c in cols if c not in drop_cols]
        if fieldnames is None:
            fieldnames = kept
        elif kept != fieldnames:
            common = [c for c in fieldnames if c in kept]
            fieldnames = common
        all_rows.extend(rows)
    if fieldnames is None:
        print(f"  (no inputs for *_{suffix}, skipping {out_path.name})")
        return
    write_rows(out_path, fieldnames, all_rows)


def merge_by_key(data_dir: pathlib.Path, suffix: str, out_path: pathlib.Path,
                 key: str):
    """Merge per-run-tag CSVs that share a common key column.

    Rows from later files overwrite earlier values for the same key. Useful for
    `mse_plot` (key=encoding) and `sweep_alpha` (key=alpha) where each profile
    only fills its own column(s)."""
    merged = OrderedDict()
    fieldnames = None
    for path in sorted(data_dir.glob(f"article_*_{suffix}")):
        cols, rows = read_rows(path)
        if fieldnames is None:
            fieldnames = list(cols)
        else:
            for c in cols:
                if c not in fieldnames:
                    fieldnames.append(c)
        for row in rows:
            k = row.get(key)
            if k is None:
                continue
            entry = merged.setdefault(k, {})
            for c, v in row.items():
                if v not in (None, "", "0", "0.0", "0.000000"):
                    entry[c] = v
                else:
                    entry.setdefault(c, v)
    if fieldnames is None or not merged:
        print(f"  (no inputs for *_{suffix}, skipping {out_path.name})")
        return
    write_rows(out_path, fieldnames, list(merged.values()))


def write_backend_placeholder(out_path: pathlib.Path):
    """Placeholder rows so the paper compiles before a backend run produces real timings."""
    if out_path.is_file():
        return
    rows = [
        {"operation": "train_epoch", "xtensor_ms": "0.00", "opencl_ms": "0.00", "speedup": "1.000"},
        {"operation": "infer_sample", "xtensor_ms": "0.00", "opencl_ms": "0.00", "speedup": "1.000"},
    ]
    write_rows(out_path, ["operation", "xtensor_ms", "opencl_ms", "speedup"], rows)


PLACEHOLDER_SCHEMAS = {
    "paper_summary_by_model.csv": (
        ["model", "mse", "mae", "r2", "spike_rate", "energy", "infer_ms",
         "train_ms", "param_count", "macs"],
        [
            {"model": "LSTM-AE", "mse": "0", "mae": "0", "r2": "0",
             "spike_rate": "0", "energy": "0", "infer_ms": "0",
             "train_ms": "0", "param_count": "0", "macs": "0"},
            {"model": "SNN-AE", "mse": "0", "mae": "0", "r2": "0",
             "spike_rate": "0", "energy": "0", "infer_ms": "0",
             "train_ms": "0", "param_count": "0", "macs": "0"},
        ],
    ),
    "paper_recon_by_encoding.csv": (
        ["model", "encoding", "mse", "mae", "r2"],
        [
            {"model": "LSTM-AE", "encoding": "direct",
             "mse": "0", "mae": "0", "r2": "0"},
            {"model": "SNN-AE", "encoding": "direct",
             "mse": "0", "mae": "0", "r2": "0"},
        ],
    ),
    "paper_eff_by_encoding.csv": (
        ["model", "encoding", "spike_rate", "energy", "param_count", "macs"],
        [
            {"model": "LSTM-AE", "encoding": "direct", "spike_rate": "0",
             "energy": "0", "param_count": "0", "macs": "0"},
            {"model": "SNN-AE", "encoding": "direct", "spike_rate": "0",
             "energy": "0", "param_count": "0", "macs": "0"},
        ],
    ),
    "paper_profiles.csv": (
        ["profile", "run_tag", "repeats", "datasets", "encodings",
         "snn_architectures", "v_th_values", "alpha_values",
         "window_size", "train_samples", "val_samples"],
        [
            {"profile": "article-lstm-ae", "run_tag": "article_lstm_ae",
             "repeats": "3", "datasets": "fsdd",
             "encodings": "direct;poisson;latency",
             "snn_architectures": "", "v_th_values": "", "alpha_values": "",
             "window_size": "256", "train_samples": "100", "val_samples": "30"},
        ],
    ),
    "paper_mse_plot.csv": (
        ["encoding", "lstm_ae", "snn_dense", "snn_conv1d", "snn_recurrent"],
        [
            {"encoding": "direct", "lstm_ae": "0", "snn_dense": "0",
             "snn_conv1d": "0", "snn_recurrent": "0"},
            {"encoding": "poisson", "lstm_ae": "0", "snn_dense": "0",
             "snn_conv1d": "0", "snn_recurrent": "0"},
            {"encoding": "latency", "lstm_ae": "0", "snn_dense": "0",
             "snn_conv1d": "0", "snn_recurrent": "0"},
        ],
    ),
    "paper_sweep_alpha.csv": (
        ["alpha", "dense", "conv1d", "recurrent"],
        [
            {"alpha": "0.8", "dense": "0", "conv1d": "0", "recurrent": "0"},
            {"alpha": "0.9", "dense": "0", "conv1d": "0", "recurrent": "0"},
            {"alpha": "0.99", "dense": "0", "conv1d": "0", "recurrent": "0"},
        ],
    ),
}


def seed_placeholder(out_path: pathlib.Path):
    """Write a placeholder file with the schema the paper expects, only if missing."""
    if out_path.is_file():
        return
    schema = PLACEHOLDER_SCHEMAS.get(out_path.name)
    if schema is None:
        return
    fieldnames, rows = schema
    write_rows(out_path, fieldnames, rows)


def main():
    data_dir = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DIR
    if not data_dir.is_dir():
        print(f"data dir not found: {data_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"aggregating into {data_dir}")

    concat_csvs(data_dir, "summary_by_model.csv",
                data_dir / "paper_summary_by_model.csv")
    concat_csvs(data_dir, "recon_by_encoding.csv",
                data_dir / "paper_recon_by_encoding.csv",
                drop_cols=["f1"])
    concat_csvs(data_dir, "efficiency_by_encoding.csv",
                data_dir / "paper_eff_by_encoding.csv")
    concat_csvs(data_dir, "profile_manifest.csv",
                data_dir / "paper_profiles.csv")
    merge_by_key(data_dir, "mse_plot.csv",
                 data_dir / "paper_mse_plot.csv", key="encoding")
    merge_by_key(data_dir, "sweep_alpha.csv",
                 data_dir / "paper_sweep_alpha.csv", key="alpha")
    write_backend_placeholder(data_dir / "paper_backend_comparison.csv")

    for name in PLACEHOLDER_SCHEMAS:
        seed_placeholder(data_dir / name)

    print("done")


if __name__ == "__main__":
    main()
