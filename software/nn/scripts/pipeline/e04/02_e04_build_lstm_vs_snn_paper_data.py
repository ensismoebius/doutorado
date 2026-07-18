#!/usr/bin/env python3
"""02_e04_build_lstm_vs_snn_paper_data.py — Aggregate Experiment 04 LSTM-AE vs
SNN-AE comparative CSVs into paper-ready tables for the Guaiaquil conference
article.

Reads per-run comparative_metrics.csv files produced by experiment04 and
aggregates them into summary CSVs suitable for LaTeX (pgfplots / tabular).
Also writes .dat files to the paper's data/ directory for pgfplots input.

Column mapping:
  model == "lstm-ae"                    → label LSTM-AE
  model == "snn-ae" + architecture      → label SNN-{dense|conv1d|recurrent}

Usage:
    python scripts/pipeline/e04/02_e04_build_lstm_vs_snn_paper_data.py \\
        --results-dir results/guayaquil \\
        --data-dir <paper-data-dir> \\
        --profiles-dir src/experiments/04/profiles

Called automatically at the end of 01_e04_run_article_profiles.sh.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
from collections import defaultdict
from typing import Dict, List, Tuple


NUMERIC_FIELDS = {
    "mse",
    "mae",
    "r2",
    "precision",
    "recall",
    "f1",
    "spike_rate",
    "energy",
    "train_ms",
    "infer_ms",
    "param_count",
    "macs",
    "v_th",
    "alpha",
}


def parse_row(raw: Dict[str, str]) -> Dict[str, object]:
    row: Dict[str, object] = dict(raw)
    for key in NUMERIC_FIELDS:
        if key in row and row[key] != "":
            row[key] = float(row[key])
    return row


def read_csv_rows(path: pathlib.Path) -> List[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [parse_row(r) for r in reader]


def model_label(row: Dict[str, object]) -> str:
    if row["model"] == "lstm-ae":
        return "LSTM-AE"
    arch = str(row["architecture"])
    if arch == "dense":
        return "SNN-dense"
    if arch == "conv1d":
        return "SNN-conv1d"
    if arch == "recurrent":
        return "SNN-recurrent"
    return str(row["model"])


def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def aggregate_all(rows: List[Dict[str, object]], data_dir: pathlib.Path) -> None:
    by_model: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    by_model_encoding: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    by_encoding_model: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    by_alpha_arch: Dict[Tuple[float, str], List[Dict[str, object]]] = defaultdict(list)

    for row in rows:
        mlabel = model_label(row)
        encoding = str(row["encoding"])
        by_model[mlabel].append(row)
        by_model_encoding[(mlabel, encoding)].append(row)
        by_encoding_model[(encoding, mlabel)].append(row)

        if row["model"] == "snn-ae" and encoding == "direct" and abs(float(row["v_th"]) - 1.0) < 1e-6:
            by_alpha_arch[(float(row["alpha"]), str(row["architecture"]))].append(row)

    summary_path = data_dir / "paper_summary_by_model.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "mse", "mae", "r2", "spike_rate", "energy", "infer_ms", "train_ms", "param_count", "macs"])
        for model in sorted(by_model):
            group = by_model[model]
            w.writerow([
                model,
                mean([float(r["mse"]) for r in group]),
                mean([float(r["mae"]) for r in group]),
                mean([float(r["r2"]) for r in group]),
                mean([float(r["spike_rate"]) for r in group]),
                mean([float(r["energy"]) for r in group]),
                mean([float(r["infer_ms"]) for r in group]),
                mean([float(r["train_ms"]) for r in group]),
                mean([float(r["param_count"]) for r in group]),
                mean([float(r["macs"]) for r in group]),
            ])

    recon_path = data_dir / "paper_recon_by_encoding.csv"
    with recon_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "encoding", "mse", "mae", "r2", "f1"])
        for key in sorted(by_model_encoding):
            group = by_model_encoding[key]
            model, encoding = key
            w.writerow([
                model,
                encoding,
                mean([float(r["mse"]) for r in group]),
                mean([float(r["mae"]) for r in group]),
                mean([float(r["r2"]) for r in group]),
                mean([float(r["f1"]) for r in group]),
            ])

    eff_path = data_dir / "paper_eff_by_encoding.csv"
    with eff_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "encoding", "spike_rate", "energy", "param_count", "macs"])
        for key in sorted(by_model_encoding):
            group = by_model_encoding[key]
            model, encoding = key
            w.writerow([
                model,
                encoding,
                mean([float(r["spike_rate"]) for r in group]),
                mean([float(r["energy"]) for r in group]),
                mean([float(r["param_count"]) for r in group]),
                mean([float(r["macs"]) for r in group]),
            ])

    mse_plot_path = data_dir / "paper_mse_plot.csv"
    with mse_plot_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["encoding", "lstm_ae", "snn_dense", "snn_conv1d", "snn_recurrent"])
        for encoding in ["direct", "poisson", "latency"]:
            def lookup(model: str) -> float:
                return mean([float(r["mse"]) for r in by_encoding_model.get((encoding, model), [])])

            w.writerow([
                encoding,
                lookup("LSTM-AE"),
                lookup("SNN-dense"),
                lookup("SNN-conv1d"),
                lookup("SNN-recurrent"),
            ])

    sweep_path = data_dir / "paper_sweep_alpha.csv"
    alphas = sorted({k[0] for k in by_alpha_arch.keys()})
    with sweep_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["alpha", "dense", "conv1d", "recurrent"])
        for alpha in alphas:
            def lookup(arch: str) -> float:
                return mean([float(r["mse"]) for r in by_alpha_arch.get((alpha, arch), [])])

            w.writerow([alpha, lookup("dense"), lookup("conv1d"), lookup("recurrent")])


def build_profile_table(profiles_dir: pathlib.Path, data_dir: pathlib.Path) -> None:
    profile_files = sorted(profiles_dir.glob("article-*.json"))
    out_path = data_dir / "paper_profiles.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "profile",
            "run_tag",
            "repeats",
            "datasets",
            "encodings",
            "snn_architectures",
            "v_th_values",
            "alpha_values",
            "window_size",
            "train_samples",
            "val_samples",
        ])
        for p in profile_files:
            cfg = json.loads(p.read_text(encoding="utf-8"))
            w.writerow([
                p.name,
                cfg["experiment"]["run_tag"],
                cfg["experiment"]["repeats"],
                ";".join(cfg["evaluation"]["datasets"]),
                ";".join(cfg["evaluation"]["encodings"]),
                ";".join(cfg["evaluation"]["snn_architectures"]),
                ";".join(str(v) for v in cfg["evaluation"]["v_th_values"]),
                ";".join(str(v) for v in cfg["evaluation"]["alpha_values"]),
                cfg["dataset"]["window_size"],
                cfg["dataset"]["max_loaded_train_samples"],
                cfg["dataset"]["max_validation_samples"],
            ])


def build_model_timing_table(rows: List[Dict[str, object]], data_dir: pathlib.Path) -> None:
    """LSTM-AE vs SNN-* inference/training timing on one (XTensor/CPU reference) backend.

    Feeds the paper's Table tab:backend ("Inference and training timing (XTensor backend,
    mean over 3 seeds)") and the Sec. Timing prose speedup figures. This is a *model*
    comparison, not a backend comparison — see build_xtensor_opencl_table below for that.
    """
    by_model: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_model[model_label(row)].append(row)

    infer_by_model = {m: mean([float(r["infer_ms"]) for r in g]) for m, g in by_model.items()}
    train_by_model = {m: mean([float(r["train_ms"]) for r in g]) for m, g in by_model.items()}
    baseline = infer_by_model.get("LSTM-AE")

    out_path = data_dir / "paper_backend_comparison.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "infer_ms", "train_ms", "infer_speedup"])
        for model in sorted(by_model):
            infer_ms = infer_by_model[model]
            speedup = (baseline / infer_ms) if baseline and infer_ms else 0.0
            w.writerow([model, infer_ms, train_by_model[model], speedup])


def build_xtensor_opencl_table(results_dir: pathlib.Path, data_dir: pathlib.Path) -> None:
    """XTensor/CPU vs OpenCL/GPU timing for the SAME model (article-backend-bench profile).

    Populated only by scripts/pipeline/e04/e04_run_backend_comparison.sh, which builds and
    runs both the max-performance and max-performance-opencl presets. Not currently
    referenced from paper.tex — kept separate from paper_backend_comparison.csv (the
    LSTM-vs-SNN model timing table) so the two comparisons never collide on one filename.
    """
    xt_path = results_dir / "article_backend_bench_xtensor_comparative_metrics.csv"
    oc_path = results_dir / "article_backend_bench_opencl_comparative_metrics.csv"

    out_path = data_dir / "paper_xtensor_opencl_comparison.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["operation", "xtensor_ms", "opencl_ms", "speedup"])

        if not xt_path.exists() or not oc_path.exists():
            return

        xt_rows = read_csv_rows(xt_path)
        oc_rows = read_csv_rows(oc_path)

        xt_train = mean([float(r["train_ms"]) for r in xt_rows])
        oc_train = mean([float(r["train_ms"]) for r in oc_rows])
        xt_infer = mean([float(r["infer_ms"]) for r in xt_rows])
        oc_infer = mean([float(r["infer_ms"]) for r in oc_rows])

        w.writerow(["train", xt_train, oc_train, (xt_train / oc_train) if oc_train else 0.0])
        w.writerow(["inference", xt_infer, oc_infer, (xt_infer / oc_infer) if oc_infer else 0.0])


def main() -> int:
    parser = argparse.ArgumentParser(description="Build paper CSV data from experiment outputs.")
    parser.add_argument("--results-dir", default="results/guayaquil")
    parser.add_argument(
        "--data-dir",
        default="/home/ensismoebius/Repos/doutorado/documentation/07-articlesProduced/conference71070Guaiaquil/data",
    )
    parser.add_argument("--profiles-dir", default="src/experiments/04/profiles")
    args = parser.parse_args()

    results_dir = pathlib.Path(args.results_dir)
    data_dir = pathlib.Path(args.data_dir)
    profiles_dir = pathlib.Path(args.profiles_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Each SNN profile also runs an incidental LSTM-AE baseline (from its linear specs,
    # not a real LSTM). Only accept lstm-ae rows from the canonical LSTM profile CSV so
    # the paper LSTM stats are not contaminated by linear-only runs.
    rows: List[Dict[str, object]] = []

    lstm_only_csv = results_dir / "article_lstm_ae_comparative_metrics.csv"
    if lstm_only_csv.exists():
        rows.extend(read_csv_rows(lstm_only_csv))

    for name in [
        "article_snn_dense_comparative_metrics.csv",
        "article_snn_conv1d_comparative_metrics.csv",
        "article_snn_recurrent_comparative_metrics.csv",
    ]:
        p = results_dir / name
        if p.exists():
            rows.extend(r for r in read_csv_rows(p) if str(r.get("model", "")) != "lstm-ae")

    if not rows:
        print("[paper-data] no article result CSV files found")
        return 1

    aggregate_all(rows, data_dir)
    build_profile_table(profiles_dir, data_dir)
    build_model_timing_table(rows, data_dir)
    build_xtensor_opencl_table(results_dir, data_dir)
    print(f"[paper-data] wrote aggregated files to {data_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
