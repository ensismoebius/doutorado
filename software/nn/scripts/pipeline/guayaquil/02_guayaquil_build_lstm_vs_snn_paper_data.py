#!/usr/bin/env python3
"""02_guayaquil_build_lstm_vs_snn_paper_data.py — Aggregate Experiment 04 LSTM-AE vs
SNN-AE comparative CSVs into paper-ready tables for the Guaiaquil conference
article.

Reads per-run comparative_metrics.csv files produced by guayaquil and
aggregates them into summary CSVs suitable for LaTeX (pgfplots / tabular).
Also writes .dat files to the paper's data/ directory for pgfplots input.

Column mapping:
  model == "lstm-ae"                    → label LSTM-AE
  model == "snn-ae" + architecture      → label SNN-{dense|conv1d|recurrent}

Usage:
    python scripts/pipeline/guayaquil/02_guayaquil_build_lstm_vs_snn_paper_data.py \\
        --results-dir results/guayaquil \\
        --data-dir <paper-data-dir> \\
        --profiles-dir src/experiments/guayaquil/profiles

Called automatically at the end of 01_guayaquil_run_article_profiles.sh.
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


def fmt_pgf(value: float, precision: int, thousands: bool = False) -> str:
    """Format a number at a fixed decimal precision, always zero-filled (e.g. 1.0 with
    precision=2 -> "1.00", not "1"). Column magnitudes are much easier to compare at a
    glance in the paper's tables when every row has the same number of decimal digits —
    see paper.tex's zerofill on the matching numeric columns (spike_rate/param_count/macs)."""
    spec = f",.{precision}f" if thousands else f".{precision}f"
    return format(value, spec)


def bold_best(
    values: List[float], precision: int, mode: str, thousands: bool = False
) -> List[str]:
    """Format a column the same way pgfplotstable would, wrapping the best value(s) in
    \\textbf. `mode` is "min" or "max"; ties for best are all bolded. Column must be
    switched to `string type` in paper.tex wherever this is used — see tab:perf,
    tab:recon, tab:eff, tab:backend."""
    best = min(values) if mode == "min" else max(values)
    out = []
    for v in values:
        s = fmt_pgf(v, precision, thousands=thousands)
        out.append(f"\\textbf{{{s}}}" if v == best else s)
    return out


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

    # Best-value bolding (tab:perf, tab:recon, tab:eff in paper.tex): columns below are
    # pre-formatted to match pgfplotstable's default `fixed` rendering exactly (see
    # fmt_pgf) and the best row(s) wrapped in \textbf. spike_rate, param_count, and macs
    # are left as plain numbers — param_count/macs are near-total ties within a table
    # (all three SNN modes share one network, see software/nn/CLAUDE.md SNN invariant #7)
    # so bolding them would highlight a tie rather than a real result, and spike rate has
    # no obvious "better" direction. The corresponding columns must be `string type` in
    # paper.tex.
    summary_models = sorted(by_model)
    summary_mse = [mean([float(r["mse"]) for r in by_model[m]]) for m in summary_models]
    summary_mae = [mean([float(r["mae"]) for r in by_model[m]]) for m in summary_models]
    summary_r2 = [mean([float(r["r2"]) for r in by_model[m]]) for m in summary_models]
    summary_spike = [mean([float(r["spike_rate"]) for r in by_model[m]]) for m in summary_models]
    summary_energy = [mean([float(r["energy"]) for r in by_model[m]]) for m in summary_models]
    summary_infer = [mean([float(r["infer_ms"]) for r in by_model[m]]) for m in summary_models]
    summary_train = [mean([float(r["train_ms"]) for r in by_model[m]]) for m in summary_models]
    summary_params = [mean([float(r["param_count"]) for r in by_model[m]]) for m in summary_models]
    summary_macs = [mean([float(r["macs"]) for r in by_model[m]]) for m in summary_models]

    # Semicolon delimiter (not comma): bold_best's thousands-grouped strings (e.g.
    # "11,842,560") contain literal commas, and pgfplotstable's `col sep=comma` reader
    # does not respect CSV quoting — it silently mis-splits quoted cells. paper.tex uses
    # `col sep=semicolon` for this file accordingly.
    summary_path = data_dir / "paper_summary_by_model.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["model", "mse", "mae", "r2", "spike_rate", "energy", "infer_ms", "train_ms", "param_count", "macs"])
        for i, model in enumerate(summary_models):
            w.writerow([
                model,
                bold_best(summary_mse, 4, "min")[i],
                bold_best(summary_mae, 4, "min")[i],
                bold_best(summary_r2, 4, "max")[i],
                summary_spike[i],
                bold_best(summary_energy, 2, "min", thousands=True)[i],
                bold_best(summary_infer, 2, "min")[i],
                summary_train[i],
                summary_params[i],
                summary_macs[i],
            ])

    recon_keys = sorted(by_model_encoding)
    recon_mse = [mean([float(r["mse"]) for r in by_model_encoding[k]]) for k in recon_keys]
    recon_mae = [mean([float(r["mae"]) for r in by_model_encoding[k]]) for k in recon_keys]
    recon_r2 = [mean([float(r["r2"]) for r in by_model_encoding[k]]) for k in recon_keys]

    # Semicolon delimiter — see comment on paper_summary_by_model.csv above.
    recon_path = data_dir / "paper_recon_by_encoding.csv"
    with recon_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["model", "encoding", "mse", "mae", "r2", "f1"])
        for i, key in enumerate(recon_keys):
            group = by_model_encoding[key]
            model, encoding = key
            w.writerow([
                model,
                encoding,
                bold_best(recon_mse, 4, "min")[i],
                bold_best(recon_mae, 4, "min")[i],
                bold_best(recon_r2, 4, "max")[i],
                mean([float(r["f1"]) for r in group]),
            ])

    eff_keys = sorted(by_model_encoding)
    eff_energy = [mean([float(r["energy"]) for r in by_model_encoding[k]]) for k in eff_keys]

    # Semicolon delimiter — see comment on paper_summary_by_model.csv above.
    eff_path = data_dir / "paper_eff_by_encoding.csv"
    with eff_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["model", "encoding", "spike_rate", "energy", "param_count", "macs"])
        for i, key in enumerate(eff_keys):
            group = by_model_encoding[key]
            model, encoding = key
            w.writerow([
                model,
                encoding,
                mean([float(r["spike_rate"]) for r in group]),
                bold_best(eff_energy, 2, "min", thousands=True)[i],
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
    # article-backend-bench.json is excluded: it is a single-seed, timing-only profile
    # for build_xtensor_opencl_table() above (CPU-vs-OpenCL wall clock), run separately by
    # guayaquil_run_backend_comparison.sh, not by 01_guayaquil_run_article_profiles.sh. Its repeats=1
    # (vs. 3 for every other profile) reads as an inconsistency, and its data feeds no
    # number or table anywhere in paper.tex — listing it in tab:profiles only confuses
    # readers about what was actually used for the reported results.
    profile_files = [
        p
        for p in sorted(profiles_dir.glob("article-*.json"))
        if "backend-bench" not in p.name
    ]
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

    models = sorted(by_model)
    infer_vals = [infer_by_model[m] for m in models]
    train_vals = [train_by_model[m] for m in models]
    speedup_vals = [
        (baseline / infer_by_model[m]) if baseline and infer_by_model[m] else 0.0 for m in models
    ]

    # Best-value bolding (tab:backend): matches pgfplotstable's default `fixed` rendering
    # (see fmt_pgf); infer_ms/train_ms/infer_speedup are `string type` in paper.tex.
    infer_fmt = bold_best(infer_vals, 2, "min")
    train_fmt = bold_best(train_vals, 0, "min", thousands=True)
    speedup_fmt = bold_best(speedup_vals, 2, "max")

    # Semicolon delimiter — see comment on paper_summary_by_model.csv in aggregate_all().
    out_path = data_dir / "paper_backend_comparison.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["model", "infer_ms", "train_ms", "infer_speedup"])
        for i, model in enumerate(models):
            w.writerow([model, infer_fmt[i], train_fmt[i], speedup_fmt[i]])


def build_xtensor_opencl_table(results_dir: pathlib.Path, data_dir: pathlib.Path) -> None:
    """XTensor/CPU vs OpenCL/GPU timing for the SAME model (article-backend-bench profile).

    Populated only by scripts/pipeline/guayaquil/guayaquil_run_backend_comparison.sh, which builds and
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
    parser.add_argument("--profiles-dir", default="src/experiments/guayaquil/profiles")
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
