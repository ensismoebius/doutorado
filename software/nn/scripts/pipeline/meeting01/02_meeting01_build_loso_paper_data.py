#!/usr/bin/env python3
"""02_meeting01_build_loso_paper_data.py — aggregate the nested-LOSO run into
paper-ready tables.

Inputs (per outer fold f, written by the meeting01 binary):
    results/meeting01/<tag>_fold<f>_comparative_metrics.csv                     (split = val | test rows)
    results/meeting01/<tag>_fold<f>_<enc>_run<r>_model_selection_manifest.json  (SNN-AE winner)
    results/meeting01/<tag>_fold<f>_<fam>_run<r>_model_selection_manifest.json  (LSTM-AE/GRU-AE/
                                                                                  Transformer-AE winner,
                                                                                  <fam> in {lstm-ae,
                                                                                  gru-ae, transformer-ae})

One manifest per (dataset, fold, run_id) per family, for all four families alike -- every
family runs its own NSGA-II search and this file records its winner (see
Meeting01Experiment.cpp's finalize_snn_selection / finalize_baseline_selection). The
filename segment right before "_run<r>_..." disambiguates: for SNN-AE it is the winning
genome's own encoding (direct/poisson/latency); for a baseline it is the family token
itself. LSTM-AE's and GRU-AE's JSON bodies are byte-for-byte identical in shape
(hidden_size, num_layers, encoding, val_mse) -- only that filename segment tells them
apart; see _classify_manifest().

Only  split == "test"  rows feed the headline tables. Reconstruction quality is reported
as mean +/- sample standard deviation across the FIVE seeds (per-seed means first). Seeds
establish optimization / reproducibility robustness around the estimate, not independent
samples -- see the paper's Statistical methods and 04_meeting01_significance_tests.py for
the inferential (recording-level) analysis.

Model inventory: FOUR trained families -- SNN-AE, LSTM-AE, GRU-AE, Transformer-AE. The SNN
pre-processing modes (dense / conv1d / recurrent) are input transforms selected per fold,
NOT separate families: every selected SNN winner is labelled "SNN-AE" here, and which mode
won on which validation speaker is reported separately in paper_loso_snn_selection.tex.
PCA and Mean are analytic reference points appended by 03_meeting01_pca_mean_baselines.py
to the per-window CSVs; their reconstruction error is folded in from there.

Outputs (--data-dir):
    paper_loso_summary.csv               model; mse; mae; r2; params; craw; train_ms; infer_ms
                                         (mse/mae/r2/timing = "mean$\\pm$std" strings, best bold)
    paper_loso_recon_by_encoding.csv     model; encoding; mse; mae; r2   (formatted, no bold)
    paper_loso_mse_plot.csv              encoding, <one column per model>  (mse means, for bars)
    paper_loso_snn_selection.tex         per (fold, encoding): winning SNN mode / V_th / alpha
    paper_loso_recurrent_selection.tex   per (family, fold): winning LSTM-AE/GRU-AE hidden
                                         size / depth (median over run_ids) + modal encoding
    paper_loso_transformer_selection.tex per fold: winning Transformer-AE d_model / heads /
                                         depth / d_ff (median over run_ids) + modal encoding

Usage:
    python scripts/pipeline/meeting01/02_meeting01_build_loso_paper_data.py \\
        --results-dir results/meeting01 --run-tag meeting01_loso --data-dir <paper-data-dir>
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import pathlib
import re
import statistics as pystat
import sys
from collections import defaultdict

import numpy as np

# "meeting01_loso_audiomnist_fold3_comparative_metrics.csv" -> dataset "audiomnist".
# Legacy files without a dataset segment ("meeting01_loso_fold3_...") fall back to "fsdd".
_DATASET_RE = re.compile(r"_(?P<ds>[a-z0-9]+)_fold\d+_")
# "mitbih" replaced 2026-09-23 by eegmmidb/chbmit in the active LOSO grid; kept here
# (falls back to DATASET_ORDER's unknown-dataset tail / DATASET_TITLE.get(d, d) otherwise)
# so any pre-existing mitbih result CSVs from before the swap still title correctly.
DATASET_ORDER = ["fsdd", "audiomnist", "eegmmidb", "chbmit", "mitbih"]
DATASET_TITLE = {
    "fsdd": "FSDD",
    "audiomnist": "AudioMNIST",
    "eegmmidb": "PhysioNet EEGMMIDB",
    "chbmit": "CHB-MIT EEG",
    "mitbih": "MIT-BIH ECG",
}


def _dataset_of(path: pathlib.Path, run_tag: str) -> str:
    rest = path.name[len(run_tag):] if path.name.startswith(run_tag) else path.name
    m = _DATASET_RE.search(rest)
    return m["ds"] if m else "fsdd"


# Selection-manifest filename: "..._fold<f>_<segment>_run<r>_model_selection_manifest.json".
# <segment> disambiguates which family wrote it -- see finalize_snn_selection /
# finalize_baseline_selection in Meeting01Experiment.cpp. SNN's segment is the winning
# genome's own encoding gene (never a grid-search sweep label, despite an older comment
# in that source file that said otherwise -- fixed 2026-09-23); a baseline's segment is
# its family token verbatim.
_MANIFEST_RE = re.compile(
    r"_fold(?P<fold>\d+)_(?P<segment>.+)_run(?P<run>\d+)_model_selection_manifest\.json$")
_SNN_ENCODINGS = {"direct", "poisson", "latency"}
# family token (as it appears in the filename) -> paper-facing label.
_BASELINE_FAMILY_LABEL = {
    "lstm-ae": "LSTM-AE",
    "gru-ae": "GRU-AE",
    "transformer-ae": "Transformer-AE",
}


def _classify_manifest(path: pathlib.Path) -> tuple[str, int, int] | None:
    """(family, fold, run_id) from a selection-manifest filename, family in
    {"snn-ae", "lstm-ae", "gru-ae", "transformer-ae"}. None if the filename's segment
    matches neither the known encodings nor the known family tokens -- e.g. a 5th
    family added later without updating this map -- so callers can warn instead of
    silently mis-sorting an unrecognised manifest into the wrong table."""
    m = _MANIFEST_RE.search(path.name)
    if not m:
        return None
    segment = m["segment"]
    if segment in _SNN_ENCODINGS:
        return ("snn-ae", int(m["fold"]), int(m["run"]))
    if segment in _BASELINE_FAMILY_LABEL:
        return (segment, int(m["fold"]), int(m["run"]))
    return None

# label -> stable column key for the wide plot CSV
PLOT_KEY = {
    "Mean": "mean",
    "PCA": "pca",
    "LSTM-AE": "lstm_ae",
    "GRU-AE": "gru_ae",
    "Transformer-AE": "transformer_ae",
    "SNN-AE": "snn_ae",
}
MODEL_ORDER = ["Mean", "PCA", "SNN-AE", "LSTM-AE", "GRU-AE", "Transformer-AE"]
ENCODINGS = ["direct", "poisson", "latency"]

# column -> (formatter precision, "min"|"max" for best-bolding)
FMT = {
    "mse": (4, "min"),
    "mae": (4, "min"),
    "r2": (4, "max"),
    "train_ms": (1, "min"),
    "infer_ms": (2, "min"),
}


def _model_label(model: str, architecture: str) -> str:
    if model == "snn-ae":
        return "SNN-AE"
    return {
        "lstm-ae": "LSTM-AE",
        "gru-ae": "GRU-AE",
        "transformer-ae": "Transformer-AE",
        "pca": "PCA",
        "mean": "Mean",
    }.get(model, model)


def _mean_std(per_seed: list[float]) -> tuple[float, float]:
    a = np.asarray(per_seed, dtype=float)
    if a.size == 0:
        return (float("nan"), float("nan"))
    return (float(a.mean()), float(a.std(ddof=1)) if a.size > 1 else 0.0)


# --------------------------------------------------------------------- loading

def load_comparative(results_dir: pathlib.Path, run_tag: str) -> list[dict]:
    """One dict per test-split comparative row (fold x seed x model x encoding)."""
    rows: list[dict] = []
    paths = sorted(results_dir.glob(f"{run_tag}*_fold*_comparative_metrics.csv"))
    for path in paths:
        ds = _dataset_of(path, run_tag)
        with path.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("split") != "test":
                    continue
                rows.append({
                    "dataset": ds,
                    "label": _model_label(r["model"], r.get("architecture", "")),
                    "encoding": r["encoding"],
                    "seed": int(r["seed"]),
                    "mse": float(r["mse"]),
                    "mae": float(r["mae"]),
                    "r2": float(r["r2"]),
                    "train_ms": float(r["train_ms"]),
                    "infer_ms": float(r["infer_ms"]),
                    "params": float(r["param_count"]),
                    "craw": float(r["macs"]),
                })
    if not paths:
        print(f"[loso-data] no {run_tag}_fold*_comparative_metrics.csv under {results_dir}",
              file=sys.stderr)
    return rows


def load_per_window_refs(results_dir: pathlib.Path, run_tag: str) -> list[dict]:
    """PCA / Mean test rows from the per-window CSVs (appended by 03_)."""
    rows: list[dict] = []
    for path in sorted(results_dir.glob(f"{run_tag}*_fold*_per_window_errors.csv")):
        ds = _dataset_of(path, run_tag)
        with path.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("split") != "test" or r["model"] not in ("pca", "mean"):
                    continue
                rows.append({
                    "dataset": ds,
                    "label": _model_label(r["model"], ""),
                    "encoding": r["encoding"],
                    "seed": int(r["seed"]),
                    "mse": float(r["mse"]),
                    "mae": float(r["mae"]),
                    "r2": float("nan"),
                    "train_ms": float("nan"),
                    "infer_ms": float("nan"),
                    "params": float("nan"),
                    "craw": float("nan"),
                })
    return rows


def load_snn_selection(results_dir: pathlib.Path, run_tag: str) -> list[dict]:
    out: list[dict] = []
    for path in sorted(results_dir.glob(f"{run_tag}*_fold*_*_model_selection_manifest.json")):
        cls = _classify_manifest(path)
        if cls is None or cls[0] != "snn-ae":
            continue
        m = json.loads(path.read_text(encoding="utf-8"))
        sel = m["selected"]
        out.append({
            "dataset": m.get("dataset", _dataset_of(path, run_tag)),
            "fold": int(m["cv_fold"]),
            # The winning genome's own encoding gene; identical to the top-level
            # m["encoding"] by construction (finalize_snn_selection's only caller
            # passes winner.genome.encoding as both), but read from `selected` since
            # that is the field that is guaranteed to mean "the gene", not "whatever
            # the caller happened to label this call".
            "encoding": sel["encoding"],
            "test_speaker": m.get("test_speaker", "?"),
            "val_speaker": m.get("selection_split", "").replace("val (speaker ", "").rstrip(")"),
            "architecture": sel["architecture"],
            "v_th": float(sel["v_th"]),
            "alpha": float(sel["alpha"]),
            "val_mse": float(sel["val_mse"]),
        })
    return out


def load_baseline_selection(results_dir: pathlib.Path, run_tag: str) -> list[dict]:
    """One row per (dataset, family, fold, run_id) LSTM-AE/GRU-AE/Transformer-AE
    winner -- the baseline-family analogue of load_snn_selection(). LSTM-AE and
    GRU-AE manifests are indistinguishable by JSON shape alone (both are
    {hidden_size, num_layers, encoding, val_mse}); _classify_manifest() disambiguates
    from the filename's family-token segment, so `family` here is authoritative even
    though it is never itself a JSON key inside the manifest body."""
    out: list[dict] = []
    for path in sorted(results_dir.glob(f"{run_tag}*_fold*_*_model_selection_manifest.json")):
        cls = _classify_manifest(path)
        if cls is None or cls[0] not in _BASELINE_FAMILY_LABEL:
            continue
        family, fold, run_id = cls
        m = json.loads(path.read_text(encoding="utf-8"))
        sel = m["selected"]
        row = {
            "dataset": m.get("dataset", _dataset_of(path, run_tag)),
            "family": family,
            "fold": fold,
            "run_id": run_id,
            "encoding": sel["encoding"],
            "test_speaker": m.get("test_speaker", "?"),
            "val_speaker": m.get("selection_split", "").replace("val (speaker ", "").rstrip(")"),
            "val_mse": float(sel["val_mse"]),
        }
        if family in ("lstm-ae", "gru-ae"):
            row["hidden_size"] = int(sel["hidden_size"])
            row["num_layers"] = int(sel["num_layers"])
        else:  # transformer-ae
            row["d_model"] = int(sel["d_model"])
            row["n_heads"] = int(sel["n_heads"])
            row["n_layers"] = int(sel["n_layers"])
            row["d_ff"] = int(sel["d_ff"])
        out.append(row)
    return out


# ---------------------------------------------------------- degeneracy caveat

# A window whose test-split reconstruction error is this tiny is not "the model
# converged" -- at these window sizes it is the signature of a near-duplicate
# input population (see the AudioMNIST case in .wiki/Experiments/Meeting01.md:
# `source_window_index` stuck at 0 samples only the recording's leading
# near-silence, so every window collapses to ~1 distinct pattern and any model
# reproduces it trivially, on train AND on the LOSO-held-out test speaker).
_DEGENERATE_MSE_THRESHOLD = 1e-4
_DEGENERATE_FRACTION_WARN = 0.02


def check_degenerate_reconstruction(rows: list[dict], dataset: str) -> str | None:
    """None if `dataset`'s test-split mse values look ordinary; else a caveat
    string naming the suspiciously-degenerate fraction, for callers to print
    and/or write next to the paper tables so it cannot be missed later."""
    if not rows:
        return None
    n_low = sum(1 for r in rows if r["mse"] < _DEGENERATE_MSE_THRESHOLD)
    frac = n_low / len(rows)
    if frac < _DEGENERATE_FRACTION_WARN:
        return None
    return (
        f"CAVEAT ({dataset}): {n_low}/{len(rows)} test rows ({frac * 100:.1f}%) have "
        f"mse < {_DEGENERATE_MSE_THRESHOLD}. Before citing these numbers, check whether "
        "the sampled windows carry real signal (known cause for this pipeline: stratified "
        "window sampling always takes source_window_index 0, which is pure recording "
        "lead-in for some corpora -- see the 'Known caveat' box in "
        ".wiki/Experiments/Meeting01.md)."
    )

# ------------------------------------------------------------------- aggregate

def _per_seed_means(rows: list[dict], keyer, metric: str) -> dict:
    by_key_seed: dict = defaultdict(list)
    for r in rows:
        v = r[metric]
        if v == v:  # skip nan
            by_key_seed[(keyer(r), r["seed"])].append(v)
    per_key: dict = defaultdict(list)
    for (key, _seed), vals in by_key_seed.items():
        per_key[key].append(float(np.mean(vals)))
    return per_key


def _order(labels) -> list[str]:
    known = [m for m in MODEL_ORDER if m in labels]
    return known + sorted(l for l in labels if l not in MODEL_ORDER)


def _fmt_cell(mean: float, std: float, prec: int, bold: bool) -> str:
    if mean != mean:
        return ""
    s = f"{mean:.{prec}f}$\\pm${std:.{prec}f}"
    return f"\\textbf{{{s}}}" if bold else s


def write_summary(rows: list[dict], data_dir: pathlib.Path, infix: str = "") -> pathlib.Path:
    agg: dict = {}
    for m in ("mse", "mae", "r2", "train_ms", "infer_ms"):
        agg[m] = {k: _mean_std(v) for k, v in
                  _per_seed_means(rows, lambda r: r["label"], m).items()}
    # params / craw: constant per label (take any finite value)
    const: dict = defaultdict(dict)
    for r in rows:
        for k in ("params", "craw"):
            if r[k] == r[k]:
                const[r["label"]][k] = r[k]

    labels = _order({r["label"] for r in rows})
    best: dict = {}
    for m, (prec, direction) in FMT.items():
        vals = {lab: agg[m][lab][0] for lab in labels if lab in agg[m] and agg[m][lab][0] == agg[m][lab][0]}
        if vals:
            best[m] = (min if direction == "min" else max)(vals, key=vals.get)

    out = data_dir / f"paper_loso_{infix}summary.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["model", "mse", "mae", "r2", "params", "craw", "train_ms", "infer_ms"])
        for lab in labels:
            cells = {}
            for m, (prec, _d) in FMT.items():
                mean, std = agg[m].get(lab, (float("nan"), float("nan")))
                cells[m] = _fmt_cell(mean, std, prec, best.get(m) == lab)
            p = const[lab].get("params")
            c = const[lab].get("craw")
            w.writerow([lab, cells["mse"], cells["mae"], cells["r2"],
                        "" if p is None else f"{int(round(p))}",
                        "" if c is None else f"{int(round(c))}",
                        cells["train_ms"], cells["infer_ms"]])
    return out


def write_recon_by_encoding(rows: list[dict], data_dir: pathlib.Path, infix: str = "") -> pathlib.Path:
    agg: dict = {}
    for m in ("mse", "mae", "r2"):
        agg[m] = {k: _mean_std(v) for k, v in
                  _per_seed_means(rows, lambda r: (r["label"], r["encoding"]), m).items()}
    out = data_dir / f"paper_loso_{infix}recon_by_encoding.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["model", "encoding", "mse", "mae", "r2"])
        keys = sorted(agg["mse"].keys(),
                      key=lambda k: (MODEL_ORDER.index(k[0]) if k[0] in MODEL_ORDER else 99,
                                     ENCODINGS.index(k[1]) if k[1] in ENCODINGS else 99))
        for lab, enc in keys:
            row = [lab, enc]
            for m in ("mse", "mae", "r2"):
                mean, std = agg[m].get((lab, enc), (float("nan"), float("nan")))
                prec = 4
                row.append("" if mean != mean else f"{mean:.{prec}f}$\\pm${std:.{prec}f}")
            w.writerow(row)
    return out


def write_mse_plot(rows: list[dict], data_dir: pathlib.Path, infix: str = "") -> pathlib.Path:
    per = _per_seed_means(rows, lambda r: (r["label"], r["encoding"]), "mse")
    present = _order({lab for (lab, _e) in per})
    cols = [PLOT_KEY[lab] for lab in present if lab in PLOT_KEY]
    out = data_dir / f"paper_loso_{infix}mse_plot.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["encoding"] + cols)
        for enc in ENCODINGS:
            row = [enc]
            for lab in present:
                if lab not in PLOT_KEY:
                    continue
                mean, _s = _mean_std(per.get((lab, enc), []))
                row.append("nan" if mean != mean else f"{mean:.6f}")
            w.writerow(row)
    return out


def write_snn_selection_tex(sel: list[dict], data_dir: pathlib.Path, infix: str = "") -> pathlib.Path:
    """One row per (fold, encoding): modal winning mode + median V_th/alpha over seeds."""
    by: dict = defaultdict(list)
    for s in sel:
        by[(s["fold"], s["encoding"])].append(s)
    lines = [
        r"% auto-generated by 02_meeting01_build_loso_paper_data.py",
        r"\begin{tabular}{llllrrr}",
        r"\toprule",
        r"Fold & Test spk. & Val spk. & Enc. & Mode & $V_{th}$ & $\alpha$ \\",
        r"\midrule",
    ]
    for (fold, enc) in sorted(by):
        grp = by[(fold, enc)]
        modes = collections.Counter(g["architecture"] for g in grp)
        mode = modes.most_common(1)[0][0]
        agree = modes[mode]
        vth = pystat.median(g["v_th"] for g in grp)
        alpha = pystat.median(g["alpha"] for g in grp)
        mode_cell = mode if agree == len(grp) else f"{mode} ({agree}/{len(grp)})"
        lines.append(
            f"{fold} & {grp[0]['test_speaker']} & {grp[0]['val_speaker']} & {enc} & "
            f"{mode_cell} & {vth:.2f} & {alpha:.2f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    out = data_dir / f"paper_loso_{infix}snn_selection.tex"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def write_recurrent_selection_tex(sel: list[dict], data_dir: pathlib.Path, infix: str = "") -> pathlib.Path:
    """One row per (family, fold) for LSTM-AE and GRU-AE: modal winning encoding +
    median hidden_size/num_layers over run_ids. Mirrors write_snn_selection_tex's
    aggregation convention (mode for the categorical gene, median for the numeric
    ones); LSTM-AE and GRU-AE share this table since their genome shape is identical."""
    by: dict = defaultdict(list)
    for s in sel:
        if s["family"] not in ("lstm-ae", "gru-ae"):
            continue
        by[(s["family"], s["fold"])].append(s)
    lines = [
        r"% auto-generated by 02_meeting01_build_loso_paper_data.py",
        r"\begin{tabular}{llllrrr}",
        r"\toprule",
        r"Family & Fold & Test spk. & Enc. & $H$ & $L$ & val MSE \\",
        r"\midrule",
    ]
    for (family, fold) in sorted(by, key=lambda k: (k[0], k[1])):
        grp = by[(family, fold)]
        encs = collections.Counter(g["encoding"] for g in grp)
        enc = encs.most_common(1)[0][0]
        agree = encs[enc]
        hidden = pystat.median(g["hidden_size"] for g in grp)
        layers = pystat.median(g["num_layers"] for g in grp)
        val_mse = pystat.median(g["val_mse"] for g in grp)
        enc_cell = enc if agree == len(grp) else f"{enc} ({agree}/{len(grp)})"
        lines.append(
            f"{_BASELINE_FAMILY_LABEL[family]} & {fold} & {grp[0]['test_speaker']} & "
            f"{enc_cell} & {hidden:.0f} & {layers:.0f} & {val_mse:.4f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    out = data_dir / f"paper_loso_{infix}recurrent_selection.tex"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def write_transformer_selection_tex(sel: list[dict], data_dir: pathlib.Path, infix: str = "") -> pathlib.Path:
    """One row per fold for Transformer-AE: modal winning encoding + median
    d_model/n_heads/n_layers/d_ff over run_ids. Same aggregation convention as
    write_recurrent_selection_tex; Transformer-AE gets its own table because its
    genome has a different shape (no hidden_size/num_layers)."""
    by: dict = defaultdict(list)
    for s in sel:
        if s["family"] != "transformer-ae":
            continue
        by[s["fold"]].append(s)
    lines = [
        r"% auto-generated by 02_meeting01_build_loso_paper_data.py",
        r"\begin{tabular}{lllrrrr}",
        r"\toprule",
        r"Fold & Test spk. & Enc. & $d_{\mathrm{model}}$ & Heads & $N$ & $d_{\mathrm{ff}}$ \\",
        r"\midrule",
    ]
    for fold in sorted(by):
        grp = by[fold]
        encs = collections.Counter(g["encoding"] for g in grp)
        enc = encs.most_common(1)[0][0]
        agree = encs[enc]
        d_model = pystat.median(g["d_model"] for g in grp)
        n_heads = pystat.median(g["n_heads"] for g in grp)
        n_layers = pystat.median(g["n_layers"] for g in grp)
        d_ff = pystat.median(g["d_ff"] for g in grp)
        enc_cell = enc if agree == len(grp) else f"{enc} ({agree}/{len(grp)})"
        lines.append(
            f"{fold} & {grp[0]['test_speaker']} & {enc_cell} & {d_model:.0f} & "
            f"{n_heads:.0f} & {n_layers:.0f} & {d_ff:.0f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    out = data_dir / f"paper_loso_{infix}transformer_selection.tex"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=pathlib.Path, default=pathlib.Path("results/meeting01"))
    ap.add_argument("--run-tag", default="meeting01_loso")
    ap.add_argument(
        "--data-dir",
        type=pathlib.Path,
        default=pathlib.Path(
            "/home/ensismoebius/Repos/doutorado/documentation/07-articlesProduced/"
            "meeting01/data"),
    )
    args = ap.parse_args(argv)
    args.data_dir.mkdir(parents=True, exist_ok=True)

    rows = load_comparative(args.results_dir, args.run_tag)
    rows += load_per_window_refs(args.results_dir, args.run_tag)
    if not rows:
        print("[loso-data] no test-split rows found", file=sys.stderr)
        return 1
    sel = load_snn_selection(args.results_dir, args.run_tag)
    baseline_sel = load_baseline_selection(args.results_dir, args.run_tag)

    datasets = sorted({r["dataset"] for r in rows},
                      key=lambda d: (DATASET_ORDER.index(d) if d in DATASET_ORDER else 99, d))
    written: list[pathlib.Path] = []
    for ds in datasets:
        infix = f"{ds}_"
        d_rows = [r for r in rows if r["dataset"] == ds]
        d_sel = [s for s in sel if s.get("dataset") == ds]
        d_baseline_sel = [s for s in baseline_sel if s.get("dataset") == ds]
        written += [
            write_summary(d_rows, args.data_dir, infix),
            write_recon_by_encoding(d_rows, args.data_dir, infix),
            write_mse_plot(d_rows, args.data_dir, infix),
        ]
        if d_sel:
            written.append(write_snn_selection_tex(d_sel, args.data_dir, infix))
        if any(s["family"] in ("lstm-ae", "gru-ae") for s in d_baseline_sel):
            written.append(write_recurrent_selection_tex(d_baseline_sel, args.data_dir, infix))
        if any(s["family"] == "transformer-ae" for s in d_baseline_sel):
            written.append(write_transformer_selection_tex(d_baseline_sel, args.data_dir, infix))
        caveat = check_degenerate_reconstruction(d_rows, ds)
        if caveat:
            print(f"[loso-data] {caveat}", file=sys.stderr)
            caveat_path = args.data_dir / f"paper_loso_{infix}CAVEATS.txt"
            caveat_path.write_text(caveat + "\n", encoding="utf-8")
            written.append(caveat_path)

    # datasets.tex: the \foreach list the paper iterates.
    dtex = args.data_dir / "paper_loso_datasets.tex"
    dtex.write_text(
        "% auto-generated by 02_meeting01_build_loso_paper_data.py\n"
        + "".join(f"\\loParseDataset{{{d}}}{{{DATASET_TITLE.get(d, d)}}}\n" for d in datasets),
        encoding="utf-8",
    )
    written.append(dtex)
    print(f"[loso-data] {len(datasets)} dataset(s): {', '.join(datasets)}")
    print("[loso-data] wrote " + ", ".join(p.name for p in written) + f" to {args.data_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
