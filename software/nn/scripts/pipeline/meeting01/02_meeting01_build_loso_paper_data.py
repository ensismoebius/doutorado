#!/usr/bin/env python3
"""02_meeting01_build_loso_paper_data.py — aggregate the nested-LOSO run into
paper-ready tables.

Inputs (per dataset d and outer fold f, prefix <tag>_<d>_fold<f>, written by the binary):
    _comparative_metrics.csv                     one test row per family winner and seed
    _per_window_errors.csv                       with 03_'s mean / pca reference rows
    _split_manifest.json                         the fold's provenance (results_format)
    _<enc>_run<r>_model_selection_manifest.json  SNN-AE winner (<enc> = its encoding gene)
    _<fam>_run<r>_model_selection_manifest.json  LSTM-AE / GRU-AE / Transformer-AE winner

One manifest per (dataset, fold, run_id) per family, for all four families alike -- every
family runs its own NSGA-II search and this file records its winner (see
Meeting01Experiment.cpp's finalize_snn_selection / finalize_baseline_selection). The
filename segment right before "_run<r>_..." disambiguates: for SNN-AE it is the winning
genome's own encoding (direct/poisson/latency); for a baseline it is the family token
itself. LSTM-AE's and GRU-AE's JSON bodies are byte-for-byte identical in shape
(hidden_size, num_layers, encoding, val_mse) -- only that filename segment tells them
apart; see selection_manifests().

Only  split == "test"  rows feed the headline tables. Reconstruction quality is reported
as mean +/- sample standard deviation across the FIVE seeds (per-seed means first, each a
mean over folds of the fold's mean window error). Seeds establish optimization /
reproducibility robustness around the estimate, not independent samples -- see the
paper's Statistical methods and 04_meeting01_significance_tests.py for the inferential
(recording-level) analysis.

Model inventory: FOUR trained families -- SNN-AE, LSTM-AE, GRU-AE, Transformer-AE. The SNN
pre-processing modes (dense / conv1d / recurrent) are input transforms selected per fold,
NOT separate families: every selected SNN winner is labelled "SNN-AE" here, and which mode
won on which validation speaker is reported separately in paper_loso_snn_selection.tex.
PCA and Mean are analytic reference points appended by 03_meeting01_pca_mean_baselines.py
to the per-window CSVs; they are averaged per fold first, exactly like a family's
comparative row (the mean window error of the fold), so both sides of the table are the
same estimand even when folds hold different numbers of test windows.

Refuses, naming cause and remedy (no fallbacks): a fold from a binary older than
2026-10-06 (meeting01_results.py); a file without a dataset segment; a fold with a
comparative CSV but no per-window CSV, or the reverse; a fold without reference rows; a
test row whose dataset / fold disagrees with its file name; a trained model with
train_ms <= 0; a selection manifest without its test row (left over from an earlier run)
or a test row without its manifest.

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
import os
import pathlib
import re
import shutil
import statistics as pystat
import sys
import tempfile
from collections import defaultdict

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
from meeting01_results import (  # noqa: E402
    PW_HEADER,
    REFERENCE_MODELS,
    REFERENCE_SEED,
    RESULTS_FORMAT,
    TRAINED_MODELS,
    ResultsError,
    check_results_format,
    fold_files,
    read_per_window,
    split_manifest,
)

# "mitbih" replaced 2026-09-23 by eegmmidb/chbmit, then chbmit replaced the same day by
# siena (disk space), in the active LOSO grid; both are kept so their titles stay known.
DATASET_ORDER = ["fsdd", "audiomnist", "eegmmidb", "siena", "chbmit", "mitbih"]
DATASET_TITLE = {
    "fsdd": "FSDD",
    "audiomnist": "AudioMNIST",
    "eegmmidb": "PhysioNet EEGMMIDB",
    "siena": "Siena Scalp EEG",
    "chbmit": "CHB-MIT EEG",
    "mitbih": "MIT-BIH ECG",
}


# Selection-manifest filename: "<tag>_<d>_fold<f>_<segment>_run<r>_model_selection_manifest.json".
# <segment> disambiguates which family wrote it -- see finalize_snn_selection /
# finalize_baseline_selection in Meeting01Experiment.cpp. SNN's segment is the winning
# genome's own encoding gene (never a grid-search sweep label, despite an older comment
# in that source file that said otherwise -- fixed 2026-09-23); a baseline's segment is
# its family token verbatim.
_MANIFEST_SUFFIX = r"(?P<segment>[a-z0-9-]+)_run(?P<run>\d+)_model_selection_manifest\.json"
_SNN_ENCODINGS = {"direct", "poisson", "latency"}
# family token (as it appears in the filename) -> paper-facing label.
_BASELINE_FAMILY_LABEL = {
    "lstm-ae": "LSTM-AE",
    "gru-ae": "GRU-AE",
    "transformer-ae": "Transformer-AE",
}
# The comparative-CSV columns this script reads (write_rows_csv, Meeting01Output.cpp).
_COMPARATIVE_COLUMNS = ("dataset", "model", "encoding", "architecture", "run", "seed", "split",
                        "cv_fold", "mse", "mae", "r2", "train_ms", "infer_ms", "param_count",
                        "macs")


def selection_manifests(results_dir: pathlib.Path, run_tag: str) -> list[dict]:
    """Every family winner's selection manifest of the run, as {dataset, fold, family,
    run_id, seed, manifest}: in the current results format, and recording the dataset /
    fold / run its file name says. A segment that is neither an SNN encoding nor a known
    family token (a fifth family added without updating this map) is refused, not
    silently left out of the tables."""
    out = []
    for path, ds, fold, m in fold_files(results_dir, run_tag, _MANIFEST_SUFFIX):
        segment, run_id = m["segment"], int(m["run"])
        if segment in _SNN_ENCODINGS:
            family = "snn-ae"
        elif segment in _BASELINE_FAMILY_LABEL:
            family = segment
        else:
            raise ResultsError(
                f"{path.name}: segment {segment!r} is neither an SNN encoding "
                f"{sorted(_SNN_ENCODINGS)} nor a family token {sorted(_BASELINE_FAMILY_LABEL)}. "
                "A new family? Add it to _BASELINE_FAMILY_LABEL so its winners are tabulated.")
        man = json.loads(path.read_text(encoding="utf-8"))
        check_results_format(man, path.name)
        if (man.get("dataset"), man.get("cv_fold"), man.get("run_id")) != (ds, fold, run_id):
            raise ResultsError(
                f"{path.name} records dataset={man.get('dataset')!r} fold={man.get('cv_fold')!r} "
                f"run={man.get('run_id')!r}: it was renamed or copied from elsewhere.")
        out.append({"dataset": ds, "fold": fold, "family": family, "run_id": run_id,
                    "seed": int(man["seed"]), "manifest": man, "name": path.name})
    return out

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
    """One dict per test-split comparative row (dataset x fold x seed x family)."""
    rows: list[dict] = []
    for path, ds, fold, _ in fold_files(results_dir, run_tag,
                                        re.escape("comparative_metrics.csv")):
        split_manifest(results_dir, run_tag, ds, fold)
        with path.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            absent = [c for c in _COMPARATIVE_COLUMNS if c not in (reader.fieldnames or [])]
            if absent:
                raise ResultsError(f"{path.name} lacks the column(s) {absent}: written by a "
                                   "meeting01 version this script does not read.")
            fold_rows = [r for r in reader if r["split"] == "test"]
        seen: set = set()
        for line, r in enumerate(fold_rows, start=2):
            where = f"{path.name} (test row {line - 1})"
            if r["dataset"] != ds or int(r["cv_fold"]) != fold:
                raise ResultsError(f"{where} says dataset={r['dataset']} fold={r['cv_fold']}, "
                                   f"its file name {ds} fold {fold}: renamed or concatenated.")
            if r["model"] not in TRAINED_MODELS:
                raise ResultsError(f"{where}: unknown model {r['model']!r}; add it to "
                                   "TRAINED_MODELS in meeting01_results.py.")
            if not r["encoding"]:
                raise ResultsError(f"{where}: {r['model']} winner without an encoding.")
            if float(r["train_ms"]) <= 0.0:
                raise ResultsError(
                    f"{where}: {r['model']} seed {r['seed']} has train_ms {r['train_ms']} -- a "
                    "trained model cannot take no time, and the cost table would print it as "
                    "the fastest. A binary before 2026-10-06 wrote 0 for every LSTM/GRU/"
                    "Transformer-AE test row (and a checkpoint without train_ms restores 0); "
                    "rerun the fold with the current binary and RESUME unset.")
            key = (r["model"], r["run"], r["seed"])
            if key in seen:
                raise ResultsError(f"{where}: {r['model']} run {r['run']} seed {r['seed']} "
                                   "appears twice.")
            seen.add(key)
            rows.append({
                "dataset": ds,
                "fold": fold,
                "model": r["model"],
                "run_id": int(r["run"]),
                "label": _model_label(r["model"], r["architecture"]),
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
    return rows


def load_per_window_refs(results_dir: pathlib.Path, run_tag: str) -> list[dict]:
    """PCA / Mean, one row per (dataset, fold, model, encoding): the mean of the fold's
    per-window test errors -- the same estimand as a family's comparative row, whose mse
    is the mean window error of that fold (evaluate_ae). Averaging all windows of all
    folds at once instead would weight the folds by their window counts, which the
    families' numbers do not."""
    rows: list[dict] = []
    for path, ds, fold, _ in fold_files(results_dir, run_tag, re.escape("per_window_errors.csv")):
        split_manifest(results_dir, run_tag, ds, fold)
        acc: dict = defaultdict(list)
        for r in read_per_window(path, fold):
            if r["split"] == "test" and r["model"] in REFERENCE_MODELS:
                acc[(r["model"], r["encoding"])].append((float(r["mse"]), float(r["mae"])))
        missing = [m for m in REFERENCE_MODELS if not any(k[0] == m for k in acc)]
        if missing:
            raise ResultsError(f"{path.name} has no {'/'.join(missing)} reference rows: run "
                               "03_meeting01_pca_mean_baselines.py first.")
        for (model, encoding), errors in sorted(acc.items()):
            mse, mae = np.mean(errors, axis=0)
            rows.append({
                "dataset": ds,
                "fold": fold,
                "label": _model_label(model, ""),
                "encoding": encoding,
                "seed": int(REFERENCE_SEED),
                "mse": float(mse),
                "mae": float(mae),
                "r2": float("nan"),
                "train_ms": float("nan"),
                "infer_ms": float("nan"),
                "params": float("nan"),
                "craw": float("nan"),
            })
    return rows


def check_folds_and_manifests(comparative: list[dict], references: list[dict],
                              manifests: list[dict]) -> None:
    """The three file families of a run must describe the same folds and the same family
    winners. A fold with test rows but no references (03_ not run since it finished), a
    selection manifest whose test row is gone (left over from an earlier run with more
    seeds, which a rerun does not delete), or a test row whose manifest is missing would
    each make one table silently disagree with another."""
    def folds(rows):
        return {(r["dataset"], r["fold"]) for r in rows}

    problems = []
    if folds(comparative) - folds(references):
        problems.append("folds with test rows but no per-window CSV: "
                        f"{sorted(folds(comparative) - folds(references))}")
    if folds(references) - folds(comparative):
        problems.append("folds with a per-window CSV but no test rows: "
                        f"{sorted(folds(references) - folds(comparative))}")
    if problems:
        raise ResultsError("; ".join(problems) + ". A fold that crashed between the two writes, "
                           "or a file left from another run: rerun those folds.")
    winners = {(r["dataset"], r["fold"], r["model"], r["run_id"], r["seed"]) for r in comparative}
    recorded = {(m["dataset"], m["fold"], m["family"], m["run_id"], m["seed"]) for m in manifests}
    stale = sorted(m["name"] for m in manifests
                   if (m["dataset"], m["fold"], m["family"], m["run_id"], m["seed"]) not in winners)
    if stale:
        problems.append(f"selection manifests without their test row: {stale} (left over from "
                        "an earlier run of the fold -- delete them)")
    if winners - recorded:
        problems.append(f"test rows without their manifest: {sorted(winners - recorded)} (the "
                        "manifest write failed: rerun the fold)")
    if problems:
        raise ResultsError("; ".join(problems) + ".")


def load_snn_selection(manifests: list[dict]) -> list[dict]:
    out: list[dict] = []
    for entry in manifests:
        if entry["family"] != "snn-ae":
            continue
        m = entry["manifest"]
        sel = m["selected"]
        out.append({
            "dataset": entry["dataset"],
            "fold": entry["fold"],
            # The winning genome's own encoding gene; identical to the top-level
            # m["encoding"] by construction (finalize_snn_selection's only caller
            # passes winner.genome.encoding as both), but read from `selected` since
            # that is the field that is guaranteed to mean "the gene", not "whatever
            # the caller happened to label this call".
            "encoding": sel["encoding"],
            "test_speaker": m["test_speaker"],
            "val_speaker": m["selection_split"].replace("val (speaker ", "").rstrip(")"),
            "architecture": sel["architecture"],
            "v_th": float(sel["v_th"]),
            "alpha": float(sel["alpha"]),
            "val_mse": float(sel["val_mse"]),
        })
    return out


def load_baseline_selection(manifests: list[dict]) -> list[dict]:
    """One row per (dataset, family, fold, run_id) LSTM-AE/GRU-AE/Transformer-AE
    winner -- the baseline-family analogue of load_snn_selection(). LSTM-AE and
    GRU-AE manifests are indistinguishable by JSON shape alone (both are
    {hidden_size, num_layers, encoding, val_mse}); selection_manifests() disambiguates
    from the filename's family-token segment, so `family` here is authoritative even
    though it is never itself a JSON key inside the manifest body."""
    out: list[dict] = []
    for entry in manifests:
        family = entry["family"]
        if family not in _BASELINE_FAMILY_LABEL:
            continue
        m = entry["manifest"]
        sel = m["selected"]
        row = {
            "dataset": entry["dataset"],
            "family": family,
            "fold": entry["fold"],
            "run_id": entry["run_id"],
            "encoding": sel["encoding"],
            "test_speaker": m["test_speaker"],
            "val_speaker": m["selection_split"].replace("val (speaker ", "").rstrip(")"),
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


def build(results_dir: pathlib.Path, run_tag: str, data_dir: pathlib.Path) -> list:
    """Reads and cross-checks the whole run, then writes the paper tables; returns the
    datasets. Raises ResultsError, before writing anything, on any inconsistency."""
    comparative = load_comparative(results_dir, run_tag)
    references = load_per_window_refs(results_dir, run_tag)
    manifests = selection_manifests(results_dir, run_tag)
    if not comparative:
        raise ResultsError(f"no {run_tag}_<dataset>_fold<f>_comparative_metrics.csv test rows "
                           f"under {results_dir}.")
    check_folds_and_manifests(comparative, references, manifests)
    rows = comparative + references
    sel = load_snn_selection(manifests)
    baseline_sel = load_baseline_selection(manifests)

    data_dir.mkdir(parents=True, exist_ok=True)
    datasets = sorted({r["dataset"] for r in rows},
                      key=lambda d: (DATASET_ORDER.index(d) if d in DATASET_ORDER else 99, d))
    written: list[pathlib.Path] = []
    for ds in datasets:
        infix = f"{ds}_"
        d_rows = [r for r in rows if r["dataset"] == ds]
        d_sel = [s for s in sel if s["dataset"] == ds]
        d_baseline_sel = [s for s in baseline_sel if s["dataset"] == ds]
        written += [
            write_summary(d_rows, data_dir, infix),
            write_recon_by_encoding(d_rows, data_dir, infix),
            write_mse_plot(d_rows, data_dir, infix),
        ]
        if d_sel:
            written.append(write_snn_selection_tex(d_sel, data_dir, infix))
        if any(s["family"] in ("lstm-ae", "gru-ae") for s in d_baseline_sel):
            written.append(write_recurrent_selection_tex(d_baseline_sel, data_dir, infix))
        if any(s["family"] == "transformer-ae" for s in d_baseline_sel):
            written.append(write_transformer_selection_tex(d_baseline_sel, data_dir, infix))
        caveat = check_degenerate_reconstruction(d_rows, ds)
        if caveat:
            print(f"[loso-data] {caveat}", file=sys.stderr)
            caveat_path = data_dir / f"paper_loso_{infix}CAVEATS.txt"
            caveat_path.write_text(caveat + "\n", encoding="utf-8")
            written.append(caveat_path)

    # datasets.tex: the \foreach list the paper iterates.
    dtex = data_dir / "paper_loso_datasets.tex"
    dtex.write_text(
        "% auto-generated by 02_meeting01_build_loso_paper_data.py\n"
        + "".join(f"\\loParseDataset{{{d}}}{{{DATASET_TITLE.get(d, d)}}}\n" for d in datasets),
        encoding="utf-8",
    )
    written.append(dtex)
    print(f"[loso-data] {len(datasets)} dataset(s): {', '.join(datasets)}")
    print("[loso-data] wrote " + ", ".join(p.name for p in written) + f" to {data_dir}")
    return datasets


# ------------------------------------------------------------------- self-test

def _write_synthetic_run(d: pathlib.Path) -> None:
    """Two fsdd folds as the binary + 03_ leave them: SNN-AE and LSTM-AE winners for runs
    1-2, and mean / pca rows. Fold 0 has ONE test window (pca error 1.0), fold 1 has THREE
    (pca 0.2 each): per fold first, PCA's MSE is (1.0 + 0.2) / 2 = 0.6, while pooling the
    four windows would give (1.0 + 3 * 0.2) / 4 = 0.4."""
    for fold, n_windows, pca_err in ((0, 1, 1.0), (1, 3, 0.2)):
        stem = d / f"t_fsdd_fold{fold}"
        (d / f"{stem.name}_split_manifest.json").write_text(json.dumps(
            {"dataset": "fsdd", "cv_fold": fold, "results_format": RESULTS_FORMAT}))
        with (d / f"{stem.name}_comparative_metrics.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(_COMPARATIVE_COLUMNS), lineterminator="\n")
            w.writeheader()
            for run_id in (1, 2):
                for model, enc, err in (("snn-ae", "latency", 0.5), ("lstm-ae", "poisson", 0.7)):
                    w.writerow({"dataset": "fsdd", "model": model, "encoding": enc,
                                "architecture": "dense" if model == "snn-ae" else "lstm",
                                "run": run_id, "seed": 41 + run_id, "split": "test",
                                "cv_fold": fold, "mse": err, "mae": err, "r2": 0.1,
                                "train_ms": 1000.0, "infer_ms": 1.0, "param_count": 100,
                                "macs": 1000})
                for family, selected in (
                        ("snn-ae", {"architecture": "dense", "encoding": "latency",
                                    "v_th": 1.0, "alpha": 0.9, "val_mse": 0.5}),
                        ("lstm-ae", {"hidden_size": 32, "num_layers": 1, "encoding": "poisson",
                                     "val_mse": 0.7})):
                    segment = "latency" if family == "snn-ae" else family
                    (d / f"{stem.name}_{segment}_run{run_id}_model_selection_manifest.json"
                     ).write_text(json.dumps({
                         "dataset": "fsdd", "cv_fold": fold, "run_id": run_id,
                         "seed": 41 + run_id, "results_format": RESULTS_FORMAT,
                         "test_speaker": str(fold), "selection_split": "val (speaker 9)",
                         "selected": selected}))
        with (d / f"{stem.name}_per_window_errors.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=PW_HEADER, lineterminator="\n")
            w.writeheader()
            for i in range(n_windows):
                ident = {"cv_fold": fold, "split": "test", "speaker_id": fold,
                         "recording_id": 10 * fold + i, "window_id": 100 * fold + i,
                         "source_window_index": 0, "v_th": 0.0, "alpha": 0.0}
                for model, err in (("mean", 1.5), ("pca", pca_err)):
                    for enc in ("latency", "poisson"):
                        w.writerow({"model": model, "encoding": enc, "architecture": model,
                                    "run_id": 0, "seed": REFERENCE_SEED, "mse": err,
                                    "mae": err, **ident})


def _self_test_checks(tmp_root: pathlib.Path) -> list:
    failures = []

    def expect(ok: bool, what: str) -> None:
        if not ok:
            failures.append(what)

    def expect_refusal(fn, needle: str, what: str) -> None:
        try:
            fn()
        except ResultsError as e:
            expect(needle in str(e), f"{what}: refused, but message lacks {needle!r}: {e}")
        else:
            failures.append(f"{what}: not refused")

    d, out = tmp_root / "results", tmp_root / "data"
    d.mkdir()
    _write_synthetic_run(d)
    expect(build(d, "t", out) == ["fsdd"], "a clean run builds")
    summary = {r[0]: r for r in csv.reader((out / "paper_loso_fsdd_summary.csv").open(),
                                          delimiter=";")}
    expect(summary["PCA"][1].startswith("0.6000"), f"PCA averaged per fold: {summary['PCA']}")
    expect(summary["SNN-AE"][1].startswith("\\textbf{0.5000"), f"SNN-AE: {summary['SNN-AE']}")
    sel = (out / "paper_loso_fsdd_recurrent_selection.tex").read_text()
    expect("LSTM-AE & 0" in sel and "LSTM-AE & 1" in sel, "recurrent selection table")

    def rewrite(name: str, edit) -> str:
        path = d / name
        before = path.read_text()
        path.write_text(edit(before))
        return before

    comp0 = "t_fsdd_fold0_comparative_metrics.csv"
    before = rewrite(comp0, lambda s: s.replace(",1000.0,1.0,", ",0.0,1.0,", 1))
    expect_refusal(lambda: build(d, "t", out), "cannot take no time", "train_ms 0")
    (d / comp0).write_text(before)

    stale = d / "t_fsdd_fold0_latency_run3_model_selection_manifest.json"
    stale.write_text((d / "t_fsdd_fold0_latency_run2_model_selection_manifest.json").read_text()
                     .replace('"run_id": 2', '"run_id": 3').replace('"seed": 43', '"seed": 44'))
    expect_refusal(lambda: build(d, "t", out), "left over from an earlier run", "stale manifest")
    stale.unlink()

    gone = d / "t_fsdd_fold1_lstm-ae_run2_model_selection_manifest.json"
    kept = gone.read_text()
    gone.unlink()
    expect_refusal(lambda: build(d, "t", out), "test rows without their manifest",
                   "missing manifest")
    gone.write_text(kept)

    manifest0 = "t_fsdd_fold0_split_manifest.json"
    before = rewrite(manifest0, lambda s: s.replace(f', "results_format": {RESULTS_FORMAT}', ""))
    expect_refusal(lambda: build(d, "t", out), "older than 2026-10-06", "older binary's fold")
    (d / manifest0).write_text(before)

    before = rewrite(comp0, lambda s: s.replace("fsdd,snn-ae", "siena,snn-ae", 1))
    expect_refusal(lambda: build(d, "t", out), "renamed or concatenated", "row of another dataset")
    (d / comp0).write_text(before)

    pw1 = d / "t_fsdd_fold1_per_window_errors.csv"
    kept = pw1.read_text()
    pw1.unlink()
    expect_refusal(lambda: build(d, "t", out), "no per-window CSV", "fold without references")
    pw1.write_text(kept)

    nameless = d / "t_fold4_comparative_metrics.csv"
    nameless.write_text("which dataset?\n")
    expect_refusal(lambda: build(d, "t", out), "no dataset segment", "file without a dataset")
    nameless.unlink()

    (d / "t_smoke_fsdd_fold0_comparative_metrics.csv").write_text("another run's\n")
    expect(build(d, "t", out) == ["fsdd"], "another run's prefix-sharing file is not read")
    return failures


def self_test() -> int:
    tmp_root = pathlib.Path(tempfile.mkdtemp(prefix="loso_paper_data_selftest_"))
    try:
        failures = _self_test_checks(tmp_root)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
    if failures:
        for f in failures:
            print(f"[self-test] FAIL: {f}", file=sys.stderr)
        return 1
    print("[self-test] all synthetic known-answer checks passed")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true",
                    help="run synthetic known-answer checks instead of reading real data")
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
    if args.self_test:
        return self_test()
    try:
        build(args.results_dir, args.run_tag, args.data_dir)
    except ResultsError as e:
        print(f"[loso-data] ERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
