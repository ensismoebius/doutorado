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
    paper_loso_summary.csv               model; mse; mae; r2; params; craw; energy_nj; train_ms;
                                         infer_ms (craw = macs + acs, the paper's C_raw; energy_nj
                                         = the memory-aware energy, ENERGY_COLUMNS in
                                         meeting01_results.py; params/craw/energy_nj = means)
                                         (mse/mae/r2/timing = "mean$\\pm$std" strings, best bold)
    paper_loso_energy.tex                per dataset x family: memory-aware E, its memory share,
                                         E at one SRAM size (8 KB / 32 KB / 1 MB), arithmetic
                                         only, and the SNN-AE's E over each other family's
    paper_loso_energy_verdict.tex        \\energyVerdict: that comparison as a sentence, and
                                         whether it survives every repricing
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
import math
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
    DATASET_ORDER,
    DATASET_SHORT,
    DESIGN_INFORMED_FOLDS,
    DEVELOPMENT_FOLDS,
    EQUIVALENCE_MARGIN,
    ENERGY_CHECKS,
    ENERGY_CHECKS_HOLD,
    ENERGY_COLUMNS,
    PW_HEADER,
    REFERENCE_MODELS,
    REFERENCE_SEED,
    RESULTS_FORMAT,
    TRAINED_MODELS,
    ResultsError,
    check_results_format,
    configured_families,
    fold_files,
    folds_text,
    is_headline_fold,
    pca_energy_nj,
    paper_datasets,
    printed_configuration,
    read_per_window,
    resolve_source,
    run_configuration,
    split_manifest,
    synthetic_run_config,
)

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
# Families that are NOT searched: ann-ae copies the SNN-AE winner's widths of the same
# (dataset, fold, run) (run_ann_twin, Meeting01Experiment.cpp), so its manifest records
# where the widths came from and has no genome or val_mse of its own. Its manifest is
# still cross-checked against its test row; it just has no selection table.
_UNSEARCHED_FAMILY_LABEL = {
    "ann-ae": "ANN-AE (control)",
}
# The comparative-CSV columns this script reads (write_rows_csv, Meeting01Output.cpp).
_COMPARATIVE_COLUMNS = ("dataset", "model", "encoding", "architecture", "run", "seed", "split",
                        "cv_fold", "mse", "mae", "r2", "train_ms", "infer_ms", "param_count",
                        "macs", "acs", *ENERGY_COLUMNS, "energy_nj_encoder", "spike_rate",
                        "hidden_silent_share", "hidden_always_share", "lif_v_th_mean",
                        "lif_v_th_min", "lif_beta_mean", "train_mse")
#: What a final fit logs about its own insides (results_format 11, RunMetrics).
_INSIDE_COLUMNS = _COMPARATIVE_COLUMNS[-6:]


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
        elif segment in _BASELINE_FAMILY_LABEL or segment in _UNSEARCHED_FAMILY_LABEL:
            family = segment
        else:
            raise ResultsError(
                f"{path.name}: segment {segment!r} is neither an SNN encoding "
                f"{sorted(_SNN_ENCODINGS)} nor a family token "
                f"{sorted(_BASELINE_FAMILY_LABEL) + sorted(_UNSEARCHED_FAMILY_LABEL)}. "
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
    "ANN-AE (control)": "ann_ae",
}
MODEL_ORDER = ["Mean", "PCA", "SNN-AE", "ANN-AE (control)", "LSTM-AE", "GRU-AE",
               "Transformer-AE"]
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
        "ann-ae": "ANN-AE (control)",
        "pca": "PCA",
        "mean": "Mean",
    }.get(model, model)


def _mean_std(per_seed: list[float]) -> tuple[float, float]:
    a = np.asarray(per_seed, dtype=float)
    if a.size == 0:
        return (float("nan"), float("nan"))
    return (float(a.mean()), float(a.std(ddof=1)) if a.size > 1 else 0.0)


# --------------------------------------------------------------------- loading

def _is_price(cell: str) -> bool:
    """A finite, non-negative number: what every energy column of a priced model holds."""
    try:
        v = float(cell)
    except ValueError:
        return False
    return math.isfinite(v) and v >= 0.0


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
            finite = ["mse", "mae", "r2", "train_ms", "infer_ms"]
            if r["model"] == "snn-ae":
                finite.append("spike_rate")  # nan only for the models without spikes
            bad = {c: r[c] for c in finite if not math.isfinite(float(r[c]))}
            if bad:
                raise ResultsError(
                    f"{where}: {r['model']} seed {r['seed']} has {bad}: "
                    "the model diverged. Leaving the seed out would flatter the model (its mean "
                    "would average the seeds that converged), keeping it makes the mean NaN -- "
                    "look into the run before tabulating it.")
            # The energy is counted from the architecture (Meeting01Energy.hpp), not measured:
            # a non-finite value is a model kind the binary did not price, not a divergence.
            unpriced = {c: r[c] for c in ENERGY_COLUMNS if not _is_price(r[c])}
            if unpriced:
                raise ResultsError(
                    f"{where}: {r['model']} seed {r['seed']} has {unpriced}: the binary did not "
                    "price this model's inference (set_energy, Meeting01Energy.hpp, was not "
                    "called for it). Fix the binary and rerun the fold.")
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
                # C_raw = sum over operation classes (paper, Sec. "Analytic computational
                # cost"): dense MACs plus spike-driven ACs, which is 0 for a non-spiking model.
                "craw": float(r["macs"]) + float(r["acs"]),
                # The memory-aware energy per window (the paper's E) and its checks.
                **{c: float(r[c]) for c in ENERGY_COLUMNS},
                "energy_nj_encoder": float(r["energy_nj_encoder"]),  # nan: not split
                # Memory priced at half: the prices are for 64-bit accesses, one word is 32.
                "energy_nj_half_mem": float(r["energy_ops_nj"]) + 0.5 * float(r["energy_mem_nj"]),
                **{c: float(r[c]) for c in _INSIDE_COLUMNS},  # nan: not applicable
                "spike_rate": float(r["spike_rate"]),  # nan for a model without spikes
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
                # The mean predictor is a stored vector, so nothing to cost; PCA's cost needs
                # the fold's k and window and is filled in by price_pca.
                **{c: float("nan") for c in ENERGY_COLUMNS},
                "energy_nj_encoder": float("nan"),
                "energy_nj_half_mem": float("nan"),
                **{c: float("nan") for c in _INSIDE_COLUMNS},
                "spike_rate": float("nan"),
            })
    return rows


def price_pca(rows: list[dict], window: int, latent: dict) -> None:
    """Fills the energy columns of every PCA row, in place, from pca_energy_nj at each dataset's
    k = latent width, so PCA sits in the energy table on the same model as the networks. It has
    one pass and no spike-dependent count, so its as-executed value is its E."""
    for r in rows:
        if r["label"] != "PCA":
            continue
        k = latent[r["dataset"]]
        base = pca_energy_nj(window, k)
        columns = {"energy_nj": base["total"], "energy_nj_encoder": base["encoder"],
                   "energy_nj_as_executed": base["total"], "arith_energy_nj": base["arith"],
                   "energy_ops_nj": base["ops"], "energy_mem_nj": base["mem"],
                   "energy_nj_half_mem": base["ops"] + 0.5 * base["mem"],
                   "mem_accesses": float("nan")}
        for name, size in (("energy_nj_sram8k", 8.0), ("energy_nj_sram32k", 32.0),
                           ("energy_nj_sram1m", 1024.0)):
            columns[name] = pca_energy_nj(window, k, size * 1024.0)["total"]
        r.update(columns)


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


def check_complete_grid(comparative: list[dict], config: dict, datasets_info: dict) -> None:
    """Every dataset that ran must hold every fold of its configuration, and every fold every
    family's winner for every run, with the run's seed. The grid is taken from the RECORDED
    configuration, not from the rows: a family or a run missing from every fold, or a whole
    fold missing, leaves no row to notice it by. Each gap is silent downstream -- test MSE is
    averaged over the folds per seed and then over the seeds (_per_seed_means), so a missing
    cell shifts a mean, a missing family prints as "--", and the text's fold macros
    (cfgHeadlineFolds*, from the configuration) would describe folds the tables lack. 04_
    refuses part of this (check_coverage), but 01_ runs 02_ first, which would already have
    written the paper's tables."""
    exp = config["experiment"]
    models = set(configured_families(config))
    runs = {r: (exp["seed"] if exp["seed_deterministic"] else exp["seed"] + r - 1)
            for r in range(1, exp["repeats"] + 1)}
    want = {(m, r, s) for m in models for r, s in runs.items()}
    cells: dict = defaultdict(set)
    for r in comparative:
        cells[(r["dataset"], r["fold"])].add((r["model"], r["run_id"], r["seed"]))
    problems = []
    for ds in sorted({d for d, _ in cells}):
        n_folds = datasets_info[ds]["source"]["cv_num_folds"]
        for fold in range(n_folds):
            have = cells.get((ds, fold), set())
            if not have:
                problems.append(f"{ds} fold {fold} has not run")
            elif have != want:
                problems.append(f"{ds} fold {fold} lacks (model, run, seed) {sorted(want - have)}"
                                + (f" and has unexpected {sorted(have - want)}"
                                   if have - want else ""))
        extra = sorted(f for d, f in cells if d == ds and f >= n_folds)
        if extra:
            problems.append(f"{ds} has folds {extra} beyond its {n_folds}")
    if problems:
        raise ResultsError("; ".join(problems) + ". The tables need the whole configured grid: "
                           "finish the run (or rerun the crashed / half-resumed folds) first.")


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
            "time_steps": int(sel["time_steps"]),
            "encoder_widths": [int(w) for w in sel["encoder_widths"]],
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


def _fmt_cell(mean: float, std: float, prec: int, bold: bool, with_std: bool = True) -> str:
    if mean != mean:
        return ""
    s = f"{mean:.{prec}f}$\\pm${std:.{prec}f}" if with_std else f"{mean:.{prec}f}"
    return f"\\textbf{{{s}}}" if bold else s


def write_summary(rows: list[dict], data_dir: pathlib.Path, infix: str = "") -> pathlib.Path:
    agg: dict = {}
    for m in ("mse", "mae", "r2", "train_ms", "infer_ms"):
        agg[m] = {k: _mean_std(v) for k, v in
                  _per_seed_means(rows, lambda r: r["label"], m).items()}
    # params / craw / energy_nj: the mean over every row of the label. They are NOT constant
    # per label: each (fold, run) searches its own winner, and an SNN's craw also depends on
    # how many spikes it fired. Until 2026-11-01 this kept whichever row came last.
    sums: dict = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for k in ("params", "craw", "energy_nj"):
            if r[k] == r[k]:
                sums[r["label"]][k].append(r[k])
    const = {lab: {k: float(np.mean(v)) for k, v in by_k.items()} for lab, by_k in sums.items()}

    labels = _order({r["label"] for r in rows})
    best: dict = {}
    for m, (prec, direction) in FMT.items():
        vals = {lab: agg[m][lab][0] for lab in labels if lab in agg[m] and agg[m][lab][0] == agg[m][lab][0]}
        if vals:
            best[m] = (min if direction == "min" else max)(vals, key=vals.get)

    out = data_dir / f"paper_loso_{infix}summary.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["model", "mse", "mae", "r2", "params", "craw", "energy_nj", "train_ms",
                    "infer_ms"])
        for lab in labels:
            cells = {}
            for m, (prec, _d) in FMT.items():
                mean, std = agg[m].get(lab, (float("nan"), float("nan")))
                cells[m] = _fmt_cell(mean, std, prec, best.get(m) == lab)
            p = const.get(lab, {}).get("params")
            c = const.get(lab, {}).get("craw")
            e = const.get(lab, {}).get("energy_nj")
            w.writerow([lab, cells["mse"], cells["mae"], cells["r2"],
                        "" if p is None else f"{int(round(p))}",
                        "" if c is None else f"{int(round(c))}",
                        "" if e is None else f"{e:.1f}",
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
    """One row per (fold, encoding): modal winning mode + median V_th/alpha/T over seeds."""
    by: dict = defaultdict(list)
    for s in sel:
        by[(s["fold"], s["encoding"])].append(s)
    lines = [
        r"% auto-generated by 02_meeting01_build_loso_paper_data.py",
        r"\begin{tabular}{llllrrrr}",
        r"\toprule",
        r"Fold & Test spk. & Val spk. & Enc. & Mode & $V_{th}$ & $\alpha$ & $T$ \\",
        r"\midrule",
    ]
    for (fold, enc) in sorted(by):
        grp = by[(fold, enc)]
        modes = collections.Counter(g["architecture"] for g in grp)
        mode = modes.most_common(1)[0][0]
        agree = modes[mode]
        vth = pystat.median(g["v_th"] for g in grp)
        alpha = pystat.median(g["alpha"] for g in grp)
        steps = pystat.median(g["time_steps"] for g in grp)
        mode_cell = mode if agree == len(grp) else f"{mode} ({agree}/{len(grp)})"
        lines.append(
            f"{fold} & {grp[0]['test_speaker']} & {grp[0]['val_speaker']} & {enc} & "
            f"{mode_cell} & {vth:.2f} & {alpha:.2f} & {steps:g} \\\\"
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


# ---------------------------------------------------------- the 8-page paper's tables
#
# The paper reports the four datasets side by side instead of one block of tables each
# (author's decision 2026-11-02: 8 pages). Every file below exists for every dataset of the
# run's configuration, also before the dataset has results ("--" / nan), so the paper
# compiles at every stage of the grid run and a missing dataset shows as a gap, not as a
# silently shorter table. All of them use the HEADLINE folds only (DEVELOPMENT_FOLDS left out);
# 04_ writes the sensitivity analysis with them in.

_NONE = "--"


def paper_labels(config: dict) -> list[str]:
    """Row order of the overview table: the two analytic references, then the families the
    configuration runs (configured_families: the spiking model, its matched control, any
    searched sequence baseline). Only those: a family left out of the run gets no "--" row."""
    return ["Mean", "PCA"] + [_model_label(m, "") for m in configured_families(config)]


def _label_stats(rows: list[dict]) -> tuple[dict, dict]:
    """({label: (mse mean, std over seeds)}, {label: mean memory-aware energy in nJ})."""
    mse = {k: _mean_std(v) for k, v in _per_seed_means(rows, lambda r: r["label"], "mse").items()}
    energy: dict = defaultdict(list)
    for r in rows:
        if r["energy_nj"] == r["energy_nj"]:
            energy[r["label"]].append(r["energy_nj"])
    return mse, {k: float(np.mean(v)) for k, v in energy.items()}


def write_overview(rows: list[dict], labels: list[str], datasets: list[str],
                   data_dir: pathlib.Path) -> pathlib.Path:
    """paper_loso_overview.csv: one row per model of `labels` (paper_labels), two columns per
    dataset -- test MSE (mean +/- std over seeds, the lowest of the dataset in bold) and
    memory-aware energy per window (nJ, mean over every winner). Column keys are
    <dataset>mse / <dataset>energy."""
    unlisted = sorted({r["label"] for r in rows} - set(labels))
    if unlisted:
        raise ResultsError(f"results of {unlisted}, which the recorded configuration does not "
                           "run: the folds and their configuration disagree.")
    stats = {ds: _label_stats([r for r in rows if r["dataset"] == ds]) for ds in datasets}
    best = {}
    for ds, (mse, _e) in stats.items():
        finite = {lab: v[0] for lab, v in mse.items() if v[0] == v[0]}
        best[ds] = min(finite, key=finite.get) if finite else None
    out = data_dir / "paper_loso_overview.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["model"] + [f"{ds}{c}" for ds in datasets for c in ("mse", "energy")])
        for lab in labels:
            cells = [lab]
            for ds in datasets:
                mse, energy = stats[ds]
                mean, std = mse.get(lab, (float("nan"), float("nan")))
                # PCA / Mean are fitted once, deterministically: no seeds, so no std to show
                # (a printed +/-0.000 would read as a measured zero spread).
                cells.append(_NONE if mean != mean else
                             _fmt_cell(mean, std, 3, best[ds] == lab,
                                       with_std=lab not in ("Mean", "PCA")))
                e = energy.get(lab)
                cells.append(_NONE if e is None else f"{e:.0f}")
            w.writerow(cells)
    return out


# Figure key and legend text of each trained family. figs/fig_pareto.tex has one mark style
# per key ("pareto <key>"): a family added here without one is a LaTeX error, not a blank.
_PARETO_KEY = {"snn-ae": ("snn", "SNN-AE"), "ann-ae": ("ann", "ANN-AE"),
               "lstm-ae": ("lstm", "LSTM-AE"), "gru-ae": ("gru", "GRU-AE"),
               "transformer-ae": ("transformer", "Transformer-AE")}


def write_pareto(rows: list[dict], families: list[str], ds: str,
                 data_dir: pathlib.Path) -> pathlib.Path:
    """paper_loso_<ds>_pareto.csv: key, mse, mse_std, energy_nj per configured family
    (configured_families), then PCA as the LAST row (paper_loso_pareto_series.tex gives its
    index). PCA is costed since 2026-11-05 (pca_energy_nj); the figure draws its error as a
    horizontal reference and its energy as a mark on it."""
    mse, energy = _label_stats([r for r in rows if r["dataset"] == ds])
    out = data_dir / f"paper_loso_{ds}_pareto.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["key", "mse", "mse_std", "energy_nj"])
        for key, lab in [(_PARETO_KEY[m][0], _model_label(m, "")) for m in families] \
                + [("pca", "PCA")]:
            mean, std = mse.get(lab, (float("nan"), float("nan")))
            e = energy.get(lab, float("nan"))
            w.writerow([key] + ["nan" if v != v else f"{v:.6g}" for v in (mean, std, e)])
    return out


def write_pareto_series(families: list[str], data_dir: pathlib.Path) -> pathlib.Path:
    """paper_loso_pareto_series.tex: what figs/fig_pareto.tex draws -- one series and one
    legend entry per configured family, the PCA row's index in the _pareto.csv files, and the
    legend's column count. Generated from the same list as the CSVs, so a family the run left
    out has no empty series and no legend entry."""
    series = "".join(f"  \\paretoSeries{{{_PARETO_KEY[m][0]}}}%\n" for m in families)
    legend = "".join(f"  \\addlegendentry{{{_PARETO_KEY[m][1]}}}%\n" for m in families)
    lines = ["% auto-generated by 02_meeting01_build_loso_paper_data.py -- do not edit",
             "\\newcommand{\\paretoAllSeries}{%", series + "}",
             "\\newcommand{\\paretoAllLegend}{%", legend
             + "  \\addlegendimage{dashed, gray}\\addlegendentry{PCA ($k=d$)}%\n}",
             f"\\newcommand{{\\paretoPcaRow}}{{{len(families)}}}",
             f"\\newcommand{{\\paretoLegendColumns}}{{{len(families) + 1}}}", ""]
    out = data_dir / "paper_loso_pareto_series.tex"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def write_snn_overview(sel: list[dict], rows: list[dict], datasets: list[str],
                       data_dir: pathlib.Path) -> pathlib.Path:
    """paper_loso_snn_overview.tex: what the SNN-AE search chose, one row per dataset over
    every (fold, run) winner -- modal encoding and input transform (with agreement k/n),
    median and range of T, median V_th / alpha / depth, and the hidden LIF neurons' mean
    spike fraction rho on the test windows; then what training did with them: the trained
    V_th (mean, and the lowest layer's, which nothing bounds), the trained beta, the shares of
    (neuron, window) pairs that never fired and that fired at every step, and the training
    error of the SNN-AE and of the ANN-AE."""
    lines = [
        r"% auto-generated by 02_meeting01_build_loso_paper_data.py",
        r"\begin{tabular}{lllrrrrrrrrrrr}",
        r"\toprule",
        # V_th and the decay are the INITIAL values the search draws (both are trained after);
        # the decay is beta in the paper, so it is not confused with the surrogate's alpha.
        (r"Dataset & Enc. & Transf. & $T$ & $V_{th}^{(0)}$ & $\beta^{(0)}$ & $L$ & $\rho$ & "
         r"$V_{th}$ & min & $\beta$ & silent & always & \multicolumn{1}{c}{train SNN / ANN} \\"),
        r"\midrule",
    ]
    for ds in datasets:
        grp = [s for s in sel if s["dataset"] == ds]
        name = DATASET_SHORT.get(ds, ds)
        if not grp:
            lines.append(f"{name} & " + " & ".join([_NONE] * 13) + " \\\\")
            continue

        def modal(field, grp=grp):
            c = collections.Counter(g[field] for g in grp)
            value, k = c.most_common(1)[0]
            return value if k == len(grp) else f"{value} ({k}/{len(grp)})"

        steps = [g["time_steps"] for g in grp]
        t_cell = (f"{pystat.median(steps):g}" if min(steps) == max(steps)
                  else f"{pystat.median(steps):g} [{min(steps)}--{max(steps)}]")
        rho = [r["spike_rate"] for r in rows
               if r["dataset"] == ds and r["label"] == "SNN-AE" and r["spike_rate"] == r["spike_rate"]]
        snn = [r for r in rows if r["dataset"] == ds and r["label"] == "SNN-AE"]
        ann = [r for r in rows if r["dataset"] == ds and r["label"] == "ANN-AE (control)"]

        def mean_of(rs, column, digits=2):
            values = [r[column] for r in rs if r[column] == r[column]]
            return f"{np.mean(values):.{digits}f}" if values else _NONE

        v_th_low = [r["lif_v_th_min"] for r in snn if r["lif_v_th_min"] == r["lif_v_th_min"]]
        lines.append(
            f"{name} & {modal('encoding')} & {modal('architecture')} & {t_cell} & "
            f"{pystat.median(g['v_th'] for g in grp):.2f} & "
            f"{pystat.median(g['alpha'] for g in grp):.2f} & "
            f"{pystat.median(len(g['encoder_widths']) for g in grp):g} & "
            f"{(f'{np.mean(rho):.2f}' if rho else _NONE)} & "
            f"{mean_of(snn, 'lif_v_th_mean')} & "
            f"{(f'{min(v_th_low):.2f}' if v_th_low else _NONE)} & "
            f"{mean_of(snn, 'lif_beta_mean')} & {mean_of(snn, 'hidden_silent_share')} & "
            f"{mean_of(snn, 'hidden_always_share')} & "
            f"{mean_of(snn, 'train_mse', 3)} / {mean_of(ann, 'train_mse', 3)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    out = data_dir / "paper_loso_snn_overview.tex"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


# ------------------------------------------------------------------------- energy

def _ratio_text(x: float) -> str:
    return f"{x:.0f}" if x >= 10 else f"{x:.1f}" if x >= 1 else f"{x:.2f}"


def _nj(x: float) -> str:
    """An energy for a table: whole nJ from 100 up, one decimal from 10, two below."""
    return f"{x:.0f}" if x >= 100 else f"{x:.1f}" if x >= 10 else f"{x:.2f}"


def _energy_means(rows: list[dict], ds: str, label: str) -> dict | None:
    """{column: mean over every winner of `label` on `ds`} for ENERGY_COLUMNS and the encoder's
    share, or None when the label has no row there (the dataset has not run yet)."""
    rs = [r for r in rows if r["dataset"] == ds and r["label"] == label]
    if not rs:
        return None
    columns = (*ENERGY_COLUMNS, "energy_nj_encoder", "energy_nj_half_mem")
    return {c: float(np.mean([r[c] for r in rs])) for c in columns}


def write_energy_tex(rows: list[dict], families: list[str], datasets: list[str],
                     data_dir: pathlib.Path) -> pathlib.Path:
    """paper_loso_energy.tex: the energy model's breakdown and its sensitivity, per dataset and
    configured family -- E (memory-aware, the paper's number; mean over every winner), the
    share of E spent on memory accesses, E with every access priced at ONE SRAM size (8 KB,
    32 KB, 1 MB: Horowitz 2014's three points), and the arithmetic-only estimate. Then one row
    per other family with the SNN-AE's energy divided by that family's, in every column, so a
    reader can see whether the comparison survives each repricing without computing it."""
    head = [r"% auto-generated by 02_meeting01_build_loso_paper_data.py -- do not edit",
            r"\begin{tabular}{lrrrrrrrrr}", r"\toprule",
            (r" & $E$ & Enc. & Mem. & \multicolumn{3}{c}{$E$, one SRAM size} & Mem. & Arith. & As "
             r"\\"),
            r"\cmidrule(lr){5-7}",
            r"Model & (nJ) & only & share & 8\,KB & 32\,KB & 1\,MB & $\div 2$ & only & exec. \\"]
    cols = ("energy_nj_sram8k", "energy_nj_sram32k", "energy_nj_sram1m", "energy_nj_half_mem",
            "arith_energy_nj", "energy_nj_as_executed")
    lines = list(head)
    for ds in datasets:
        lines += [r"\midrule", rf"\multicolumn{{10}}{{l}}{{\emph{{{DATASET_SHORT.get(ds, ds)}}}}} \\"]
        priced = [(m, _PARETO_KEY[m][1], _model_label(m, "")) for m in families]
        priced.append(("pca", "PCA", "PCA"))
        means = {m: _energy_means(rows, ds, label) for m, _, label in priced}
        for m, name, _ in priced:
            e = means[m]
            if e is None:
                lines.append(f"{name} & " + " & ".join([_NONE] * 9) + r" \\")
                continue
            share = (f"{100.0 * e['energy_mem_nj'] / e['energy_nj']:.0f}\\%"
                     if e["energy_mem_nj"] == e["energy_mem_nj"] else _NONE)
            lines.append(f"{name} & {_nj(e['energy_nj'])} & {_nj(e['energy_nj_encoder'])} & "
                         f"{share} & " + " & ".join(_nj(e[c]) for c in cols) + r" \\")
        snn = means.get("snn-ae")
        for m, name, _ in priced:
            if m == "snn-ae":
                continue
            other = means[m]
            label = rf"SNN-AE $\div$ {name}"
            if snn is None or other is None:
                lines.append(f"{label} & " + " & ".join([_NONE] * 9) + r" \\")
                continue
            lines.append(f"{label} & {_ratio_text(snn['energy_nj'] / other['energy_nj'])} & "
                         f"{_ratio_text(snn['energy_nj_encoder'] / other['energy_nj_encoder'])} & "
                         f"{_NONE} & "
                         + " & ".join(_ratio_text(snn[c] / other[c]) for c in cols) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    out = data_dir / "paper_loso_energy.tex"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def _cost_name(model: str) -> str:
    return "PCA" if model == "pca" else _PARETO_KEY[model][1]


def energy_verdict(rows: list[dict], families: list[str], datasets: list[str]) -> str:
    """The sentence the paper prints about energy, generated so the text cannot claim more
    than the table shows: per dataset, how many times the energy of each other configured
    family the SNN-AE needs (memory-aware E), then whether each of those orderings -- which of
    the two needs less -- holds under every repricing (ENERGY_CHECKS), naming the first that
    turns one around otherwise."""
    pairs = []
    for ds in datasets:
        snn = _energy_means(rows, ds, _model_label("snn-ae", ""))
        for m in families[1:] + ["pca"]:
            other = _energy_means(rows, ds, "PCA" if m == "pca" else _model_label(m, ""))
            if snn is not None and other is not None:
                pairs.append((ds, m, snn, other))
    if not pairs:
        return "[the energy comparison has not run yet]"
    parts = [f"{_ratio_text(snn['energy_nj'] / other['energy_nj'])} times the "
             f"{_cost_name(m)}'s energy on {DATASET_SHORT.get(ds, ds)}"
             for ds, m, snn, other in pairs]
    # The paper prints it after "On energy per window,".
    claim = "the SNN-AE needs " + (parts[0] if len(parts) == 1
                                   else ", ".join(parts[:-1]) + " and " + parts[-1])
    for column, what in ENERGY_CHECKS:
        for ds, m, snn, other in pairs:
            if (snn[column] < other[column]) != (snn["energy_nj"] < other["energy_nj"]):
                return (f"{claim}; that ordering does not hold {what}, under which the SNN-AE "
                        f"needs {_ratio_text(snn[column] / other[column])} times the "
                        f"{_cost_name(m)}'s energy on {DATASET_SHORT.get(ds, ds)}")
    return f"{claim}, an ordering that holds {ENERGY_CHECKS_HOLD}"


def write_energy_verdict(rows: list[dict], families: list[str], datasets: list[str],
                         data_dir: pathlib.Path) -> pathlib.Path:
    out = data_dir / "paper_loso_energy_verdict.tex"
    out.write_text("% auto-generated by 02_meeting01_build_loso_paper_data.py -- do not edit\n"
                   f"\\newcommand{{\\energyVerdict}}{{{energy_verdict(rows, families, datasets)}}}\n",
                   encoding="utf-8")
    return out


# --------------------------------------------------------------- configuration macros

def _num(x) -> str:
    """A profile number as the paper prints it: float noise removed (0.8999999761 -> 0.9),
    a power of ten as 10^{k} (for learning rates)."""
    v = float(f"{float(x):.6g}")
    if v > 0 and v < 0.01:
        k = round(np.log10(v))
        if abs(v - 10.0 ** k) < 1e-12:
            return f"10^{{{k}}}"
    return f"{v:g}"


def _interval(lo, hi) -> str:
    """A searched range for the paper's math intervals, $[\\cfgX]$: "lo, hi"."""
    return f"{_num(lo)}, {_num(hi)}"


_COUNT_WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")


def _count_word(n: int) -> str:
    if not 0 <= n < len(_COUNT_WORDS):
        raise ResultsError(f"no word for the count {n}: extend _COUNT_WORDS.")
    return _COUNT_WORDS[n]


def write_config_tex(config: dict, datasets_info: dict, status: str,
                     data_dir: pathlib.Path) -> pathlib.Path:
    """paper_config.tex: the paper's configuration numbers as macros, from the configuration
    the folds RECORDED (run_configuration) -- or, before the run, from the binary's
    --print-resolved-config -- never typed by hand, so the text cannot describe a run that
    did not happen. Refuses a configuration the text has no wording for (one family searched
    with another budget, LSTM-AE and GRU-AE with other bounds, a biophysical learning rate
    left to the binary's default)."""
    exp, ds_cfg, tr, mdl, ga = (config["experiment"], config["dataset"], config["training"],
                                config["model"], config["evaluation"]["ga"])
    if tr["learning_rate_biophysical"] <= 0:
        raise ResultsError("training.learning_rate_biophysical is 0 (the binary then uses 0.1 x "
                           "lr): set it explicitly so the paper states the value that ran.")
    # Only the families that ran: a GA block of a family not in the run is never read.
    ran = ({"snn"} if config["evaluation"]["snn_architectures"] else set()) | {
        {"lstm-ae": "lstm", "gru-ae": "gru", "transformer-ae": "transformer"}[b]
        for b in config["evaluation"]["baselines"] if b != "ann-ae"}
    budget = {fam: (g["population_size"], g["generations"], g["winner_seeds"])
              for fam, g in ga.items() if fam in ran}
    if len(set(budget.values())) != 1:
        raise ResultsError(f"the families' GA budgets differ {budget}: the paper describes one "
                           "matched evaluation budget.")
    if {"lstm", "gru"} <= ran and \
            {k: ga["lstm"][k] for k in ("min_hidden", "max_hidden", "min_layers", "max_layers")} != \
            {k: ga["gru"][k] for k in ("min_hidden", "max_hidden", "min_layers", "max_layers")}:
        raise ResultsError("LSTM-AE and GRU-AE search different bounds: the paper gives one range.")
    pop, gens, winner_seeds = next(iter(budget.values()))
    folds = {s["cv_num_folds"] for s in (i["source"] for i in datasets_info.values())}
    if len(folds) != 1:
        raise ResultsError(f"the datasets use different fold counts {sorted(folds)}.")
    k_folds = folds.pop()
    seeds = ([exp["seed"]] * exp["repeats"] if exp["seed_deterministic"]
             else list(range(exp["seed"], exp["seed"] + exp["repeats"])))
    t_lo, t_hi = ga["snn"]["time_steps_min"], ga["snn"]["time_steps_max"]
    if t_lo == 0 and t_hi == 0:
        t_lo = t_hi = mdl["time_steps"]
    windows = {(i["source"]["loso_max_train_windows"], i["source"]["loso_max_val_windows"],
                i["source"]["loso_max_test_windows"]) for i in datasets_info.values()}
    if len(windows) != 1:
        raise ResultsError(f"the datasets use different window caps {sorted(windows)}.")
    train_w, val_w, test_w = windows.pop()

    macros = {
        "cfgDataStatus": status,
        "cfgResultsFormat": str(RESULTS_FORMAT),
        "cfgRuns": str(exp["repeats"]),
        "cfgSeeds": folds_text(sorted(set(seeds))),
        "cfgFolds": str(k_folds),
        "cfgEpochs": str(tr["epochs"]),
        "cfgPatience": str(tr["early_stop_patience"]),
        # s, the share below the mean predictor's monitor loss that arms the patience, in %.
        "cfgMinSkill": f"{100.0 * float(tr['early_stop_min_skill']):.4g}",
        "cfgBatch": str(tr["samples_per_batch"]),
        "cfgLr": _num(tr["learning_rate"]),
        "cfgLrBio": _num(tr["learning_rate_biophysical"]),
        "cfgWindow": str(ds_cfg["window_size"]),
        "cfgTrainWindows": str(train_w),
        "cfgValWindows": str(val_w),
        "cfgTestWindows": str(test_w),
        "cfgInit": mdl["initializer"].capitalize() or _NONE,
        "cfgTmin": str(t_lo),
        "cfgTmax": str(t_hi),
        "cfgVth": _interval(ga["snn"]["voltage_threshold_min"], ga["snn"]["voltage_threshold_max"]),
        "cfgAlpha": _interval(ga["snn"]["alpha_min"], ga["snn"]["alpha_max"]),
        "cfgSnnDepth": _interval(ga["snn"]["min_layers"], ga["snn"]["max_layers"]),
        # repair_widths always keeps one hidden layer, and at most max_layers - 1 of them.
        "cfgSnnHidden": _interval(1, ga["snn"]["max_layers"] - 1),
        "cfgSnnWidthMin": str(ga["snn"]["min_width"]),
        "cfgSnnWidthMax": str(ga["snn"]["max_width"]),
        "cfgGaPop": str(pop),
        "cfgGaGen": str(gens),
        "cfgGaEvals": str(pop * (1 + gens)),
        "cfgGaWinnerSeeds": str(winner_seeds),
        # The Holm family is every model the SNN-AE is tested against, the one 04_'s
        # analyse() corrects over: "six references" stayed in the text when the grid lost
        # three of them, a smaller correction than the sentence claimed.
        "cfgHolmRefs": _count_word(len([m for m in configured_families(config) if m != "snn-ae"])
                                   + len(REFERENCE_MODELS)),
    }
    # A family's macros exist only when the family runs, so a sentence about a family the run
    # left out is an "undefined control sequence" when the paper compiles, not a sentence
    # describing a search that never happened.
    if ran & {"lstm", "gru", "transformer"}:
        macros["cfgFrame"] = str(mdl["lstm_frame_size"])
        macros["cfgSeqLen"] = str(ds_cfg["window_size"] * mdl["recurrent_time_steps"]
                                  // mdl["lstm_frame_size"])
    if ran & {"lstm", "gru"}:
        rnn = ga["lstm"] if "lstm" in ran else ga["gru"]
        macros["cfgRnnHidden"] = _interval(rnn["min_hidden"], rnn["max_hidden"])
        macros["cfgRnnLayers"] = _interval(rnn["min_layers"], rnn["max_layers"])
    if "transformer" in ran:
        tf = ga["transformer"]
        macros["cfgTfDmodel"] = _interval(tf["min_d_model"], tf["max_d_model"])
        macros["cfgTfHeads"] = ", ".join(str(h) for h in tf["head_choices"])
        macros["cfgTfLayers"] = _interval(tf["min_layers"], tf["max_layers"])
        macros["cfgTfDff"] = _interval(tf["min_d_ff"], tf["max_d_ff"])
    for ds, info in datasets_info.items():
        tag = "".join(c for c in ds.capitalize() if c.isalpha())
        src = info["source"]
        cap = src["max_windows_per_recording"]
        macros[f"cfgLatent{tag}"] = str(info["latent_dim"])
        # repair_widths (Meeting01GaGenome.cpp) raises a hidden width drawn below d + 1 to
        # d + 1: the share of uniform draws in [min_width, max_width] that end at that floor.
        lo_w, hi_w, floor = ga["snn"]["min_width"], ga["snn"]["max_width"], info["latent_dim"] + 1
        if not lo_w <= floor <= hi_w:
            raise ResultsError(f"{ds}: the hidden-width floor d + 1 = {floor} is outside the GA's "
                               f"[{lo_w}, {hi_w}]; the paper's width sentence assumes it is inside.")
        macros[f"cfgWidthFloorShare{tag}"] = f"{100.0 * (floor - lo_w + 1) / (hi_w - lo_w + 1):.0f}"
        macros[f"cfgCap{tag}"] = str(cap) if cap > 0 else _NONE
        macros[f"cfgRate{tag}"] = str(src["sample_rate"])
        dev = sorted(DEVELOPMENT_FOLDS.get(ds, frozenset()))
        macros[f"cfgDesignFolds{tag}"] = folds_text(
            sorted(DESIGN_INFORMED_FOLDS.get(ds, frozenset())))
        macros[f"cfgHeadlineFolds{tag}"] = folds_text(
            [f for f in range(src["cv_num_folds"]) if is_headline_fold(ds, f)])
        macros[f"cfgDevFolds{tag}"] = folds_text(dev)
    macros["cfgMargin"] = f"{EQUIVALENCE_MARGIN:g}"
    lines = ["% auto-generated by 02_meeting01_build_loso_paper_data.py -- do not edit;",
             f"% {status}: from the configuration "
             + ("the folds recorded" if status == "results" else "the binary printed") + "."]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()]
    out = data_dir / "paper_config.tex"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def write_paper_tables(rows: list[dict], sel: list[dict], config: dict, datasets_info: dict,
                       status: str, data_dir: pathlib.Path) -> list[pathlib.Path]:
    """Every file the 8-page paper reads, on the headline folds of `rows` / `sel`."""
    datasets = paper_datasets(config)
    resolved = {ds: resolve_source(config, ds) for ds in datasets}
    for ds, info in datasets_info.items():
        if resolved.get(ds) != info:
            raise ResultsError(
                f"{ds}: the folds recorded {info}, the configuration resolves to "
                f"{resolved.get(ds)}: resolve_source no longer mirrors Dataset::resolve.")
    datasets_info = resolved
    families = configured_families(config)
    data_dir.mkdir(parents=True, exist_ok=True)
    written = [write_config_tex(config, datasets_info, status, data_dir),
               write_overview(rows, paper_labels(config), datasets, data_dir),
               write_snn_overview(sel, rows, datasets, data_dir),
               write_pareto_series(families, data_dir),
               write_energy_tex(rows, families, datasets, data_dir),
               write_energy_verdict(rows, families, datasets, data_dir)]
    written += [write_pareto(rows, families, ds, data_dir) for ds in datasets]
    return written


def build_placeholders(printed_config: pathlib.Path, data_dir: pathlib.Path) -> list:
    """The paper's files before any fold has run: configuration from the binary's
    --print-resolved-config, every table cell "--", every figure point nan."""
    config, datasets_info = printed_configuration(printed_config)
    written = write_paper_tables([], [], config, datasets_info, "placeholder", data_dir)
    print("[loso-data] placeholders: wrote " + ", ".join(p.name for p in written)
          + f" to {data_dir}")
    return paper_datasets(config)


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
    config, datasets_info = run_configuration(results_dir, run_tag)
    check_complete_grid(comparative, config, datasets_info)
    # Every check above ran on all folds; every table below uses the headline folds only.
    rows = [r for r in comparative + references if is_headline_fold(r["dataset"], r["fold"])]
    manifests = [m for m in manifests if is_headline_fold(m["dataset"], m["fold"])]
    sel = load_snn_selection(manifests)
    baseline_sel = load_baseline_selection(manifests)
    price_pca(rows, int(config["dataset"]["window_size"]),
              {ds: info["latent_dim"] for ds, info in datasets_info.items()})

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
    written += write_paper_tables(rows, sel, config, datasets_info, "results", data_dir)
    print(f"[loso-data] {len(datasets)} dataset(s): {', '.join(datasets)}")
    print("[loso-data] wrote " + ", ".join(p.name for p in written) + f" to {data_dir}")
    return datasets


# ------------------------------------------------------------------- self-test

# The energy checks of a synthetic row as multiples of its energy_nj: equal for every model,
# so every repricing keeps every ratio unless a test passes `energy_factors` for one model.
_SYN_ENERGY = {"energy_nj": 1.0, "energy_ops_nj": 0.4, "energy_mem_nj": 0.6, "mem_accesses": 0.05,
               "energy_nj_sram8k": 0.8, "energy_nj_sram32k": 1.1, "energy_nj_sram1m": 3.0,
               "arith_energy_nj": 0.3, "energy_nj_as_executed": 1.5, "energy_nj_encoder": 0.5}


def _write_synthetic_run(d: pathlib.Path, energy_factors: dict | None = None) -> None:
    """Three fsdd folds as the binary + 03_ leave them: SNN-AE, LSTM-AE and ann-ae winners
    for runs 1-2, and mean / pca rows. Fold 0 is a DEVELOPMENT fold (DEVELOPMENT_FOLDS) whose
    errors are 10x the others', so any of its rows in a table shows. Of the headline folds,
    fold 1 has ONE test window (pca error 1.0) and fold 2 has THREE (pca 0.2 each): per fold
    first, PCA's MSE is (1.0 + 0.2) / 2 = 0.6, while pooling the four windows would give
    (1.0 + 3 * 0.2) / 4 = 0.4."""
    for fold, n_windows, pca_err, scale in ((0, 2, 5.0, 10.0), (1, 1, 1.0, 1.0),
                                            (2, 3, 0.2, 1.0)):
        stem = d / f"t_fsdd_fold{fold}"
        config = synthetic_run_config(fold)
        (d / f"{stem.name}_split_manifest.json").write_text(json.dumps(
            {"dataset": "fsdd", "cv_fold": fold, "results_format": RESULTS_FORMAT,
             "config": config, "source": resolve_source(config, "fsdd")["source"]}))
        with (d / f"{stem.name}_comparative_metrics.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(_COMPARATIVE_COLUMNS), lineterminator="\n")
            w.writeheader()
            for run_id in (1, 2):
                for model, enc, err in (("snn-ae", "latency", 0.5 * scale),
                                        ("lstm-ae", "poisson", 0.7 * scale),
                                        ("ann-ae", "direct", 0.4 * scale)):
                    w.writerow({"dataset": "fsdd", "model": model, "encoding": enc,
                                "architecture": {"snn-ae": "dense", "lstm-ae": "lstm",
                                                 "ann-ae": "dense"}[model],
                                "run": run_id, "seed": 41 + run_id, "split": "test",
                                "cv_fold": fold, "mse": err, "mae": err, "r2": 0.1,
                                "train_ms": 1000.0, "infer_ms": 1.0, "param_count": 100,
                                "macs": 1000, "acs": 0,
                                **{c: f"{f * e:.6g}" for c, f in
                                   {**_SYN_ENERGY, **(energy_factors or {}).get(model, {})}.items()
                                   for e in [{"snn-ae": 3.0, "lstm-ae": 50.0,
                                              "ann-ae": 4.6}[model] * scale]},
                                "spike_rate": 0.25 * scale if model == "snn-ae" else "nan",
                                "hidden_silent_share": 0.4 if model == "snn-ae" else "nan",
                                "hidden_always_share": 0.1 if model == "snn-ae" else "nan",
                                "lif_v_th_mean": 0.9 if model == "snn-ae" else "nan",
                                "lif_v_th_min": 0.7 if model == "snn-ae" else "nan",
                                "lif_beta_mean": 0.8 if model == "snn-ae" else "nan",
                                "train_mse": 0.5 * err})
                for family, selected in (
                        ("snn-ae", {"architecture": "dense", "encoding": "latency",
                                    "v_th": 1.0, "alpha": 0.9, "time_steps": 4 + 4 * run_id
                                    + (100 if fold == 0 else 0),
                                    "encoder_widths": [64, 16], "val_mse": 0.5}),
                        ("lstm-ae", {"hidden_size": 32, "num_layers": 1, "encoding": "poisson",
                                     "val_mse": 0.7}),
                        ("ann-ae", {"copied_from": "snn-ae winner of the same (dataset, fold, run)",
                                    "encoder_widths": [8, 4], "encoding": "direct",
                                    "val_mse": None})):
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
                for model, err in (("mean", 1.5 * scale), ("pca", pca_err)):
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
    expect(summary["SNN-AE"][1].startswith("0.5000"), f"SNN-AE: {summary['SNN-AE']}")
    expect(summary["ANN-AE (control)"][1].startswith("\\textbf{0.4000"),
           f"ann-ae is tabulated: {summary.get('ANN-AE (control)')}")
    plot_head = (out / "paper_loso_fsdd_mse_plot.csv").read_text().splitlines()[0]
    expect(plot_head == "encoding,mean,pca,snn_ae,ann_ae,lstm_ae", f"plot columns: {plot_head}")
    sel = (out / "paper_loso_fsdd_recurrent_selection.tex").read_text()
    expect("LSTM-AE & 1" in sel and "LSTM-AE & 2" in sel and "LSTM-AE & 0" not in sel,
           f"recurrent selection table, headline folds only: {sel}")

    # The 8-page paper's files: headline folds only, every configured dataset present.
    overview = {r[0]: r for r in csv.reader((out / "paper_loso_overview.csv").open(),
                                           delimiter=";")}
    expect(overview["model"] == ["model", "fsddmse", "fsddenergy", "sienamse", "sienaenergy"],
           f"overview columns: {overview['model']}")
    expect(overview["SNN-AE"][1].startswith("0.500") and overview["SNN-AE"][2] == "3",
           f"overview SNN-AE (dev fold 0 left out): {overview['SNN-AE']}")
    expect(overview["ANN-AE (control)"][1].startswith("\\textbf{0.400"),
           f"overview best in bold: {overview['ANN-AE (control)']}")
    expect(overview["PCA"][1].startswith("0.600") and overview["PCA"][2] == "158",
           f"overview PCA per fold, costed: {overview['PCA']}")
    expect(overview["SNN-AE"][3:] == ["--", "--"], f"siena not run yet: {overview['SNN-AE']}")
    # Rows of the configured families only (baselines: lstm-ae, ann-ae): a family the run
    # left out gets no "--" row, which would read as "not run yet".
    expect(list(overview)[1:] == ["Mean", "PCA", "SNN-AE", "ANN-AE (control)", "LSTM-AE"],
           f"overview rows: {list(overview)}")
    snn = (out / "paper_loso_snn_overview.tex").read_text()
    expect("FSDD & latency & dense & 10 [8--12] & 1.00 & 0.90 & 2 & 0.25" in snn,
           f"SNN overview (median T 10 over runs, dev fold's T 108 left out): {snn}")
    expect("Siena & -- &" in snn, f"SNN overview row for a dataset not run: {snn}")
    pareto = (out / "paper_loso_fsdd_pareto.csv").read_text().splitlines()
    expect(pareto[0] == "key,mse,mse_std,energy_nj" and "snn,0.5,0,3" in pareto
           and "pca,0.6,0,157.985" in pareto, f"pareto: {pareto}")
    expect([r.split(",")[0] for r in pareto[1:]] == ["snn", "ann", "lstm", "pca"],
           f"pareto rows: the configured families, PCA last: {pareto}")
    series = (out / "paper_loso_pareto_series.tex").read_text()
    expect(all(f"\\paretoSeries{{{k}}}" in series for k in ("snn", "ann", "lstm"))
           and "{gru}" not in series and "{transformer}" not in series
           and "\\addlegendentry{LSTM-AE}" in series and "GRU" not in series
           and "\\newcommand{\\paretoPcaRow}{3}" in series
           and "\\newcommand{\\paretoLegendColumns}{4}" in series, f"pareto series: {series}")
    # The energy table: memory-aware E, its memory share, the three one-SRAM-size repricings
    # arithmetic alone and E as executed, then the SNN-AE's energy over each other family's in
    # every column.
    energy = (out / "paper_loso_energy.tex").read_text()
    for needle in ("\nSNN-AE & 3.00 & 1.50 & 60\\% & 2.40 & 3.30 & 9.00 & 2.10 & 0.90 & 4.50 \\\\",
                   "\nANN-AE & 4.60 & 2.30 & 60\\% & 3.68 & 5.06 & 13.8 & 3.22 & 1.38 & 6.90 \\\\",
                   "\nSNN-AE $\\div$ ANN-AE & 0.65 & 0.65 & -- & 0.65 & 0.65 & 0.65 & 0.65 & 0.65 & 0.65 \\\\",
                   "\nSNN-AE $\\div$ LSTM-AE & 0.06 & 0.06 & -- &",
                   "\\multicolumn{10}{l}{\\emph{Siena}} \\\\\nSNN-AE & -- & -- &"):
        expect(needle in energy, f"energy table lacks {needle!r}: {energy}")
    verdict = (out / "paper_loso_energy_verdict.tex").read_text()
    expect("{the SNN-AE needs 0.65 times the ANN-AE's energy on FSDD, 0.06 times "
           "the LSTM-AE's energy on FSDD and 0.02 times the PCA's energy on FSDD, an ordering that holds with every memory access priced "
           "at one SRAM size, 8 KB, 32 KB or 1 MB, with every memory price halved, with arithmetic "
           "alone, and with the first layer of a direct-coded input recomputed at every time "
           "step, as the code runs it}" in verdict,
           f"energy verdict: {verdict}")
    macros = (out / "paper_config.tex").read_text()
    for needle in ("\\newcommand{\\cfgTmin}{2}", "\\newcommand{\\cfgTmax}{32}",
                   "\\newcommand{\\cfgLr}{10^{-3}}", "\\newcommand{\\cfgLrBio}{10^{-4}}",
                   "\\newcommand{\\cfgAlpha}{0.8, 0.99}", "\\newcommand{\\cfgSeeds}{42--43}",
                   "\\newcommand{\\cfgSeqLen}{32}", "\\newcommand{\\cfgGaEvals}{48}",
                   "\\newcommand{\\cfgRnnHidden}{8, 256}",
                   "\\newcommand{\\cfgHolmRefs}{four}",  # ANN-AE, LSTM-AE, PCA, mean
                   "\\newcommand{\\cfgHeadlineFoldsFsdd}{1--2}",
                   "\\newcommand{\\cfgHeadlineFoldsSiena}{0--2}",
                   "\\newcommand{\\cfgLatentSiena}{64}", "\\newcommand{\\cfgCapFsdd}{--}",
                   "\\newcommand{\\cfgInit}{Xavier}", "\\newcommand{\\cfgDataStatus}{results}",
                   "\\newcommand{\\cfgMinSkill}{10}"):
        expect(needle in macros, f"config macro {needle} missing")
    expect("cfgTf" not in macros, f"Transformer macros without a Transformer-AE run: {macros}")

    # Placeholders before the run: the same files, empty cells.
    printed = tmp_root / "printed.json"
    printed.write_text(json.dumps({"results_format": RESULTS_FORMAT,
                                   "config": synthetic_run_config(0), "datasets": {}}))
    ph = tmp_root / "placeholders"
    expect(build_placeholders(printed, ph) == ["fsdd", "siena"], "placeholders build")
    ph_overview = list(csv.reader((ph / "paper_loso_overview.csv").open(), delimiter=";"))
    expect(all(c == "--" for r in ph_overview[1:] for c in r[1:]), f"empty cells: {ph_overview}")
    expect("cfgDataStatus}{placeholder}" in (ph / "paper_config.tex").read_text(),
           "placeholder status macro")

    # The production shape (2026-11-02): no searched sequence baseline, only the matched
    # control. No row, series, legend entry or macro of the three removed families.
    printed.write_text(json.dumps({"results_format": RESULTS_FORMAT,
                                   "config": synthetic_run_config(0, ("ann-ae",)),
                                   "datasets": {}}))
    ph1 = tmp_root / "placeholders_ann_only"
    build_placeholders(printed, ph1)
    rows1 = [r[0] for r in csv.reader((ph1 / "paper_loso_overview.csv").open(), delimiter=";")]
    expect(rows1 == ["model", "Mean", "PCA", "SNN-AE", "ANN-AE (control)"], f"rows: {rows1}")
    series1 = (ph1 / "paper_loso_pareto_series.tex").read_text()
    expect("lstm" not in series1 and "LSTM" not in series1
           and "\\newcommand{\\paretoPcaRow}{2}" in series1, f"series: {series1}")
    pareto1 = (ph1 / "paper_loso_fsdd_pareto.csv").read_text().splitlines()
    expect([r.split(",")[0] for r in pareto1[1:]] == ["snn", "ann", "pca"], f"pareto: {pareto1}")
    macros1 = (ph1 / "paper_config.tex").read_text()
    expect(not any(m in macros1 for m in ("cfgRnn", "cfgTf", "cfgSeqLen", "cfgFrame")),
           f"sequence-baseline macros without a sequence baseline: {macros1}")
    expect("\\newcommand{\\cfgHolmRefs}{three}" in macros1, f"Holm family: {macros1}")

    # An ordering that one repricing turns around is named with it: arithmetic alone makes
    # the ANN-AE 0.3 x 0.5 x 4.6 = 0.69 nJ against the SNN-AE's 0.9, 1.3 times.
    _write_synthetic_run(d, {"ann-ae": {"arith_energy_nj": 0.15}})
    build(d, "t", out)
    verdict = (out / "paper_loso_energy_verdict.tex").read_text()
    expect("; that ordering does not hold with arithmetic alone, under which the SNN-AE needs "
           "1.3 times the ANN-AE's energy on FSDD}" in verdict, f"turned-around verdict: {verdict}")
    _write_synthetic_run(d, {"ann-ae": {"energy_nj_sram1m": float("nan")}})
    expect_refusal(lambda: build(d, "t", out), "did not price", "an unpriced energy check")
    _write_synthetic_run(d)
    ph_energy = (ph / "paper_loso_energy.tex").read_text()
    expect("SNN-AE & -- & -- & -- & -- & -- & -- & -- & -- & -- \\\\" in ph_energy
           and "{[the energy comparison has not run yet]}"
           in (ph / "paper_loso_energy_verdict.tex").read_text(), f"energy placeholders: {ph_energy}")

    # A fold run with another configuration (a redeploy mid-run) is refused.
    m2 = d / "t_fsdd_fold2_split_manifest.json"
    kept = m2.read_text()
    m2.write_text(kept.replace('"epochs": 30', '"epochs": 60'))
    expect_refusal(lambda: build(d, "t", out), "different configurations", "mixed config")
    m2.write_text(kept)

    def rewrite(name: str, edit) -> str:
        path = d / name
        before = path.read_text()
        path.write_text(edit(before))
        return before

    comp0 = "t_fsdd_fold0_comparative_metrics.csv"
    before = rewrite(comp0, lambda s: s.replace(",1000.0,1.0,", ",0.0,1.0,", 1))
    expect_refusal(lambda: build(d, "t", out), "cannot take no time", "train_ms 0")
    (d / comp0).write_text(before)

    before = rewrite(comp0, lambda s: s.replace(",test,0,5.0,5.0,", ",test,0,nan,nan,", 1))
    expect_refusal(lambda: build(d, "t", out), "diverged", "NaN test error")
    (d / comp0).write_text(before)

    comp1 = "t_fsdd_fold1_comparative_metrics.csv"
    gone = d / "t_fsdd_fold1_lstm-ae_run2_model_selection_manifest.json"
    kept = gone.read_text()
    gone.unlink()
    before = rewrite(comp1, lambda s: "".join(
        l for l in s.splitlines(keepends=True) if not l.startswith("fsdd,lstm-ae,poisson,lstm,2,")))
    expect_refusal(lambda: build(d, "t", out), "lacks (model, run, seed)", "fold missing one seed")
    (d / comp1).write_text(before)
    gone.write_text(kept)

    # A family missing from EVERY fold leaves no row to notice it by: the grid comes from the
    # recorded configuration (baselines: lstm-ae, ann-ae), not from the rows.
    saved = {}
    for f in range(3):
        saved[f"t_fsdd_fold{f}_comparative_metrics.csv"] = rewrite(
            f"t_fsdd_fold{f}_comparative_metrics.csv", lambda s: "".join(
                l for l in s.splitlines(keepends=True) if not l.startswith("fsdd,lstm-ae,")))
        for run in (1, 2):
            m = d / f"t_fsdd_fold{f}_lstm-ae_run{run}_model_selection_manifest.json"
            saved[m.name] = m.read_text()
            m.unlink()
    expect_refusal(lambda: build(d, "t", out), "lacks (model, run, seed)",
                   "a family missing from every fold")
    for name, text in saved.items():
        (d / name).write_text(text)

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


def _pca_checks() -> list:
    """Known answers for PCA's price, worked by hand for window 256, k 16: 4096 MACs each way at
    4.6 pJ, 256 mean adds at 0.9 pJ, 272 address adds at 0.1 pJ; 4096 reads of a 16 KB memory at
    10 + 10 * (16 - 8) / 24 = 13.33 pJ, 16 latent and 512 input/mean accesses at 10 pJ each."""
    enc = (4096 * 4.6 + 256 * 0.9 + 272 * 0.1) + (4096 * (10 + 10 * 8 / 24) + 528 * 10.0)
    got = pca_energy_nj(256, 16)
    failures = []
    if abs(got["encoder"] * 1000.0 - enc) > 1e-6 or abs(got["total"] - 2 * got["encoder"]) > 1e-9:
        failures.append(f"PCA price: {got}, by hand encoder {enc} pJ")
    if pca_energy_nj(256, 16, 8.0 * 1024)["total"] >= got["total"]:
        failures.append("PCA at one 8 KB SRAM should cost less than at its own 16 KB weights")
    return failures


def self_test() -> int:
    tmp_root = pathlib.Path(tempfile.mkdtemp(prefix="loso_paper_data_selftest_"))
    # The fixtures predate DESIGN_INFORMED_FOLDS and use fold 1 as a headline fold; what that
    # table does is checked on its own in meeting01_results' tests of is_headline_fold.
    design = dict(DESIGN_INFORMED_FOLDS)
    DESIGN_INFORMED_FOLDS.clear()
    try:
        failures = _self_test_checks(tmp_root) + _pca_checks()
    finally:
        DESIGN_INFORMED_FOLDS.update(design)
        shutil.rmtree(tmp_root, ignore_errors=True)
    if is_headline_fold("fsdd", 1) or is_headline_fold("eegmmidb", 0) \
            or not is_headline_fold("fsdd", 2) or not is_headline_fold("siena", 1):
        failures.append("is_headline_fold: folds 0 and 1 of FSDD/EEGMMIDB out, the rest in")
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
    ap.add_argument("--placeholders", type=pathlib.Path, metavar="PRINTED_CONFIG",
                    help="write the paper's files with empty cells instead of reading results: "
                         "PRINTED_CONFIG is `meeting01 --comparative-config <profile> "
                         "--print-resolved-config` output")
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test()
    try:
        if args.placeholders:
            build_placeholders(args.placeholders, args.data_dir)
            return 0
        build(args.results_dir, args.run_tag, args.data_dir)
    except ResultsError as e:
        print(f"[loso-data] ERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
