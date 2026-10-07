#!/usr/bin/env python3
"""03_meeting01_pca_mean_baselines.py — linear reference baselines for the nested-LOSO
comparison.

Why references at all
---------------------
A trained family's test MSE alone cannot say whether it learned anything a linear map
would not. Two references answer that, for every held-out test window x:

    mean : x_hat = mu                                   (predicts no structure at all)
    pca  : x_hat = mu + (x - mu) V_k V_k^T              (the best LINEAR code of width k)

with k = the dataset's autoencoder bottleneck (profile: dataset.sources[].latent_dim,
resolved exactly like the binary does). "Family beats PCA" then means "beats every linear
compression through the same k-wide hole": k = 16 for fsdd/audiomnist, 64 for EEG.

What makes the comparison fair (each row was a real defect until 2026-10-06)
-------------------------------------------------------------------------------
    same ...   the references                         the trained families
    k          the dataset's own latent_dim           built with that latent_dim
               (was: 32 for every dataset)
    target     the analog window x                    make_reconstruction_target = x
               (was: the ENCODED window -- 0/1          repeated over T steps
                spikes under poisson/latency)
    data       fitted on train ∪ val                  final fit on (train \\ monitor) ∪ val,
               (was: train only)                       early-stopped on monitor ⊂ train
    metric     masked MSE/MAE over real samples       masked MSE/MAE (valid_length)
               (was: unmasked)

The T-fold repeat does not change the number: a reference predicts the same x_hat at
every step, so its MSE against x repeated T times equals its MSE against x. One dump per
fold therefore serves every encoding, and the reference value is IDENTICAL for every
encoding label it is written under (the label only lets 02_/04_ pair rows).

Padding: a window cut from the end of a recording is zero-padded after valid_length
samples. Those samples are treated as missing, not as data: excluded from the error, and
replaced by the position mean before fitting and before projecting, so they carry no
variance and pull no principal axis.

Inputs (written by the meeting01 binary at fold start -- write_reference_inputs):
    <tag>_fold<f>_target_{train,val,test}_windows.npy       float32 (N, window_size)
    <tag>_fold<f>_target_{train,val,test}_windows_meta.csv  speaker_id,recording_id,
                                                            window_id,source_window_index,
                                                            valid_length
    <tag>_fold<f>_split_manifest.json                       the run's split record
    <tag>_fold<f>_per_window_errors.csv                     the families' rows (output too)

Output: model in {mean, pca} rows in <tag>_fold<f>_per_window_errors.csv (split=test,
seed=0, run_id=0), one per test window per encoding of the profile, REPLACING any
mean/pca rows already there -- re-running gives the same file, byte for byte.

Refuses, naming cause and remedy, instead of guessing (no fallbacks):
  - a dataset with no single k (latent_dim unset: each family derived its own);
  - a fold whose manifest records a different latent_dim / window_size / split than the
    profile and dumps say (the profile is not the run's, or the dumps are not the run's);
  - a fold with family rows but no target dumps (a run made by an older binary -- rebuild
    them, no retraining: meeting01 --comparative-config <profile> --dataset <d>
    --cv-fold <f> --dump-reference-inputs-only);
  - target test windows that are not exactly the windows the families were scored on.
Problems are collected over every fold first; nothing is written unless all folds pass.
A fold with dumps but no per-window CSV is still running (the binary writes that CSV once,
at fold end): it is skipped and listed.

Usage:
    python scripts/pipeline/meeting01/03_meeting01_pca_mean_baselines.py \\
        --results-dir results/meeting01 --run-tag meeting01_loso \\
        --profile src/experiments/meeting01/profiles/meeting01-loso.json
    python scripts/pipeline/meeting01/03_meeting01_pca_mean_baselines.py --self-test
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import re
import shutil
import stat
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass

import numpy as np

PW_HEADER = [
    "model", "encoding", "architecture", "v_th", "alpha", "run_id", "seed",
    "cv_fold", "split", "speaker_id", "recording_id", "window_id",
    "source_window_index", "mse", "mae",
]
META_HEADER = ["speaker_id", "recording_id", "window_id", "source_window_index", "valid_length"]
KEY_FIELDS = ("speaker_id", "recording_id", "window_id", "source_window_index")
REFERENCE_MODELS = ("mean", "pca")
KNOWN_ENCODINGS = ("direct", "poisson", "latency")
PARTS = ("train", "val", "test")
_NESTED_SECTIONS = ("experiment", "dataset", "training", "model", "evaluation")


class BaselineInputError(RuntimeError):
    """The references cannot be computed faithfully; the message names cause and remedy."""


# ------------------------------------------------------------------- profile

@dataclass(frozen=True)
class Profile:
    path: pathlib.Path
    run_tag: str
    datasets: tuple
    encodings: tuple
    dataset: dict
    model: dict
    sources: tuple

    def resolve(self, name: str) -> tuple:
        """(latent_dim, window_size) for dataset `name`, by the binary's own chain:
        Dataset::resolve takes the source entry's value when > 0, else the dataset-level
        one; resolve_dataset_config then lets a resolved latent_dim > 0 replace
        model.latent_dim, else model.latent_dim stands."""
        latent = int(self.dataset.get("latent_dim", 0))
        window = int(self.dataset.get("window_size", 0))
        for src in self.sources:
            if src.get("name") != name:
                continue
            if int(src.get("window_size", 0)) > 0:
                window = int(src["window_size"])
            if int(src.get("latent_dim", 0)) > 0:
                latent = int(src["latent_dim"])
            break
        if latent <= 0:
            latent = int(self.model.get("latent_dim", 0))
        if latent <= 0:
            raise BaselineInputError(
                f"{self.path}: no bottleneck width for dataset '{name}' -- latent_dim is unset "
                "at source, dataset and model level, so each family derived its own from "
                "encoder_layer_spec and PCA has no single k to match. Set "
                f"dataset.sources[name={name}].latent_dim (or model.latent_dim) in the profile "
                "the run used.")
        if window <= 0:
            raise BaselineInputError(
                f"{self.path}: no window_size for dataset '{name}'. Set dataset.window_size.")
        return latent, window


def load_profile(path: pathlib.Path) -> Profile:
    raw = json.loads(path.read_text(encoding="utf-8"))
    # Same dispatch as Meeting01Cli.cpp::load_config: nested iff all five sections exist.
    if all(k in raw for k in _NESTED_SECTIONS):
        exp, dat, mdl, evl = (raw[k] for k in ("experiment", "dataset", "model", "evaluation"))
        sources = dat.get("sources", [])
    else:
        # Flat schema (Meeting01Config::from_flat_json): one namespace, sources under
        # "dataset_sources", and "latent_dim" feeds BOTH dataset.latent_dim and
        # model.latent_dim.
        exp = dat = mdl = evl = raw
        sources = raw.get("dataset_sources", [])
    encodings = tuple(evl.get("encodings", ()))
    unknown = [e for e in encodings if e not in KNOWN_ENCODINGS]
    if not encodings or unknown:
        raise BaselineInputError(
            f"{path}: evaluation.encodings must be a non-empty subset of {KNOWN_ENCODINGS}, "
            f"got {list(encodings)}.")
    return Profile(path=path, run_tag=str(exp.get("run_tag", "")),
                   datasets=tuple(evl.get("datasets", ())), encodings=encodings,
                   dataset=dat, model=mdl, sources=tuple(sources))


# ------------------------------------------------------------------- the references

def reference_errors(fit_x: np.ndarray, fit_valid: np.ndarray, test_x: np.ndarray,
                     test_valid: np.ndarray, k: int) -> dict:
    """{"mean": (mse, mae), "pca": (mse, mae)}, each an array over the test windows.

    Rows are windows in sample order; *_valid[i] is window i's count of real samples
    (the rest is zero padding). Fitted on fit_x only."""
    n_fit, width = fit_x.shape
    if test_x.shape[1] != width:
        raise BaselineInputError(f"fit windows have {width} samples, test windows "
                                 f"{test_x.shape[1]}; one dataset has one window_size.")
    if not 0 < k <= min(n_fit, width):
        raise BaselineInputError(
            f"k={k} principal components cannot be fitted from {n_fit} windows of {width} "
            "samples (needs 0 < k <= min of the two); lower latent_dim or raise the "
            "loso_max_train/val_windows caps.")
    if np.any(test_valid <= 0):
        raise BaselineInputError("a test window has valid_length 0 -- no real sample to "
                                 "score; the loader never emits one, so the dump is corrupt.")
    positions = np.arange(width)
    fit_mask = positions[None, :] < fit_valid[:, None]
    test_mask = positions[None, :] < test_valid[:, None]

    counts = fit_mask.sum(axis=0)
    if np.any(counts == 0):
        raise BaselineInputError(
            f"sample positions {np.flatnonzero(counts == 0).tolist()} are padding in every "
            "fit window, so no mean can be estimated there.")
    mu = np.where(fit_mask, fit_x, 0.0).sum(axis=0) / counts

    # Padding as missing data: centred, an imputed sample is exactly 0, so it adds no
    # variance to the fit and no weight to a test window's projection.
    fit_c = np.where(fit_mask, fit_x - mu, 0.0)
    test_c = np.where(test_mask, test_x - mu, 0.0)
    _, _, vt = np.linalg.svd(fit_c, full_matrices=False)  # rows of vt: principal axes
    vk = vt[:k]
    predictions = {"mean": np.broadcast_to(mu, test_x.shape),
                   "pca": mu + (test_c @ vk.T) @ vk}

    n_valid = test_mask.sum(axis=1)
    out = {}
    for model, x_hat in predictions.items():
        diff = np.where(test_mask, test_x - x_hat, 0.0)
        out[model] = ((diff ** 2).sum(axis=1) / n_valid, np.abs(diff).sum(axis=1) / n_valid)
    return out


# ------------------------------------------------------------------- one fold

@dataclass(frozen=True)
class FoldResult:
    dataset: str
    fold: int
    pw_path: pathlib.Path
    family_rows: list
    reference_rows: list
    summary: str


def _key(row: dict) -> tuple:
    return tuple(int(row[f]) for f in KEY_FIELDS)


def _read_csv(path: pathlib.Path, header: list) -> list:
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != header:
            raise BaselineInputError(
                f"{path.name}: header {reader.fieldnames} != expected {header}; written by a "
                "different meeting01 version than this script understands.")
        return list(reader)


def _load_part(results_dir: pathlib.Path, stem: str, part: str, width: int):
    base = results_dir / f"{stem}_target_{part}_windows"
    x = np.load(f"{base}.npy")
    if x.ndim != 2 or x.shape[1] != width:
        raise BaselineInputError(f"{base.name}.npy has shape {x.shape}, expected (N, {width}).")
    meta = _read_csv(pathlib.Path(f"{base}_meta.csv"), META_HEADER)
    if len(meta) != x.shape[0]:
        raise BaselineInputError(f"{base.name}: {x.shape[0]} windows but {len(meta)} meta rows.")
    valid = np.array([int(m["valid_length"]) for m in meta], dtype=np.int64)
    if np.any((valid < 0) | (valid > width)):
        raise BaselineInputError(f"{base.name}: valid_length outside [0, {width}].")
    return x.astype(np.float64), meta, valid


def _check_manifest(manifest: dict, dataset: str, fold: int, k: int, width: int,
                    parts: dict) -> str:
    if manifest.get("dataset") != dataset or manifest.get("cv_fold") != fold:
        raise BaselineInputError(
            f"manifest says dataset={manifest.get('dataset')} fold={manifest.get('cv_fold')}.")
    if "latent_dim" in manifest:
        if int(manifest["latent_dim"]) != k:
            raise BaselineInputError(
                f"the run built every family with latent_dim {manifest['latent_dim']}, but the "
                f"profile resolves k={k}: pass the profile the run actually used (or undo the "
                "edit made to it since).")
        k_note = "k matches the run's manifest"
    else:
        k_note = "k from the profile (manifest predates the latent_dim record)"
    if "window_size" in manifest and int(manifest["window_size"]) != width:
        raise BaselineInputError(
            f"the run's windows had {manifest['window_size']} samples, the profile says {width}.")
    try:
        counts, recordings = manifest["window_counts"], manifest["recordings"]
    except KeyError as e:
        raise BaselineInputError(f"the split manifest lacks {e}, so the target windows cannot "
                                 "be checked against the run's split.") from None
    for part, (_, meta, _) in parts.items():
        expected_n = int(counts[part])
        expected_rec = {int(r["recording_id"]): int(r["window_count"]) for r in recordings[part]}
        got_rec = Counter(int(m["recording_id"]) for m in meta)
        if len(meta) != expected_n or got_rec != expected_rec:
            raise BaselineInputError(
                f"the {part} target windows ({len(meta)} windows, {len(got_rec)} recordings) "
                f"are not the run's {part} split ({expected_n} windows, {len(expected_rec)} "
                "recordings): they were rebuilt from different data or a binary whose split "
                "logic differs. Rebuild them from the run's dataset with the run's profile.")
    return k_note


def _process_fold(results_dir: pathlib.Path, run_tag: str, dataset: str, fold: int,
                  profile: Profile) -> FoldResult:
    stem = f"{run_tag}_{dataset}_fold{fold}"
    if dataset not in profile.datasets:
        raise BaselineInputError(
            f"dataset '{dataset}' is not in {profile.path.name}'s evaluation.datasets "
            f"{list(profile.datasets)}: this is not the profile of the run that wrote it.")
    k, width = profile.resolve(dataset)

    pw_path = results_dir / f"{stem}_per_window_errors.csv"
    rows = _read_csv(pw_path, PW_HEADER)
    family_rows = [r for r in rows if r["model"] not in REFERENCE_MODELS]
    family_test = [r for r in family_rows if r["split"] == "test"]
    if not family_test:
        raise BaselineInputError(f"{pw_path.name} has no trained-family test rows to pair with.")

    rebuild = ("meeting01 --comparative-config <the run's profile> "
               f"--dataset {dataset} --cv-fold {fold} --dump-reference-inputs-only")
    dump_files = [f"{stem}_target_{part}_windows{suffix}"
                  for part in PARTS for suffix in (".npy", "_meta.csv")]
    missing = [name for name in dump_files if not (results_dir / name).exists()]
    if len(missing) == len(dump_files):
        legacy = sorted(p.name for p in results_dir.glob(f"{stem}_*_train_windows.npy")
                        if "_target_" not in p.name)
        raise BaselineInputError(
            "no target dumps (*_target_*_windows.npy) -- the run was made by a meeting01 "
            "binary that predates them"
            + (f"; its per-encoding dumps ({', '.join(legacy)}) hold the ENCODED window "
               "(0/1 spikes under poisson/latency) without valid_length, which is not what "
               "the families are scored against" if legacy else "")
            + f". Rebuild them without retraining: {rebuild}")
    if missing:
        raise BaselineInputError(
            f"incomplete target dump, missing {', '.join(missing)} (the write was "
            f"interrupted). Rebuild it without retraining: {rebuild}")

    parts = {part: _load_part(results_dir, stem, part, width) for part in PARTS}
    manifest_path = results_dir / f"{stem}_split_manifest.json"
    if not manifest_path.exists():
        raise BaselineInputError(
            f"{manifest_path.name} is missing: without the run's split record the target "
            "windows cannot be checked against the windows the families saw.")
    k_note = _check_manifest(json.loads(manifest_path.read_text(encoding="utf-8")),
                             dataset, fold, k, width, parts)

    test_x, test_meta, test_valid = parts["test"]
    target_keys = [_key(m) for m in test_meta]
    if len(set(target_keys)) != len(target_keys):
        raise BaselineInputError("duplicate window identities among the target test windows.")
    family_keys = {_key(r) for r in family_test}
    if family_keys != set(target_keys):
        raise BaselineInputError(
            f"the families were scored on {len(family_keys)} test windows, the target dump "
            f"holds {len(target_keys)}; {len(family_keys - set(target_keys))} family windows "
            "are missing from the dump. The dump is not this run's test split -- rebuild it "
            "with the run's profile, binary and dataset.")
    folds_seen = {int(r["cv_fold"]) for r in family_rows}
    if folds_seen != {fold}:
        raise BaselineInputError(f"{pw_path.name} carries rows of folds {sorted(folds_seen)}.")
    family_encodings = {r["encoding"] for r in family_rows} - {""}
    if not family_encodings <= set(profile.encodings):
        raise BaselineInputError(
            f"families used encodings {sorted(family_encodings)}, not all in the profile's "
            f"{list(profile.encodings)}: this is not the profile the run used.")

    fit_x = np.vstack([parts["train"][0], parts["val"][0]])
    fit_valid = np.concatenate([parts["train"][2], parts["val"][2]])
    errors = reference_errors(fit_x, fit_valid, test_x, test_valid, k)

    reference_rows = []
    for model in REFERENCE_MODELS:
        mse, mae = errors[model]
        for encoding in profile.encodings:
            for m, mse_v, mae_v in zip(test_meta, mse, mae):
                reference_rows.append({
                    "model": model, "encoding": encoding, "architecture": model,
                    "v_th": "0.00000000", "alpha": "0.00000000", "run_id": "0", "seed": "0",
                    "cv_fold": str(fold), "split": "test",
                    "speaker_id": m["speaker_id"], "recording_id": m["recording_id"],
                    "window_id": m["window_id"],
                    "source_window_index": m["source_window_index"],
                    "mse": f"{mse_v:.8f}", "mae": f"{mae_v:.8f}",
                })
    n_padded = int(np.sum(test_valid < width))
    summary = (f"{dataset} fold {fold}: k={k} ({k_note}); fit {fit_x.shape[0]} windows "
               f"(train {parts['train'][0].shape[0]} + val {parts['val'][0].shape[0]}), "
               f"test {test_x.shape[0]} ({n_padded} padded); "
               f"mean MSE {errors['mean'][0].mean():.5f}  pca MSE {errors['pca'][0].mean():.5f}")
    return FoldResult(dataset, fold, pw_path, family_rows, reference_rows, summary)


# ------------------------------------------------------------------- io / driver

def _write_atomically(path: pathlib.Path, rows: list) -> None:
    mode = stat.S_IMODE(path.stat().st_mode)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=PW_HEADER, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def run(results_dir: pathlib.Path, run_tag: str, profile: Profile) -> list:
    """Computes every fold's references, then writes them -- only if every fold passed."""
    if profile.run_tag != run_tag:
        raise BaselineInputError(
            f"--run-tag {run_tag} but {profile.path.name} has experiment.run_tag "
            f"{profile.run_tag!r}: pass the profile of the run whose results these are.")
    fold_re = re.compile(rf"^{re.escape(run_tag)}_(?P<ds>[^_]+)_fold(?P<fold>\d+)_"
                         r"(?:per_window_errors\.csv|target_test_windows\.npy)$")
    with_csv, with_dump = set(), set()
    for p in results_dir.iterdir():
        m = fold_re.match(p.name)
        if m:
            target = with_csv if p.name.endswith(".csv") else with_dump
            target.add((m["ds"], int(m["fold"])))
    if not with_csv and not with_dump:
        raise BaselineInputError(f"no {run_tag}_<dataset>_fold<f> per-window CSV or target "
                                 f"dump under {results_dir}.")

    results, problems = [], []
    for dataset, fold in sorted(with_csv):
        try:
            results.append(_process_fold(results_dir, run_tag, dataset, fold, profile))
        except BaselineInputError as e:
            problems.append(f"{dataset} fold {fold}: {e}")
    if problems:
        raise BaselineInputError("nothing written; fix these first:\n  " + "\n  ".join(problems))

    for r in results:
        _write_atomically(r.pw_path, r.family_rows + r.reference_rows)
        print(f"[pca-mean] {r.summary}")
    for dataset, fold in sorted(with_dump - with_csv):
        print(f"[pca-mean] {dataset} fold {fold}: skipped -- no per-window CSV yet "
              "(fold still running or crashed; the binary writes it once, at fold end)",
              file=sys.stderr)
    print(f"[pca-mean] wrote mean/pca references for {len(results)} fold(s) "
          f"(encodings {', '.join(profile.encodings)})")
    return results


# ------------------------------------------------------------------- self-test

def _write_fold(d: pathlib.Path, stem: str, parts: dict, manifest: dict,
                family_rows: list) -> None:
    for part, (x, meta) in parts.items():
        np.save(d / f"{stem}_target_{part}_windows.npy", x.astype(np.float32))
        with (d / f"{stem}_target_{part}_windows_meta.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=META_HEADER, lineterminator="\n")
            w.writeheader()
            w.writerows(meta)
    (d / f"{stem}_split_manifest.json").write_text(json.dumps(manifest))
    with (d / f"{stem}_per_window_errors.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PW_HEADER, lineterminator="\n")
        w.writeheader()
        w.writerows(family_rows)


def _synthetic_fold(rng, width=8, n=(12, 6, 5), latent=2):
    """Windows on a 2-D affine subspace; the last train and test windows are padded."""
    basis = rng.standard_normal((2, width))
    centre = rng.standard_normal(width)
    parts, wid = {}, 0
    for part, count in zip(PARTS, n):
        x = centre + rng.standard_normal((count, 2)) @ basis
        valid = np.full(count, width)
        if part != "val":
            valid[-1] = width - 3
            x[-1, width - 3:] = 0.0
        meta = []
        for i in range(count):
            p_idx = PARTS.index(part)
            meta.append({"speaker_id": p_idx, "recording_id": 100 * p_idx + i,
                         "window_id": wid, "source_window_index": 0,
                         "valid_length": int(valid[i])})
            wid += 1
        parts[part] = (x, meta)
    manifest = {"dataset": "fsdd", "cv_fold": 0, "latent_dim": latent, "window_size": width,
                "window_counts": {p: len(parts[p][1]) for p in PARTS},
                "recordings": {p: [{"recording_id": m["recording_id"], "window_count": 1}
                                   for m in parts[p][1]] for p in PARTS}}
    family = [{"model": "lstm-ae", "encoding": "poisson", "architecture": "lstm",
               "v_th": "0.00000000", "alpha": "0.00000000", "run_id": "1", "seed": "42",
               "cv_fold": "0", "split": "test", "speaker_id": str(m["speaker_id"]),
               "recording_id": str(m["recording_id"]), "window_id": str(m["window_id"]),
               "source_window_index": "0", "mse": "0.12345678", "mae": "0.23456789"}
              for m in parts["test"][1]]
    return parts, manifest, family


def _self_test_checks(tmp_root: pathlib.Path) -> list:
    """Every failed check as a message; empty when all pass."""
    failures = []

    def expect(ok: bool, what: str) -> None:
        if not ok:
            failures.append(what)

    def expect_refusal(fn, needle: str, what: str) -> None:
        try:
            fn()
        except BaselineInputError as e:
            expect(needle in str(e), f"{what}: refused, but message lacks {needle!r}: {e}")
        else:
            failures.append(f"{what}: not refused")

    rng = np.random.default_rng(20261006)

    # 1. k resolution follows the binary: source > dataset > model; flat feeds both.
    nested = {"experiment": {"run_tag": "t"}, "training": {},
              "dataset": {"window_size": 8, "sources": [{"name": "a", "latent_dim": 3},
                                                        {"name": "b", "window_size": 16}]},
              "model": {"latent_dim": 5},
              "evaluation": {"datasets": ["a", "b", "c"], "encodings": ["direct"]}}
    p = tmp_root / "nested.json"
    p.write_text(json.dumps(nested))
    prof = load_profile(p)
    expect(prof.resolve("a") == (3, 8), f"source latent: {prof.resolve('a')}")
    expect(prof.resolve("b") == (5, 16), f"model fallback chain: {prof.resolve('b')}")
    nested["dataset"]["latent_dim"] = 4
    p.write_text(json.dumps(nested))
    expect(load_profile(p).resolve("c") == (4, 8), "dataset-level latent beats model-level")
    flat = {"run_tag": "t", "window_size": 8, "latent_dim": 6, "datasets": ["a"],
            "encodings": ["direct"]}
    p.write_text(json.dumps(flat))
    expect(load_profile(p).resolve("a") == (6, 8), "flat latent_dim")
    del flat["latent_dim"]
    p.write_text(json.dumps(flat))
    expect_refusal(lambda: load_profile(p).resolve("a"), "no bottleneck width", "no k")

    # 2. PCA is exact on its own subspace; one component short is not; mean is the
    #    column mean.
    width = 8
    basis = rng.standard_normal((2, width))
    fit = rng.standard_normal(width) + rng.standard_normal((40, 2)) @ basis
    test = fit.mean(axis=0) + rng.standard_normal((5, 2)) @ basis
    full = np.full(5, width)
    e2 = reference_errors(fit, np.full(40, width), test, full, 2)
    e1 = reference_errors(fit, np.full(40, width), test, full, 1)
    expect(float(e2["pca"][0].max()) < 1e-20, f"rank-2 data, k=2: {e2['pca'][0]}")
    expect(float(e1["pca"][0].min()) > 1e-6, f"rank-2 data, k=1: {e1['pca'][0]}")
    expect(np.allclose(e2["mean"][0], ((test - fit.mean(axis=0)) ** 2).mean(axis=1)),
           "mean reference = column mean")

    # 3. padding is missing data: whatever sits in a test tail changes nothing, and a
    #    padded fit window does not drag the position mean.
    valid = np.array([width, width - 3, width, width - 5, width])
    zero_tail, junk_tail = test.copy(), test.copy()
    for i, v in enumerate(valid):
        zero_tail[i, v:] = 0.0
        junk_tail[i, v:] = 1e6
    a = reference_errors(fit, np.full(40, width), zero_tail, valid, 2)
    b = reference_errors(fit, np.full(40, width), junk_tail, valid, 2)
    expect(all(np.array_equal(a[m][j], b[m][j]) for m in a for j in (0, 1)),
           "test tail values leak into the error")
    unpadded = valid == width
    expect(all(np.allclose(a[m][j][unpadded], e2[m][j][unpadded]) for m in a for j in (0, 1)),
           "a padded neighbour changed an unpadded window's error")
    fit_pad, fit_valid = fit.copy(), np.full(40, width)
    fit_pad[0, width - 2:] = 0.0
    fit_valid[0] = width - 2
    mu_expected = fit.mean(axis=0)
    mu_expected[width - 2:] = fit[1:, width - 2:].mean(axis=0)
    c = reference_errors(fit_pad, fit_valid, test, full, 1)
    expect(np.allclose(c["mean"][0], ((test - mu_expected) ** 2).mean(axis=1)),
           "padded fit samples drag the position mean")
    fit_junk = fit_pad.copy()
    fit_junk[0, width - 2:] = 1e6
    j = reference_errors(fit_junk, fit_valid, test, full, 1)
    expect(all(np.allclose(c[m][i], j[m][i]) for m in c for i in (0, 1)),
           "padded fit samples steer the principal axes")
    expect_refusal(lambda: reference_errors(fit, np.full(40, width), test, full, 9),
                   "cannot be fitted", "k > width")

    # 4. the families' T-repeated masked metric equals the per-window one.
    x, x_hat = rng.standard_normal(width), rng.standard_normal(width)
    m = (np.arange(width) < 5).astype(float)
    m_t = np.tile(m, 16)
    t_mse = (m_t * (np.tile(x, 16) - np.tile(x_hat, 16)) ** 2).sum() / m_t.sum()
    expect(abs(t_mse - (m * (x - x_hat) ** 2).sum() / m.sum()) < 1e-12, "T-repeat changes MSE")

    # 5. end to end: correct rows, families untouched, idempotent, every refusal fires.
    d = tmp_root / "results"
    d.mkdir()
    prof_path = tmp_root / "e2e.json"
    prof_path.write_text(json.dumps({
        "experiment": {"run_tag": "t"}, "training": {},
        "dataset": {"window_size": width, "sources": [{"name": "fsdd", "latent_dim": 2}]},
        "model": {"latent_dim": 7},
        "evaluation": {"datasets": ["fsdd"], "encodings": ["direct", "poisson"]}}))
    prof = load_profile(prof_path)
    parts, manifest, family = _synthetic_fold(rng)
    _write_fold(d, "t_fsdd_fold0", parts, manifest, family)
    pw = d / "t_fsdd_fold0_per_window_errors.csv"
    before = pw.read_text()
    run(d, "t", prof)
    first = pw.read_text()
    run(d, "t", prof)
    expect(pw.read_text() == first, "second run changed the file (not idempotent)")
    rows = list(csv.DictReader(first.splitlines()))
    n_test = len(parts["test"][1])
    expect(first.startswith(before), "family rows were not preserved verbatim")
    expect(Counter((r["model"], r["encoding"]) for r in rows) == Counter(
        {("lstm-ae", "poisson"): n_test, **{(mm, e): n_test for mm in REFERENCE_MODELS
                                            for e in ("direct", "poisson")}}), "row counts")
    fit_x = np.vstack([parts["train"][0], parts["val"][0]]).astype(np.float32).astype(np.float64)
    fit_v = np.array([mm["valid_length"] for p_ in ("train", "val") for mm in parts[p_][1]])
    test_v = np.array([mm["valid_length"] for mm in parts["test"][1]])
    want = reference_errors(fit_x, fit_v, parts["test"][0].astype(np.float32).astype(np.float64),
                            test_v, 2)
    for model in REFERENCE_MODELS:
        for encoding in ("direct", "poisson"):
            got = [float(r["mse"]) for r in rows
                   if r["model"] == model and r["encoding"] == encoding]
            expect(np.allclose(got, want[model][0], atol=1e-8), f"{model}/{encoding} values")

    _write_fold(d, "t_fsdd_fold0", parts, dict(manifest, latent_dim=3), family)
    staged = pw.read_text()
    expect_refusal(lambda: run(d, "t", prof), "pass the profile the run actually used",
                   "manifest latent_dim mismatch")
    expect(pw.read_text() == staged, "a refused run wrote to the per-window CSV")
    _write_fold(d, "t_fsdd_fold0", parts, manifest, family[1:])
    expect_refusal(lambda: run(d, "t", prof), "not this run's test split", "test-key mismatch")
    _write_fold(d, "t_fsdd_fold0", parts, manifest, family)
    (d / "t_fsdd_fold0_target_val_windows_meta.csv").unlink()
    expect_refusal(lambda: run(d, "t", prof), "incomplete", "partial target dump")
    for f in d.glob("t_fsdd_fold0_target_*"):
        f.unlink()
    np.save(d / "t_fsdd_fold0_poisson_train_windows.npy", np.zeros((2, width), np.float32))
    expect_refusal(lambda: run(d, "t", prof), "--dump-reference-inputs-only", "no target dumps")
    expect_refusal(lambda: run(d, "other", prof), "run_tag", "run-tag mismatch")
    return failures


def self_test() -> int:
    tmp_root = pathlib.Path(tempfile.mkdtemp(prefix="pca_mean_selftest_"))
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


# ------------------------------------------------------------------- main

def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--self-test", action="store_true",
                    help="run synthetic known-answer checks instead of reading real data")
    ap.add_argument("--results-dir", type=pathlib.Path, default=pathlib.Path("results/meeting01"))
    ap.add_argument("--run-tag", default="meeting01_loso")
    ap.add_argument("--profile", type=pathlib.Path,
                    help="the profile the run used: k and the encodings come from it")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()
    if args.profile is None:
        ap.error("--profile is required: k (each dataset's bottleneck) comes from the "
                 "run's profile")
    try:
        run(args.results_dir, args.run_tag, load_profile(args.profile))
    except BaselineInputError as e:
        print(f"[pca-mean] ERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
