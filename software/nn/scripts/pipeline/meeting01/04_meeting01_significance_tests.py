#!/usr/bin/env python3
"""04_meeting01_significance_tests.py — hierarchical significance analysis for the
nested-LOSO comparison.

Reads every  results/meeting01/<tag>_<dataset>_fold<f>_per_window_errors.csv  (written by
the meeting01 binary, with the mean / pca reference rows added by
03_meeting01_pca_mean_baselines.py) and compares each trained family against every other
model at four levels of aggregation, from the paper's primary estimand down to the
descriptive:

  recording-level  (PRIMARY)  d_r = E_{m,r} - E_{b,r},  E_{m,r} = mean_{i in r} MSE
                              report mean(d_r), bootstrap-over-recordings 95% CI,
                              Wilcoxon signed-rank on {d_r}, Holm-Bonferroni across b.
  speaker-level     (robustness) paired Wilcoxon / sign test on per-group mean errors, one
                              group = one held-out speaker or subject. NOT population
                              inference.
  window-level      (descriptive) paired bootstrap — explicitly pseudoreplicated.
  seed-level        (optimization robustness) one mean error per seed, paired by seed.

Effect size: absolute Delta = mean(d_r) and relative Delta / mean(E_{b,.}) * 100 %.

Pairing against a reference (mean, pca). A reference is fitted once, deterministically, on
train + val: it HAS no seed (03_ writes it with seed 0 to mark that). Every seed of a
family is therefore paired with the one reference error of the same window -- the window-
and seed-level comparisons against references used to pair on seed VALUE, found no common
seed, and returned nothing.

Coverage. Both sides of a comparison must cover exactly the same windows, and a family
must have every window under every one of its seeds. A missing fold, recording or seed
(a crashed fold, a fold resumed from checkpoints) is refused instead of compared on the
shared part only, which would silently shrink n and change what the mean is a mean of.

Small n. With n non-zero paired differences the exact two-sided Wilcoxon test cannot go
below 2 / 2^n. Each small-n level reports that floor (min_attainable_p) next to its p:
5 seeds give 0.0625, so the seed level can describe agreement but never reach 0.05.

  --self-test   synthetic known-answer checks instead of reading real data (identical
                predictions -> p ~ 1, CI spans 0; constant offset -> significant, CI
                excludes 0, Delta matches; label permutation -> not significant; references
                paired across seeds; every refusal). Exits non-zero on any failure.

Usage:
    python scripts/pipeline/meeting01/04_meeting01_significance_tests.py \\
        --results-dir results/meeting01 --run-tag meeting01_loso \\
        --out-dir <paper-data-dir>
    python scripts/pipeline/meeting01/04_meeting01_significance_tests.py --self-test
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pathlib
import re
import shutil
import sys
import tempfile
from collections import defaultdict

import numpy as np
from scipy import stats

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
    fold_files,
    read_per_window,
    split_manifest,
)

# "mitbih" replaced 2026-09-23 by eegmmidb/chbmit, then chbmit replaced the same day by
# siena (disk space); both kept here so pre-swap datasets still sort deterministically.
DATASET_ORDER = ["fsdd", "audiomnist", "eegmmidb", "siena", "chbmit", "mitbih"]
BOOT = 10000
RNG = np.random.default_rng(20260908)

_WINDOW_KEY = ("cv_fold", "window_id", "seed")


# --------------------------------------------------------------------------- io

def load_per_window(results_dir: pathlib.Path, run_tag: str) -> list[dict]:
    """Test rows of every fold of the run, typed. Each fold must be in the current results
    format (meeting01_results.split_manifest), hold one run's rows (read_per_window), and
    already carry 03_'s reference rows."""
    rows: list[dict] = []
    for path, ds, fold, _ in fold_files(results_dir, run_tag, re.escape("per_window_errors.csv")):
        split_manifest(results_dir, run_tag, ds, fold)
        fold_rows = [r for r in read_per_window(path, fold) if r["split"] == "test"]
        missing = [m for m in REFERENCE_MODELS if not any(r["model"] == m for r in fold_rows)]
        if missing:
            raise ResultsError(
                f"{path.name} has no {'/'.join(missing)} reference rows: run "
                "03_meeting01_pca_mean_baselines.py first (it adds them; this script "
                "compares against them).")
        for r in fold_rows:
            mse, mae = float(r["mse"]), float(r["mae"])
            if not (math.isfinite(mse) and math.isfinite(mae)):
                raise ResultsError(
                    f"{path.name}: {r['model']} window {r['window_id']} seed {r['seed']} has "
                    f"mse={r['mse']} mae={r['mae']}: the model diverged on it. Dropping the "
                    "window would flatter the model, keeping it makes every mean NaN -- look "
                    "into the run before comparing.")
            rows.append({
                "dataset": ds,
                "model": r["model"],
                "encoding": r["encoding"],
                "seed": int(r["seed"]),
                "cv_fold": fold,
                "speaker_id": int(r["speaker_id"]),
                "recording_id": int(r["recording_id"]),
                "window_id": int(r["window_id"]),
                "mse": mse,
            })
    return rows


# ------------------------------------------------------------------ statistics

def _wilcoxon_p(d: np.ndarray) -> float:
    """Two-sided Wilcoxon signed-rank p (scipy uses the exact null distribution at these
    sample sizes). Zero differences carry no sign and are dropped; if every difference is
    zero there is no evidence of any difference at all: p = 1."""
    if np.allclose(d, 0.0):
        return 1.0
    return float(stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided").pvalue)


def _min_attainable_p(d: np.ndarray) -> float:
    """The smallest two-sided p the exact Wilcoxon test can return for these differences:
    every non-zero difference of one sign, probability 2 / 2^n under the null."""
    n = int(np.count_nonzero(~np.isclose(d, 0.0)))
    return min(1.0, 2.0 / 2.0 ** n)


def _sign_test_p(d: np.ndarray) -> float:
    d = d[d != 0.0]
    if d.size == 0:
        return 1.0
    pos = int((d > 0).sum())
    return float(stats.binomtest(pos, d.size, 0.5).pvalue)


def _bootstrap_ci(values: np.ndarray, n: int = BOOT, alpha: float = 0.05):
    idx = RNG.integers(0, values.size, size=(n, values.size))
    means = values[idx].mean(axis=1)
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def _holm(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (key, p) in enumerate(items):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[key] = running
    return adjusted


def _without_seed(ref: str, key_fields: tuple) -> tuple:
    """The key a model's rows are matched on: a reference has no seed, so against one,
    every seed of the other model meets the same reference row."""
    return tuple(k for k in key_fields if not (ref in REFERENCE_MODELS and k == "seed"))


def _means(rows: list[dict], model: str, key_fields: tuple) -> dict:
    acc: dict = defaultdict(list)
    for r in rows:
        if r["model"] == model:
            acc[tuple(r[k] for k in key_fields)].append(r["mse"])
    return {k: float(np.mean(v)) for k, v in acc.items()}


def check_coverage(rows: list[dict], model: str, ref: str) -> None:
    """Refuses a comparison whose two sides do not cover the same windows -- and, against a
    reference, a model that lacks some window under some of its seeds (the seed-level mean
    of that seed would then average fewer windows than the reference's)."""
    have = {tuple(r[k] for k in _WINDOW_KEY) for r in rows if r["model"] == model}
    if ref in REFERENCE_MODELS:
        windows = {(r["cv_fold"], r["window_id"]) for r in rows if r["model"] == ref}
        seeds = {key[2] for key in have}
        want = {(f, w, s) for (f, w) in windows for s in seeds}
    else:
        want = {tuple(r[k] for k in _WINDOW_KEY) for r in rows if r["model"] == ref}
    if have == want:
        return
    only_want, only_have = sorted(want - have), sorted(have - want)
    raise ResultsError(
        f"{model} vs {ref}: {len(only_want)} (fold, window, seed) combination(s) the "
        f"comparison needs have no {model} row (first: {only_want[:3]}), and {len(only_have)} "
        f"{model} row(s) have nothing to pair with (first: {only_have[:3]}). Pairing on the "
        "shared part only would silently shrink n: a fold, recording or seed is missing from "
        "one side (crashed fold? fold resumed from checkpoints, which does not regenerate "
        "per-window rows?). Complete the run first.")


def _paired_by_key(rows, model, ref, key_fields):
    """Aligned per-key means of `model` and `ref` (see _without_seed); check_coverage()
    has already guaranteed both sides have the same keys."""
    a = _means(rows, model, key_fields)
    b = _means(rows, ref, _without_seed(ref, key_fields))
    project = [key_fields.index(k) for k in _without_seed(ref, key_fields)]
    keys = sorted(a)
    return (np.array([a[k] for k in keys]),
            np.array([b[tuple(k[i] for i in project)] for k in keys]), keys)


def recording_level(rows, model, ref):
    m, b, keys = _paired_by_key(rows, model, ref, ("cv_fold", "recording_id"))
    d = m - b
    lo, hi = _bootstrap_ci(d)
    base = float(np.mean(b))
    return {
        "n_recordings": len(keys),
        "mean_d": float(np.mean(d)),
        "ci95": [lo, hi],
        "wilcoxon_p": _wilcoxon_p(d),
        "delta_abs": float(np.mean(d)),
        "delta_rel_pct": float(np.mean(d) / base * 100.0) if base else math.nan,
    }


def speaker_level(rows, model, ref):
    m, b, keys = _paired_by_key(rows, model, ref, ("cv_fold", "speaker_id"))
    d = m - b
    return {
        "n_speakers": len(keys),
        "mean_d": float(np.mean(d)),
        "wilcoxon_p": _wilcoxon_p(d),
        "min_attainable_p": _min_attainable_p(d),
        "sign_test_p": _sign_test_p(d),
        "note": "n = held-out groups (speakers or subjects) over all folds; robustness only, "
                "not population inference",
    }


def window_level(rows, model, ref):
    m, b, keys = _paired_by_key(rows, model, ref, _WINDOW_KEY)
    d = m - b
    lo, hi = _bootstrap_ci(d)
    note = "pseudoreplicated (windows of one recording are not independent) -- descriptive only"
    if ref in REFERENCE_MODELS:
        note += f"; each of the {model} seeds is paired with the same {ref} error"
    return {"n_windows": len(keys), "mean_d": float(np.mean(d)), "ci95": [lo, hi], "note": note}


def seed_level(rows, model, ref):
    m, b, keys = _paired_by_key(rows, model, ref, ("seed",))
    d = m - b
    return {
        "n_seeds": len(keys),
        "mean_d": float(np.mean(d)),
        "d_per_seed": {str(k[0]): float(v) for k, v in zip(keys, d)},
        "wilcoxon_p": _wilcoxon_p(d),
        "min_attainable_p": _min_attainable_p(d),
    }


def analyse(rows) -> dict:
    models = sorted({r["model"] for r in rows})
    out: dict = {"models_present": models, "comparisons": {}}
    refs = [x for x in (list(TRAINED_MODELS[1:]) + list(REFERENCE_MODELS)) if x in models]
    for model in [m for m in TRAINED_MODELS if m in models]:
        rec_p: dict[str, float] = {}
        per_ref: dict[str, dict] = {}
        for ref in refs:
            if ref == model:
                continue
            check_coverage(rows, model, ref)
            rec = recording_level(rows, model, ref)
            rec_p[ref] = rec["wilcoxon_p"]
            per_ref[ref] = {
                "recording_level": rec,
                "speaker_level": speaker_level(rows, model, ref),
                "window_level": window_level(rows, model, ref),
                "seed_level": seed_level(rows, model, ref),
            }
        holm = _holm(rec_p) if rec_p else {}
        for ref, adj in holm.items():
            per_ref[ref]["recording_level"]["wilcoxon_p_holm"] = adj
        out["comparisons"][model] = per_ref
    return out


def write_recording_tex(analysis: dict, path: pathlib.Path):
    lines = [
        r"% auto-generated by 04_meeting01_significance_tests.py",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Model & Reference & $n_{\text{rec}}$ & $\overline{d_r}$ & 95\% CI & $p_{\text{Holm}}$ \\",
        r"\midrule",
    ]
    for model, refs in analysis["comparisons"].items():
        for ref, blk in refs.items():
            rec = blk["recording_level"]
            lo, hi = rec["ci95"]
            p = rec.get("wilcoxon_p_holm", rec["wilcoxon_p"])
            lines.append(
                f"{model} & {ref} & {rec['n_recordings']} & {rec['mean_d']:.4f} & "
                f"[{lo:.4f}, {hi:.4f}] & {p:.3g} \\\\"
            )
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(results_dir: pathlib.Path, run_tag: str, out_dir: pathlib.Path) -> list[str]:
    rows = load_per_window(results_dir, run_tag)
    if not rows:
        raise ResultsError(f"no {run_tag}_<dataset>_fold<f>_per_window_errors.csv with test "
                           f"rows under {results_dir}.")
    out_dir.mkdir(parents=True, exist_ok=True)
    datasets = sorted({r["dataset"] for r in rows},
                      key=lambda d: (DATASET_ORDER.index(d) if d in DATASET_ORDER else 99, d))
    combined: dict = {}
    for ds in datasets:
        analysis = analyse([r for r in rows if r["dataset"] == ds])
        combined[ds] = analysis
        write_recording_tex(analysis, out_dir / f"{run_tag}_{ds}_significance_recording.tex")
    (out_dir / f"{run_tag}_significance.json").write_text(
        json.dumps(combined, indent=2), encoding="utf-8")
    return datasets


# ------------------------------------------------------------------- self test

def _synthetic_rows(effect: float, seed_noise: float, permute: bool,
                    model_noise: float = 0.02) -> list[dict]:
    rng = np.random.default_rng(7)
    rows: list[dict] = []
    for fold in range(6):
        speaker = fold
        for rec in range(fold * 20, fold * 20 + 20):
            base = rng.uniform(0.2, 0.6)
            for seed in range(5):
                for w in range(8):
                    wid = rec * 100 + w
                    err_ref = base + rng.normal(0, 0.02) + rng.normal(0, seed_noise)
                    err_mod = err_ref + effect + rng.normal(0, model_noise)
                    m1, m2 = "snn-ae", "lstm-ae"
                    if permute and rng.random() < 0.5:
                        err_mod, err_ref = err_ref, err_mod
                    rows.append({"model": m1, "encoding": "direct", "seed": seed,
                                 "cv_fold": fold, "speaker_id": speaker,
                                 "recording_id": rec, "window_id": wid, "mse": err_mod})
                    rows.append({"model": m2, "encoding": "direct", "seed": seed,
                                 "cv_fold": fold, "speaker_id": speaker,
                                 "recording_id": rec, "window_id": wid, "mse": err_ref})
    return rows


def _family_vs_reference_rows(offset: float) -> list[dict]:
    """2 folds x 3 recordings x 4 windows. pca: one error per window, written under two
    encodings (as 03_ does). snn-ae: seeds 42/43/44 at pca + offset + (-0.01, 0, +0.01)."""
    rng = np.random.default_rng(11)
    rows: list[dict] = []
    for fold in range(2):
        for rec in range(fold * 3, fold * 3 + 3):
            for w in range(4):
                wid = rec * 10 + w
                ref_err = float(rng.uniform(0.5, 1.0))
                ident = {"cv_fold": fold, "speaker_id": fold, "recording_id": rec,
                         "window_id": wid}
                for enc in ("direct", "poisson"):
                    rows.append({"model": "pca", "encoding": enc, "seed": 0, "mse": ref_err,
                                 **ident})
                for seed in (42, 43, 44):
                    rows.append({"model": "snn-ae", "encoding": "latency", "seed": seed,
                                 "mse": ref_err + offset + 0.01 * (seed - 43), **ident})
    return rows


def _write_fold(d: pathlib.Path, stem: str, manifest: dict, rows: list[dict]) -> None:
    (d / f"{stem}_split_manifest.json").write_text(json.dumps(manifest))
    with (d / f"{stem}_per_window_errors.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PW_HEADER, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def _csv_fold(fold: int) -> tuple:
    """One fold's manifest and per-window rows as 03_ leaves them: snn-ae seeds 42/43,
    mean and pca under two encodings, two windows."""
    manifest = {"dataset": "fsdd", "cv_fold": fold, "results_format": RESULTS_FORMAT}
    rows = []
    for w in range(2):
        ident = {"cv_fold": str(fold), "split": "test", "speaker_id": str(fold),
                 "recording_id": str(10 * fold + w), "window_id": str(100 * fold + w),
                 "source_window_index": "0", "v_th": "0.00000000", "alpha": "0.00000000"}
        for seed in ("42", "43"):
            rows.append({"model": "snn-ae", "encoding": "latency", "architecture": "dense",
                         "run_id": str(int(seed) - 41), "seed": seed,
                         "mse": "0.40000000", "mae": "0.50000000", **ident})
        for model, err in (("mean", "1.00000000"), ("pca", "0.60000000")):
            for enc in ("direct", "latency"):
                rows.append({"model": model, "encoding": enc, "architecture": model,
                             "run_id": "0", "seed": REFERENCE_SEED, "mse": err, "mae": err,
                             **ident})
    return manifest, rows


def _self_test_checks(tmp_root: pathlib.Path) -> list[str]:
    failures: list[str] = []

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

    # 1. identical predictions -> p == 1, CI spans 0, Delta == 0.
    rec = recording_level(
        _synthetic_rows(0.0, 0.0, permute=False, model_noise=0.0), "snn-ae", "lstm-ae")
    expect(rec["wilcoxon_p"] > 0.99 and rec["ci95"][0] <= 0 <= rec["ci95"][1]
           and abs(rec["mean_d"]) < 1e-9, f"identical: {rec}")

    # 2. constant offset -> significant, CI excludes 0, Delta matches injected offset.
    rec = recording_level(_synthetic_rows(0.10, 0.0, permute=False), "snn-ae", "lstm-ae")
    expect(rec["wilcoxon_p"] < 0.01 and rec["ci95"][0] > 0 and abs(rec["mean_d"] - 0.10) < 0.02,
           f"offset: {rec}")

    # 3. label permutation -> not significant.
    ps = [recording_level(_synthetic_rows(0.0, 0.01, permute=True), "snn-ae", "lstm-ae")
          ["wilcoxon_p"] for _ in range(5)]
    expect(np.mean(ps) >= 0.1, f"permutation p too small: {ps}")

    # 4. hierarchy sanity: recording n < window n, speaker n == 6.
    rows = _synthetic_rows(0.05, 0.0, permute=False)
    r = recording_level(rows, "snn-ae", "lstm-ae")
    w = window_level(rows, "snn-ae", "lstm-ae")
    s = speaker_level(rows, "snn-ae", "lstm-ae")
    expect(r["n_recordings"] == 120 and w["n_windows"] == 120 * 8 * 5 and s["n_speakers"] == 6,
           f"counts: rec={r['n_recordings']} win={w['n_windows']} spk={s['n_speakers']}")

    # 5. against a reference every level answers, each seed meets the same reference error,
    #    and the per-seed differences are exactly the injected ones.
    ref_rows = _family_vs_reference_rows(0.2)
    blk = analyse(ref_rows)["comparisons"]["snn-ae"]["pca"]
    win, seed = blk["window_level"], blk["seed_level"]
    expect(win["n_windows"] == 2 * 3 * 4 * 3 and abs(win["mean_d"] - 0.2) < 1e-12,
           f"window level vs reference: {win}")
    expect(seed["n_seeds"] == 3 and all(
        abs(seed["d_per_seed"][str(s_)] - (0.2 + 0.01 * (s_ - 43))) < 1e-12 for s_ in (42, 43, 44)),
        f"seed level vs reference: {seed}")
    expect(seed["min_attainable_p"] == 0.25, f"3 seeds: floor {seed['min_attainable_p']}")
    expect(blk["recording_level"]["n_recordings"] == 6
           and abs(blk["recording_level"]["mean_d"] - 0.2) < 1e-12,
           f"recording level vs reference: {blk['recording_level']}")

    # 6. a fold, or one seed's windows, missing on one side is refused, not shrunk away.
    no_fold1 = [x for x in ref_rows if not (x["model"] == "snn-ae" and x["cv_fold"] == 1)]
    expect_refusal(lambda: analyse(no_fold1), "silently shrink n", "family lacks a fold")
    no_seed44_fold1 = [x for x in ref_rows if not (x["model"] == "snn-ae" and x["cv_fold"] == 1
                                                  and x["seed"] == 44)]
    expect_refusal(lambda: analyse(no_seed44_fold1), "silently shrink n",
                   "one seed lacks a fold (seed-level mean over fewer windows)")
    rows4 = _synthetic_rows(0.05, 0.0, permute=False)
    lopsided = [x for x in rows4 if not (x["model"] == "lstm-ae" and x["cv_fold"] == 5)]
    expect_refusal(lambda: analyse(lopsided), "silently shrink n", "family vs family, a fold short")

    # 7. files: the format gate, references required, the name rules, one run per file.
    d = tmp_root / "results"
    d.mkdir()
    for fold in (0, 1):
        _write_fold(d, f"t_fsdd_fold{fold}", *_csv_fold(fold))
    out = tmp_root / "out"
    expect(run(d, "t", out) == ["fsdd"], "a clean run is read")
    sig = json.loads((out / "t_significance.json").read_text())
    blk = sig["fsdd"]["comparisons"]["snn-ae"]["pca"]
    expect(blk["seed_level"]["n_seeds"] == 2 and blk["window_level"]["n_windows"] == 8,
           f"file round trip: {blk['seed_level']} {blk['window_level']}")
    expect(abs(blk["recording_level"]["mean_d"] - (0.4 - 0.6)) < 1e-9, "file round trip: d")

    (d / "t_smoke_fsdd_fold0_per_window_errors.csv").write_text("another run's\n")
    expect(run(d, "t", out) == ["fsdd"], "another run's prefix-sharing file is not read")
    (d / "t_smoke_fsdd_fold0_per_window_errors.csv").unlink()

    nameless = d / "t_fold3_per_window_errors.csv"
    nameless.write_text("which dataset?\n")
    expect_refusal(lambda: run(d, "t", out), "no dataset segment", "file without a dataset")
    nameless.unlink()

    manifest, rows0 = _csv_fold(0)
    _write_fold(d, "t_fsdd_fold0", {k: v for k, v in manifest.items() if k != "results_format"},
                rows0)
    expect_refusal(lambda: run(d, "t", out), "older than 2026-10-06", "older binary's fold")
    _write_fold(d, "t_fsdd_fold0", manifest, [x for x in rows0 if x["model"] != "pca"])
    expect_refusal(lambda: run(d, "t", out), "run 03_", "no reference rows")
    _write_fold(d, "t_fsdd_fold0", manifest, rows0 + rows0[:1])
    expect_refusal(lambda: run(d, "t", out), "mixes two runs", "two runs in one file")
    _write_fold(d, "t_fsdd_fold0", manifest, [dict(rows0[0], encoding="")] + rows0[1:])
    expect_refusal(lambda: run(d, "t", out), "without an encoding", "empty encoding")
    _write_fold(d, "t_fsdd_fold0", manifest, [dict(rows0[0], model="vae")] + rows0[1:])
    expect_refusal(lambda: run(d, "t", out), "unknown model", "unknown model")
    _write_fold(d, "t_fsdd_fold0", manifest, [dict(rows0[0], mse="nan")] + rows0[1:])
    expect_refusal(lambda: run(d, "t", out), "diverged", "NaN error")
    return failures


def self_test() -> int:
    tmp_root = pathlib.Path(tempfile.mkdtemp(prefix="significance_selftest_"))
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


# ------------------------------------------------------------------------ main

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--results-dir", type=pathlib.Path, default=pathlib.Path("results/meeting01"))
    ap.add_argument("--run-tag", default="meeting01_loso")
    ap.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("results/meeting01"))
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()
    try:
        datasets = run(args.results_dir, args.run_tag, args.out_dir)
    except ResultsError as e:
        print(f"[significance] ERROR: {e}", file=sys.stderr)
        return 1
    print(f"[significance] wrote {args.run_tag}_significance.json + per-dataset "
          f"_significance_recording.tex ({', '.join(datasets)}) to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
