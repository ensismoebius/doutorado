#!/usr/bin/env python3
"""04_guayaquil_significance_tests.py — hierarchical significance analysis for the
nested-LOSO comparison.

Reads every  results/guayaquil/<tag>_fold*_per_window_errors.csv  (the append-only
per-window dumps written by the guayaquil binary and by
03_guayaquil_pca_mean_baselines.py) and compares each model against each reference at
four levels of aggregation, from the paper's primary estimand down to the descriptive:

  recording-level  (PRIMARY)  d_r = E_{m,r} - E_{b,r},  E_{m,r} = mean_{i in r} MSE
                              report mean(d_r), bootstrap-over-recordings 95% CI,
                              Wilcoxon signed-rank on {d_r}, Holm-Bonferroni across b.
  speaker-level     (robustness, n = 6 held-out speakers) paired Wilcoxon / sign test
                              on per-speaker mean errors. NOT population inference.
  window-level      (descriptive) paired bootstrap — explicitly pseudoreplicated.
  seed-level        (optimization robustness) paired Wilcoxon across the per-seed means.

Effect size: absolute Delta = mean(d_r) and relative Delta / mean(E_{b,.}) * 100 %.

  --self-test   run synthetic known-answer checks instead of reading real data
                (identical predictions -> p ~ 1, CI spans 0; constant offset ->
                significant, CI excludes 0, Delta matches; label permutation ->
                approx-uniform p). Exits non-zero on any failure. Safe for CI.

Usage:
    python scripts/pipeline/guayaquil/04_guayaquil_significance_tests.py \\
        --results-dir results/guayaquil --run-tag article_loso \\
        --out-dir <paper-data-dir>
    python scripts/pipeline/guayaquil/04_guayaquil_significance_tests.py --self-test
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

TRAINED = ["snn-ae", "lstm-ae", "gru-ae", "transformer-ae"]
REFERENCES = ["pca", "mean"]
BOOT = 10000
RNG = np.random.default_rng(20260908)


# --------------------------------------------------------------------------- io

def load_per_window(results_dir: pathlib.Path, run_tag: str) -> list[dict]:
    rows: list[dict] = []
    files = sorted(results_dir.glob(f"{run_tag}_fold*_per_window_errors.csv"))
    for path in files:
        with path.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("split") != "test":
                    continue
                rows.append({
                    "model": r["model"],
                    "encoding": r["encoding"],
                    "seed": int(r["seed"]),
                    "cv_fold": int(r["cv_fold"]),
                    "speaker_id": int(r["speaker_id"]),
                    "recording_id": int(r["recording_id"]),
                    "window_id": int(r["window_id"]),
                    "mse": float(r["mse"]),
                })
    return rows


# ------------------------------------------------------------------ statistics

def _wilcoxon_p(d: np.ndarray) -> float:
    d = d[np.isfinite(d)]
    if d.size < 3 or np.allclose(d, 0.0):
        return 1.0
    try:
        return float(stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided").pvalue)
    except ValueError:
        return 1.0


def _sign_test_p(d: np.ndarray) -> float:
    d = d[np.isfinite(d) & (d != 0.0)]
    if d.size == 0:
        return 1.0
    pos = int((d > 0).sum())
    return float(stats.binomtest(pos, d.size, 0.5).pvalue)


def _bootstrap_ci(values: np.ndarray, n: int = BOOT, alpha: float = 0.05):
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (math.nan, math.nan)
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


def _paired_by_key(rows, model, ref, key_fields, value="mse"):
    """mean `value` per (key_fields) for `model` and `ref`, returned as aligned arrays
    over the intersection of keys."""
    def agg(target):
        acc = defaultdict(list)
        for r in rows:
            if r["model"] != target:
                continue
            acc[tuple(r[k] for k in key_fields)].append(r[value])
        return {k: float(np.mean(v)) for k, v in acc.items()}

    a, b = agg(model), agg(ref)
    keys = sorted(set(a) & set(b))
    return np.array([a[k] for k in keys]), np.array([b[k] for k in keys]), keys


def recording_level(rows, model, ref):
    m, b, keys = _paired_by_key(rows, model, ref, ("cv_fold", "recording_id"))
    if keys == []:
        return None
    d = m - b
    lo, hi = _bootstrap_ci(d)
    base = float(np.mean(b)) if b.size else math.nan
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
    if keys == []:
        return None
    d = m - b
    return {
        "n_speakers": len(keys),
        "mean_d": float(np.mean(d)),
        "wilcoxon_p": _wilcoxon_p(d),
        "sign_test_p": _sign_test_p(d),
        "note": "n is the number of held-out FSDD speakers; not population inference",
    }


def window_level(rows, model, ref):
    m, b, keys = _paired_by_key(rows, model, ref, ("window_id", "seed"))
    if keys == []:
        return None
    d = m - b
    lo, hi = _bootstrap_ci(d)
    return {
        "n_windows": len(keys),
        "mean_d": float(np.mean(d)),
        "ci95": [lo, hi],
        "note": "pseudoreplicated — descriptive only",
    }


def seed_level(rows, model, ref):
    m, b, keys = _paired_by_key(rows, model, ref, ("seed",))
    if len(keys) < 2:
        return None
    d = m - b
    return {"n_seeds": len(keys), "mean_d": float(np.mean(d)), "wilcoxon_p": _wilcoxon_p(d)}


def analyse(rows) -> dict:
    models = sorted({r["model"] for r in rows})
    out: dict = {"models_present": models, "comparisons": {}}
    refs = [x for x in (TRAINED[1:] + REFERENCES) if x in models]
    for model in [m for m in TRAINED if m in models]:
        rec_p: dict[str, float] = {}
        per_ref: dict[str, dict] = {}
        for ref in refs:
            if ref == model:
                continue
            rec = recording_level(rows, model, ref)
            if rec is None:
                continue
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
        r"% auto-generated by 04_guayaquil_significance_tests.py",
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


def self_test() -> int:
    failures = []

    # 1. identical predictions -> p == 1, CI spans 0, Delta == 0.
    rec = recording_level(
        _synthetic_rows(0.0, 0.0, permute=False, model_noise=0.0), "snn-ae", "lstm-ae")
    if not (rec["wilcoxon_p"] > 0.99 and rec["ci95"][0] <= 0 <= rec["ci95"][1]
            and abs(rec["mean_d"]) < 1e-9):
        failures.append(f"identical: {rec}")

    # 2. constant offset -> significant, CI excludes 0, Delta matches injected offset.
    rec = recording_level(_synthetic_rows(0.10, 0.0, permute=False), "snn-ae", "lstm-ae")
    if not (rec["wilcoxon_p"] < 0.01 and rec["ci95"][0] > 0
            and abs(rec["mean_d"] - 0.10) < 0.02):
        failures.append(f"offset: {rec}")

    # 3. label permutation -> not significant.
    ps = [recording_level(_synthetic_rows(0.0, 0.01, permute=True), "snn-ae", "lstm-ae")["wilcoxon_p"]
          for _ in range(5)]
    if np.mean(ps) < 0.1:
        failures.append(f"permutation p too small: {ps}")

    # 4. hierarchy sanity: recording n < window n, speaker n == 6.
    rows = _synthetic_rows(0.05, 0.0, permute=False)
    r = recording_level(rows, "snn-ae", "lstm-ae")
    w = window_level(rows, "snn-ae", "lstm-ae")
    s = speaker_level(rows, "snn-ae", "lstm-ae")
    if not (r["n_recordings"] == 120 and w["n_windows"] == 120 * 8 * 5 and s["n_speakers"] == 6):
        failures.append(f"counts: rec={r['n_recordings']} win={w['n_windows']} spk={s['n_speakers']}")

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
    ap.add_argument("--results-dir", type=pathlib.Path, default=pathlib.Path("results/guayaquil"))
    ap.add_argument("--run-tag", default="article_loso")
    ap.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("results/guayaquil"))
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()

    rows = load_per_window(args.results_dir, args.run_tag)
    if not rows:
        print(f"[significance] no test-split per-window rows under {args.results_dir}",
              file=sys.stderr)
        return 1

    analysis = analyse(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"{args.run_tag}_significance.json").write_text(
        json.dumps(analysis, indent=2), encoding="utf-8")
    write_recording_tex(analysis, args.out_dir / f"{args.run_tag}_significance_recording.tex")
    print(f"[significance] wrote {args.run_tag}_significance.json + _recording.tex to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
