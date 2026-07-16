#!/usr/bin/env python3
"""Rank Experiment05 Phase 00 feature extractors by paraconsistent D_truth.

Phase 00 runs one profile per (wavelet x scale x signal) plus the autoencoder,
each writing results/phase00/e05_<run_tag>[_repK]_paraconsistent.csv with a
single D_truth. This script collates all of them, averages D_truth across the
repeat runs, ranks per signal (voice, eeg), and reports the winning extractor.

The winner per signal is the combination with the smallest mean D_truth (closest
to the paraconsistent "Truth" corner -> best speaker separability before any
classifier is trained).

Usage:
    python3 scripts/pipeline/e05/01_e05_phase00_rank.py \
        --profiles-dir src/experiments/05/profiles/phase00 \
        --results-dir  results/phase00 \
        --out          results/phase00/winners.json

The --out JSON feeds 02_e05_apply_winner.py, which injects each winner into the
Phase 01 profiles.
"""
import argparse
import csv
import glob
import json
import math
import os
import re
import sys


# Contradiction penalty weight (must match kContradictionPenalty in
# E05Paraconsistent.hpp): 2 - sqrt(2), so the three non-Truth vertices are
# penalized equally.
CONTRADICTION_PENALTY = 2.0 - math.sqrt(2.0)


def load_scores(csv_path):
    """Return (d_truth, d_penalized) of the first data row, or None.

    d_penalized is the primary selection metric. Newer CSVs carry it as a
    column; for older CSVs (written before the column existed) we recompute it
    from d_truth and g2 — d_penalized = d_truth + PENALTY*|g2| is a pure
    function of values already stored, so no re-run is needed to re-rank.
    """
    try:
        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
    except OSError:
        return None
    if not rows:
        return None
    try:
        d_truth = float(rows[0]["d_truth"])
    except (KeyError, ValueError):
        return None
    try:
        d_penalized = float(rows[0]["d_penalized"])
    except (KeyError, ValueError):
        try:
            g2 = float(rows[0]["g2"])
            d_penalized = d_truth + CONTRADICTION_PENALTY * abs(g2)
        except (KeyError, ValueError):
            d_penalized = d_truth  # neither column present: degrade gracefully
    return d_truth, d_penalized


def extractor_label(fe):
    """Human-readable extractor tag from a feature_extraction block."""
    if fe.get("strategy") == "handcrafted":
        hc = fe.get("handcrafted", {})
        return f"handcrafted/{hc.get('wavelet', '?')}/{hc.get('scale', '?')}"
    if fe.get("strategy") == "autoencoder":
        return f"autoencoder/{fe.get('autoencoder', {}).get('model', '?')}"
    return fe.get("strategy", "?")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profiles-dir", default="src/experiments/05/profiles/phase00")
    ap.add_argument("--results-dir", default="results/phase00")
    ap.add_argument("--out", default=None, help="Write winners JSON here.")
    ap.add_argument("--top", type=int, default=5, help="Rows to print per signal.")
    args = ap.parse_args()

    profiles = sorted(glob.glob(os.path.join(args.profiles_dir, "*.json")))
    if not profiles:
        sys.exit(f"No profiles found in {args.profiles_dir}")

    # signal -> list of {label, signal, d_truth, n_reps, feature_extraction, profile}
    per_signal = {}
    missing = []
    for prof_path in profiles:
        with open(prof_path) as f:
            prof = json.load(f)
        tag = prof["experiment"]["run_tag"]
        signal = prof["dataset"]["modality"]
        fe = prof["feature_extraction"]

        # Match both the no-repeat file and the _repK variants.
        pattern = os.path.join(args.results_dir, f"e05_{tag}*_paraconsistent.csv")
        csvs = sorted(glob.glob(pattern))
        scored = [s for s in (load_scores(c) for c in csvs) if s is not None]
        if not scored:
            missing.append(tag)
            continue

        n = len(scored)
        per_signal.setdefault(signal, []).append({
            "label": extractor_label(fe),
            "signal": signal,
            "d_truth": sum(s[0] for s in scored) / n,
            "d_penalized": sum(s[1] for s in scored) / n,
            "n_reps": n,
            "feature_extraction": fe,
            "profile": os.path.relpath(prof_path),
        })

    if not per_signal:
        sys.exit(f"No paraconsistent results found under {args.results_dir}. "
                 f"Run the Phase 00 profiles first. Missing tags: {len(missing)}")

    winners = {}
    for signal in sorted(per_signal):
        # Selection is by d_penalized (contradiction-penalized truth distance),
        # not raw d_truth: the latter can be gamed by a collapsed latent.
        ranked = sorted(per_signal[signal], key=lambda e: e["d_penalized"])
        winners[signal] = {
            "profile": ranked[0]["profile"],
            "label": ranked[0]["label"],
            "d_penalized": ranked[0]["d_penalized"],
            "d_truth": ranked[0]["d_truth"],
            "feature_extraction": ranked[0]["feature_extraction"],
        }
        print(f"\n=== {signal}  ({len(ranked)} combinations ranked) ===")
        print(f"{'rank':>4}  {'D_penalized':>12}  {'D_truth':>12}  {'reps':>4}  extractor")
        for i, e in enumerate(ranked[:args.top]):
            mark = "  <-- winner" if i == 0 else ""
            print(f"{i + 1:>4}  {e['d_penalized']:>12.8f}  {e['d_truth']:>12.8f}  "
                  f"{e['n_reps']:>4}  {e['label']}{mark}")

    if missing:
        print(f"\n[warn] {len(missing)} profile(s) had no results yet "
              f"(skipped): {', '.join(missing[:5])}{' ...' if len(missing) > 5 else ''}")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(winners, f, indent=2)
            f.write("\n")
        print(f"\nWinners written to {args.out}")
        print("Next: python3 scripts/pipeline/e05/02_e05_apply_winner.py "
              f"--winners {args.out}")


if __name__ == "__main__":
    main()
