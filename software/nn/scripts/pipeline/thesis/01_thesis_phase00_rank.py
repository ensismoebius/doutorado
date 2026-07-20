#!/usr/bin/env python3
"""Rank Experiment05 Phase 00 feature extractors by paraconsistent D_truth.

Phase 00 runs one profile per (wavelet x scale x signal) plus the autoencoder,
each writing results/thesis/phase00/e05_<run_tag>[_repK]_paraconsistent.csv with a
single D_truth. This script collates all of them, averages D_truth across the
repeat runs, ranks per signal (voice, eeg), and reports the winning extractor.

The winner per signal is the combination with the smallest mean D_truth (closest
to the paraconsistent "Truth" corner -> best speaker separability before any
classifier is trained).

Usage:
    python3 scripts/pipeline/e05/01_e05_phase00_rank.py \
        --profiles-dir src/experiments/05/profiles/phase00 \
        --results-dir  results/thesis/phase00 \
        --out          results/thesis/phase00/winners.json

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
    """Human-readable extractor tag from a feature_extraction block.

    Must identify the extractor UNIQUELY: this label is what gets quoted in the thesis, so
    two different extractors sharing one label is a reporting bug. It therefore carries the
    cepstral category (c1 = band energies, c2 = log+DCT-II cepstral coefficients) and, for
    autoencoders, the temporal encoding and latent size -- without those, e.g. haar/lfcc/c1
    and haar/lfcc/c2 both printed as "handcrafted/haar/lfcc" and appeared to be duplicate
    rows with different scores.
    """
    if fe.get("strategy") == "handcrafted":
        hc = fe.get("handcrafted", {})
        cat = "c2" if hc.get("cepstral") else "c1"
        return f"handcrafted/{hc.get('wavelet', '?')}/{hc.get('scale', '?')}/{cat}"
    if fe.get("strategy") == "autoencoder":
        ae = fe.get("autoencoder", {})
        model = ae.get("model", "?")
        parts = [f"autoencoder/{model}"]
        if model == "snn-ae":
            parts.append(ae.get("encoding", "?"))
        spec = ae.get("encoder_layer_spec") or []
        if spec:
            # last encoder stage width == latent size (tiny/small/base = 8/16/32)
            try:
                parts.append(f"latent{spec[-1].split(':')[1]}")
            except (IndexError, ValueError):
                pass
        return "/".join(parts)
    return fe.get("strategy", "?")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profiles-dir", default="src/experiments/05/profiles/phase00")
    ap.add_argument("--results-dir", default="results/thesis/phase00")
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

    # Two combinations whose d_penalized agree to within this are treated as tied.
    # They are not "almost equal" -- in practice they came out bit-identical because the
    # extractors are literally the same function of the same data (see TIE_EPS rationale in
    # the module docstring / .wiki/Guides/Engineering-Fixes-Log.md D6). The tolerance only absorbs float noise, e.g. the
    # ~1e-6 drift seen on daub32 profiles that were re-run individually.
    TIE_EPS = 1e-5

    winners = {}
    for signal in sorted(per_signal):
        # Selection is by d_penalized (contradiction-penalized truth distance),
        # not raw d_truth: the latter can be gamed by a collapsed latent.
        ranked = sorted(per_signal[signal], key=lambda e: e["d_penalized"])

        # Anything tied with rank 1. Reporting only ranked[0] would silently promote an
        # arbitrary sort tie-break into a scientific claim: for EEG, haar/bark, haar/mel and
        # haar/lfcc scored bit-identically, and "bark won" was purely an artifact of ordering
        # (.wiki/Guides/Engineering-Fixes-Log.md D6). Callers must know when the winner is not uniquely determined.
        best = ranked[0]["d_penalized"]
        tied = [e for e in ranked if abs(e["d_penalized"] - best) <= TIE_EPS]

        winners[signal] = {
            "profile": ranked[0]["profile"],
            "label": ranked[0]["label"],
            "d_penalized": ranked[0]["d_penalized"],
            "d_truth": ranked[0]["d_truth"],
            "feature_extraction": ranked[0]["feature_extraction"],
            # Provenance of the choice: when tied_with is non-empty the winner was picked by
            # tie-break among equals, so no property of the chosen label caused it to win.
            "tie_count": len(tied),
            "tied_with": [e["label"] for e in tied[1:]],
        }
        print(f"\n=== {signal}  ({len(ranked)} combinations ranked) ===")
        print(f"{'rank':>4}  {'D_penalized':>12}  {'D_truth':>12}  {'reps':>4}  extractor")
        for i, e in enumerate(ranked[:args.top]):
            mark = "  <-- winner" if i == 0 else ""
            if i > 0 and abs(e["d_penalized"] - best) <= TIE_EPS:
                mark = "  <-- TIED with winner"
            print(f"{i + 1:>4}  {e['d_penalized']:>12.8f}  {e['d_truth']:>12.8f}  "
                  f"{e['n_reps']:>4}  {e['label']}{mark}")

        if len(tied) > 1:
            print(f"\n[TIE] {signal}: {len(tied)} extractors share the best D_penalized "
                  f"({best:.8f}) to within {TIE_EPS:g}:")
            for e in tied:
                print(f"         {e['label']}")
            print("       The reported winner was chosen by tie-break, NOT because it is "
                  "better.\n       Do not attribute the result to any property that only "
                  "the chosen label has.")

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
