#!/usr/bin/env python3
"""Inject Phase 00 winning extractors into the Experiment05 Phase 01 profiles.

Phase 01 profiles ship with a placeholder feature_extraction block. After
Phase 00 has been ranked (01_thesis_phase00_rank.py -> winners.json), this script
rewrites each Phase 01 profile's feature_extraction to the winner for that
profile's signal source:

    source          -> winner used
    voice           -> winners["voice"]
    eeg             -> winners["eeg"]
    fused-early/late -> winners[--fused]   (default: voice)

Fused runs extract both signals through the same handcrafted config, so a single
winner is applied to both halves; --fused chooses whether that is the voice or
the eeg winner (the audio-dominant voice winner is the default).

Usage:
    python3 scripts/pipeline/thesis/02_thesis_apply_winner.py \
        --winners       results/thesis/phase00/winners.json \
        --profiles-dir  src/experiments/thesis/profiles/phase01 \
        --fused         voice

Re-run any time Phase 00 is re-ranked; it overwrites the feature_extraction
block in place and leaves everything else (classifier, training, source) intact.
"""
import argparse
import glob
import json
import os
import sys


def source_of(profile):
    """Classify a Phase 01 profile as voice / eeg / fused."""
    ds = profile["dataset"]
    if ds["modality"] == "fused":
        return "fused"
    return ds["modality"]  # "voice" | "eeg"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--winners", required=True, help="winners.json from 01_thesis_phase00_rank.py")
    ap.add_argument("--profiles-dir", default="src/experiments/thesis/profiles/phase01")
    ap.add_argument("--fused", choices=["voice", "eeg"], default="voice",
                    help="Which signal's winner to use for fused sources (default voice).")
    ap.add_argument("--dry-run", action="store_true", help="Report changes without writing.")
    args = ap.parse_args()

    with open(args.winners) as f:
        winners = json.load(f)
    for sig in ("voice", "eeg"):
        if sig not in winners:
            sys.exit(f"winners.json is missing '{sig}'. Re-run 01_thesis_phase00_rank.py "
                     f"after all Phase 00 {sig} profiles have results.")

    profiles = sorted(glob.glob(os.path.join(args.profiles_dir, "*.json")))
    if not profiles:
        sys.exit(f"No profiles found in {args.profiles_dir}")

    changed = 0
    for prof_path in profiles:
        with open(prof_path) as f:
            prof = json.load(f)

        src = source_of(prof)
        winner_key = args.fused if src == "fused" else src
        winner = winners[winner_key]
        fe = winner["feature_extraction"]

        if prof.get("feature_extraction") == fe:
            continue  # already applied

        prof["feature_extraction"] = fe
        changed += 1
        dpen = winner.get("d_penalized", winner.get("d_truth"))
        print(f"{os.path.basename(prof_path):45s}  <- {winner_key}: {winner['label']} "
              f"(D_penalized={dpen:.6f})")
        if not args.dry_run:
            with open(prof_path, "w") as f:
                json.dump(prof, f, indent=2)
                f.write("\n")

    verb = "would update" if args.dry_run else "updated"
    print(f"\n{verb} {changed}/{len(profiles)} Phase 01 profiles "
          f"(fused <- {args.fused} winner).")


if __name__ == "__main__":
    main()
