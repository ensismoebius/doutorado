#!/usr/bin/env python3
"""Inject the Phase 00 winning extractors into the thesis Phase 01 profiles.

The Phase 01 profiles ship with a placeholder feature_extraction block
(00_thesis_generate_profiles.py). After Phase 00 has been ranked
(01_thesis_phase00_rank.py -> winners.json), this script rewrites each profile's
feature_extraction:

    profile                         dataset                    winner used
    p01_cmp-<family>_<signal>_*     voice / eeg (condition)    winners[signal].by_family[family]
    any other                       voice                      winners["voice"]
                                    eeg, pronounced            winners["eegpron"]
                                    eeg, imagined              winners["eegimag"]
                                    fused, condition c         --fused voice -> winners["voice"]
                                                               --fused eeg   -> winners[eeg of c]

A fused run extracts both signals through one config, so one winner serves both halves;
--fused states which signal's winner that is (no default: it is a design choice).

An autoencoder winner trains in Phase 01 under the profile's training block, which also
trains the classifier. Phase 00 ranked it at AE_EPOCHS epochs, so a profile whose
training.epochs differs is refused: running it would evaluate an autoencoder that is not the
one Phase 00 ranked. (The comparison profiles are generated with AE_EPOCHS.)

Usage:
    python3 scripts/pipeline/thesis/02_thesis_apply_winner.py \\
        --winners       results/thesis/phase00/winners.json \\
        --profiles-dir  src/experiments/thesis/profiles/phase01 \\
        --fused         voice

Re-run any time Phase 00 is re-ranked; it overwrites feature_extraction in place and leaves
everything else intact.
"""
import argparse
import glob
import json
import os
import re
import sys

SIGNALS = ("voice", "eegpron", "eegimag")
FAMILIES = ("hc", "ae", "pca")
AE_EPOCHS = 100  # the Phase 00 autoencoder budget (00_thesis_generate_profiles.AE_TRAINING)
EEG_SIGNAL = {"pronounced": "eegpron", "imagined": "eegimag"}


def winner_for(prof, name, winners, fused):
    ds = prof["dataset"]
    m = re.match(r"p01_cmp-([a-z]+)_([a-z]+)_", name)
    if m:
        fam, sig = m.groups()
        if fam not in FAMILIES or sig not in SIGNALS:
            raise SystemExit(f"{name}: unknown comparison family/signal {fam!r}/{sig!r}")
        expected = "voice" if ds["modality"] == "voice" else EEG_SIGNAL[ds["eeg_condition"]]
        if ds["modality"] == "fused" or expected != sig:
            raise SystemExit(f"{name}: its name says {sig}, its dataset says {expected}")
        return f"{sig}/{fam}", winners[sig]["by_family"][fam]
    if ds["modality"] == "voice":
        return "voice", winners["voice"]
    if ds["modality"] == "eeg":
        sig = EEG_SIGNAL[ds["eeg_condition"]]
        return sig, winners[sig]
    if ds["modality"] == "fused":
        sig = "voice" if fused == "voice" else EEG_SIGNAL[ds["eeg_condition"]]
        return sig, winners[sig]
    raise SystemExit(f"{name}: unknown modality {ds['modality']!r}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--winners", required=True, help="winners.json from 01_thesis_phase00_rank.py")
    ap.add_argument("--profiles-dir", required=True)
    ap.add_argument("--fused", choices=["voice", "eeg"], required=True,
                    help="Which signal's winner the fused sources use.")
    ap.add_argument("--dry-run", action="store_true", help="Report changes without writing.")
    args = ap.parse_args()

    with open(args.winners) as f:
        winners = json.load(f)
    for sig in SIGNALS:
        if sig not in winners or "by_family" not in winners[sig]:
            sys.exit(f"winners.json lacks '{sig}' (or its by_family). Re-run "
                     f"01_thesis_phase00_rank.py after all Phase 00 profiles have results.")

    profiles = sorted(glob.glob(os.path.join(args.profiles_dir, "*.json")))
    if not profiles:
        sys.exit(f"No profiles found in {args.profiles_dir}")

    planned = []
    for prof_path in profiles:
        with open(prof_path) as f:
            prof = json.load(f)
        name = os.path.splitext(os.path.basename(prof_path))[0]
        key, winner = winner_for(prof, name, winners, args.fused)
        fe = winner["feature_extraction"]
        if fe["strategy"] == "autoencoder":
            epochs = prof["training"].get("epochs")
            if epochs != AE_EPOCHS:
                sys.exit(f"{name}: the {key} winner is an autoencoder ranked at {AE_EPOCHS} "
                         f"epochs, but this profile trains {epochs}. Generate the profile with "
                         f"epochs={AE_EPOCHS} or rank without it; nothing was written.")
        planned.append((prof_path, prof, key, winner))

    # Validate everything before writing anything: a refusal leaves no half-applied set.
    changed = 0
    for prof_path, prof, key, winner in planned:
        fe = winner["feature_extraction"]
        if prof["feature_extraction"] == fe:
            continue
        prof["feature_extraction"] = fe
        changed += 1
        print(f"{os.path.basename(prof_path):48s}  <- {key}: {winner['label']} "
              f"(D_penalized={winner['d_penalized']:.6f})")
        if not args.dry_run:
            with open(prof_path, "w") as f:
                json.dump(prof, f, indent=2)
                f.write("\n")

    verb = "would update" if args.dry_run else "updated"
    print(f"\n{verb} {changed}/{len(profiles)} Phase 01 profiles (fused <- {args.fused} winner).")


if __name__ == "__main__":
    main()
