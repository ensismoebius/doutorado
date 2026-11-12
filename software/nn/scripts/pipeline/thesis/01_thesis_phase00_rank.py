#!/usr/bin/env python3
"""Rank the thesis Phase 00 feature extractors by paraconsistent D_penalized.

Phase 00 runs one profile per extractor and signal (handcrafted wavelet x scale x
category, ANN-AE, SNN-AE, PCA), each writing
results/thesis/phase00/e05_<run_tag>[_rep<k>]_paraconsistent.csv with one scored row.
This script reads exactly the files each profile declares (its run_tag and repeats),
averages D_penalized across the repetitions, ranks per signal (voice, eeg) and writes
the winner of each.

The winner per signal is the extractor with the smallest mean D_penalized: the distance
to the paraconsistent "Truth" corner plus a penalty on |G_ct|, so a collapsed latent
cannot win on D_truth alone.

Nothing is optional: every argument is required, and a missing or incomplete result set
is refused -- a winner picked from a partial set is not the Phase 00 winner.

Usage:
    python3 scripts/pipeline/thesis/01_thesis_phase00_rank.py \
        --profiles-dir src/experiments/thesis/profiles/phase00 \
        --results-dir  results/thesis/phase00 \
        --out          results/thesis/phase00/winners.json

The --out JSON feeds 02_thesis_apply_winner.py, which injects each winner into the
Phase 01 profiles.
"""
import argparse
import csv
import glob
import json
import os
import sys


def load_scores(csv_path):
    """Return (d_truth, d_penalized) of the one data row of a Phase 00 paraconsistent CSV.

    A Phase 00 profile scores exactly one feature set (one signal, one extractor), so the
    file must hold exactly one data row: with more, picking rows[0] would silently rank
    whichever set happened to be written first. d_penalized is the selection metric and is
    read, never recomputed -- a CSV without the column predates it and must be re-run, not
    patched up from d_truth and g2.
    """
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1:
        raise ValueError(f"{csv_path}: {len(rows)} data rows; a Phase 00 run scores exactly one "
                         f"feature set. Re-run the profile.")
    row = rows[0]
    for col in ("d_truth", "d_penalized"):
        if col not in row:
            raise ValueError(f"{csv_path}: no {col!r} column (the file predates it). Re-run the "
                             f"profile with the current thesis binary.")
    return float(row["d_truth"]), float(row["d_penalized"])


def expected_csvs(results_dir, prof):
    """The paraconsistent CSVs a profile writes, from its own declared run_tag and repeats.

    thesis.cpp appends _rep<k> to the run_tag only when repeats > 1; the output file adds
    the e05_ prefix. Building the exact names (instead of globbing e05_<tag>*) means a tag
    that is a prefix of another (..._k8_voice vs ..._k8_voice_x) can never borrow its
    neighbour's results, and a missing repetition is detected instead of averaged around.
    """
    tag = prof["experiment"]["run_tag"]
    repeats = prof["experiment"]["repeats"]
    if repeats == 1:
        tags = [tag]
    else:
        tags = [f"{tag}_rep{k}" for k in range(repeats)]
    return [os.path.join(results_dir, f"e05_{t}_paraconsistent.csv") for t in tags]


def extractor_label(fe):
    """Human-readable extractor tag from a feature_extraction block.

    Must identify the extractor UNIQUELY: this label is what gets quoted in the thesis, so
    two different extractors sharing one label is a reporting bug. It therefore carries the
    cepstral category (c1 = band energies, c2 = log+DCT-II cepstral coefficients) and, for
    autoencoders, the temporal encoding and latent size -- without those, e.g. haar/lfcc/c1
    and haar/lfcc/c2 both printed as "handcrafted/haar/lfcc" and appeared to be duplicate
    rows with different scores.
    """
    strategy = fe["strategy"]
    if strategy == "handcrafted":
        hc = fe["handcrafted"]
        cat = "c2" if hc["cepstral"] else "c1"
        return f"handcrafted/{hc['wavelet']}/{hc['scale']}/{cat}"
    if strategy == "autoencoder":
        ae = fe["autoencoder"]
        parts = [f"autoencoder/{ae['model']}"]
        if ae["model"] == "snn-ae":
            parts.append(ae["encoding"])
        # last encoder stage width == latent size (tiny/small/base = 8/16/32)
        parts.append(f"latent{ae['encoder_layer_spec'][-1].split(':')[1]}")
        return "/".join(parts)
    if strategy == "pca":
        # k == the AE latent sizes it is compared against (8/16/32)
        return f"pca/k{fe['pca']['n_components']}"
    raise ValueError(f"unknown feature_extraction.strategy {strategy!r}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profiles-dir", required=True)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--out", required=True, help="Write winners JSON here.")
    args = ap.parse_args()

    profiles = sorted(glob.glob(os.path.join(args.profiles_dir, "*.json")))
    if not profiles:
        sys.exit(f"No profiles found in {args.profiles_dir}")

    # signal -> list of {label, signal, d_truth, d_penalized, n_reps, feature_extraction, profile}
    per_signal = {}
    missing = []
    for prof_path in profiles:
        with open(prof_path) as f:
            prof = json.load(f)
        signal = prof["dataset"]["modality"]
        fe = prof["feature_extraction"]

        csvs = expected_csvs(args.results_dir, prof)
        absent = [c for c in csvs if not os.path.isfile(c)]
        if absent:
            missing.extend(absent)
            continue
        scored = [load_scores(c) for c in csvs]

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

    # A winner picked from a partial set is not the Phase 00 winner, and 02_ would write it
    # into every Phase 01 profile. Refuse rather than rank what happens to be there.
    if missing:
        listed = "\n  ".join(missing[:20])
        more = f"\n  ... and {len(missing) - 20} more" if len(missing) > 20 else ""
        sys.exit(f"{len(missing)} expected result file(s) are missing under {args.results_dir}:"
                 f"\n  {listed}{more}\nRun those Phase 00 profiles first; the ranking is only "
                 f"defined over the complete set.")

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
        for i, e in enumerate(ranked):
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

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(winners, f, indent=2)
        f.write("\n")
    print(f"\nWinners written to {args.out}")
    print("Next: python3 scripts/pipeline/thesis/02_thesis_apply_winner.py "
          f"--winners {args.out} --profiles-dir <phase01 dir> --fused <voice|eeg>")


if __name__ == "__main__":
    main()
