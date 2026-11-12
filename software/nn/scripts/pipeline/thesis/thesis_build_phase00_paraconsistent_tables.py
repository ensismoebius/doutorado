#!/usr/bin/env python3
"""thesis_build_phase00_paraconsistent_tables.py — Rank Experiment 05 phase00
configs by d_penalized.

For every Phase 00 profile, reads exactly the summaries it declares
(e05_<run_tag>[_rep<k>]_summary.json, from its own run_tag and repeats), averages
best_d_penalized/best_d_truth/best_alpha/best_beta across the repetitions, and writes
ranked CSV data files (one per {autoencoder, handcrafted, pca} x {voice, eegpron, eegimag}
group, sorted ASCENDING by mean d_penalized). The profile set, not whatever is in results/,
defines the tables: a result file with no profile is never read, and a profile with a
missing repetition or an empty group is refused (a stale table would otherwise stay in
the thesis looking current). The thesis LaTeX
typesets these directly via csvsimple — this script only produces data,
never hand-edited .tex, so the tables stay a clean generate/typeset split.

RANKING KEY. Rank 1 is the BEST configuration: both d_truth and d_penalized are
distances to the paraconsistent "Truth" vertex, so lower is better, and the sort
is ascending. This file previously sorted descending, which silently made every
"#1" row the *worst* configuration of its table and buried each signal's actual
winner at the bottom (the EEG winner sat at rank 46 of 46).

The ranking key is d_penalized, NOT d_truth — matching 01_thesis_phase00_rank.py,
which is what actually picks the winner written into winners.json. The two orders
genuinely differ: for voice the lowest-d_truth config (daub22/lfcc/c1) is not the
winner, because the |g2| contradiction penalty demotes it below haar/lfcc/c1. Both
columns are emitted so a reader can see why the winner is the winner (see the
D_penalizado section of the thesis's chapter 2).

Per-signal winners are marked with a dagger. The winner is chosen across BOTH the
handcrafted and autoencoder groups, so the marked row appears in only one of the
two tables for that signal; the other table's rank 1 is merely the best of its own
family. That distinction matters here: ranked by d_truth alone the best EEG
autoencoder (1.4944) would appear to beat the handcrafted winner (1.5143), and it
is only the contradiction penalty that reverses them.

UNCERTAINTY (issue 8). A handcrafted or PCA extractor is deterministic: its repetitions are
the same number, so their spread is not printed ("—"). What is printed instead comes from the
speaker subsets every profile scores (e05_<tag>_paraconsistent_subsets.csv):
  loso_sd    the standard deviation of d_penalized over the 15 leave-one-speaker-out sets --
             how much the score moves when one speaker is missing;
  fold_wins  in how many of the 15 Phase 01 training-fold subsets (k = 5 x 3 repetitions)
             this configuration has the best d_penalized of its signal, ties included --
             how often a selection made on the training speakers alone picks it.

LABELS (issue 17). c1 is the band energies grouped on a frequency scale, c2 their cepstrum
(log + DCT-II). Only c2 is cepstral, so c1 is named by its scale ("Linear", "Bark", "Mel") and
c2 by the cepstral coefficient name ("LFCC", "BFCC", "MFCC").

Usage:
    python3 scripts/pipeline/thesis/thesis_build_phase00_paraconsistent_tables.py \\
        --profiles-dir src/experiments/thesis/profiles/phase00 \\
        --results-dir results/thesis/phase00 \\
        --tables-dir ../../documentation/00-thesis/monography/tables
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re
import sys
import statistics
from collections import defaultdict
from typing import Dict, List

WAVELET_LABELS = {"haar": "Haar"}


def wavelet_label(tag: str) -> str:
    if tag in WAVELET_LABELS:
        return WAVELET_LABELS[tag]
    m = re.match(r"daub(\d+)$", tag)
    if m:
        return f"Daubechies {m.group(1)}"
    raise ValueError(f"no table label for wavelet {tag!r}; add it to WAVELET_LABELS")


C1_SCALE = {"lfcc": "Linear", "bark": "Bark", "mel": "Mel"}
C2_SCALE = {"lfcc": "LFCC", "bark": "BFCC", "mel": "MFCC"}
SIGNALS = ("voice", "eegpron", "eegimag")
DETERMINISTIC = {"hc", "pca"}


def signal_of(prof: dict) -> str:
    ds = prof["dataset"]
    if ds["modality"] == "voice":
        return "voice"
    return {"pronounced": "eegpron", "imagined": "eegimag"}[ds["eeg_condition"]]


def parse_profile(name: str, modality: str) -> Dict[str, str]:
    """name is the profile stem, e.g. 'p00_ae_snn_direct_base_eeg' or
    'p00_hc_daub10_bark_c1_voice' or 'p00_pca_k16_eeg'.
    Returns {'group': 'ae'|'hc'|'pca', 'label': str}."""
    body = name[len("p00_"):]
    suffix = f"_{modality}"
    assert body.endswith(suffix), f"{body} does not end with {suffix}"
    body = body[: -len(suffix)]

    if body.startswith("ae_ann_"):
        size = body[len("ae_ann_"):]
        return {"group": "ae", "label": f"ANN-AE ({size})"}

    m = re.match(r"^ae_snn_(direct|latency|poisson)_(.+)$", body)
    if m:
        encoding, size = m.group(1), m.group(2)
        return {"group": "ae", "label": f"SNN-AE {encoding} ({size})"}

    m = re.match(r"^pca_k(\d+)$", body)
    if m:
        return {"group": "pca", "label": f"PCA (k = {m.group(1)})"}

    m = re.match(r"^hc_(.+)_(bark|lfcc|mel)_(c\d)$", body)
    if m:
        wavelet, feature, category = m.group(1), m.group(2), m.group(3)
        scale = (C1_SCALE if category == "c1" else C2_SCALE)[feature]
        return {"group": "hc", "label": f"{wavelet_label(wavelet)} + {scale} ({category})"}

    raise ValueError(f"Unrecognized phase00 profile stem: {name}")


def expected_summaries(results_dir: pathlib.Path, prof: dict) -> List[pathlib.Path]:
    """The summaries a profile writes: thesis.cpp appends _rep<k> to the run_tag only when
    repeats > 1, and the output file adds the e05_ prefix."""
    tag = prof["experiment"]["run_tag"]
    repeats = prof["experiment"]["repeats"]
    tags = [tag] if repeats == 1 else [f"{tag}_rep{k}" for k in range(repeats)]
    return [results_dir / f"e05_{t}_summary.json" for t in tags]


def collect(profiles_dir: pathlib.Path,
            results_dir: pathlib.Path) -> Dict[str, List[Dict[str, object]]]:
    profiles = sorted(profiles_dir.glob("*.json"))
    if not profiles:
        raise SystemExit(f"[phase00-tables] no profiles in {profiles_dir}")
    by_profile: Dict[str, List[dict]] = {}
    subsets_of: Dict[str, Dict[str, float]] = {}
    signal_by: Dict[str, str] = {}
    missing: List[pathlib.Path] = []
    for prof_path in profiles:
        prof = json.loads(prof_path.read_text(encoding="utf-8"))
        files = expected_summaries(results_dir, prof)
        subs = [f.with_name(f.name.replace("_summary.json", "_paraconsistent_subsets.csv"))
                for f in files]
        absent = [f for f in files + subs if not f.is_file()]
        if absent:
            missing.extend(absent)
            continue
        by_profile[prof_path.stem] = [json.loads(f.read_text(encoding="utf-8")) for f in files]
        signal_by[prof_path.stem] = signal_of(prof)
        per_rep = []
        for f in subs:
            with f.open(newline="", encoding="utf-8") as fh:
                per_rep.append({r["subset"]: float(r["d_penalized"]) for r in csv.DictReader(fh)})
        if any(set(x) != set(per_rep[0]) for x in per_rep):
            raise SystemExit(f"[phase00-tables] {prof_path}: repetitions scored different subsets")
        subsets_of[prof_path.stem] = {k: statistics.mean(x[k] for x in per_rep)
                                      for k in per_rep[0]}
    if missing:
        listed = "\n  ".join(str(m) for m in missing[:20])
        more = f"\n  ... and {len(missing) - 20} more" if len(missing) > 20 else ""
        raise SystemExit(f"[phase00-tables] {len(missing)} expected summary file(s) missing:"
                         f"\n  {listed}{more}\nRun those Phase 00 profiles first.")

    tables: Dict[str, List[dict]] = defaultdict(list)
    for profile, runs in by_profile.items():
        modality = signal_by[profile]
        parsed = parse_profile(profile, modality)
        d_truths = [float(r["best_d_truth"]) for r in runs]
        alphas = [float(r["best_alpha"]) for r in runs]
        betas = [float(r["best_beta"]) for r in runs]
        # Average the per-repetition d_penalized rather than recomputing it from the
        # mean alpha/beta: d_penalized is non-linear in |g2|, so the two agree only
        # when the repeats are identical (true for handcrafted, where std=0, but not
        # for the autoencoders). 01_thesis_phase00_rank.py averages per repetition, and
        # the winner marked below has to match the winner it wrote to winners.json.
        d_pens = [float(r["best_d_penalized"]) for r in runs]
        row = {
            "label": parsed["label"],
            "modality": modality,
            "n_reps": len(runs),
            "mean_d_penalized": statistics.mean(d_pens),
            "mean_d_truth": statistics.mean(d_truths),
            # A deterministic extractor's repetitions are one measurement, and one
            # repetition has no spread: None prints as an explicit "—", never as 0.
            "std_d_penalized": (statistics.pstdev(d_pens)
                                if len(d_pens) > 1 and parsed["group"] not in DETERMINISTIC
                                else None),
            "subsets": subsets_of[profile],
            "loso_sd": statistics.stdev(v for k, v in subsets_of[profile].items()
                                        if k.startswith("loso:")),
            "mean_alpha": statistics.mean(alphas),
            "mean_beta": statistics.mean(betas),
        }
        tables[f"{parsed['group']}_{modality}"].append(row)

    # Ascending: d_penalized is a distance to the Truth vertex, so rank 1 is the best.
    for key in tables:
        tables[key].sort(key=lambda r: r["mean_d_penalized"])

    # fold_wins: per Phase 01 training-fold subset, the best d_penalized of the signal.
    TIE_EPS = 1e-5  # as 01_thesis_phase00_rank.py
    for modality in {r["modality"] for rows in tables.values() for r in rows}:
        cands = [r for rows in tables.values() for r in rows if r["modality"] == modality]
        folds = sorted(k for k in cands[0]["subsets"] if k.startswith("fold:"))
        if any(set(r["subsets"]) != set(cands[0]["subsets"]) for r in cands):
            raise SystemExit(f"[phase00-tables] {modality}: the profiles scored different subsets")
        for r in cands:
            r["fold_wins"] = 0
            r["n_folds"] = len(folds)
        for k in folds:
            best = min(r["subsets"][k] for r in cands)
            for r in cands:
                if r["subsets"][k] - best <= TIE_EPS:
                    r["fold_wins"] += 1

    # Mark each signal's winner, chosen across BOTH groups (the selection in
    # 01_thesis_phase00_rank.py is per signal, not per family), so exactly one row per
    # modality carries the mark and the other table's rank 1 is not mistaken for it.
    for modality in {r["modality"] for rows in tables.values() for r in rows}:
        candidates = [r for rows in tables.values() for r in rows if r["modality"] == modality]
        min(candidates, key=lambda r: r["mean_d_penalized"])["winner"] = True

    return tables


def write_table(key: str, rows: List[dict], out_dir: pathlib.Path) -> None:
    out_path = out_dir / f"phase00_{key}.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "configuration", "mean_d_penalized", "std_d_penalized",
                    "loso_sd", "fold_wins", "mean_d_truth", "mean_alpha", "mean_beta"])
        for i, row in enumerate(rows, start=1):
            # The dagger is emitted as a LaTeX macro, not a literal "†": the CSV is read by
            # csvsimple straight into a tabular cell, so the marker has to survive as markup.
            # No "\," thin space before it -- that macro contains a literal comma, which would
            # force the writer to quote the field and leave the quoting for csvsimple to undo.
            rank = f"{i}\\textsuperscript{{$\\dagger$}}" if row.get("winner") else str(i)
            w.writerow([
                rank,
                row["label"],
                f"{row['mean_d_penalized']:.4f}",
                "—" if row["std_d_penalized"] is None else f"{row['std_d_penalized']:.4f}",
                f"{row['loso_sd']:.4f}",
                f"{row['fold_wins']}/{row['n_folds']}",
                f"{row['mean_d_truth']:.4f}",
                f"{row['mean_alpha']:.4f}",
                f"{row['mean_beta']:.4f}",
            ])
    marked = sum(1 for r in rows if r.get("winner"))
    print(f"[phase00-tables] wrote {out_path} ({len(rows)} rows"
          f"{', winner marked' if marked else ''})")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build ranked phase00 d_truth CSV tables.")
    parser.add_argument("--profiles-dir", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--tables-dir", required=True)
    args = parser.parse_args()

    results_dir = pathlib.Path(args.results_dir)
    tables_dir = pathlib.Path(args.tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)

    tables = collect(pathlib.Path(args.profiles_dir), results_dir)
    keys = tuple(f"{g}_{s}" for g in ("ae", "hc", "pca") for s in SIGNALS)
    empty = [k for k in keys if k not in tables]
    if empty:
        print(f"[phase00-tables] no profile in groups {empty}: every group the thesis typesets "
              f"must be produced, or its table would stay stale", file=sys.stderr)
        return 1
    for key in keys:
        write_table(key, tables[key], tables_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
