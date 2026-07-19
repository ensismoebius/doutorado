#!/usr/bin/env python3
"""e05_build_phase00_paraconsistent_tables.py — Rank Experiment 05 phase00
configs by d_penalized.

Reads every e05_e05_p00_*_rep{0,1,2}_summary.json under results/thesis/phase00,
averages best_d_penalized/best_d_truth/best_alpha/best_beta across the 3 repeats
per profile, and writes ranked CSV data files (one per {autoencoder, handcrafted}
x {eeg, voice} group, sorted ASCENDING by mean d_penalized). The thesis LaTeX
typesets these directly via csvsimple — this script only produces data,
never hand-edited .tex, so the tables stay a clean generate/typeset split.

RANKING KEY. Rank 1 is the BEST configuration: both d_truth and d_penalized are
distances to the paraconsistent "Truth" vertex, so lower is better, and the sort
is ascending. This file previously sorted descending, which silently made every
"#1" row the *worst* configuration of its table and buried each signal's actual
winner at the bottom (the EEG winner sat at rank 46 of 46).

The ranking key is d_penalized, NOT d_truth — matching 01_e05_phase00_rank.py,
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

Usage:
    python3 scripts/pipeline/e05/e05_build_phase00_paraconsistent_tables.py \\
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

RUN_RE = re.compile(r"^e05_e05_(p00_.+)_rep(\d+)_summary\.json$")

WAVELET_LABELS = {"haar": "Haar"}


def wavelet_label(tag: str) -> str:
    if tag in WAVELET_LABELS:
        return WAVELET_LABELS[tag]
    m = re.match(r"daub(\d+)$", tag)
    if m:
        return f"Daubechies {m.group(1)}"
    return tag


def parse_profile(name: str, modality: str) -> Dict[str, str]:
    """name is the profile stem, e.g. 'p00_ae_snn_direct_base_eeg' or
    'p00_hc_daub10_bark_c1_voice'. Returns {'group': 'ae'|'hc', 'label': str}."""
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

    m = re.match(r"^hc_(.+)_(bark|lfcc|mel)_(c\d)$", body)
    if m:
        wavelet, feature, category = m.group(1), m.group(2), m.group(3)
        return {
            "group": "hc",
            "label": f"{wavelet_label(wavelet)} + {feature.upper()} ({category})",
        }

    raise ValueError(f"Unrecognized phase00 profile stem: {name}")


# EEG results for the bark/mel scales, produced before those profiles were retired.
# The tables are built from whatever is in results/, not from the current profile set, so
# without this filter the retired runs would still emit bark/mel rows for EEG -- and they
# are bit-identical duplicates of the lfcc rows (.wiki/Guides/Engineering-Fixes-Log.md D6: on EEG the Nyquist
# normalization stretches the curve ~5x, the mapping becomes injective and the grouping
# degenerates to exactly lfcc's). Printing them as if they were distinct results is exactly
# the misreading D6 exists to prevent. The result files themselves are deliberately left on
# disk -- they are long-run data, and they are not wrong, merely redundant.
RETIRED_EEG_SCALES = ("bark", "mel")


def is_retired_eeg_scale_run(profile: str, modality: str) -> bool:
    if modality != "eeg":
        return False
    m = re.match(r"^p00_hc_.+_(bark|lfcc|mel)_c\d$", profile[: -len("_eeg")])
    return bool(m) and m.group(1) in RETIRED_EEG_SCALES


def collect(results_dir: pathlib.Path) -> Dict[str, List[Dict[str, object]]]:
    by_profile: Dict[str, List[dict]] = defaultdict(list)
    skipped = 0
    for f in results_dir.glob("e05_e05_p00_*_rep*_summary.json"):
        m = RUN_RE.match(f.name)
        if not m:
            continue
        profile = m.group(1)
        if profile.endswith("_eeg") and is_retired_eeg_scale_run(profile, "eeg"):
            skipped += 1
            continue
        data = json.loads(f.read_text(encoding="utf-8"))
        by_profile[profile].append(data)
    if skipped:
        print(f"[info] skipped {skipped} retired EEG bark/mel run(s) — duplicates of lfcc "
              f"(.wiki/Guides/Engineering-Fixes-Log.md D6); result files left on disk", file=sys.stderr)

    tables: Dict[str, List[dict]] = defaultdict(list)
    for profile, runs in by_profile.items():
        modality = runs[0]["modality"]
        parsed = parse_profile(profile, modality)
        d_truths = [float(r["best_d_truth"]) for r in runs]
        alphas = [float(r["best_alpha"]) for r in runs]
        betas = [float(r["best_beta"]) for r in runs]
        # Average the per-repetition d_penalized rather than recomputing it from the
        # mean alpha/beta: d_penalized is non-linear in |g2|, so the two agree only
        # when the repeats are identical (true for handcrafted, where std=0, but not
        # for the autoencoders). 01_e05_phase00_rank.py averages per repetition, and
        # the winner marked below has to match the winner it wrote to winners.json.
        d_pens = [float(r["best_d_penalized"]) for r in runs]
        row = {
            "label": parsed["label"],
            "modality": modality,
            "n_reps": len(runs),
            "mean_d_penalized": statistics.mean(d_pens),
            "mean_d_truth": statistics.mean(d_truths),
            "std_d_truth": statistics.pstdev(d_truths) if len(d_truths) > 1 else 0.0,
            "mean_alpha": statistics.mean(alphas),
            "mean_beta": statistics.mean(betas),
        }
        tables[f"{parsed['group']}_{modality}"].append(row)

    # Ascending: d_penalized is a distance to the Truth vertex, so rank 1 is the best.
    for key in tables:
        tables[key].sort(key=lambda r: r["mean_d_penalized"])

    # Mark each signal's winner, chosen across BOTH groups (the selection in
    # 01_e05_phase00_rank.py is per signal, not per family), so exactly one row per
    # modality carries the mark and the other table's rank 1 is not mistaken for it.
    for modality in {r["modality"] for rows in tables.values() for r in rows}:
        candidates = [r for rows in tables.values() for r in rows if r["modality"] == modality]
        min(candidates, key=lambda r: r["mean_d_penalized"])["winner"] = True

    return tables


def write_table(key: str, rows: List[dict], out_dir: pathlib.Path) -> None:
    out_path = out_dir / f"phase00_{key}.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "configuration", "mean_d_penalized", "mean_d_truth",
                    "std_d_truth", "mean_alpha", "mean_beta"])
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
                f"{row['mean_d_truth']:.4f}",
                f"{row['std_d_truth']:.4f}",
                f"{row['mean_alpha']:.4f}",
                f"{row['mean_beta']:.4f}",
            ])
    marked = sum(1 for r in rows if r.get("winner"))
    print(f"[phase00-tables] wrote {out_path} ({len(rows)} rows"
          f"{', winner marked' if marked else ''})")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build ranked phase00 d_truth CSV tables.")
    parser.add_argument("--results-dir", default="results/thesis/phase00")
    parser.add_argument(
        "--tables-dir",
        default="/home/ensismoebius/Repos/doutorado/documentation/00-thesis/monography/tables",
    )
    args = parser.parse_args()

    results_dir = pathlib.Path(args.results_dir)
    tables_dir = pathlib.Path(args.tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)

    tables = collect(results_dir)
    if not tables:
        print("[phase00-tables] no phase00 summary.json files found")
        return 1

    for key in ("ae_eeg", "ae_voice", "hc_eeg", "hc_voice"):
        if key in tables:
            write_table(key, tables[key], tables_dir)
        else:
            print(f"[phase00-tables] WARNING: no rows for {key}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
