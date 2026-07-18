#!/usr/bin/env python3
"""e05_build_phase00_paraconsistent_tables.py — Rank Experiment 05 phase00
configs by d_truth.

Reads every e05_e05_p00_*_rep{0,1,2}_summary.json under results/thesis/phase00,
averages best_d_truth/best_alpha/best_beta across the 3 repeats per profile,
and writes ranked CSV data files (one per {autoencoder, handcrafted} x
{eeg, voice} group, sorted descending by mean d_truth). The thesis LaTeX
typesets these directly via csvsimple — this script only produces data,
never hand-edited .tex, so the tables stay a clean generate/typeset split.

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
# are bit-identical duplicates of the lfcc rows (fixme.md D6: on EEG the Nyquist
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
              f"(fixme.md D6); result files left on disk", file=sys.stderr)

    tables: Dict[str, List[dict]] = defaultdict(list)
    for profile, runs in by_profile.items():
        modality = runs[0]["modality"]
        parsed = parse_profile(profile, modality)
        d_truths = [float(r["best_d_truth"]) for r in runs]
        alphas = [float(r["best_alpha"]) for r in runs]
        betas = [float(r["best_beta"]) for r in runs]
        row = {
            "label": parsed["label"],
            "n_reps": len(runs),
            "mean_d_truth": statistics.mean(d_truths),
            "std_d_truth": statistics.pstdev(d_truths) if len(d_truths) > 1 else 0.0,
            "mean_alpha": statistics.mean(alphas),
            "mean_beta": statistics.mean(betas),
        }
        tables[f"{parsed['group']}_{modality}"].append(row)

    for key in tables:
        tables[key].sort(key=lambda r: r["mean_d_truth"], reverse=True)

    return tables


def write_table(key: str, rows: List[dict], out_dir: pathlib.Path) -> None:
    out_path = out_dir / f"phase00_{key}.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "configuration", "mean_d_truth", "std_d_truth", "mean_alpha", "mean_beta"])
        for i, row in enumerate(rows, start=1):
            w.writerow([
                i,
                row["label"],
                f"{row['mean_d_truth']:.4f}",
                f"{row['std_d_truth']:.4f}",
                f"{row['mean_alpha']:.4f}",
                f"{row['mean_beta']:.4f}",
            ])
    print(f"[phase00-tables] wrote {out_path} ({len(rows)} rows)")


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
