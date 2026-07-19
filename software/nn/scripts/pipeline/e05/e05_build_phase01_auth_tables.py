#!/usr/bin/env python3
"""e05_build_phase01_auth_tables.py — Rank Experiment 05 phase01 (DSNN
authentication) configs by EER.

Reads every e05_e05_p01_*_rep{0,1,2}_summary.json under results/thesis/phase01,
averages mean_eer/mean_auc across the 3 repeats per profile, and writes one
ranked CSV (ascending EER — lower is better) covering all 32 configurations.
The thesis LaTeX typesets this directly via csvsimple — this script only
produces data, never hand-edited .tex.

Usage:
    python3 scripts/pipeline/e05/e05_build_phase01_auth_tables.py \\
        --results-dir results/thesis/phase01 \\
        --tables-dir ../../documentation/00-thesis/monography/tables
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re
import statistics
import sys
from collections import defaultdict
from typing import Dict, List

RUN_RE = re.compile(r"^e05_e05_(p01_.+)_rep(\d+)_summary\.json$")
PROFILE_RE = re.compile(
    r"^p01_dsnn_(eeg|voice|fused_early|fused_late)_(dep|indep)_(flat|nested)_(raw|std)$"
)

SOURCE_LABELS = {
    "eeg": "EEG",
    "voice": "Voz",
    "fused_early": "Fusão precoce",
    "fused_late": "Fusão tardia",
}
TEXT_MODE_LABELS = {"dep": "dependente de texto", "indep": "independente de texto"}
CV_LABELS = {"flat": "$k$-fold plano", "nested": "$k$-fold aninhado"}
STD_LABELS = {"raw": "bruto", "std": "padronizado"}


def parse_profile(name: str) -> str:
    m = PROFILE_RE.match(name)
    if not m:
        raise ValueError(f"Unrecognized phase01 profile stem: {name}")
    source, text_mode, cv, std = m.groups()
    return (
        f"{SOURCE_LABELS[source]} · {TEXT_MODE_LABELS[text_mode]} · "
        f"{CV_LABELS[cv]} · {STD_LABELS[std]}"
    )


def collect(results_dir: pathlib.Path) -> List[Dict[str, object]]:
    by_profile: Dict[str, List[dict]] = defaultdict(list)
    for f in results_dir.glob("e05_e05_p01_*_rep*_summary.json"):
        m = RUN_RE.match(f.name)
        if not m:
            continue
        by_profile[m.group(1)].append(json.loads(f.read_text(encoding="utf-8")))

    rows: List[dict] = []
    for profile, runs in by_profile.items():
        label = parse_profile(profile)
        eers = [float(r["results"][0]["mean_eer"]) for r in runs]
        aucs = [float(r["results"][0]["mean_auc"]) for r in runs]
        rows.append({
            "label": label,
            "n_reps": len(runs),
            "mean_eer": statistics.mean(eers),
            "std_eer": statistics.pstdev(eers) if len(eers) > 1 else 0.0,
            "mean_auc": statistics.mean(aucs),
            "std_auc": statistics.pstdev(aucs) if len(aucs) > 1 else 0.0,
        })

    rows.sort(key=lambda r: r["mean_eer"])  # ascending — lower EER is better
    return rows


def write_table(rows: List[dict], out_dir: pathlib.Path) -> pathlib.Path:
    out_path = out_dir / "phase01_auth.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "configuration", "mean_eer", "std_eer", "mean_auc", "std_auc"])
        for i, row in enumerate(rows, start=1):
            w.writerow([
                i,
                row["label"],
                f"{row['mean_eer']:.4f}",
                f"{row['std_eer']:.4f}",
                f"{row['mean_auc']:.4f}",
                f"{row['std_auc']:.4f}",
            ])
    print(f"[phase01-tables] wrote {out_path} ({len(rows)} rows)")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build ranked phase01 EER/AUC CSV table.")
    parser.add_argument("--results-dir", default="results/thesis/phase01")
    parser.add_argument(
        "--tables-dir",
        default="/home/ensismoebius/Repos/doutorado/documentation/00-thesis/monography/tables",
    )
    args = parser.parse_args()

    results_dir = pathlib.Path(args.results_dir)
    tables_dir = pathlib.Path(args.tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)

    rows = collect(results_dir)
    if not rows:
        print("[phase01-tables] no phase01 summary.json files found")
        return 1
    if len(rows) != 32:
        print(f"[phase01-tables] WARNING: expected 32 configurations, found {len(rows)}",
              file=sys.stderr)

    write_table(rows, tables_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
