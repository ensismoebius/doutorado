#!/usr/bin/env python3
"""thesis_build_phase01_auth_tables.py — Rank the thesis Phase 01 (DSNN
authentication) configurations by EER.

For every Phase 01 profile, reads exactly the summaries it declares
(e05_<run_tag>_rep<k>_summary.json, from its own run_tag and repeats), averages
mean_eer/mean_auc across the repetitions, and writes one ranked CSV (ascending EER
— lower is better). The thesis LaTeX typesets it directly via csvsimple — this
script only produces data, never hand-edited .tex.

Nothing is guessed or skipped:
  * every argument is required;
  * the profile set defines the table: a missing summary is refused (a table over a
    partial set would rank a different experiment), and a result file with no
    profile is never read;
  * each row's label is built from the profile's declared config (modality,
    fusion_mode, text_mode, nested_cv, standardize_features), not from its file
    name, and two profiles with the same label are refused;
  * a summary must hold exactly one classification result with a finite EER and AUC;
  * with one repetition there is no spread: the std columns print "—", never 0.

Usage:
    python3 scripts/pipeline/thesis/thesis_build_phase01_auth_tables.py \\
        --profiles-dir src/experiments/thesis/profiles/phase01 \\
        --results-dir  results/thesis/phase01 \\
        --tables-dir   ../../documentation/00-thesis/monography/tables
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import statistics
import sys

SOURCE_LABELS = {
    ("eeg", None): "EEG",
    ("voice", None): "Voz",
    ("fused", "early"): "Fusão precoce",
    ("fused", "late"): "Fusão tardia",
}
TEXT_MODE_LABELS = {"dependent": "dependente de texto", "independent": "independente de texto"}
CV_LABELS = {False: "$k$-fold plano", True: "$k$-fold aninhado"}
STD_LABELS = {False: "bruto", True: "padronizado"}


def label_of(prof: dict, path: pathlib.Path) -> str:
    """Table label from the declared config. KeyError on an undeclared key."""
    ds = prof["dataset"]
    source = (ds["modality"], ds["fusion_mode"] if ds["modality"] == "fused" else None)
    if source not in SOURCE_LABELS:
        raise SystemExit(f"[phase01-tables] {path}: no label for source {source}")
    if not prof["classifier"]["enabled"]:
        raise SystemExit(f"[phase01-tables] {path}: classifier.enabled is false — not a "
                         f"Phase 01 (authentication) profile")
    tr = prof["training"]
    return (f"{SOURCE_LABELS[source]} · {TEXT_MODE_LABELS[prof['classifier']['text_mode']]} · "
            f"{CV_LABELS[tr['nested_cv']]} · {STD_LABELS[tr['standardize_features']]}")


def expected_summaries(results_dir: pathlib.Path, prof: dict) -> list[pathlib.Path]:
    """thesis.cpp appends _rep<k> to the run_tag only when repeats > 1."""
    tag = prof["experiment"]["run_tag"]
    repeats = prof["experiment"]["repeats"]
    tags = [tag] if repeats == 1 else [f"{tag}_rep{k}" for k in range(repeats)]
    return [results_dir / f"e05_{t}_summary.json" for t in tags]


def finite(value: object, what: str, path: pathlib.Path) -> float:
    v = float(value)
    if not math.isfinite(v):
        raise SystemExit(f"[phase01-tables] {path}: {what} is {v}; the run did not produce a "
                         f"usable score. Inspect the run, do not average around it.")
    return v


def collect(profiles_dir: pathlib.Path, results_dir: pathlib.Path) -> list[dict[str, object]]:
    profiles = sorted(profiles_dir.glob("*.json"))
    if not profiles:
        raise SystemExit(f"[phase01-tables] no profiles in {profiles_dir}")

    rows: list[dict] = []
    missing: list[pathlib.Path] = []
    seen: dict[str, pathlib.Path] = {}
    for prof_path in profiles:
        prof = json.loads(prof_path.read_text(encoding="utf-8"))
        label = label_of(prof, prof_path)
        if label in seen:
            raise SystemExit(f"[phase01-tables] {prof_path} and {seen[label]} share the label "
                             f"{label!r}: the table could not tell them apart")
        seen[label] = prof_path

        files = expected_summaries(results_dir, prof)
        absent = [f for f in files if not f.is_file()]
        if absent:
            missing.extend(absent)
            continue

        eers, aucs = [], []
        for f in files:
            summary = json.loads(f.read_text(encoding="utf-8"))
            results = summary["results"]
            if len(results) != 1:
                raise SystemExit(f"[phase01-tables] {f}: {len(results)} classification results; "
                                 f"a Phase 01 run classifies exactly one feature set")
            eers.append(finite(results[0]["mean_eer"], "mean_eer", f))
            aucs.append(finite(results[0]["mean_auc"], "mean_auc", f))
        rows.append({
            "label": label,
            "n_reps": len(files),
            "mean_eer": statistics.mean(eers),
            "std_eer": statistics.pstdev(eers) if len(eers) > 1 else None,
            "mean_auc": statistics.mean(aucs),
            "std_auc": statistics.pstdev(aucs) if len(aucs) > 1 else None,
        })

    if missing:
        listed = "\n  ".join(str(m) for m in missing[:20])
        more = f"\n  ... and {len(missing) - 20} more" if len(missing) > 20 else ""
        raise SystemExit(f"[phase01-tables] {len(missing)} expected summary file(s) missing:"
                         f"\n  {listed}{more}\nRun those Phase 01 profiles first.")

    rows.sort(key=lambda r: r["mean_eer"])  # ascending — lower EER is better
    return rows


def spread(std: float | None) -> str:
    return "—" if std is None else f"{std:.4f}"


def write_table(rows: list[dict], out_dir: pathlib.Path) -> pathlib.Path:
    out_path = out_dir / "phase01_auth.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "configuration", "mean_eer", "std_eer", "mean_auc", "std_auc"])
        for i, row in enumerate(rows, start=1):
            w.writerow([
                i,
                row["label"],
                f"{row['mean_eer']:.4f}",
                spread(row["std_eer"]),
                f"{row['mean_auc']:.4f}",
                spread(row["std_auc"]),
            ])
    print(f"[phase01-tables] wrote {out_path} ({len(rows)} rows)")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the ranked Phase 01 EER/AUC CSV table.")
    parser.add_argument("--profiles-dir", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--tables-dir", required=True)
    args = parser.parse_args()

    tables_dir = pathlib.Path(args.tables_dir)
    rows = collect(pathlib.Path(args.profiles_dir), pathlib.Path(args.results_dir))
    tables_dir.mkdir(parents=True, exist_ok=True)
    write_table(rows, tables_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
