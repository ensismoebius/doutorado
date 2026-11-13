#!/usr/bin/env python3
"""thesis_build_biomex_tables.py — the tables of the BIOMEX-DB replication (thesis chapter 9).

The question each table answers is the replication's: does a Coretto result hold on a second
base? So every BIOMEX-DB cell is printed next to the SAME cell of the Coretto Phase 01 (same
source, text mode, classifier, flat CV, standardized features, frozen winner). This script only
produces data, never .tex.

  biomex_auth.csv        per source x text mode: Coretto DSNN EER; BIOMEX-DB DSNN, ANN twin and
                         cosine EER (mean and spread over the 3 repetitions), and the DSNN/ANN
                         inference energy (nJ per test sample, primary count)
  biomex_dysphonia.csv   per source x scope: the BIOMEX-DB EER at whisper levels 0 (the clean
                         DSNN cell) to 100, and the Coretto rise from 0 to 100 for comparison

Nothing is guessed or skipped: every argument is required; the profile set defines the tables
(a missing summary is refused); a profile whose name belongs to no BIOMEX set is refused; the
name is checked against the declared config; the Coretto cell each row is compared with must
exist. The per-run reading (mean/spread over repetitions, finite scores, energy block) is the
Phase 01 builder's own (thesis_build_phase01_auth_tables.read_runs).

Usage (from software/nn):
    python3 scripts/pipeline/thesis/thesis_build_biomex_tables.py \\
        --profiles-dir src/experiments/thesis/profiles/biomex --results-dir results/thesis/biomex \\
        --coretto-profiles-dir src/experiments/thesis/profiles/phase01 \\
        --coretto-results-dir results/thesis/phase01 \\
        --tables-dir ../../documentation/00-thesis/monography/tables
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("p01", HERE / "thesis_build_phase01_auth_tables.py")
P01 = importlib.util.module_from_spec(_spec)
sys.modules["p01"] = P01
_spec.loader.exec_module(P01)

SET_PATTERNS = {
    "grid": re.compile(r"^bx01_dsnn_([a-z]+)_(dep|indep)_flat_std$"),
    "ann": re.compile(r"^bx01_ann_([a-z]+)_(dep|indep)_flat_std$"),
    "cosine": re.compile(r"^bx01_cosine_([a-z]+)_(dep|indep)_flat_std$"),
    "dysph": re.compile(r"^bx01_dysph_([a-z]+)_(probes|test)_l(\d+)$"),
}
SOURCES = ("voice", "eegpron", "earlypron", "latepron")
DYSPH_SOURCES = ("voice", "earlypron", "latepron")


def fail(msg: str) -> None:
    raise SystemExit(f"[biomex-tables] {msg}")


def collect(profiles_dir: pathlib.Path, results_dir: pathlib.Path) -> dict:
    profiles = sorted(profiles_dir.glob("*.json"))
    if not profiles:
        fail(f"no profiles in {profiles_dir}")
    sets: dict[str, dict] = {k: {} for k in SET_PATTERNS}
    missing: list[pathlib.Path] = []
    for path in profiles:
        name = path.stem
        prof = json.loads(path.read_text(encoding="utf-8"))
        if prof["dataset"]["source"] != "biomex2021":
            fail(f"{path}: dataset.source is {prof['dataset']['source']!r}, not biomex2021")
        matched = [(k, m) for k, p in SET_PATTERNS.items() if (m := p.match(name))]
        if len(matched) != 1:
            fail(f"{path}: the name belongs to no BIOMEX-DB set (or to several)")
        kind, m = matched[0]
        src = P01.source_of(prof)
        tr, clf = prof["training"], prof["classifier"]
        P01.check(name, "source", src, m[1])
        if kind == "dysph":
            dg = prof["degradation"]
            P01.check(name, "degradation.scope", dg["scope"], m[2])
            P01.check(name, "degradation.level", round(dg["level"] * 100), int(m[3]))
            P01.check(name, "text mode", clf["text_mode"], "dependent")
            key = (src, m[2], int(m[3]))
        else:
            P01.check(name, "text mode", clf["text_mode"], P01.TEXT_MODE[m[2]])
            P01.check(name, "classifier", clf["type"], {"grid": "dsnn"}.get(kind, kind))
            P01.check(name, "nested_cv", tr["nested_cv"], False)
            P01.check(name, "standardize", tr["standardize_features"], True)
            key = (src, m[2])
        if key in sets[kind]:
            fail(f"{path}: a second profile for {kind} cell {key}")
        runs = P01.read_runs(prof, results_dir, missing)
        if runs is not None:
            runs["profile"] = prof
            sets[kind][key] = runs
    if missing:
        listed = "\n  ".join(str(p) for p in missing[:20])
        fail(f"{len(missing)} expected summary file(s) missing:\n  {listed}")
    return sets


def auth_table(bx: dict, co: dict, out_dir: pathlib.Path) -> None:
    rows = []
    for s in SOURCES:
        for t in ("dep", "indep"):
            c = P01.require(co["grid"], (s, t, "flat", "std"), "Coretto DSNN")
            d = P01.require(bx["grid"], (s, t), "BIOMEX-DB DSNN")
            a = P01.require(bx["ann"], (s, t), "BIOMEX-DB ANN")
            k = P01.require(bx["cosine"], (s, t), "BIOMEX-DB cosine")
            rows.append([P01.SOURCE_LABELS[s], P01.TEXT_LABELS[t], P01.f4(c["mean_eer"]),
                         P01.f4(d["mean_eer"]), P01.f4(d["std_eer"]),
                         P01.f4(a["mean_eer"]), P01.f4(a["std_eer"]), P01.f4(k["mean_eer"]),
                         f"{d['nj']:.2f}", f"{a['nj']:.2f}", f"{d['nj'] / a['nj']:.2f}"])
    P01.write(out_dir / "biomex_auth.csv",
              ["source", "text", "coretto_dsnn_eer", "dsnn_eer", "dsnn_std", "ann_eer", "ann_std",
               "cosine_eer", "dsnn_nj", "ann_nj", "dsnn_over_ann"], rows)


def dysphonia_table(bx: dict, co: dict, out_dir: pathlib.Path) -> None:
    rows = []
    for s in DYSPH_SOURCES:
        clean = P01.require(bx["grid"], (s, "dep"), "BIOMEX-DB dysphonia (level 0)")
        co_clean = P01.require(co["grid"], (s, "dep", "flat", "std"), "Coretto dysphonia (level 0)")
        for scope in ("probes", "test"):
            cells = [P01.require(bx["dysph"], (s, scope, lv), "BIOMEX-DB dysphonia")
                     for lv in P01.LEVELS]
            co_full = P01.require(co["dysph"], (s, scope, 100), "Coretto dysphonia")
            rows.append([P01.SOURCE_LABELS[s], P01.SCOPE_LABELS[scope], P01.f4(clean["mean_eer"])]
                        + [P01.f4(c["mean_eer"]) for c in cells]
                        + [P01.f4(cells[-1]["mean_eer"] - clean["mean_eer"]),
                           P01.f4(co_full["mean_eer"] - co_clean["mean_eer"])])
    P01.write(out_dir / "biomex_dysphonia.csv",
              ["source", "scope", "eer_l0"] + [f"eer_l{lv}" for lv in P01.LEVELS]
              + ["rise", "coretto_rise"], rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the BIOMEX-DB replication CSV tables.")
    for arg in ("--profiles-dir", "--results-dir", "--coretto-profiles-dir",
                "--coretto-results-dir", "--tables-dir"):
        ap.add_argument(arg, required=True)
    args = ap.parse_args()
    bx = collect(pathlib.Path(args.profiles_dir), pathlib.Path(args.results_dir))
    co = P01.collect(pathlib.Path(args.coretto_profiles_dir), pathlib.Path(args.coretto_results_dir))
    out = pathlib.Path(args.tables_dir)
    out.mkdir(parents=True, exist_ok=True)
    auth_table(bx, co, out)
    dysphonia_table(bx, co, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
