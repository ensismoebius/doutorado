#!/usr/bin/env python3
"""thesis_build_phase01_auth_tables.py — the thesis Phase 01 (authentication) tables.

Phase 01 holds five experimental sets (00_thesis_generate_profiles.py). Each becomes one CSV
that the thesis typesets with csvsimple; this script only produces data, never .tex.

  phase01_auth.csv         p01_dsnn_*: the grid, ranked by EER
                           (source x text mode x CV scheme x raw/standardized)
  phase01_controls.csv     DSNN vs its non-spiking twin (p01_ann_*) vs cosine on the features
                           (p01_cosine_*), per source and text mode, flat CV, standardized;
                           with the measured inference energy of the two networks
  phase01_dysphonia.csv    p01_dysph_*: EER per source and scope at each whisper level; level
                           0 is the clean DSNN cell of the grid (dependent, flat, std)
  phase01_extractors.csv   p01_cmp-<family>_*: the three extractor families through the same
                           classifier, per signal and text mode (Question 3)
  phase01_energy.csv       the measured inference energy (nJ per test sample, classifier only)
                           of the DSNN and its twin, per source (text mode dependent: the
                           networks are the same in both modes), under every check: the primary
                           memory-aware count, the DSNN as executed (first layer at every step),
                           every access at one SRAM size (8 KB, 32 KB, 1 MB), and arithmetic
                           alone; `holds` says whether the DSNN/ANN order is the same in all

Nothing is guessed or skipped: every argument is required; the profile set defines the tables
(a missing summary is refused); a profile whose name does not belong to one of the five sets is
refused; labels come from the declared config, and the name is checked against it; with one
repetition there is no spread (the std columns print "—", never 0).

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
import re
import statistics
import sys

SET_PATTERNS = {
    "grid": re.compile(r"^p01_dsnn_([a-z]+)_(dep|indep)_(flat|nested)_(raw|std)$"),
    "ann": re.compile(r"^p01_ann_([a-z]+)_(dep|indep)_flat_std$"),
    "cosine": re.compile(r"^p01_cosine_([a-z]+)_(dep|indep)_flat_std$"),
    "dysph": re.compile(r"^p01_dysph_([a-z]+)_(probes|test)_l(\d+)$"),
    "cmp": re.compile(r"^p01_cmp-(hc|ae|pca)_([a-z]+)_(dep|indep)$"),
}
SOURCE_LABELS = {
    "voice": "Voz",
    "eegpron": "EEG (fala pronunciada)",
    "eegimag": "EEG (fala imaginada)",
    "earlypron": "Fusão precoce (pronunciada)",
    "earlyimag": "Fusão precoce (imaginada)",
    "latepron": "Fusão tardia (pronunciada)",
    "lateimag": "Fusão tardia (imaginada)",
}
SOURCE_ORDER = list(SOURCE_LABELS)
TEXT_LABELS = {"dep": "frases vistas", "indep": "frases não vistas"}
TEXT_MODE = {"dep": "dependent", "indep": "independent"}
CV_LABELS = {"flat": "$k$-fold plano", "nested": "$k$-fold aninhado"}
STD_LABELS = {"raw": "bruto", "std": "padronizado"}
SCOPE_LABELS = {"probes": "só sondas", "test": "cadastro e sondas"}
FAMILY_LABELS = {"hc": "Manual", "ae": "Autoencoder", "pca": "PCA"}
LEVELS = (25, 50, 75, 100)


def fail(msg: str) -> None:
    raise SystemExit(f"[phase01-tables] {msg}")


def source_of(prof: dict) -> str:
    """The source key from the declared dataset block."""
    ds = prof["dataset"]
    cond = {"pronounced": "pron", "imagined": "imag"}
    if ds["modality"] == "voice":
        return "voice"
    if ds["modality"] == "eeg":
        return "eeg" + cond[ds["eeg_condition"]]
    if ds["modality"] == "fused":
        return ds["fusion_mode"] + cond[ds["eeg_condition"]]
    fail(f"unknown modality {ds['modality']!r}")
    return ""


def check(name: str, what: str, declared: object, named: object) -> None:
    if declared != named:
        fail(f"{name}: its name says {what} = {named!r}, its config declares {declared!r}")


def expected_summaries(results_dir: pathlib.Path, prof: dict) -> list[pathlib.Path]:
    """thesis.cpp appends _rep<k> to the run_tag only when repeats > 1."""
    tag = prof["experiment"]["run_tag"]
    repeats = prof["experiment"]["repeats"]
    tags = [tag] if repeats == 1 else [f"{tag}_rep{k}" for k in range(repeats)]
    return [results_dir / f"e05_{t}_summary.json" for t in tags]


def finite(value: object, what: str, path: pathlib.Path) -> float:
    v = float(value)
    if not math.isfinite(v):
        fail(f"{path}: {what} is {v}; the run did not produce a usable score. Inspect the run, "
             f"do not average around it.")
    return v


def read_runs(prof: dict, results_dir: pathlib.Path, missing: list) -> dict | None:
    """Mean/std of EER, AUC and (networks) energy over the profile's repetitions."""
    files = expected_summaries(results_dir, prof)
    absent = [f for f in files if not f.is_file()]
    if absent:
        missing.extend(absent)
        return None
    eers, aucs, energies = [], [], []
    for f in files:
        results = json.loads(f.read_text(encoding="utf-8"))["results"]
        if len(results) != 1:
            fail(f"{f}: {len(results)} classification results; a Phase 01 run classifies exactly "
                 f"one feature set")
        r = results[0]
        eers.append(finite(r["mean_eer"], "mean_eer", f))
        aucs.append(finite(r["mean_auc"], "mean_auc", f))
        is_net = prof["classifier"]["type"] != "cosine"
        if is_net != ("energy" in r):
            fail(f"{f}: a {prof['classifier']['type']} result "
                 f"{'without' if is_net else 'with'} an energy block")
        if is_net:
            energies.append({k: finite(v, f"energy.{k}", f) for k, v in r["energy"].items()})

    def spread(v):
        return statistics.pstdev(v) if len(v) > 1 else None

    return {"n_reps": len(files),
            "mean_eer": statistics.mean(eers), "std_eer": spread(eers),
            "mean_auc": statistics.mean(aucs), "std_auc": spread(aucs),
            "nj": statistics.mean(e["nj"] for e in energies) if energies else None,
            "energy": ({k: statistics.mean(e[k] for e in energies) for k in energies[0]}
                       if energies else None)}


def collect(profiles_dir: pathlib.Path, results_dir: pathlib.Path) -> dict:
    profiles = sorted(profiles_dir.glob("*.json"))
    if not profiles:
        fail(f"no profiles in {profiles_dir}")
    sets: dict[str, dict] = {k: {} for k in SET_PATTERNS}
    missing: list[pathlib.Path] = []
    for path in profiles:
        name = path.stem
        prof = json.loads(path.read_text(encoding="utf-8"))
        if not prof["classifier"]["enabled"]:
            fail(f"{path}: classifier.enabled is false — not a Phase 01 profile")
        matched = [(k, m) for k, p in SET_PATTERNS.items() if (m := p.match(name))]
        if len(matched) != 1:
            fail(f"{path}: the name belongs to no Phase 01 set (or to several); the tables are "
                 f"defined by the generated sets only")
        kind, m = matched[0]
        src = source_of(prof)
        tr, clf = prof["training"], prof["classifier"]
        if kind == "grid":
            check(name, "source", src, m[1])
            check(name, "text mode", clf["text_mode"], TEXT_MODE[m[2]])
            check(name, "nested_cv", tr["nested_cv"], m[3] == "nested")
            check(name, "standardize", tr["standardize_features"], m[4] == "std")
            check(name, "classifier", clf["type"], "dsnn")
            key = (src, m[2], m[3], m[4])
        elif kind in ("ann", "cosine"):
            check(name, "source", src, m[1])
            check(name, "text mode", clf["text_mode"], TEXT_MODE[m[2]])
            check(name, "classifier", clf["type"], kind)
            key = (src, m[2])
        elif kind == "dysph":
            check(name, "source", src, m[1])
            dg = prof["degradation"]
            check(name, "degradation.scope", dg["scope"], m[2])
            check(name, "degradation.level", round(dg["level"] * 100), int(m[3]))
            check(name, "text mode", clf["text_mode"], "dependent")
            key = (src, m[2], int(m[3]))
        else:  # cmp
            check(name, "signal", src, m[2])
            check(name, "text mode", clf["text_mode"], TEXT_MODE[m[3]])
            fam = {"handcrafted": "hc", "autoencoder": "ae", "pca": "pca"}[
                prof["feature_extraction"]["strategy"]]
            check(name, "extractor family", fam, m[1])
            key = (src, m[3], m[1])
        if key in sets[kind]:
            fail(f"{path}: a second profile for {kind} cell {key}")
        runs = read_runs(prof, results_dir, missing)
        if runs is not None:
            runs["profile"] = prof
            sets[kind][key] = runs
    if missing:
        listed = "\n  ".join(str(p) for p in missing[:20])
        more = f"\n  ... and {len(missing) - 20} more" if len(missing) > 20 else ""
        fail(f"{len(missing)} expected summary file(s) missing:\n  {listed}{more}\n"
             f"Run those Phase 01 profiles first.")
    return sets


def f4(x: float | None) -> str:
    return "—" if x is None else f"{x:.4f}"


def write(out: pathlib.Path, header: list[str], rows: list[list]) -> None:
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print(f"[phase01-tables] wrote {out} ({len(rows)} rows)")


def grid_table(sets: dict, out_dir: pathlib.Path) -> None:
    rows = sorted(sets["grid"].items(), key=lambda kv: kv[1]["mean_eer"])
    write(out_dir / "phase01_auth.csv",
          ["rank", "configuration", "mean_eer", "std_eer", "mean_auc", "std_auc"],
          [[i, f"{SOURCE_LABELS[s]} · {TEXT_LABELS[t]} · {CV_LABELS[c]} · {STD_LABELS[n]}",
            f4(r["mean_eer"]), f4(r["std_eer"]), f4(r["mean_auc"]), f4(r["std_auc"])]
           for i, ((s, t, c, n), r) in enumerate(rows, start=1)])


def require(table: dict, key: tuple, what: str) -> dict:
    if key not in table:
        fail(f"{what}: no profile for cell {key}; the set is incomplete")
    return table[key]


def controls_table(sets: dict, out_dir: pathlib.Path) -> None:
    rows = []
    for s in SOURCE_ORDER:
        for t in ("dep", "indep"):
            d = require(sets["grid"], (s, t, "flat", "std"), "controls (DSNN)")
            a = require(sets["ann"], (s, t), "controls (ANN)")
            c = require(sets["cosine"], (s, t), "controls (cosine)")
            rows.append([SOURCE_LABELS[s], TEXT_LABELS[t], f4(d["mean_eer"]), f4(d["std_eer"]),
                         f4(a["mean_eer"]), f4(a["std_eer"]), f4(c["mean_eer"]),
                         f"{d['nj']:.2f}", f"{a['nj']:.2f}", f"{d['nj'] / a['nj']:.2f}"])
    write(out_dir / "phase01_controls.csv",
          ["source", "text", "dsnn_eer", "dsnn_std", "ann_eer", "ann_std", "cosine_eer",
           "dsnn_nj", "ann_nj", "dsnn_over_ann"], rows)


def dysphonia_table(sets: dict, out_dir: pathlib.Path) -> None:
    sources = sorted({k[0] for k in sets["dysph"]}, key=SOURCE_ORDER.index)
    if not sources:
        fail("dysphonia: no p01_dysph_* profile")
    rows = []
    for s in sources:
        clean = require(sets["grid"], (s, "dep", "flat", "std"), "dysphonia (level 0)")
        for scope in ("probes", "test"):
            cells = [require(sets["dysph"], (s, scope, lv), "dysphonia") for lv in LEVELS]
            rows.append([SOURCE_LABELS[s], SCOPE_LABELS[scope], f4(clean["mean_eer"])] +
                        [f4(c["mean_eer"]) for c in cells])
    write(out_dir / "phase01_dysphonia.csv",
          ["source", "scope", "eer_l0"] + [f"eer_l{lv}" for lv in LEVELS], rows)


def extractor_label(fe: dict) -> str:
    if fe["strategy"] == "handcrafted":
        hc = fe["handcrafted"]
        return f"{hc['wavelet']}/{hc['scale']}/{'c2' if hc['cepstral'] else 'c1'}"
    if fe["strategy"] == "autoencoder":
        ae = fe["autoencoder"]
        enc = f"/{ae['encoding']}" if ae["model"] == "snn-ae" else ""
        return f"{ae['model']}{enc}/latente {ae['encoder_layer_spec'][-1].split(':')[1]}"
    return f"k = {fe['pca']['n_components']}"


def extractors_table(sets: dict, out_dir: pathlib.Path) -> None:
    rows = []
    for s in ("voice", "eegpron", "eegimag"):
        for t in ("dep", "indep"):
            row = [SOURCE_LABELS[s], TEXT_LABELS[t]]
            for fam in ("hc", "ae", "pca"):
                r = require(sets["cmp"], (s, t, fam), "extractor comparison")
                row += [extractor_label(r["profile"]["feature_extraction"]), f4(r["mean_eer"]),
                        f4(r["std_eer"])]
            rows.append(row)
    header = ["signal", "text"]
    for fam in ("hc", "ae", "pca"):
        header += [f"{fam}_extractor", f"{fam}_eer", f"{fam}_std"]
    write(out_dir / "phase01_extractors.csv", header, rows)


ENERGY_CHECKS = (("nj", "primária"), ("nj_as_executed", "como executada"),
                 ("nj_sram8k", "SRAM 8 KB"), ("nj_sram32k", "SRAM 32 KB"),
                 ("nj_sram1m", "SRAM 1 MB"), ("ops_nj", "só aritmética"))


def energy_table(sets: dict, out_dir: pathlib.Path) -> None:
    rows = []
    for s in SOURCE_ORDER:
        d = require(sets["grid"], (s, "dep", "flat", "std"), "energy (DSNN)")["energy"]
        a = require(sets["ann"], (s, "dep"), "energy (ANN)")["energy"]
        ratios = [d[k] / a[k] for k, _ in ENERGY_CHECKS]
        holds = all(r < 1 for r in ratios) or all(r > 1 for r in ratios)
        rows.append([SOURCE_LABELS[s], f"{d['spikes']:.1f}"] +
                    [f"{d[k]:.2f}" for k, _ in ENERGY_CHECKS] +
                    [f"{a[k]:.2f}" for k, _ in ENERGY_CHECKS] +
                    [f"{r:.2f}" for r in ratios] + ["sim" if holds else "não"])
    header = ["source", "dsnn_spikes"] + [f"dsnn_{k}" for k, _ in ENERGY_CHECKS] + \
             [f"ann_{k}" for k, _ in ENERGY_CHECKS] + [f"ratio_{k}" for k, _ in ENERGY_CHECKS] + \
             ["holds"]
    write(out_dir / "phase01_energy.csv", header, rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the Phase 01 CSV tables.")
    parser.add_argument("--profiles-dir", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--tables-dir", required=True)
    args = parser.parse_args()

    sets = collect(pathlib.Path(args.profiles_dir), pathlib.Path(args.results_dir))
    out_dir = pathlib.Path(args.tables_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    grid_table(sets, out_dir)
    controls_table(sets, out_dir)
    dysphonia_table(sets, out_dir)
    extractors_table(sets, out_dir)
    energy_table(sets, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
