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

Two protocols (--protocol, required; .wiki/Experiments/Thesis-Per-Fold-Protocol.md):

  fixed     the profiles of profiles/phase01 (p01_*): one extractor (the Phase 00 winner) and
            one classifier configuration for every fold. Tables phase01_*.csv, as above.
  per-fold  the profiles of profiles/perfold (pf01_*): every outer fold chooses its extractor
            (D_pen on its training speakers) and the classifier's hyperparameters (inner
            search). The same five sets, without the CV-scheme and raw/standardized axes (the
            selection replaces nested CV; raw features stay a fixed-grid ablation). Tables
            perfold_*.csv, the same five, where the extractor column of perfold_extractors.csv
            reports the choice most folds made and how many made it, plus:
              perfold_selection.csv   per cell: the folds' extractor and grid choices (the
                                      stability §2 of the protocol asks to report)
              perfold_vs_fixed.csv    per cell: the per-fold EER beside the fixed one and their
                                      difference -- descriptive, no test (none is declared)
            It needs --fixed-profiles-dir and --fixed-results-dir for the second table.

The summary's "protocol" must agree with the profile: "per-fold" for a per-fold profile; a fixed
profile's summary says "fixed" or, written before the field existed (2026-11-18), nothing --
every earlier binary ran the fixed protocol only.

Usage:
    python3 scripts/pipeline/thesis/thesis_build_phase01_auth_tables.py --protocol fixed \\
        --profiles-dir src/experiments/thesis/profiles/phase01 \\
        --results-dir  results/thesis/phase01 \\
        --tables-dir   ../../documentation/00-thesis/monography/tables
    python3 scripts/pipeline/thesis/thesis_build_phase01_auth_tables.py --protocol per-fold \\
        --profiles-dir src/experiments/thesis/profiles/perfold \\
        --results-dir  results/thesis/perfold \\
        --fixed-profiles-dir src/experiments/thesis/profiles/phase01 \\
        --fixed-results-dir  results/thesis/phase01 \\
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
import importlib.util

SET_PATTERNS = {
    "fixed": {
        "grid": re.compile(r"^p01_dsnn_([a-z]+)_(dep|indep)_(flat|nested)_(raw|std)$"),
        "ann": re.compile(r"^p01_ann_([a-z]+)_(dep|indep)_flat_std$"),
        "cosine": re.compile(r"^p01_cosine_([a-z]+)_(dep|indep)_flat_std$"),
        "dysph": re.compile(r"^p01_dysph_([a-z]+)_(probes|test)_l(\d+)$"),
        "cmp": re.compile(r"^p01_cmp-(hc|ae|pca)_([a-z]+)_(dep|indep)$"),
    },
    "per-fold": {
        "grid": re.compile(r"^pf01_dsnn_([a-z]+)_(dep|indep)$"),
        "ann": re.compile(r"^pf01_ann_([a-z]+)_(dep|indep)$"),
        "cosine": re.compile(r"^pf01_cosine_([a-z]+)_(dep|indep)$"),
        "dysph": re.compile(r"^pf01_dysph_([a-z]+)_(probes|test)_l(\d+)$"),
        "cmp": re.compile(r"^pf01_cmp-(hc|ae|pca)_([a-z]+)_(dep|indep)$"),
    },
}
TABLE_PREFIX = {"fixed": "phase01", "per-fold": "perfold"}
FAMILY_OF_STRATEGY = {"handcrafted": "hc", "autoencoder": "ae", "pca": "pca"}
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


def fold_choice(prof: dict, fold: dict, f: pathlib.Path) -> dict:
    """One fold's choices under the per-fold protocol: the extractor (labelled from the
    profile's own candidate, so the label carries wavelet and scale) and the grid point."""
    if "selection" not in fold:
        fail(f"{f}: fold {fold.get('fold')} has no selection record; not a per-fold run")
    sel = fold["selection"]
    cands = prof["selection"]["extractor_candidates"]
    idx = sel["extractor_index"]
    if not 0 <= idx < len(cands) or len(sel["extractors"]) != len(cands):
        fail(f"{f}: fold {fold['fold']} ranks {len(sel['extractors'])} extractors and chose "
             f"#{idx}; the profile lists {len(cands)} candidates. The result is not this "
             f"profile's run.")
    choice = {"extractor": extractor_label(cands[idx]["feature_extraction"]), "grid": None}
    if "grid_index" in sel:
        g = sel["grid"][sel["grid_index"]]
        choice["grid"] = (f"{g['hidden_width']}/{g['residual_blocks']}/"
                          f"{g['learning_rate']:.0e}/{g['weight_decay']:.0e}")
    elif prof["classifier"]["type"] != "cosine":
        fail(f"{f}: fold {fold['fold']} of a {prof['classifier']['type']} run has no grid choice")
    return choice


def read_runs(prof: dict, results_dir: pathlib.Path, missing: list, protocol: str) -> dict | None:
    """Mean/std of EER, AUC and (networks) energy over the profile's repetitions; under the
    per-fold protocol also every fold's choices."""
    files = expected_summaries(results_dir, prof)
    absent = [f for f in files if not f.is_file()]
    if absent:
        missing.extend(absent)
        return None
    eers, aucs, energies, choices = [], [], [], []
    for f in files:
        summary = json.loads(f.read_text(encoding="utf-8"))
        written = summary.get("protocol")
        if (protocol == "per-fold") != (written == "per-fold") or written not in (None, "fixed",
                                                                                  "per-fold"):
            fail(f"{f}: the summary says protocol {written!r}; the tables were asked for the "
                 f"{protocol} protocol")
        results = summary["results"]
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
        if protocol == "per-fold":
            choices += [fold_choice(prof, fold, f) for fold in r["fold_models"]]

    def spread(v):
        return statistics.pstdev(v) if len(v) > 1 else None

    return {"n_reps": len(files),
            "mean_eer": statistics.mean(eers), "std_eer": spread(eers),
            "mean_auc": statistics.mean(aucs), "std_auc": spread(aucs),
            "nj": statistics.mean(e["nj"] for e in energies) if energies else None,
            "energy": ({k: statistics.mean(e[k] for e in energies) for k in energies[0]}
                       if energies else None),
            "choices": choices}


def collect(profiles_dir: pathlib.Path, results_dir: pathlib.Path, protocol: str) -> dict:
    profiles = sorted(profiles_dir.glob("*.json"))
    if not profiles:
        fail(f"no profiles in {profiles_dir}")
    patterns = SET_PATTERNS[protocol]
    sets: dict[str, dict] = {k: {} for k in patterns}
    missing: list[pathlib.Path] = []
    for path in profiles:
        name = path.stem
        prof = json.loads(path.read_text(encoding="utf-8"))
        if not prof["classifier"]["enabled"]:
            fail(f"{path}: classifier.enabled is false — not a Phase 01 profile")
        if prof["selection"]["enabled"] != (protocol == "per-fold"):
            fail(f"{path}: selection.enabled is {prof['selection']['enabled']}; the tables were "
                 f"asked for the {protocol} protocol")
        matched = [(k, m) for k, p in patterns.items() if (m := p.match(name))]
        if len(matched) != 1:
            fail(f"{path}: the name belongs to no {protocol} Phase 01 set (or to several); the "
                 f"tables are defined by the generated sets only")
        kind, m = matched[0]
        src = source_of(prof)
        tr, clf = prof["training"], prof["classifier"]
        if kind == "grid" and protocol == "per-fold":
            check(name, "source", src, m[1])
            check(name, "text mode", clf["text_mode"], TEXT_MODE[m[2]])
            check(name, "standardize", tr["standardize_features"], True)
            check(name, "classifier", clf["type"], "dsnn")
            key = (src, m[2])
        elif kind == "grid":
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
            if protocol == "per-fold":
                fams = {FAMILY_OF_STRATEGY[c["feature_extraction"]["strategy"]]
                        for c in prof["selection"]["extractor_candidates"]}
                if len(fams) != 1:
                    fail(f"{path}: an extractor comparison whose candidates span {sorted(fams)}")
                fam = fams.pop()
            else:
                fam = FAMILY_OF_STRATEGY[prof["feature_extraction"]["strategy"]]
            check(name, "extractor family", fam, m[1])
            key = (src, m[3], m[1])
        if key in sets[kind]:
            fail(f"{path}: a second profile for {kind} cell {key}")
        runs = read_runs(prof, results_dir, missing, protocol)
        if runs is not None:
            runs["profile"] = prof
            sets[kind][key] = runs
    if missing:
        listed = "\n  ".join(str(p) for p in missing[:20])
        more = f"\n  ... and {len(missing) - 20} more" if len(missing) > 20 else ""
        fail(f"{len(missing)} expected summary file(s) missing:\n  {listed}{more}\n"
             f"Run those Phase 01 profiles first.")
    return sets


def with_protocol(sets: dict, protocol: str) -> dict:
    """The sets plus the protocol the table functions read ("protocol" is not a set: collect's
    callers elsewhere iterate over the sets, so it is added here only)."""
    return {**sets, "protocol": protocol}


def f4(x: float | None) -> str:
    return "—" if x is None else f"{x:.4f}"


def write(out: pathlib.Path, header: list[str], rows: list[list]) -> None:
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print(f"[phase01-tables] wrote {out} ({len(rows)} rows)")


def table_path(sets: dict, out_dir: pathlib.Path, name: str) -> pathlib.Path:
    return out_dir / f"{TABLE_PREFIX[sets['protocol']]}_{name}.csv"


def configuration(key: tuple) -> str:
    """The grid cell's label: (source, text) under the per-fold protocol, plus the CV scheme
    and the raw/standardized axis under the fixed one."""
    parts = [SOURCE_LABELS[key[0]], TEXT_LABELS[key[1]]]
    if len(key) == 4:
        parts += [CV_LABELS[key[2]], STD_LABELS[key[3]]]
    return " · ".join(parts)


def dsnn_cell(sets: dict, source: str, text: str, what: str) -> dict:
    """The DSNN cell the controls, dysphonia and energy tables compare with: flat CV,
    standardized features under the fixed protocol; the only one under the per-fold one."""
    key = (source, text) if sets["protocol"] == "per-fold" else (source, text, "flat", "std")
    return require(sets["grid"], key, what)


def grid_table(sets: dict, out_dir: pathlib.Path) -> None:
    rows = sorted(sets["grid"].items(), key=lambda kv: kv[1]["mean_eer"])
    write(table_path(sets, out_dir, "auth"),
          ["rank", "configuration", "mean_eer", "std_eer", "mean_auc", "std_auc"],
          [[i, configuration(key), f4(r["mean_eer"]), f4(r["std_eer"]), f4(r["mean_auc"]),
            f4(r["std_auc"])]
           for i, (key, r) in enumerate(rows, start=1)])


def require(table: dict, key: tuple, what: str) -> dict:
    if key not in table:
        fail(f"{what}: no profile for cell {key}; the set is incomplete")
    return table[key]


def controls_table(sets: dict, out_dir: pathlib.Path) -> None:
    rows = []
    for s in SOURCE_ORDER:
        for t in ("dep", "indep"):
            d = dsnn_cell(sets, s, t, "controls (DSNN)")
            a = require(sets["ann"], (s, t), "controls (ANN)")
            c = require(sets["cosine"], (s, t), "controls (cosine)")
            rows.append([SOURCE_LABELS[s], TEXT_LABELS[t], f4(d["mean_eer"]), f4(d["std_eer"]),
                         f4(a["mean_eer"]), f4(a["std_eer"]), f4(c["mean_eer"]),
                         f"{d['nj']:.2f}", f"{a['nj']:.2f}", f"{d['nj'] / a['nj']:.2f}"])
    write(table_path(sets, out_dir, "controls"),
          ["source", "text", "dsnn_eer", "dsnn_std", "ann_eer", "ann_std", "cosine_eer",
           "dsnn_nj", "ann_nj", "dsnn_over_ann"], rows)


def dysphonia_table(sets: dict, out_dir: pathlib.Path) -> None:
    sources = sorted({k[0] for k in sets["dysph"]}, key=SOURCE_ORDER.index)
    if not sources:
        fail("dysphonia: no dysphonia profile (p01_dysph_* / pf01_dysph_*)")
    rows = []
    for s in sources:
        clean = dsnn_cell(sets, s, "dep", "dysphonia (level 0)")
        for scope in ("probes", "test"):
            cells = [require(sets["dysph"], (s, scope, lv), "dysphonia") for lv in LEVELS]
            rows.append([SOURCE_LABELS[s], SCOPE_LABELS[scope], f4(clean["mean_eer"])] +
                        [f4(c["mean_eer"]) for c in cells])
    write(table_path(sets, out_dir, "dysphonia"),
          ["source", "scope", "eer_l0"] + [f"eer_l{lv}" for lv in LEVELS], rows)


def _phase00_tables():
    path = pathlib.Path(__file__).resolve().parent / "thesis_build_phase00_paraconsistent_tables.py"
    spec = importlib.util.spec_from_file_location("phase00_tables", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


P00 = _phase00_tables()


def extractor_label(fe: dict) -> str:
    if fe["strategy"] == "handcrafted":
        hc = fe["handcrafted"]
        category = "c2" if hc["cepstral"] else "c1"
        scale = (P00.C1_SCALE if category == "c1" else P00.C2_SCALE)[hc["scale"]]
        return f"{P00.wavelet_label(hc['wavelet'])} + {scale} ({category})"
    if fe["strategy"] == "autoencoder":
        # The Phase 00 tables' label (thesis_build_phase00_paraconsistent_tables.py), so one
        # extractor reads the same in both chapters: size name from the latent width.
        ae = fe["autoencoder"]
        latent = int(ae["encoder_layer_spec"][-1].split(":")[1])
        sizes = {8: "tiny", 16: "small", 32: "base"}
        if latent not in sizes:
            raise SystemExit(f"[auth-tables] ERROR: autoencoder latent {latent} is not one of "
                             f"the Phase 00 sizes {sorted(sizes)}")
        if ae["model"] == "ann-ae":
            return f"ANN-AE ({sizes[latent]})"
        if ae["model"] == "snn-ae":
            return f"SNN-AE {ae['encoding']} ({sizes[latent]})"
        raise SystemExit(f"[auth-tables] ERROR: unknown autoencoder model {ae['model']}")
    return f"PCA (k = {fe['pca']['n_components']})"


MAX_TIED_LISTED = 2


def most_chosen(values: list) -> str:
    """The value most folds chose and how many: "<value> (n/N)". A tie is never broken
    silently: up to MAX_TIED_LISTED tied values are listed ("a = b (n/N)"); more are counted
    ("5 empatados (2/15 cada)"), since a list of 15 one-fold winners says only "no mode"."""
    counts: dict = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    top = max(counts.values())
    tied = sorted(v for v, n in counts.items() if n == top)
    if len(tied) > MAX_TIED_LISTED:
        return f"{len(tied)} empatados ({top}/{len(values)} cada)"
    return f"{' = '.join(tied)} ({top}/{len(values)})"


def extractor_of(sets: dict, r: dict) -> str:
    """The extractor a cell used: the profile's own (fixed), or the folds' choice (per-fold)."""
    if sets["protocol"] == "per-fold":
        return most_chosen([c["extractor"] for c in r["choices"]])
    return extractor_label(r["profile"]["feature_extraction"])


def selection_table(sets: dict, out_dir: pathlib.Path, stem: str = "perfold_selection") -> None:
    """Per cell of the per-fold protocol: what its folds chose (protocol §2: the stability of
    the choices is reported, not hidden). Grid point = width/blocks/lr/weight decay. `stem`
    names the file; the BIOMEX-DB builder passes its own and has no extractor-comparison set."""
    rows = []
    cells = [("dsnn", k, r) for k, r in sets["grid"].items()] + \
            [(kind, k, r) for kind in ("ann", "cosine") for k, r in sets[kind].items()] + \
            [(f"cmp-{k[2]}", k[:2], r) for k, r in sets.get("cmp", {}).items()]
    order = {"dsnn": 0, "ann": 1, "cosine": 2, "cmp-hc": 3, "cmp-ae": 4, "cmp-pca": 5}
    cells.sort(key=lambda c: (order[c[0]], SOURCE_ORDER.index(c[1][0]), c[1][1]))
    for kind, (s, t), r in cells:
        ext = [c["extractor"] for c in r["choices"]]
        grid = [c["grid"] for c in r["choices"] if c["grid"] is not None]
        rows.append([kind, SOURCE_LABELS[s], TEXT_LABELS[t], len(ext), most_chosen(ext),
                     len(set(ext)), most_chosen(grid) if grid else "—",
                     len(set(grid)) if grid else "—"])
    write(out_dir / f"{stem}.csv",
          ["classifier", "source", "text", "folds", "extractor", "n_extractors", "grid_point",
           "n_grid_points"], rows)


def vs_fixed_table(sets: dict, fixed: dict, out_dir: pathlib.Path) -> None:
    """The per-fold EER of every DSNN/ANN/cosine cell beside the fixed-configuration one
    (flat CV, standardized: the cell it mirrors). Descriptive only: no test is declared."""
    rows = []
    for kind in ("dsnn", "ann", "cosine"):
        for s in SOURCE_ORDER:
            for t in ("dep", "indep"):
                if kind == "dsnn":
                    p, f = dsnn_cell(sets, s, t, "vs fixed"), dsnn_cell(fixed, s, t, "vs fixed")
                else:
                    p = require(sets[kind], (s, t), f"vs fixed ({kind}, per-fold)")
                    f = require(fixed[kind], (s, t), f"vs fixed ({kind}, fixed)")
                rows.append([kind, SOURCE_LABELS[s], TEXT_LABELS[t], f4(p["mean_eer"]),
                             f4(f["mean_eer"]), f"{p['mean_eer'] - f['mean_eer']:+.4f}"])
    write(table_path(sets, out_dir, "vs_fixed"),
          ["classifier", "source", "text", "perfold_eer", "fixed_eer", "difference"], rows)


def extractors_table(sets: dict, out_dir: pathlib.Path) -> None:
    rows = []
    for s in ("voice", "eegpron", "eegimag"):
        for t in ("dep", "indep"):
            row = [SOURCE_LABELS[s], TEXT_LABELS[t]]
            for fam in ("hc", "ae", "pca"):
                r = require(sets["cmp"], (s, t, fam), "extractor comparison")
                row += [extractor_of(sets, r), f4(r["mean_eer"]), f4(r["std_eer"])]
            rows.append(row)
    header = ["signal", "text"]
    for fam in ("hc", "ae", "pca"):
        header += [f"{fam}_extractor", f"{fam}_eer", f"{fam}_std"]
    write(table_path(sets, out_dir, "extractors"), header, rows)


ENERGY_CHECKS = (("nj", "primária"), ("nj_as_executed", "como executada"),
                 ("nj_sram8k", "SRAM 8 KB"), ("nj_sram32k", "SRAM 32 KB"),
                 ("nj_sram1m", "SRAM 1 MB"), ("ops_nj", "só aritmética"))


def energy_table(sets: dict, out_dir: pathlib.Path) -> None:
    rows = []
    for s in SOURCE_ORDER:
        d = dsnn_cell(sets, s, "dep", "energy (DSNN)")["energy"]
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
    write(table_path(sets, out_dir, "energy"), header, rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the Phase 01 CSV tables.")
    parser.add_argument("--protocol", required=True, choices=sorted(SET_PATTERNS))
    parser.add_argument("--profiles-dir", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--tables-dir", required=True)
    parser.add_argument("--fixed-profiles-dir",
                        help="per-fold only, required there: the fixed grid's profiles")
    parser.add_argument("--fixed-results-dir",
                        help="per-fold only, required there: the fixed grid's results")
    args = parser.parse_args()
    fixed_args = (args.fixed_profiles_dir, args.fixed_results_dir)
    if args.protocol == "per-fold" and None in fixed_args:
        fail("--protocol per-fold needs --fixed-profiles-dir and --fixed-results-dir "
             "(perfold_vs_fixed.csv compares every cell with the fixed grid's)")
    if args.protocol == "fixed" and fixed_args != (None, None):
        fail("--fixed-profiles-dir/--fixed-results-dir belong to --protocol per-fold; the fixed "
             "tables read no other set")

    sets = with_protocol(collect(pathlib.Path(args.profiles_dir), pathlib.Path(args.results_dir),
                                 args.protocol), args.protocol)
    out_dir = pathlib.Path(args.tables_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    grid_table(sets, out_dir)
    controls_table(sets, out_dir)
    dysphonia_table(sets, out_dir)
    extractors_table(sets, out_dir)
    energy_table(sets, out_dir)
    if args.protocol == "per-fold":
        fixed = with_protocol(collect(pathlib.Path(args.fixed_profiles_dir),
                                      pathlib.Path(args.fixed_results_dir), "fixed"), "fixed")
        selection_table(sets, out_dir)
        vs_fixed_table(sets, fixed, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
