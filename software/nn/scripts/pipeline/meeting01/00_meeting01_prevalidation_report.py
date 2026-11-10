#!/usr/bin/env python3
"""00_meeting01_prevalidation_report.py -- plan and report of the baseline pre-validation.

Why it exists. The grid (meeting01-loso.json) compared the SNN-AE with its matched control
(ann-ae), PCA, the mean frame AND three searched sequence autoencoders (LSTM-AE, GRU-AE,
Transformer-AE). On 2026-11-02 the author removed the three. A removal is a design decision
the paper has to defend with evidence, not with "they were slow": this script produces that
evidence, from runs of the same binary, with the same windows and training, on the folds the
paper already sets aside for design decisions (DEVELOPMENT_FOLDS: fsdd fold 0, eegmmidb fold 0).

What it measures, per family and dataset (each family through the SAME reduced NSGA-II
budget, meeting01-prevalidation.json):

    quality         test MSE of the family's winner (mean, sd over seeds), next to PCA's
    training cost   wall time of the family's fold, from its fold_begin to its fold_end event
                    (search, re-scoring, final fit, test; dataset loading, ~1 min on eegmmidb,
                    excluded), and its ratio to the SNN-AE's on the same fold; the winner's
                    final fit in ms per training window and epoch
    inference cost  the cost of processing data after training: the winner's forward pass in
                    ms per window (measured), and its energy per window in nJ (analytic,
                    memory-aware: Lemaire et al. 2022's operations + addressing + memory model
                    at Horowitz 2014's 45 nm prices, Meeting01Energy.hpp -- the paper's number),
                    with the two checks the paper prints beside it: the same counts priced at
                    one on-chip memory size (8 KB, 32 KB, 1 MB), and arithmetic alone

Usage (from software/nn; 00_meeting01_prevalidate_baselines.sh calls the first two):
    00_meeting01_prevalidation_report.py --plan --profile P --out-root D
        writes D/<family>/profile.json, prints one "family dataset fold profile" task per line
    00_meeting01_prevalidation_report.py --report --profile P --out-root D --data-dir X
        D/prevalidation_summary.csv, X/paper_prevalidation.tex, X/paper_prevalidation_config.tex
    00_meeting01_prevalidation_report.py --placeholders --profile P --data-dir X
        the same two paper files before the run, every cell "--"
    00_meeting01_prevalidation_report.py --self-test

Refuses, naming cause and remedy (no fallbacks): a profile that is not a pre-validation (a
family missing, a dataset without a development fold, a headline fold), a task that did not
finish or exited non-zero, a fold of another results_format, families run under different
configurations, a missing seed, a selection manifest without the final_fit record (binary
older than 2026-11-02), and a family dir with results of a model it does not run.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import pathlib
import shutil
import sys
import tempfile

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
from meeting01_results import (  # noqa: E402
    DATASET_SHORT,
    DEVELOPMENT_FOLDS,
    ENERGY_CHECKS,
    ENERGY_CHECKS_HOLD,
    ENERGY_COLUMNS,
    PW_HEADER,
    READS_WITHOUT_ANN_SHARE,
    REFERENCE_SEED,
    RESULTS_FORMAT,
    ResultsError,
    folds_text,
    read_per_window,
    split_manifest,
)

#: One process per family: the evaluation.baselines it runs and whether it runs the SNN arm.
#: ann-ae copies the SNN-AE winner of the same run, so it lives in the SNN-AE's process.
FAMILIES = {
    "snn-ae": (["ann-ae"], True),
    "lstm-ae": (["lstm-ae"], False),
    "gru-ae": (["gru-ae"], False),
    "transformer-ae": (["transformer-ae"], False),
}
#: Table row order and labels; the three the grid no longer runs come last.
MODELS = ["snn-ae", "ann-ae", "lstm-ae", "gru-ae", "transformer-ae"]
REMOVED = ["lstm-ae", "gru-ae", "transformer-ae"]
LABEL = {"snn-ae": "SNN-AE", "ann-ae": "ANN-AE", "lstm-ae": "LSTM-AE", "gru-ae": "GRU-AE",
         "transformer-ae": "Transformer-AE"}
PROCESS_OF = {"snn-ae": "snn-ae", "ann-ae": "snn-ae", "lstm-ae": "lstm-ae",
              "gru-ae": "gru-ae", "transformer-ae": "transformer-ae"}
_NONE = "--"
_COMPARATIVE = "comparative_metrics.csv"
# Recorded-config fields that differ between the family processes by design.
_PER_FAMILY_FIELDS = (("dataset", "cv_fold"), ("dataset", "results_dir"),
                      ("evaluation", "datasets"), ("evaluation", "baselines"),
                      ("evaluation", "snn_architectures"), ("model", "latent_dim"))


# ----------------------------------------------------------------------------- plan

def load_profile(path: pathlib.Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def plan_tasks(profile: dict) -> list[tuple[str, int]]:
    """(dataset, fold) pairs of the pre-validation: every development fold of every
    profile dataset. Refuses a profile that is not a whole pre-validation."""
    ev = profile["evaluation"]
    want = sorted(b for fam in FAMILIES.values() for b in fam[0])
    if sorted(ev["baselines"]) != want or not ev["snn_architectures"]:
        raise ResultsError(
            f"{profile.get('_profile_name', 'the profile')} runs baselines {ev['baselines']} "
            f"and SNN architectures {ev['snn_architectures']}: a pre-validation runs every "
            f"family ({want} and the SNN arm), or it compares a subset.")
    tasks = []
    for ds in ev["datasets"]:
        if ds not in DEVELOPMENT_FOLDS:
            raise ResultsError(
                f"{ds} has no development fold (DEVELOPMENT_FOLDS in meeting01_results.py): "
                "the pre-validation would look at a headline fold's test windows. Run it on "
                "the development folds only.")
        tasks += [(ds, f) for f in sorted(DEVELOPMENT_FOLDS[ds])]
    missing = sorted(set(DEVELOPMENT_FOLDS) - set(ev["datasets"]))
    if missing:
        raise ResultsError(f"the profile leaves out the development folds of {missing}: add "
                           "them to evaluation.datasets so the evidence covers both domains.")
    return tasks


def write_plan(profile_path: pathlib.Path, out_root: pathlib.Path) -> list[str]:
    profile = load_profile(profile_path)
    tasks = plan_tasks(profile)
    lines = []
    for fam, (baselines, snn) in FAMILIES.items():
        derived = copy.deepcopy(profile)
        derived["_derived_from"] = (f"{profile_path.name}, narrowed to the {fam} process by "
                                    "00_meeting01_prevalidation_report.py --plan")
        derived["evaluation"]["baselines"] = list(baselines)
        if not snn:
            derived["evaluation"]["snn_architectures"] = []
        derived["dataset"]["results_dir"] = str(out_root / fam)
        (out_root / fam).mkdir(parents=True, exist_ok=True)
        path = out_root / fam / "profile.json"
        path.write_text(json.dumps(derived, indent=4) + "\n", encoding="utf-8")
        lines += [f"{fam}\t{ds}\t{fold}\t{path}" for ds, fold in tasks]
    return lines


# --------------------------------------------------------------------------- report

def _without(config: dict, fields) -> dict:
    out = json.loads(json.dumps(config))
    for section, key in fields:
        out.get(section, {}).pop(key, None)
    return out


def _seeds(profile: dict) -> dict[int, int]:
    exp = profile["experiment"]
    return {r: (exp["seed"] if exp["seed_deterministic"] else exp["seed"] + r - 1)
            for r in range(1, exp["repeats"] + 1)}


def _test_rows(path: pathlib.Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return [r for r in csv.DictReader(f) if r["split"] == "test"]


def _finite(x, where: str) -> float:
    v = float(x)
    if not math.isfinite(v):
        raise ResultsError(f"{where} is {x}: the model diverged; the pre-validation cannot "
                           "report it as a cost or an error. Rerun the task.")
    return v


def _energy(row: dict, column: str, where: str) -> float:
    """One energy column of a test row. Not a divergence when it is missing or not finite: the
    energy is analytic (counted from the architecture, Meeting01Energy.hpp), so a NaN there is
    a model kind the binary did not price, and a missing column a binary older than the format
    that introduced it (7 for the memory-aware energy, 8 for energy_nj_as_executed)."""
    if column not in row:
        raise ResultsError(f"{where}: the test row has no {column} column; the binary that wrote "
                           "it predates this energy column (memory-aware energy: results_format "
                           "7; energy_nj_as_executed: 8). Rebuild meeting01 and rerun the task.")
    v = float(row[column])
    if not math.isfinite(v) or v < 0:
        raise ResultsError(f"{where}: {column} is {row[column]}; the binary did not price this "
                           "model's inference (set_energy, Meeting01Energy.hpp, was not called "
                           "for it). Fix the binary and rerun the task.")
    return v


def _fold_seconds(events: pathlib.Path, where: str) -> float:
    """Seconds from the fold's fold_begin event to its fold_end: every search, final fit and
    test of the fold, without loading the dataset. Loading is the same for every family (about
    a minute on eegmmidb), so counted in it would pull every ratio towards 1."""
    stamps: dict = {"fold_begin": [], "fold_end": []}
    for line in events.read_text(encoding="utf-8").splitlines():
        e = json.loads(line)
        if e.get("type") in stamps:
            stamps[e["type"]].append(float(e["ts_unix"]))
    if len(stamps["fold_begin"]) != 1 or len(stamps["fold_end"]) != 1:
        raise ResultsError(f"{where}: {events.name} has {len(stamps['fold_begin'])} fold_begin "
                           f"and {len(stamps['fold_end'])} fold_end events, expected one each "
                           "(a resumed task appends a second session: rerun it fresh).")
    return stamps["fold_end"][0] - stamps["fold_begin"][0]


def collect(profile: dict, out_root: pathlib.Path) -> tuple[list[dict], dict, dict]:
    """([per (model, dataset, run) record], {(dataset, family): (search s, process wall s)},
    {dataset: PCA test MSE})."""
    tag = profile["experiment"]["run_tag"]
    seeds = _seeds(profile)
    tasks = plan_tasks(profile)
    records, walls, pca = [], {}, {}
    reference_config, reference_from = None, None
    for fam, (baselines, snn) in FAMILIES.items():
        d = out_root / fam
        models = (["snn-ae"] if snn else []) + baselines
        for ds, fold in tasks:
            where = f"{fam} {ds} fold {fold}"
            proc_path = d / f"{ds}_fold{fold}_process.json"
            if not proc_path.exists():
                raise ResultsError(f"{where}: no {proc_path.name}, so the task did not finish "
                                   "(00_meeting01_prevalidate_baselines.sh writes it last).")
            proc = json.loads(proc_path.read_text(encoding="utf-8"))
            if proc["exit"] != 0:
                raise ResultsError(f"{where} exited {proc['exit']}: see {proc['log']}; rerun "
                                   "it (RESUME=1) before reporting.")
            walls[(ds, fam)] = (_fold_seconds(d / f"{tag}_{ds}_fold{fold}_events.jsonl", where),
                                float(proc["wall_s"]))
            # 11 or 12: the report never reads the ann-ae's hidden_silent_share (format 12)
            man = split_manifest(d, tag, ds, fold, accept=READS_WITHOUT_ANN_SHARE)
            common = _without(man["config"], _PER_FAMILY_FIELDS)
            if reference_config is None:
                reference_config, reference_from = common, where
            elif common != reference_config:
                diff = sorted(f"{s}.{k}" for s in common for k in common[s]
                              if reference_config.get(s, {}).get(k) != common[s][k])
                raise ResultsError(f"{where} and {reference_from} ran under different "
                                   f"configurations ({diff}): the families would not be "
                                   "compared on equal terms. Rerun them from one profile.")
            if sorted(man["config"]["evaluation"]["baselines"]) != sorted(baselines):
                raise ResultsError(f"{where}: its fold recorded baselines "
                                   f"{man['config']['evaluation']['baselines']}, not {baselines}:"
                                   " the directory holds another process's results.")
            rows = _test_rows(d / f"{tag}_{ds}_fold{fold}_{_COMPARATIVE}")
            unexpected = sorted({r["model"] for r in rows} - set(models))
            if unexpected:
                raise ResultsError(f"{where}: test rows of {unexpected}, which this process "
                                   "does not run.")
            for model in models:
                for run, seed in seeds.items():
                    hit = [r for r in rows if r["model"] == model and int(r["run"]) == run
                           and int(r["seed"]) == seed]
                    if len(hit) != 1:
                        raise ResultsError(f"{where}: {len(hit)} test rows for {model} run "
                                           f"{run} (seed {seed}), expected 1.")
                    row = hit[0]
                    segment = row["encoding"] if model == "snn-ae" else model
                    mpath = d / f"{tag}_{ds}_fold{fold}_{segment}_run{run}_model_selection_manifest.json"
                    sel = json.loads(mpath.read_text(encoding="utf-8"))
                    if "final_fit" not in sel:
                        raise ResultsError(
                            f"{mpath.name} has no final_fit record: written by a meeting01 "
                            "binary older than 2026-11-02, which did not record the final fit's "
                            "epochs and window counts. Rebuild and rerun the task.")
                    ff = sel["final_fit"]
                    if ff["epochs_run"] <= 0 or ff["fit_windows"] <= 0 or ff["monitor_windows"] <= 0:
                        raise ResultsError(f"{mpath.name}: final_fit {ff} has an empty count.")
                    w = f"{where} {model} run {run}"
                    records.append({
                        "model": model, "dataset": ds, "fold": fold, "run": run, "seed": seed,
                        "test_mse": _finite(row["mse"], f"{w} test MSE"),
                        "val_mse": sel["selected"].get("val_mse"),
                        **{k: _energy(row, k, w) for k in ENERGY_COLUMNS},
                        "macs": float(row["macs"]),
                        "params": int(row["param_count"]),
                        "train_ms_per_window_epoch": ff["train_ms"] / (ff["epochs_run"] * ff["fit_windows"]),
                        "infer_ms_per_window": ff["monitor_infer_ms"] / ff["monitor_windows"],
                        "epochs_run": ff["epochs_run"],
                        "selected": sel["selected"],
                    })
            if fam == "snn-ae":
                pw = read_per_window(d / f"{tag}_{ds}_fold{fold}_per_window_errors.csv", fold)
                windows = {}
                for r in pw:
                    if r["model"] == "pca" and r["split"] == "test":
                        windows[r["window_id"]] = float(r["mse"])
                if not windows:
                    raise ResultsError(f"{where}: no PCA rows; run 03_ on {d} first.")
                pca[ds] = sum(windows.values()) / len(windows)
    return records, walls, pca


def _mean(xs) -> float:
    return sum(xs) / len(xs)


def _sd(xs) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def summarize(records, walls, pca, datasets) -> list[dict]:
    out = []
    for ds in datasets:
        snn_search = walls[(ds, "snn-ae")][0]
        for model in MODELS:
            rs = [r for r in records if r["model"] == model and r["dataset"] == ds]
            search, wall = walls[(ds, PROCESS_OF[model])]
            vals = [r["val_mse"] for r in rs if r["val_mse"] is not None]
            out.append({
                "dataset": ds, "model": model, "n_seeds": len(rs),
                "test_mse_mean": _mean([r["test_mse"] for r in rs]),
                "test_mse_sd": _sd([r["test_mse"] for r in rs]),
                "val_mse_mean": _mean(vals) if vals else float("nan"),
                "pca_test_mse": pca[ds],
                "fold_wall_h": search / 3600.0,
                "process_wall_h": wall / 3600.0,
                "fold_wall_ratio_vs_snn": search / snn_search,
                "train_ms_per_window_epoch": _mean([r["train_ms_per_window_epoch"] for r in rs]),
                "infer_ms_per_window": _mean([r["infer_ms_per_window"] for r in rs]),
                **{f"{k}_per_window": _mean([r[k] for r in rs]) for k in ENERGY_COLUMNS},
                "energy_nj_half_mem_per_window": _mean(
                    [r["energy_ops_nj"] + 0.5 * r["energy_mem_nj"] for r in rs]),
                "macs_per_window": _mean([r["macs"] for r in rs]),
                "params": _mean([r["params"] for r in rs]),
                "epochs_run": _mean([r["epochs_run"] for r in rs]),
                "winners": json.dumps([r["selected"] for r in rs], sort_keys=True),
            })
    return out


# ---------------------------------------------------------------------------- write

def _g(x: float) -> str:
    """A cost for the table: whole numbers from 100 up, one decimal from 10, two below
    (never the exponent form %g falls into, 1e+02)."""
    return f"{x:.0f}" if x >= 100 else f"{x:.1f}" if x >= 10 else f"{x:.2f}"


def write_table(summary: list[dict] | None, datasets: list[str], data_dir: pathlib.Path) -> None:
    """paper_prevalidation.tex: per development fold, every family and PCA. Fold: the
    family's fold_begin to fold_end (search, re-scoring, final fit, test; dataset loading
    excluded). ANN-AE is not searched and trains inside the SNN-AE's fold, so it has no fold
    time of its own and the SNN-AE's includes it -- a larger denominator, so every ratio
    errs in the SNN-AE's disfavour. One column wide: the dataset is a group row, the units a
    second header row."""
    lines = [r"% auto-generated by 00_meeting01_prevalidation_report.py -- do not edit",
             r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r" & Test & Fold & & Train & Infer. & $E$ \\",
             r"Model & MSE & (h) & $\times$SNN & (ms/win.) & (ms/win.) & (nJ) \\"]
    for ds in datasets:
        lines += [r"\midrule", rf"\multicolumn{{7}}{{l}}{{\emph{{{DATASET_SHORT.get(ds, ds)}}}}} \\"]
        for model in MODELS:
            s = next((x for x in (summary or []) if x["dataset"] == ds and x["model"] == model),
                     None)
            if s is None:
                lines.append(f"{LABEL[model]} & " + " & ".join([_NONE] * 6) + r" \\")
                continue
            searched = model != "ann-ae"
            lines.append(
                f"{LABEL[model]} & ${s['test_mse_mean']:.3f}\\pm{s['test_mse_sd']:.3f}$ & "
                + (f"{_g(s['fold_wall_h'])} & {_ratio(s['fold_wall_ratio_vs_snn'])} & " if searched
                   else f"{_NONE} & {_NONE} & ")
                + f"{_g(s['train_ms_per_window_epoch'])} & {_g(s['infer_ms_per_window'])} & "
                f"{s['energy_nj_per_window']:.0f} \\\\")
        pca = next((x["pca_test_mse"] for x in (summary or []) if x["dataset"] == ds), None)
        lines.append("PCA ($k=d$) & " + (_NONE if pca is None else f"{pca:.3f}")
                     + " & " + " & ".join([_NONE] * 5) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "paper_prevalidation.tex").write_text("\n".join(lines), encoding="utf-8")


def _verdict(summary: list[dict]) -> str:
    """Which removed family reached PCA's test error (mean over seeds), as the sentence the
    paper prints -- generated, so the text cannot claim more than the table shows."""
    hits = [s for s in summary if s["model"] in REMOVED
            and s["test_mse_mean"] <= s["pca_test_mse"]]
    if not hits:
        return ("none of the three reached the test error of PCA on either development fold")
    parts = [f"{LABEL[s['model']]} on {DATASET_SHORT.get(s['dataset'], s['dataset'])} "
             f"({s['test_mse_mean']:.3f} against PCA's {s['pca_test_mse']:.3f})" for s in hits]
    return "the test error of PCA was reached by " + ", ".join(parts)


def _ratio(x: float) -> str:
    return f"{x:.0f}" if x >= 10 else f"{x:.1f}" if x >= 1 else f"{x:.2f}"


def _cost_verdict(summary: list[dict]) -> str:
    """The cost sentence the paper prints, generated for the same reason as _verdict. One
    clause per cost: the training side (the fold's wall time) and the two after training (the
    measured time to reconstruct a window, the estimated memory-aware energy). Each clause gives
    the smallest ratio over the removed families and folds, so it says "at least" only when
    every one of them is the costlier, and names the exception otherwise. A last clause says
    whether the energy comparison -- which of each removed family and the SNN-AE needs less --
    is the same under every repricing (one memory size for every access, arithmetic alone),
    and names the first repricing that turns one around otherwise: an energy claim that
    holds only under one memory model is what a reviewer is right to reject."""
    removed = [s for s in summary if s["model"] in REMOVED]
    snn = {s["dataset"]: s for s in summary if s["model"] == "snn-ae"}

    def clause(key: str, what: str, verb: str, every: str, one: str) -> str:
        low = min(removed, key=lambda s: s[key] / snn[s["dataset"]][key])
        r = low[key] / snn[low["dataset"]][key]
        if r >= 1:
            return f"each {every} {verb} at least {_ratio(r)} times the SNN-AE's {what}"
        ds = DATASET_SHORT.get(low["dataset"], low["dataset"])
        return (f"the {LABEL[low['model']]} {one} on {ds} {verb} only {_ratio(r)} times the "
                f"SNN-AE's {what}")

    return ", ".join((
        clause("fold_wall_h", "wall time on the same fold", "took", "of the three families' runs",
               "run"),
        clause("infer_ms_per_window", "time to reconstruct a window", "took", "winner", "winner"),
        clause("energy_nj_per_window", "energy per window (operations and memory accesses)",
               "needed", "winner", "winner"),
        "and " + _energy_robustness(removed, snn)))


def _energy_robustness(removed: list[dict], snn: dict) -> str:
    """The clause on whether the energy comparison survives each repricing (ENERGY_CHECKS)."""
    def cheaper(s: dict, key: str) -> bool:
        return s[key] < snn[s["dataset"]][key]

    for column, what in ENERGY_CHECKS:
        key = f"{column}_per_window"
        for s in removed:
            if cheaper(s, key) != cheaper(s, "energy_nj_per_window"):
                ds = DATASET_SHORT.get(s["dataset"], s["dataset"])
                r = s[key] / snn[s["dataset"]][key]
                return (f"that energy comparison does not hold {what}: there the "
                        f"{LABEL[s['model']]} winner on {ds} needs {_ratio(r)} times the "
                        "SNN-AE's")
    return "that energy comparison holds " + ENERGY_CHECKS_HOLD


def write_macros(profile: dict, summary: list[dict] | None, data_dir: pathlib.Path) -> None:
    ga_all = profile["evaluation"]["ga"]
    ga, rnn, tf, mdl = ga_all["snn"], ga_all["lstm"], ga_all["transformer"], profile["model"]
    bounds = ("min_hidden", "max_hidden", "min_layers", "max_layers")
    if {k: rnn[k] for k in bounds} != {k: ga_all["gru"][k] for k in bounds}:
        raise ResultsError("LSTM-AE and GRU-AE search different bounds: the paper gives one range.")
    for key in ("lstm_frame_size", "recurrent_time_steps"):
        if key not in mdl:
            raise ResultsError(f"the profile does not declare model.{key}: the paper states the "
                               "sequence the three families read, so it must be in the profile.")
    seeds = sorted(set(_seeds(profile).values()))
    tasks = plan_tasks(profile)
    folds = " and ".join(f"{DATASET_SHORT.get(ds, ds)} fold {f}" for ds, f in tasks)
    macros = {
        "preDataStatus": "results" if summary else "placeholder",
        "preGaPop": str(ga["population_size"]),
        "preGaGen": str(ga["generations"]),
        "preGaEvals": str(ga["population_size"] * (1 + ga["generations"])),
        "preGaWinnerSeeds": str(ga["winner_seeds"]),
        "preRuns": str(profile["experiment"]["repeats"]),
        "preSeeds": folds_text(seeds),
        "preFolds": folds,
        "preFrame": str(mdl["lstm_frame_size"]),
        "preSeqLen": str(profile["dataset"]["window_size"] * mdl["recurrent_time_steps"]
                         // mdl["lstm_frame_size"]),
        "preRnnHidden": f"{rnn['min_hidden']}, {rnn['max_hidden']}",
        "preRnnLayers": f"{rnn['min_layers']}, {rnn['max_layers']}",
        "preTfDmodel": f"{tf['min_d_model']}, {tf['max_d_model']}",
        "preTfHeads": ", ".join(str(h) for h in tf["head_choices"]),
        "preTfLayers": f"{tf['min_layers']}, {tf['max_layers']}",
        "preTfDff": f"{tf['min_d_ff']}, {tf['max_d_ff']}",
    }
    if summary:
        macros["preCostVerdict"] = _cost_verdict(summary)
        macros["preQualityVerdict"] = _verdict(summary)
    else:
        macros["preCostVerdict"] = "[the pre-validation has not run yet]"
        macros["preQualityVerdict"] = "[the pre-validation has not run yet]"
    lines = ["% auto-generated by 00_meeting01_prevalidation_report.py -- do not edit"]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()]
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "paper_prevalidation_config.tex").write_text("\n".join(lines) + "\n",
                                                              encoding="utf-8")


def report(profile_path: pathlib.Path, out_root: pathlib.Path, data_dir: pathlib.Path) -> list:
    profile = load_profile(profile_path)
    datasets = list(dict.fromkeys(ds for ds, _ in plan_tasks(profile)))
    records, walls, pca = collect(profile, out_root)
    summary = summarize(records, walls, pca, datasets)
    with (out_root / "prevalidation_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(summary)
    write_table(summary, datasets, data_dir)
    write_macros(profile, summary, data_dir)
    return summary


def placeholders(profile_path: pathlib.Path, data_dir: pathlib.Path) -> None:
    profile = load_profile(profile_path)
    datasets = list(dict.fromkeys(ds for ds, _ in plan_tasks(profile)))
    write_table(None, datasets, data_dir)
    write_macros(profile, None, data_dir)


# ------------------------------------------------------------------------ self-test

def _synthetic_profile() -> dict:
    return {
        "_profile_name": "t",
        "experiment": {"run_tag": "t", "seed": 42, "repeats": 2, "seed_deterministic": False},
        "dataset": {"results_dir": "r", "cv_fold": 0, "window_size": 256},
        "training": {"epochs": 30},
        "model": {"latent_dim": 16, "lstm_frame_size": 8, "recurrent_time_steps": 1},
        "evaluation": {"datasets": ["fsdd", "eegmmidb"], "snn_architectures": ["dense"],
                       "baselines": ["ann-ae", "lstm-ae", "gru-ae", "transformer-ae"],
                       "ga": {"snn": {"population_size": 4, "generations": 2, "winner_seeds": 2},
                              "lstm": {"min_hidden": 8, "max_hidden": 256, "min_layers": 1,
                                       "max_layers": 3},
                              "gru": {"min_hidden": 8, "max_hidden": 256, "min_layers": 1,
                                      "max_layers": 3},
                              "transformer": {"min_d_model": 16, "max_d_model": 128,
                                              "head_choices": [1, 2, 4, 8], "min_layers": 1,
                                              "max_layers": 4, "min_d_ff": 32,
                                              "max_d_ff": 256}}},
    }


# (test MSE, energy nJ, train ms per window-epoch, fold wall s) of each model on fsdd / eegmmidb.
# The energy is the memory-aware energy_nj; its checks are fixed multiples of it (so every
# repricing keeps every ratio) unless a test's row_mutate changes one.
_SYN_ENERGY = {"energy_nj": 1.0, "energy_ops_nj": 0.4, "energy_mem_nj": 0.6, "mem_accesses": 0.05,
               "energy_nj_sram8k": 0.8, "energy_nj_sram32k": 1.1, "energy_nj_sram1m": 3.0,
               "arith_energy_nj": 0.3, "energy_nj_as_executed": 1.5}
_SYN = {"snn-ae": (0.78, 2000.0, 2.0, 3600.0), "ann-ae": (0.66, 150.0, 0.5, None),
        "lstm-ae": (0.87, 380000.0, 300.0, 360000.0), "gru-ae": (0.86, 290000.0, 250.0, 300000.0),
        "transformer-ae": (1.00, 500000.0, 400.0, 540000.0)}


def _write_synthetic(out_root: pathlib.Path, profile: dict, mutate=None, row_mutate=None) -> None:
    for fam, (baselines, snn) in FAMILIES.items():
        d = out_root / fam
        d.mkdir(parents=True, exist_ok=True)
        models = (["snn-ae"] if snn else []) + baselines
        for ds, fold in plan_tasks(profile):
            scale = 1.0 if ds == "fsdd" else 0.5
            cfg = copy.deepcopy(profile)
            cfg["evaluation"]["baselines"] = baselines
            cfg["evaluation"]["snn_architectures"] = ["dense"] if snn else []
            cfg["dataset"]["results_dir"] = str(d)
            if mutate:
                mutate(fam, cfg)
            stem = f"t_{ds}_fold{fold}"
            (d / f"{stem}_split_manifest.json").write_text(json.dumps(
                {"dataset": ds, "cv_fold": fold, "results_format": RESULTS_FORMAT,
                 "config": {k: v for k, v in cfg.items() if not k.startswith("_")}}))
            rows = []
            for run, seed in _seeds(profile).items():
                for model in models:
                    mse, nj, ms, _ = _SYN[model]
                    mse = mse * scale + 0.01 * (run - 1.5)
                    enc = "latency" if model == "snn-ae" else "direct"
                    row = {"model": model, "encoding": enc, "run": run, "seed": seed,
                           "split": "test", "mse": mse, "macs": nj / 4.6e-3, "param_count": 1000,
                           **{k: f * nj for k, f in _SYN_ENERGY.items()}}
                    if row_mutate:
                        row_mutate(ds, row)
                    rows.append(row)
                    segment = enc if model == "snn-ae" else model
                    (d / f"{stem}_{segment}_run{run}_model_selection_manifest.json").write_text(
                        json.dumps({"selected": {"val_mse": None if model == "ann-ae" else mse},
                                    "final_fit": {"fit_windows": 1000, "monitor_windows": 150,
                                                  "epochs_run": 10,
                                                  "train_ms": ms * 1000 * 10,
                                                  "monitor_infer_ms": ms * 150 / 3}}))
            with (d / f"{stem}_{_COMPARATIVE}").open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
                w.writeheader()
                w.writerows(rows)
            if snn:
                with (d / f"{stem}_per_window_errors.csv").open("w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=PW_HEADER, lineterminator="\n")
                    w.writeheader()
                    for i, e in enumerate((0.60 * scale, 0.70 * scale)):
                        for enc in ("direct", "latency"):
                            w.writerow({"model": "pca", "encoding": enc, "architecture": "pca",
                                        "v_th": 0, "alpha": 0, "run_id": 0,
                                        "seed": REFERENCE_SEED, "cv_fold": fold, "split": "test",
                                        "speaker_id": 0, "recording_id": i, "window_id": i,
                                        "source_window_index": 0, "mse": e, "mae": e})
            search = _SYN[fam][3] * scale
            (d / f"{stem}_events.jsonl").write_text("\n".join(json.dumps(e) for e in (
                {"type": "session_begin", "ts_unix": 1000.0},
                {"type": "fold_begin", "ts_unix": 1060.0},  # a minute of loading
                {"type": "config_begin", "ts_unix": 1060.0},
                {"type": "fold_end", "ts_unix": 1060.0 + search},
                {"type": "session_end", "ts_unix": 1060.0 + search})) + "\n")
            (d / f"{ds}_fold{fold}_process.json").write_text(json.dumps(
                {"family": fam, "dataset": ds, "fold": fold, "exit": 0, "wall_s": search + 60.0,
                 "log": "x.log"}))


def _self_test_checks(tmp: pathlib.Path) -> list[str]:
    failures: list[str] = []

    def expect(ok: bool, what: str) -> None:
        if not ok:
            failures.append(what)

    def expect_refusal(fn, needle: str, what: str) -> None:
        try:
            fn()
        except ResultsError as e:
            expect(needle in str(e), f"{what}: refused, but message lacks {needle!r}: {e}")
        else:
            failures.append(f"{what}: not refused")

    profile = _synthetic_profile()
    ppath = tmp / "profile.json"
    ppath.write_text(json.dumps(profile))

    # The plan: one profile per family process, narrowed, results kept apart.
    plan = write_plan(ppath, tmp / "plan")
    expect(len(plan) == 4 * 2, f"4 families x 2 development folds: {plan}")
    lstm = json.loads((tmp / "plan" / "lstm-ae" / "profile.json").read_text())
    expect(lstm["evaluation"]["baselines"] == ["lstm-ae"]
           and lstm["evaluation"]["snn_architectures"] == []
           and lstm["dataset"]["results_dir"] == str(tmp / "plan" / "lstm-ae"),
           f"lstm process profile: {lstm['evaluation']} {lstm['dataset']}")
    snn = json.loads((tmp / "plan" / "snn-ae" / "profile.json").read_text())
    expect(snn["evaluation"]["baselines"] == ["ann-ae"] and snn["evaluation"]["snn_architectures"],
           "the SNN process carries ann-ae, whose winner it copies")
    expect_refusal(lambda: plan_tasks(dict(profile, evaluation=dict(
        profile["evaluation"], datasets=["fsdd", "eegmmidb", "siena"]))),
        "no development fold", "a dataset without a development fold")
    expect_refusal(lambda: plan_tasks(dict(profile, evaluation=dict(
        profile["evaluation"], baselines=["ann-ae", "lstm-ae"]))),
        "every family", "a family left out")
    expect_refusal(lambda: plan_tasks(dict(profile, evaluation=dict(
        profile["evaluation"], datasets=["fsdd"]))), "leaves out", "eegmmidb left out")

    # The report on a complete synthetic run.
    out = tmp / "run"
    _write_synthetic(out, profile)
    data = tmp / "data"
    summary = report(ppath, out, data)
    by = {(s["dataset"], s["model"]): s for s in summary}
    lstm_f = by[("fsdd", "lstm-ae")]
    expect(abs(lstm_f["test_mse_mean"] - 0.87) < 1e-12 and abs(lstm_f["test_mse_sd"]
           - 0.01 * math.sqrt(0.5)) < 1e-12, f"lstm fsdd mse: {lstm_f}")
    # The ratio is of the folds (fold_begin to fold_end), not of the processes, whose
    # minute of dataset loading would have made it (360000 + 60) / (3600 + 60) = 98.4.
    expect(abs(lstm_f["fold_wall_ratio_vs_snn"] - 100.0) < 1e-9, f"search ratio: {lstm_f}")
    expect(abs(lstm_f["process_wall_h"] - (360000 + 60) / 3600) < 1e-9, f"process wall: {lstm_f}")
    expect(abs(lstm_f["train_ms_per_window_epoch"] - 300.0) < 1e-9
           and abs(lstm_f["infer_ms_per_window"] - 100.0) < 1e-9, f"per-step costs: {lstm_f}")
    expect(abs(by[("fsdd", "snn-ae")]["pca_test_mse"] - 0.65) < 1e-12, "PCA over windows")
    expect(math.isnan(by[("fsdd", "ann-ae")]["val_mse_mean"]), "ann-ae has no val_mse")
    table = (data / "paper_prevalidation.tex").read_text()
    expect("\\multicolumn{7}{l}{\\emph{FSDD}} \\\\\nSNN-AE & $0.780\\pm0.007$ & 1.00 & 1.0 & 2.00 & 0.67 & 2000 \\\\" in table,
           f"snn row: {table}")
    expect("\nANN-AE & $0.660\\pm0.007$ & -- & -- & 0.50 & 0.17 & 150 \\\\" in table,
           f"ann row (not searched): {table}")
    expect("\nLSTM-AE & $0.870\\pm0.007$ & 100 & 100 & 300 & 100 & 380000 \\\\" in table,
           f"lstm row: {table}")
    expect("\nTransformer-AE & $0.500\\pm0.007$ & 75.0 & 150 &" in table,
           f"eegmmidb transformer row: {table}")
    expect("\nPCA ($k=d$) & 0.650 & -- &" in table, f"pca row: {table}")
    macros = (data / "paper_prevalidation_config.tex").read_text()
    for needle in ("\\newcommand{\\preGaEvals}{12}", "\\newcommand{\\preSeeds}{42--43}",
                   # smallest ratios, all GRU-AE: 300000/3600 s, 83.3/0.667 ms, 290000/2000 nJ
                   ("\\newcommand{\\preCostVerdict}{each of the three families' runs took at least 83 "
                    "times the SNN-AE's wall time on the same fold, each winner took at least "
                    "125 times the SNN-AE's time to reconstruct a window, each winner "
                    "needed at least 145 times the SNN-AE's energy per window (operations and "
                    "memory accesses), and that energy comparison holds with every memory access "
                    "priced at one SRAM size, 8 KB, 32 KB or 1 MB, with every memory price halved, "
                    "with arithmetic alone, and with the "
                    "first layer of a direct-coded input recomputed at every time step, as the "
                    "code runs it}"),
                   "\\newcommand{\\preQualityVerdict}{none of the three reached",
                   "\\newcommand{\\preFolds}{FSDD fold 0 and EEGMMIDB fold 0}",
                   "\\newcommand{\\preSeqLen}{32}", "\\newcommand{\\preRnnHidden}{8, 256}",
                   "\\newcommand{\\preTfHeads}{1, 2, 4, 8}",
                   "\\newcommand{\\preDataStatus}{results}"):
        expect(needle in macros, f"macro {needle} missing: {macros}")

    # A removed family that reaches PCA, or costs less than the SNN-AE, is named, not hidden
    # behind a fixed sentence.
    _SYN["gru-ae"] = (0.60, 1000.0, 250.0, 300000.0)
    _write_synthetic(out, profile)
    report(ppath, out, data)
    macros = (data / "paper_prevalidation_config.tex").read_text()
    expect("reached by GRU-AE on FSDD (0.600 against PCA's 0.650)" in macros,
           f"verdict names the hit: {macros}")
    expect("the GRU-AE winner on FSDD needed only 0.50 times the SNN-AE's energy per window" in macros
           and "each winner took at least 125 times" in macros,
           f"cost verdict names a cheaper winner, and only for that cost: {macros}")
    _SYN["gru-ae"] = (0.86, 290000.0, 250.0, 300000.0)

    # An energy comparison that one repricing turns around is named, with the repricing: here
    # the GRU-AE on FSDD costs 145 times the SNN-AE's memory-aware energy, but under the
    # arithmetic-only count it needs 100 nJ against the SNN-AE's 0.3 x 2000 = 600.
    def cheap_gru_arith(ds, row):
        if row["model"] == "gru-ae" and ds == "fsdd":
            row["arith_energy_nj"] = 100.0
    _write_synthetic(out, profile, row_mutate=cheap_gru_arith)
    report(ppath, out, data)
    macros = (data / "paper_prevalidation_config.tex").read_text()
    expect("and that energy comparison does not hold with arithmetic alone: there the GRU-AE "
           "winner on FSDD needs 0.17 times the SNN-AE's}" in macros,
           f"robustness clause names the repricing that turns a comparison around: {macros}")
    expect("each winner needed at least 145 times" in macros,
           f"the primary energy clause is unchanged by a check: {macros}")
    summary_csv = (out / "prevalidation_summary.csv").read_text()
    expect("energy_nj_sram1m_per_window" in summary_csv and "arith_energy_nj_per_window" in summary_csv,
           f"the summary keeps every energy check: {summary_csv.splitlines()[0]}")
    _write_synthetic(out, profile)
    bad = copy.deepcopy(profile)
    bad["evaluation"]["ga"]["gru"]["max_hidden"] = 128
    expect_refusal(lambda: write_macros(bad, None, tmp / "badmacros"), "different bounds",
                   "LSTM and GRU with different bounds")

    # Refusals.
    proc = out / "gru-ae" / "fsdd_fold0_process.json"
    kept = proc.read_text()
    proc.unlink()
    expect_refusal(lambda: report(ppath, out, data), "did not finish", "a task that did not end")
    proc.write_text(kept.replace('"exit": 0', '"exit": 2'))
    expect_refusal(lambda: report(ppath, out, data), "exited 2", "a task that failed")
    proc.write_text(kept)
    ev = out / "gru-ae" / "t_fsdd_fold0_events.jsonl"
    kept = ev.read_text()
    ev.write_text(kept + json.dumps({"type": "fold_begin", "ts_unix": 9.0}) + "\n")
    expect_refusal(lambda: report(ppath, out, data), "2 fold_begin", "two sessions in one log")
    ev.write_text(kept)

    def other_epochs(fam, cfg):
        if fam == "transformer-ae":
            cfg["training"]["epochs"] = 60
    _write_synthetic(out, profile, other_epochs)
    expect_refusal(lambda: report(ppath, out, data), "different configurations",
                   "families under different configurations")
    _write_synthetic(out, profile)

    man = out / "lstm-ae" / "t_eegmmidb_fold0_lstm-ae_run2_model_selection_manifest.json"
    kept = man.read_text()
    man.write_text(json.dumps({"selected": {"val_mse": 0.4}}))
    expect_refusal(lambda: report(ppath, out, data), "no final_fit", "a pre-2026-11-02 manifest")
    man.write_text(kept)

    comp = out / "lstm-ae" / f"t_fsdd_fold0_{_COMPARATIVE}"
    kept = comp.read_text()
    comp.write_text("".join(l for l in kept.splitlines(keepends=True) if ",2,43," not in l))
    expect_refusal(lambda: report(ppath, out, data), "0 test rows for lstm-ae run 2",
                   "a missing seed")
    comp.write_text(kept.replace("lstm-ae,direct,1,42,test,0.865", "lstm-ae,direct,1,42,test,nan"))
    expect_refusal(lambda: report(ppath, out, data), "diverged", "a NaN test error")
    head, *body = kept.splitlines(keepends=True)
    cols = head.rstrip("\n").split(",")
    i = cols.index("energy_nj_sram1m")
    body = [",".join(c if j != i else "nan" for j, c in enumerate(l.rstrip("\n").split(","))) + "\n"
            for l in body]
    comp.write_text(head + "".join(body))
    expect_refusal(lambda: report(ppath, out, data), "energy_nj_sram1m",
                   "a NaN energy check (a binary that did not price one SRAM size)")
    comp.write_text(kept)

    # Placeholders: the same two files, every cell "--".
    ph = tmp / "ph"
    placeholders(ppath, ph)
    pt = (ph / "paper_prevalidation.tex").read_text()
    expect("\nLSTM-AE & -- & -- & -- & -- & -- & -- \\\\" in pt and "PCA ($k=d$) & --" in pt,
           f"placeholder table: {pt}")
    expect("\\preDataStatus}{placeholder}" in (ph / "paper_prevalidation_config.tex").read_text(),
           "placeholder status")
    return failures


def self_test() -> int:
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="prevalidation_selftest_"))
    try:
        failures = _self_test_checks(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if failures:
        for f in failures:
            print(f"[self-test] FAIL: {f}", file=sys.stderr)
        return 1
    print("[self-test] all pre-validation checks passed")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--report", action="store_true")
    mode.add_argument("--placeholders", action="store_true")
    mode.add_argument("--self-test", action="store_true")
    ap.add_argument("--profile", type=pathlib.Path)
    ap.add_argument("--out-root", type=pathlib.Path)
    ap.add_argument("--data-dir", type=pathlib.Path)
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test()
    try:
        if args.profile is None:
            raise ResultsError("--profile is required.")
        if args.plan:
            if args.out_root is None:
                raise ResultsError("--plan needs --out-root.")
            print("\n".join(write_plan(args.profile, args.out_root)))
        elif args.report:
            if args.out_root is None or args.data_dir is None:
                raise ResultsError("--report needs --out-root and --data-dir.")
            report(args.profile, args.out_root, args.data_dir)
            print(f"[prevalidation] wrote {args.out_root}/prevalidation_summary.csv, "
                  f"{args.data_dir}/paper_prevalidation.tex, paper_prevalidation_config.tex")
        else:
            if args.data_dir is None:
                raise ResultsError("--placeholders needs --data-dir.")
            placeholders(args.profile, args.data_dir)
            print(f"[prevalidation] placeholders: wrote {args.data_dir}/paper_prevalidation.tex, "
                  "paper_prevalidation_config.tex")
    except ResultsError as e:
        print(f"[prevalidation] REFUSED: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
