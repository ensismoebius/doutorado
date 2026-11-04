#!/usr/bin/env python3
"""05_meeting01_t1_transformer.py -- T1: does warm-up or pre-norm get the Transformer-AE off the
mean-predictor plateau?

Why it exists. In the pre-validation (GridUnesp job 5315335) the Transformer-AE never left the
plateau of the mean predictor in 55 trainings. Xiong et al. (ICML 2020) give an explanation that
is not "the architecture cannot do it": a post-norm Transformer at a fixed Adam rate needs a
learning-rate warm-up, and a pre-norm one does not. The paper removes the Transformer-AE from the
grid, so it must test that explanation before leaning on the removal. This script is that test.

Design, fixed before the first run (src/experiments/meeting01/questionsAndTasks.md, T1):

    arm  block        rate                                   profile switch
    A    post-norm    Adam 1e-3 constant (the protocol)      --
    B    post-norm    linear warm-up over 4000 steps (~4     training.lr_warmup_steps = 4000
                      epochs at 1000 steps/epoch), then 1e-3
    C    pre-norm     Adam 1e-3 constant                     model.transformer_norm_first = true

    configurations  the two Transformer-AE winners of job 5315335 on each development fold
                    (WINNERS below), every other setting the pre-validation profile's
    seeds           the profile's (42, 43); since results_format 9 a seed reaches the weights and
                    the window order, so each process trains 4 distinct networks (a GA evaluation
                    and a final fit per seed)
    "left the plateau"  the training armed its patience: monitor loss <= (1 - early_stop_min_skill)
                    x the mean predictor's (Meeting01Stopping.hpp)

Each process is the pre-validation's transformer-ae process with the NSGA-II pinned to one genome
(population 1, 0 generations, winner_seeds 1): the GA trains that genome once and the final fit
retrains it exactly as the pre-validation does.

Usage (from software/nn; 05_meeting01_t1_transformer.sh calls the first two):
    05_meeting01_t1_transformer.py --profiles --base P --out-root D
        writes D/profiles/<name>.json, prints one "name<TAB>dataset<TAB>fold<TAB>profile" per line
    05_meeting01_t1_transformer.py --report --out-root D --data-dir X
        D/t1_summary.csv (one row per training), X/paper_t1.tex
    05_meeting01_t1_transformer.py --placeholders --data-dir X
    05_meeting01_t1_transformer.py --self-test

Refuses (no fallbacks): a base profile without the transformer-ae family or a winner's dataset, a
process that did not finish or exited non-zero, a fold of another results_format, a missing seed or
training.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import pathlib
import sys
import tempfile

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
from meeting01_results import (  # noqa: E402
    DATASET_SHORT,
    DEVELOPMENT_FOLDS,
    ResultsError,
    split_manifest,
)

#: The Transformer-AE winners of GridUnesp job 5315335 (results_format 7, code fd4ed177), one per
#: seed and development fold: (config label, genome, the seed whose search picked it). Typed here,
#: not read, because T1 tests THOSE networks and the job's files are not kept in the tree.
WINNERS = {
    "fsdd": [("c1", {"d_model": 27, "d_ff": 135, "n_heads": 1, "n_layers": 1}, 42),
             ("c2", {"d_model": 24, "d_ff": 101, "n_heads": 4, "n_layers": 3}, 43)],
    "eegmmidb": [("c1", {"d_model": 96, "d_ff": 135, "n_heads": 8, "n_layers": 1}, 42),
                 ("c2", {"d_model": 79, "d_ff": 105, "n_heads": 1, "n_layers": 1}, 43)],
}
WARMUP_STEPS = 4000
#: arm -> (label, training.* switches, model.* switches)
ARMS = {
    "A": ("control", {}, {}),
    "B": ("warm-up", {"lr_warmup_steps": WARMUP_STEPS}, {}),
    "C": ("pre-norm", {}, {"transformer_norm_first": True}),
}
KEY = ("speaker_id", "recording_id", "window_id", "source_window_index")
SUMMARY_FIELDS = ["dataset", "config", "arm", "seed", "training", "armed_epoch", "best_val",
                  "test_mse", "mean_predictor_test_mse", "share_windows_10pct_below_mean"]


# ------------------------------------------------------------------------- profiles
def build_profiles(base: dict) -> list[tuple[str, str, int, dict]]:
    """(name, dataset, fold, profile) for every dataset x winner x arm."""
    ev = base.get("evaluation", {})
    if "transformer-ae" not in ev.get("baselines", []):
        raise ResultsError("the base profile does not run transformer-ae: T1 is built from the "
                           "pre-validation profile (profiles/meeting01-prevalidation.json)")
    out = []
    for ds, winners in WINNERS.items():
        if ds not in ev.get("datasets", []):
            raise ResultsError(f"the base profile has no dataset {ds}, which T1 tests")
        folds = sorted(DEVELOPMENT_FOLDS.get(ds, ()))
        if len(folds) != 1:
            raise ResultsError(f"{ds}: T1 runs on the one development fold, found {folds}")
        for cfg_label, genome, picked_by in winners:
            for arm, (arm_label, trn, mdl) in ARMS.items():
                name = f"meeting01_t1_{ds}_{cfg_label}_{arm}"
                p = copy.deepcopy(base)
                p["_profile_name"] = name
                p["_comment"] = (
                    f"T1 (05_meeting01_t1_transformer.py), arm {arm} ({arm_label}), {ds} fold "
                    f"{folds[0]}, config {cfg_label} = {json.dumps(genome)}: the Transformer-AE "
                    f"winner of GridUnesp job 5315335 for seed {picked_by}. The pre-validation "
                    "profile with the NSGA-II pinned to this genome (population 1, 0 generations, "
                    "winner_seeds 1), one dataset, the direct code, the arm's switch.")
                p["experiment"]["run_tag"] = name
                p["dataset"]["results_dir"] = f"__OUT__/{ds}_{cfg_label}_{arm}"
                p["evaluation"]["datasets"] = [ds]
                p["evaluation"]["encodings"] = ["direct"]
                p["evaluation"]["baselines"] = ["transformer-ae"]
                p["evaluation"]["snn_architectures"] = []
                p["evaluation"]["ga"]["transformer"].update(
                    population_size=1, generations=0, winner_seeds=1,
                    min_d_model=genome["d_model"], max_d_model=genome["d_model"],
                    head_choices=[genome["n_heads"]],
                    min_layers=genome["n_layers"], max_layers=genome["n_layers"],
                    min_d_ff=genome["d_ff"], max_d_ff=genome["d_ff"])
                p["training"].update(trn)
                p["model"].update(mdl)
                out.append((name, ds, folds[0], p))
    return out


def write_profiles(base_path: pathlib.Path, out_root: pathlib.Path) -> list[str]:
    base = json.loads(base_path.read_text(encoding="utf-8"))
    prof_dir = out_root / "profiles"
    prof_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    for name, ds, fold, p in build_profiles(base):
        text = json.dumps(p, indent=4).replace("__OUT__", str(out_root)) + "\n"
        path = prof_dir / f"{name}.json"
        path.write_text(text, encoding="utf-8")
        lines.append(f"{name}\t{ds}\t{fold}\t{path}")
    return lines


# --------------------------------------------------------------------------- report
def _mean_predictor_errors(d: pathlib.Path, tag: str, ds: str, fold: int):
    """Per test window: the masked MSE of the position-wise mean of the training targets."""
    p = f"{tag}_{ds}_fold{fold}_target_"
    tr = np.load(d / (p + "train_windows.npy")).astype(np.float64)
    with (d / (p + "train_windows_meta.csv")).open(encoding="utf-8") as f:
        tr_len = [int(r["valid_length"]) for r in csv.DictReader(f)]
    m_tr = np.arange(tr.shape[1])[None, :] < np.array(tr_len)[:, None]
    count = m_tr.sum(0)
    mean = np.where(count > 0, (tr * m_tr).sum(0) / np.maximum(count, 1), 0.0)
    te = np.load(d / (p + "test_windows.npy")).astype(np.float64)
    with (d / (p + "test_windows_meta.csv")).open(encoding="utf-8") as f:
        te_meta = list(csv.DictReader(f))
    m_te = np.arange(te.shape[1])[None, :] < np.array([int(r["valid_length"]) for r in te_meta])[:, None]
    err = (((te - mean) ** 2) * m_te).sum(1) / m_te.sum(1)
    return err, {tuple(r[k] for k in KEY): i for i, r in enumerate(te_meta)}


def collect(out_root: pathlib.Path) -> list[dict]:
    rows = []
    for ds, winners in WINNERS.items():
        fold = sorted(DEVELOPMENT_FOLDS[ds])[0]
        for cfg_label, _genome, _ in winners:
            for arm in ARMS:
                name = f"meeting01_t1_{ds}_{cfg_label}_{arm}"
                rec = out_root / f"{name}_process.json"
                if not rec.is_file():
                    raise ResultsError(f"{rec}: the process did not finish; rerun "
                                       "05_meeting01_t1_transformer.sh")
                proc = json.loads(rec.read_text(encoding="utf-8"))
                if proc.get("exit") != 0:
                    raise ResultsError(f"{name} exited {proc.get('exit')}: see {out_root}/{name}.log")
                d = out_root / f"{ds}_{cfg_label}_{arm}"
                split_manifest(d, name, ds, fold)  # checks the fold's results_format
                events = [json.loads(line) for line in (d / f"{name}_{ds}_fold{fold}_events.jsonl")
                          .read_text(encoding="utf-8").splitlines() if line.strip()]
                ends = [e for e in events if e.get("type") == "train_end"]
                with (d / f"{name}_{ds}_fold{fold}_comparative_metrics.csv").open(encoding="utf-8") as f:
                    test = {r["seed"]: r for r in csv.DictReader(f) if r["split"] == "test"}
                with (d / f"{name}_{ds}_fold{fold}_per_window_errors.csv").open(encoding="utf-8") as f:
                    pw = [r for r in csv.DictReader(f) if r["split"] == "test"]
                e_mean, idx = _mean_predictor_errors(d, name, ds, fold)
                if len(test) != 2:
                    raise ResultsError(f"{name}: {len(test)} test rows, expected one per seed (2)")
                for seed, trow in sorted(test.items()):
                    run = trow["run"]
                    final = [e for e in ends if e["config_id"] == f"transformer-ae_direct_seed{seed}_run{run}"]
                    ga = [e for e in ends if "_ga_" in e["config_id"] and e["config_id"].endswith(f"_seed{seed}")]
                    if len(final) != 1 or len(ga) != 1:
                        raise ResultsError(f"{name} seed {seed}: {len(final)} final fits and {len(ga)} "
                                           "GA evaluations in the events, expected one each")
                    ws = [w for w in pw if w["run_id"] == run]
                    ii = np.array([idx[tuple(w[k] for k in KEY)] for w in ws])
                    e = np.array([float(w["mse"]) for w in ws])
                    em = e_mean[ii]
                    for training, ev in (("ga_eval", ga[0]), ("final", final[0])):
                        rows.append({
                            "dataset": ds, "config": cfg_label, "arm": arm, "seed": seed,
                            "training": training, "armed_epoch": int(ev["armed_epoch"]),
                            "best_val": float(ev["best_val_loss"]),
                            "test_mse": float(trow["mse"]) if training == "final" else float("nan"),
                            "mean_predictor_test_mse": float(em.mean()),
                            "share_windows_10pct_below_mean":
                                float(np.mean(e < 0.9 * em)) if training == "final" else float("nan"),
                        })
    return rows


def verdict(rows: list[dict]) -> str:
    """One clause per dataset: how many trainings of each arm armed, and the test errors."""
    parts = []
    for ds in WINNERS:
        r = [x for x in rows if x["dataset"] == ds]
        if not r:
            raise ResultsError(f"no T1 rows for {ds}")

        def armed(arm):
            a = [x for x in r if x["arm"] == arm]
            return sum(1 for x in a if x["armed_epoch"] > 0), len(a)

        def finals(arm):
            return [x["test_mse"] for x in r if x["arm"] == arm and x["training"] == "final"]

        l0 = float(np.mean([x["mean_predictor_test_mse"] for x in r if x["training"] == "final"]))
        best_arm = min(ARMS, key=lambda a: min(finals(a)))
        (ab, nb), (ac, nc), (aa, na) = armed("B"), armed("C"), armed("A")
        parts.append(
            f"on {DATASET_SHORT.get(ds, ds)} fold {sorted(DEVELOPMENT_FOLDS[ds])[0]}, "
            f"{ab} of {nb} warm-up trainings and {ac} of {nc} pre-norm trainings left the "
            f"mean-predictor plateau (control: {aa} of {na}); the lowest test MSE of any arm was "
            f"{min(finals(best_arm)):.3f} ({ARMS[best_arm][0]}), against {np.mean(finals('A')):.3f} "
            f"for the control and {l0:.3f} for the mean predictor")
    return "; ".join(parts)


def write_macros(rows: list[dict] | None, data_dir: pathlib.Path) -> None:
    macros = {
        "tOneDataStatus": "results" if rows else "placeholder",
        "tOneWarmupSteps": str(WARMUP_STEPS),
        "tOneVerdict": verdict(rows) if rows else "[T1 has not run yet]",
    }
    lines = ["% auto-generated by 05_meeting01_t1_transformer.py -- do not edit"]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()]
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "paper_t1.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def report(out_root: pathlib.Path, data_dir: pathlib.Path) -> list[dict]:
    rows = collect(out_root)
    with (out_root / "t1_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    write_macros(rows, data_dir)
    for x in rows:
        print(f"{x['dataset']:9s} {x['config']} {x['arm']} s{x['seed']} {x['training']:7s} "
              f"armed@{x['armed_epoch']:2d} best_val {x['best_val']:.4f} test {x['test_mse']:.4f} "
              f"L0 {x['mean_predictor_test_mse']:.4f} share {x['share_windows_10pct_below_mean']:.3f}")
    print("[t1] verdict:", verdict(rows))
    return rows


# ------------------------------------------------------------------------ self-test
def self_test() -> int:
    failures = []
    base = {"experiment": {"run_tag": "x"}, "dataset": {"results_dir": "r"},
            "training": {"learning_rate": 1e-3}, "model": {},
            "evaluation": {"datasets": ["fsdd", "eegmmidb"], "encodings": ["direct", "poisson"],
                           "baselines": ["ann-ae", "transformer-ae"], "snn_architectures": ["dense"],
                           "ga": {"transformer": {"population_size": 4, "generations": 2,
                                                  "winner_seeds": 2}}}}
    profs = build_profiles(base)
    if len(profs) != 12:
        failures.append(f"expected 12 profiles (2 datasets x 2 winners x 3 arms), got {len(profs)}")
    by = {n: p for n, _, _, p in profs}
    b = by["meeting01_t1_fsdd_c1_B"]
    if b["training"].get("lr_warmup_steps") != WARMUP_STEPS or "transformer_norm_first" in b["model"]:
        failures.append("arm B must set the warm-up and only it")
    c = by["meeting01_t1_eegmmidb_c2_C"]
    if c["model"].get("transformer_norm_first") is not True or "lr_warmup_steps" in c["training"]:
        failures.append("arm C must set pre-norm and only it")
    tr = c["evaluation"]["ga"]["transformer"]
    if (tr["min_d_model"], tr["max_d_model"], tr["head_choices"], tr["population_size"],
            tr["generations"], tr["winner_seeds"]) != (79, 79, [1], 1, 0, 1):
        failures.append(f"eegmmidb c2 not pinned to its genome: {tr}")
    if c["evaluation"]["baselines"] != ["transformer-ae"] or c["evaluation"]["datasets"] != ["eegmmidb"]:
        failures.append("a T1 profile must run transformer-ae on its one dataset")
    if base["training"] != {"learning_rate": 1e-3}:
        failures.append("build_profiles modified its base")
    try:
        build_profiles({**base, "evaluation": {**base["evaluation"], "baselines": ["ann-ae"]}})
        failures.append("a base without transformer-ae was accepted")
    except ResultsError:
        pass

    def row(ds, arm, armed, test, training="final"):
        return {"dataset": ds, "config": "c1", "arm": arm, "seed": "42", "training": training,
                "armed_epoch": armed, "best_val": 0.9, "test_mse": test,
                "mean_predictor_test_mse": 1.0, "share_windows_10pct_below_mean": 0.1}
    rows = []
    for ds in WINNERS:
        rows += [row(ds, "A", 0, 0.93), row(ds, "A", 0, float("nan"), "ga_eval"),
                 row(ds, "B", 11, 0.90), row(ds, "B", 0, float("nan"), "ga_eval"),
                 row(ds, "C", 0, 0.94), row(ds, "C", 0, float("nan"), "ga_eval")]
    v = verdict(rows)
    want = ("on FSDD fold 0, 1 of 2 warm-up trainings and 0 of 2 pre-norm trainings left the "
            "mean-predictor plateau (control: 0 of 2); the lowest test MSE of any arm was 0.900 "
            "(warm-up), against 0.930 for the control and 1.000 for the mean predictor; on EEGMMIDB")
    if not v.startswith(want):
        failures.append(f"verdict text:\n  got  {v}\n  want {want}...")
    with tempfile.TemporaryDirectory() as tmp:
        write_macros(None, pathlib.Path(tmp))
        text = (pathlib.Path(tmp) / "paper_t1.tex").read_text(encoding="utf-8")
        for needle in ("\\newcommand{\\tOneDataStatus}{placeholder}",
                       f"\\newcommand{{\\tOneWarmupSteps}}{{{WARMUP_STEPS}}}",
                       "\\newcommand{\\tOneVerdict}{[T1 has not run yet]}"):
            if needle not in text:
                failures.append(f"placeholder macros lack {needle}")
    for f in failures:
        print("[self-test] FAIL:", f)
    if not failures:
        print("[self-test] all T1 checks passed")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--profiles", action="store_true")
    mode.add_argument("--report", action="store_true")
    mode.add_argument("--placeholders", action="store_true")
    mode.add_argument("--self-test", action="store_true")
    ap.add_argument("--base", type=pathlib.Path)
    ap.add_argument("--out-root", type=pathlib.Path)
    ap.add_argument("--data-dir", type=pathlib.Path)
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    try:
        if a.profiles:
            if not (a.base and a.out_root):
                ap.error("--profiles needs --base and --out-root")
            print("\n".join(write_profiles(a.base, a.out_root)))
        elif a.report:
            if not (a.out_root and a.data_dir):
                ap.error("--report needs --out-root and --data-dir")
            report(a.out_root, a.data_dir)
        else:
            if not a.data_dir:
                ap.error("--placeholders needs --data-dir")
            write_macros(None, a.data_dir)
    except ResultsError as e:
        print(f"[t1] REFUSED: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
