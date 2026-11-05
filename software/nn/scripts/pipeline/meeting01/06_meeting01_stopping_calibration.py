#!/usr/bin/env python3
"""06_meeting01_stopping_calibration.py -- measures the stopping rule's skill threshold s.

Why it exists. Patience counts only once the monitor loss is at most (1 - s) x L0, L0 being the
mean predictor's (Meeting01Stopping.hpp). s must sit above what a network on the plateau reaches
by noise, or a plateau run gets stopped and reported as a trained model. s = 0.10 came from
format-7 runs whose FSDD window 0 was the WAV header: the plateau sat at 0.962 because of it, and
its excursions were header-sized. Format 10 removed the header, so the measurement is repeated.

Protected data. Only monitor losses are read: a slice of the training groups, on the development
folds. No validation or test number enters the choice.

The rule, fixed before the first run. For each training, skill(e) = 1 - monitor(e) / L0:

    plateau     last skill < 0.05, and skill >= 0.05 in at most 2 epochs
    collapsed   last skill < 0.05, otherwise (it held a skill for a while, then lost it)
    learned     last skill >= 0.05

    s = the smallest of CANDIDATES strictly above the deepest excursion (peak skill) of a plateau
        run. If no candidate qualifies, or s exceeds the lowest peak of a learned run (s would
        keep a real learner from ever arming), no value is chosen and a person must look.

Outcome (2026-11-05, results_format 10, 84 trainings). The rule returned no value: 14 of the 54
learned runs (all FSDD, sequence families) peak below 0.10, the lowest at 0.0519. The author kept
0.10: no plateau run passed 0.0505, and replaying the curves at s = 0.05 saves 13 of 2349 epochs and
changes no SNN-AE or ANN-AE training (`what_if`, printed by --report). The numbers and the reasoning
are in .wiki/Experiments/Meeting01.md, "Why s = 0.10".

Usage (from software/nn):
    06_meeting01_stopping_calibration.py --profile --base P --out F    the reduced profile
    06_meeting01_stopping_calibration.py --report --out-root D         D/stopping_calibration.csv
    06_meeting01_stopping_calibration.py --self-test
The run itself is 00_meeting01_prevalidate_baselines.sh with PREVALIDATION_PROFILE=F and
PREVALIDATION_DIR=D, SKIP_POSTPROCESS=1.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import sys
import tempfile

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
from meeting01_results import RESULTS_FORMAT, ResultsError  # noqa: E402

PLATEAU_LAST_SKILL = 0.05
PLATEAU_MAX_EPOCHS_ABOVE = 2
CANDIDATES = (0.10, 0.15, 0.20, 0.25)
CURRENT_S = 0.10
WHAT_IF = (0.05, 0.06, 0.08)
PATIENCE = 5  # training.early_stop_patience in the production profiles
MIN_DELTA = 1e-8  # EarlyStoppingCallback's default, the one meeting01 passes
FIELDS = ["dataset", "family", "config_id", "epochs", "trivial_loss", "peak_skill", "last_skill",
          "epochs_above", "kind"]


def reduced_profile(base: dict) -> dict:
    p = json.loads(json.dumps(base))
    p["_profile_name"] = "meeting01_s_calibration"
    p["_comment"] = ("06_meeting01_stopping_calibration.py: the pre-validation profile with each "
                     "family's NSGA-II cut to its 4 random genomes (no generations) and 1 "
                     "re-scoring seed, for the monitor curves of the stopping rule.")
    p["experiment"]["run_tag"] = "meeting01_s_calibration"
    ga = p["evaluation"]["ga"]
    for family in ("snn", "lstm", "gru", "transformer"):
        if family not in ga:
            raise ResultsError(f"the base profile has no GA block for {family}")
        ga[family].update(population_size=4, generations=0, winner_seeds=1)
    return p


def classify(trivial: float, losses: list[float]) -> dict:
    if not losses:
        raise ResultsError("a training with no epoch")
    if not trivial > 0.0:
        raise ResultsError(f"a training with mean-predictor loss {trivial}")
    skills = [1.0 - v / trivial for v in losses]
    above = sum(1 for x in skills if x >= PLATEAU_LAST_SKILL)
    if skills[-1] >= PLATEAU_LAST_SKILL:
        kind = "learned"
    elif above <= PLATEAU_MAX_EPOCHS_ABOVE:
        kind = "plateau"
    else:
        kind = "collapsed"
    return {"epochs": len(skills), "trivial_loss": trivial, "peak_skill": max(skills),
            "last_skill": skills[-1], "epochs_above": above, "kind": kind}


def choose(rows: list[dict]) -> tuple[float | None, str]:
    plateau = [r["peak_skill"] for r in rows if r["kind"] == "plateau"]
    learned = [r["peak_skill"] for r in rows if r["kind"] == "learned"]
    if not plateau:
        return None, "no plateau run: nothing to calibrate against"
    deepest = max(plateau)
    above = [c for c in CANDIDATES if c > deepest]
    if not above:
        return None, f"deepest plateau excursion {deepest:.4f} is above every candidate"
    s = above[0]
    if learned and s > min(learned):
        return None, (f"s = {s} is above the lowest peak of a learned run ({min(learned):.4f}): "
                      "it would keep that run from arming")
    low = f"{min(learned):.4f}" if learned else "none"
    return s, f"deepest plateau excursion {deepest:.4f}, lowest learned peak {low}"


def replay(trivial: float, losses: list[float], s: float, patience: int = PATIENCE) -> tuple[int, float]:
    """(epochs run, best monitor loss) of a training under threshold s, from its recorded curve.

    EarlyStoppingCallback's rule, epoch by epoch. Exact for an s below the one the run used: arming
    earlier only cuts the recorded curve sooner. It cannot extend a curve, so it says nothing
    about an s above the run's.
    """
    arm_below = (1.0 - s) * trivial
    best = float("inf")
    armed = False
    bad = 0
    for epoch, loss in enumerate(losses, start=1):
        armed = armed or loss <= arm_below
        if loss < best - MIN_DELTA:
            best = loss
            bad = 0
        elif armed:
            bad += 1
            if bad >= patience:
                return epoch, best
    return len(losses), best


def what_if(rows: list[dict], s: float, base: float) -> dict:
    """What a lower s would have changed, against the same curves replayed at `base`."""
    if s >= base:
        raise ResultsError(f"replay cannot extend a curve: s {s} must be below {base}")
    out = {"epochs": 0, "base_epochs": 0, "stopped_sooner": 0, "worse": 0, "worst_loss": 0.0,
           "plateau_stopped": 0}
    for r in rows:
        ep, best = replay(r["trivial_loss"], r["losses"], s)
        base_ep, base_best = replay(r["trivial_loss"], r["losses"], base)
        out["epochs"] += ep
        out["base_epochs"] += base_ep
        out["stopped_sooner"] += ep < base_ep
        loss = (best - base_best) / r["trivial_loss"]
        out["worse"] += loss > 1e-3
        out["worst_loss"] = max(out["worst_loss"], loss)
        out["plateau_stopped"] += r["kind"] != "learned" and ep < base_ep
    return out


def read_events(path: pathlib.Path) -> list[dict]:
    """One record per training, in file order."""
    out: list[dict] = []
    current = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        kind = e.get("type")
        if kind == "config_begin":
            current = {"config_id": e["config_id"], "trivial": e["trivial_val_loss"], "losses": []}
            out.append(current)
        elif kind == "epoch":
            if current is None or e["config_id"] != current["config_id"]:
                raise ResultsError(f"{path}: an epoch of {e['config_id']} outside its training")
            current["losses"].append(float(e["val_loss"]))
    return out


def collect(out_root: pathlib.Path) -> list[dict]:
    rows = []
    files = sorted(out_root.glob("*/*_events.jsonl"))
    if not files:
        raise ResultsError(f"no events file under {out_root}")
    for f in files:
        family = f.parent.name
        dataset = "fsdd" if "_fsdd_fold" in f.name else "eegmmidb"
        manifest = f.with_name(f.name.replace("_events.jsonl", "_split_manifest.json"))
        if not manifest.is_file():
            raise ResultsError(f"{f}: no split manifest next to it")
        got = json.loads(manifest.read_text(encoding="utf-8")).get("results_format")
        if got != RESULTS_FORMAT:
            raise ResultsError(f"{manifest}: results_format {got}, need {RESULTS_FORMAT}")
        for t in read_events(f):
            rows.append({"dataset": dataset, "family": family, "config_id": t["config_id"],
                         "losses": t["losses"], **classify(float(t["trivial"]), t["losses"])})
    return rows


def report(out_root: pathlib.Path) -> tuple[float | None, str]:
    rows = collect(out_root)
    with (out_root / "stopping_calibration.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    for ds in sorted({r["dataset"] for r in rows}):
        sub = [r for r in rows if r["dataset"] == ds]
        counts = {k: sum(1 for r in sub if r["kind"] == k)
                  for k in ("plateau", "collapsed", "learned")}
        l0 = sum(r["trivial_loss"] for r in sub) / len(sub)
        print(f"{ds}: {len(sub)} trainings, {counts}, mean L0 {l0:.4f}")
        for fam in sorted({r["family"] for r in sub}):
            fs = [r for r in sub if r["family"] == fam]
            print(f"  {fam}: n={len(fs)}, "
                  f"plateau {sum(1 for r in fs if r['kind'] == 'plateau')}, "
                  f"peak skill {min(r['peak_skill'] for r in fs):.4f}.."
                  f"{max(r['peak_skill'] for r in fs):.4f}")
    for r in sorted(rows, key=lambda r: -r["peak_skill"]):
        if r["kind"] != "learned":
            print(f"  {r['kind']}: {r['dataset']} {r['family']} peak {r['peak_skill']:.4f} "
                  f"last {r['last_skill']:.4f} epochs above {r['epochs_above']}/{r['epochs']}")
    below = [r for r in rows if r["kind"] == "learned" and r["peak_skill"] < CURRENT_S]
    print(f"learned runs that never reach {CURRENT_S}: {len(below)} of "
          f"{sum(1 for r in rows if r['kind'] == 'learned')}")
    for label, sub in (("all families", rows),
                       ("SNN-AE + ANN-AE", [r for r in rows if r["family"] == "snn-ae"])):
        for lower in WHAT_IF:
            w = what_if(sub, lower, CURRENT_S)
            print(f"  {label}, s = {lower} instead of {CURRENT_S}: epochs {w['base_epochs']} -> "
                  f"{w['epochs']}, {w['stopped_sooner']} runs stop sooner, {w['worse']} end with "
                  f"a worse best monitor loss (worst +{w['worst_loss']:.4f} of L0), "
                  f"{w['plateau_stopped']} of them not learners")
    s, why = choose(rows)
    print(f"s = {s}: {why}")
    return s, why


def self_test() -> int:
    assert classify(1.0, [1.0, 0.99, 0.98])["kind"] == "plateau"
    assert classify(1.0, [1.0, 0.93, 0.99, 0.99])["kind"] == "plateau"
    assert classify(1.0, [1.0, 0.9, 0.9, 0.9, 0.99])["kind"] == "collapsed"
    assert classify(1.0, [1.0, 0.7])["kind"] == "learned"
    r = classify(2.0, [2.0, 1.0])
    assert abs(r["peak_skill"] - 0.5) < 1e-12 and abs(r["last_skill"] - 0.5) < 1e-12

    def row(kind, peak):
        return {"kind": kind, "peak_skill": peak}
    # Armed at 0.9, then 5 epochs without a new best: stops at the 6th epoch, keeps 0.85.
    assert replay(1.0, [1.0, 0.85, 0.9, 0.9, 0.9, 0.9, 0.9, 0.7], 0.10) == (7, 0.85)
    # Never armed: every epoch runs, whatever the bad streak.
    assert replay(1.0, [0.99, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.98], 0.10) == (8, 0.98)
    # The epoch crossing the level is a new best: the count starts at zero there.
    assert replay(1.0, [0.95, 0.96, 0.96, 0.96, 0.96, 0.96, 0.8], 0.05) == (6, 0.95)
    curve = {"trivial_loss": 1.0, "kind": "learned",
             "losses": [0.95, 0.96, 0.96, 0.96, 0.96, 0.96, 0.8]}
    w = what_if([curve], 0.05, 0.10)
    assert (w["epochs"], w["base_epochs"], w["stopped_sooner"], w["worse"]) == (6, 7, 1, 1)
    try:
        what_if([curve], 0.10, 0.10)
    except ResultsError:
        pass
    else:
        raise AssertionError("a replay above the run's s must be refused")

    assert choose([row("plateau", 0.066), row("learned", 0.3)])[0] == 0.10
    assert choose([row("plateau", 0.10), row("learned", 0.3)])[0] == 0.15, "strictly above"
    assert choose([row("plateau", 0.3)])[0] is None
    assert choose([row("plateau", 0.12), row("learned", 0.14)])[0] is None, "above a learner"
    assert choose([row("learned", 0.3)])[0] is None

    assert reduced_profile({"experiment": {}, "evaluation": {"ga": {
        k: {"population_size": 8} for k in ("snn", "lstm", "gru", "transformer")}}})[
            "evaluation"]["ga"]["gru"] == {"population_size": 4, "generations": 0, "winner_seeds": 1}
    try:
        reduced_profile({"experiment": {}, "evaluation": {"ga": {}}})
    except ResultsError:
        pass
    else:
        raise AssertionError("a missing GA block must be refused")

    with tempfile.TemporaryDirectory() as d:
        root = pathlib.Path(d)
        fam = root / "gru-ae"
        fam.mkdir()
        name = "t_fsdd_fold0"
        (fam / f"{name}_split_manifest.json").write_text(
            json.dumps({"results_format": RESULTS_FORMAT}), encoding="utf-8")
        ev = [{"type": "config_begin", "config_id": "a", "trivial_val_loss": 1.0},
              {"type": "epoch", "config_id": "a", "val_loss": 0.99},
              {"type": "config_begin", "config_id": "b", "trivial_val_loss": 1.0},
              {"type": "epoch", "config_id": "b", "val_loss": 0.6}]
        (fam / f"{name}_events.jsonl").write_text(
            "\n".join(json.dumps(e) for e in ev) + "\n", encoding="utf-8")
        rows = collect(root)
        assert [r["kind"] for r in rows] == ["plateau", "learned"]
        assert rows[0]["dataset"] == "fsdd" and rows[0]["family"] == "gru-ae"
        (fam / f"{name}_split_manifest.json").write_text(
            json.dumps({"results_format": RESULTS_FORMAT - 1}), encoding="utf-8")
        try:
            collect(root)
        except ResultsError:
            pass
        else:
            raise AssertionError("another results_format must be refused")
    print("self-test: ok")
    return 0


def main(argv: list[str] | None = None) -> int:
    a = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    m = a.add_mutually_exclusive_group(required=True)
    m.add_argument("--profile", action="store_true")
    m.add_argument("--report", action="store_true")
    m.add_argument("--self-test", action="store_true")
    a.add_argument("--base", type=pathlib.Path)
    a.add_argument("--out", type=pathlib.Path)
    a.add_argument("--out-root", type=pathlib.Path)
    args = a.parse_args(argv)
    if args.self_test:
        return self_test()
    if args.profile:
        base = json.loads(args.base.read_text(encoding="utf-8"))
        args.out.write_text(json.dumps(reduced_profile(base), indent=4) + "\n", encoding="utf-8")
        return 0
    s, _ = report(args.out_root)
    return 0 if s is not None else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ResultsError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
