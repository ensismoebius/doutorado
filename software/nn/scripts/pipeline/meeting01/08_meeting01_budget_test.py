#!/usr/bin/env python3
"""08_meeting01_budget_test.py -- does the training budget, not the model, decide two results of
the pre-validation? One test, two questions, on the design study's twin (07_, imported as is).

Why it exists. Two readings of the development folds could be artefacts of the budget:

  W  PCA beats the ANN-AE on EEGMMIDB by 0.11 (test 0.192 vs 0.303), and the ANN-AE fits the
     training windows worse than PCA (0.19 vs 0.105). The GA's width repair clamps every hidden
     width drawn below latent + 1 to that floor, so 32 of 36 EEG candidates had a hidden layer of
     exactly 65 for a 64-wide latent. Argued, not measured: a leaky-ReLU layer passes a signed
     value undistorted only with two units (LReLU(t) - LReLU(-t) = 1.01 t), so PCA's 64 signed
     coordinates need ~128 units, and at 65 the linear solution is out of reach.
  E  The 30-epoch cap binds more on the SNN: in the design study 65 of 112 LIF trainings had
     their best epoch at 28-30, against 9 of 24 leaky-ReLU ones. Equal epochs need not be equal
     convergence, which would bias the matched comparison against the SNN.

Design, fixed before the first run (2026-11-09; framing A chosen by the author the same day):

    twin       07_meeting01_design_study.py's network, data, stopping rule and seeds (42-45),
               unchanged; only the epoch cap and, in W, the hidden width differ
    budget     at most 60 epochs (twice the protocol's 30). One run gives both budgets: the first
               30 epochs of a 60-epoch run are the 30-epoch run (same seed, same window order,
               same stopping decisions), so the network the 30-epoch protocol would keep (best
               monitor epoch <= 30, or the earlier patience stop) is snapshotted and scored, then
               training continues. Its scores must equal the design study's rows for the same
               cells (checked: a determinism test of the twin).
    W  EEGMMIDB fold 0, ANN-AE (leaky ReLU, latent mean, leaky-ReLU decoder, Xavier), direct, widths
       [65, 64] (the winners'), [128, 64], [256, 64] (2k and 4k), 4 seeds         -> 12 trainings
    E  the four SNN-AE winners: the adopted SNN-AE and its ANN-AE control at the winner's widths,
       4 seeds; EEG controls are W's [65, 64] runs                                -> 24 more
    reported   per training and budget: test, validation and training MSE, best epoch, armed
               epoch, stop reason; per cell mean over seeds; seed-paired contrasts with Student-t
               95% intervals (df 3) of
                 W  train and test MSE of width 128 and 256 minus 65, at each budget; the share
                    of the ANN-AE's lead lost to PCA that a width closes, on train and test
                 E  (SNN - ANN) at 60 minus at 30 epochs, per winner: did the gap move?
                    and each family's 60-minus-30 change
    readings, stated before the run
                 W  "the width floor explains part of PCA's EEG lead" when the 256 run's train
                    MSE is below the 65 run's at the same budget with the interval excluding 0
                 E  "the cap biases the matched comparison" when a winner's gap change interval
                    excludes 0; it matters for the paper when the change exceeds Delta = 0.02
                    (EQUIVALENCE_MARGIN), the margin the headline uses
    not tested, said here: a wider SNN-AE (W changes only the control, which is the network PCA
    is compared with); more than 60 epochs; other folds. Development folds only.

PCA references (fit on train + val, k = latent; computed 2026-11-09 from the same windows):
FSDD train 0.568, test 0.675; EEGMMIDB train 0.105, test 0.192.

Usage (from software/nn; 08_meeting01_budget_test.sh calls --run and --report):
    08_meeting01_budget_test.py --plan
    08_meeting01_budget_test.py --run --out-root O [--jobs N]
    08_meeting01_budget_test.py --report --out-root O --data-dir P [--design-root results/meeting01_design]
    08_meeting01_budget_test.py --placeholders --data-dir P
    08_meeting01_budget_test.py --self-test

--report also writes P/paper_budget.tex: the numbers the paper's budget paragraph states, after
checking every claim its sentences make about them (budget_findings; refused when one fails).
Added 2026-11-09 after the run, report code only: the training code is the run's
(results/meeting01_budget/08_meeting01_budget_test.py.used).
"""

from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
import math
import multiprocessing as mp
import os
import pathlib
import sys
import time

import numpy as np

_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("design07", _SCRIPT_DIR / "07_meeting01_design_study.py")
D = importlib.util.module_from_spec(_spec)
sys.modules["design07"] = D
_spec.loader.exec_module(D)
ResultsError = D.ResultsError

# the headline's Delta (meeting01_results); this script used to keep a copy, 0.02
sys.path.insert(0, str(_SCRIPT_DIR))
from meeting01_results import EQUIVALENCE_MARGIN

MAX_EPOCHS = 60
PROTOCOL_EPOCHS = 30
W_DATASET = "eegmmidb"
W_WIDTHS = ([65, 64], [128, 64], [256, 64])
PCA ={"fsdd": {"train": 0.568, "test": 0.675}, "eegmmidb": {"train": 0.105, "test": 0.192}}
BUDGETS = ("e30", "e60")
ROW_FIELDS = ["name", "part", "dataset", "run", "family", "widths", "encoding", "time_steps", "seed",
              "budget", "best_epoch", "armed_epoch", "epochs_run", "stop_reason", "test_mse",
              "val_mse", "train_mse", "source_revision", "wall_s"]


# ------------------------------------------------------------------------- tasks
def build_tasks(winners: list[dict]) -> list[dict]:
    out, seen = [], set()

    def add(name, part, w, family, widths, seed):
        if name in seen:
            return
        seen.add(name)
        spiking = family == "snn"
        out.append({"name": name, "part": part, "winner": w, "family": family, "widths": list(widths),
                    "seed": seed, "steps": w["time_steps"] if spiking else 1,
                    "encoding": w["encoding"] if spiking else "direct",
                    "neuron": "lif" if spiking else "lrelu"})

    eeg = [w for w in winners if w["dataset"] == W_DATASET]
    if not eeg:
        raise ResultsError(f"no {W_DATASET} winner to take the fold from")
    for widths in W_WIDTHS:
        for seed in D.SEEDS:
            add(f"{W_DATASET}_w{D.widths_key(widths)}_ann_s{seed}", "W", eeg[0], "ann", widths, seed)
    for w in winners:
        for seed in D.SEEDS:
            add(f"{D.winner_key(w)}_snn_s{seed}", "E", w, "snn", w["widths"], seed)
            add(f"{w['dataset']}_w{D.widths_key(w['widths'])}_ann_s{seed}", "E", w, "ann", w["widths"], seed)
    out.sort(key=lambda t: -(t["steps"] * 10 + max(t["widths"]) / 100))
    return out


# ------------------------------------------------------------------------- training
def snapshot_scores(model, evaluate) -> dict:
    return {"test_mse": evaluate("test")[1], "val_mse": evaluate("val")[1],
            "train_mse": evaluate("train")[1]}


def train_budget(task: dict, prevalidation: str, out_root: str, source_revision: str) -> list[dict]:
    torch = D._torch()
    torch.set_num_threads(1)
    classes = D.make_twin_classes()
    w, seed, steps, enc = task["winner"], task["seed"], task["steps"], task["encoding"]
    pre = pathlib.Path(prevalidation)
    data = {s: D.load_split(pre, w["dataset"], w["fold"], s) for s in ("train", "val", "test")}

    def code(split, idx):
        x = data[split][0]
        if enc == "poisson":
            return torch.from_numpy(D.poisson_code(x, steps, np.random.default_rng([seed, 2, idx])))
        return torch.from_numpy(x).unsqueeze(0)

    inputs = {s: code(s, k) for k, s in enumerate(("train", "val", "test"))}
    xs = {s: torch.from_numpy(data[s][0]) for s in data}
    ms = {s: torch.from_numpy(data[s][1]) for s in data}
    gen = torch.Generator().manual_seed(seed)
    model = classes["Twin"](task["widths"], xs["train"].shape[1], task["neuron"], "lat_mean", "lrelu",
                            "xavier", w["v_th"], w["alpha"], steps, gen)
    bio = model.biophysical()
    bio_ids = {id(p) for p in bio}
    groups = [{"params": [p for p in model.parameters() if id(p) not in bio_ids], "lr": D.LR}]
    if bio:
        groups.append({"params": bio, "lr": D.LR_BIO})
    opt = torch.optim.Adam(groups, betas=(0.9, 0.999), eps=1e-8)

    def evaluate(split):
        model.eval()
        mon, one = [], []
        with torch.no_grad():
            for i in range(0, xs[split].shape[0], D.EVAL_CHUNK):
                y = model(inputs[split][:, i:i + D.EVAL_CHUNK])
                x, m = xs[split][i:i + D.EVAL_CHUNK], ms[split][i:i + D.EVAL_CHUNK]
                mon.append(D.monitor_loss(y, x, m, "lat_mean"))
                one.append(D.masked_mse(y.mean(dim=0), x, m))
        model.train()
        return float(torch.cat(mon).mean()), float(torch.cat(one).mean())

    l0 = D.trivial_loss(data["train"][0], data["train"][1], data["val"][0], data["val"][1])
    stopper = D.Stopper((1.0 - D.MIN_SKILL) * l0)
    best_state, snap30 = None, None
    order_rng = np.random.default_rng([seed, 1])
    log_path = pathlib.Path(out_root) / "logs" / f"{task['name']}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    epochs_run, stop_reason = 0, "max_epochs"
    with log_path.open("w", encoding="utf-8") as log:
        log.write(json.dumps({"type": "start", "name": task["name"], "trivial_loss": l0,
                              "source_revision": source_revision}) + "\n")
        for epoch in range(1, MAX_EPOCHS + 1):
            total = 0.0
            for i in order_rng.permutation(xs["train"].shape[0]):
                y = model(inputs["train"][:, i:i + 1])
                loss = D.monitor_loss(y, xs["train"][i:i + 1], ms["train"][i:i + 1], "lat_mean").mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                total += loss.item()
            mon = evaluate("val")[0]
            if not math.isfinite(mon):
                stop_reason, epochs_run = "non_finite", epoch
                break
            if stopper.update(epoch, mon):
                best_state = copy.deepcopy(model.state_dict())
            epochs_run = epoch
            log.write(json.dumps({"type": "epoch", "epoch": epoch, "train_loss": total / xs["train"].shape[0],
                                  "monitor": mon, "armed": stopper.armed_epoch > 0,
                                  "best_epoch": stopper.best_epoch}) + "\n")
            log.flush()
            if epoch == PROTOCOL_EPOCHS and not stopper.stop:
                snap30 = (copy.deepcopy(best_state), stopper.best_epoch, stopper.armed_epoch, epoch, "max_epochs")
            if stopper.stop:
                stop_reason = "patience"
                break
        if best_state is None:
            raise ResultsError(f"{task['name']}: no finite monitor epoch")
        if snap30 is None:  # stopped by patience at or before epoch 30: both budgets keep this network
            snap30 = (copy.deepcopy(best_state), stopper.best_epoch, stopper.armed_epoch, epochs_run, stop_reason)
        rows = []
        for budget, (state, best_ep, armed_ep, ep_run, reason) in (
                ("e30", snap30),
                ("e60", (best_state, stopper.best_epoch, stopper.armed_epoch, epochs_run, stop_reason))):
            model.load_state_dict(state)
            sc = snapshot_scores(model, evaluate)
            rows.append({"name": task["name"], "part": task["part"], "dataset": w["dataset"], "run": w["run"],
                         "family": task["family"], "widths": D.widths_key(task["widths"]),
                         "encoding": enc, "time_steps": steps, "seed": seed, "budget": budget,
                         "best_epoch": best_ep, "armed_epoch": armed_ep, "epochs_run": ep_run,
                         "stop_reason": reason, **sc, "source_revision": source_revision,
                         "wall_s": time.time() - t0})
        log.write(json.dumps({"type": "end", "rows": rows}) + "\n")
    rows_dir = pathlib.Path(out_root) / "rows"
    rows_dir.mkdir(parents=True, exist_ok=True)
    tmp = rows_dir / f"{task['name']}.json.tmp"
    tmp.write_text(json.dumps(rows), encoding="utf-8")
    tmp.replace(rows_dir / f"{task['name']}.json")
    return rows


def _worker(args):
    task, prevalidation, out_root, rev = args
    try:
        return task["name"], train_budget(task, prevalidation, out_root, rev), None
    except Exception as exc:
        return task["name"], None, f"{type(exc).__name__}: {exc}"


def run(prevalidation: pathlib.Path, out_root: pathlib.Path, jobs: int) -> int:
    rev = os.environ.get("MEETING01_GIT_COMMIT", "")
    if not rev:
        raise ResultsError("MEETING01_GIT_COMMIT is not set: run through 08_meeting01_budget_test.sh")
    tasks = build_tasks(D.read_winners(prevalidation))
    todo = []
    for t in tasks:
        p = out_root / "rows" / f"{t['name']}.json"
        if p.is_file():
            old = {r["source_revision"] for r in json.loads(p.read_text(encoding="utf-8"))}
            if old != {rev}:
                raise ResultsError(f"{p}: made by {sorted(old)}, this run is {rev}; move {out_root} aside")
            continue
        todo.append(t)
    print(f"[budget] {len(tasks)} trainings, {len(tasks) - len(todo)} done, {len(todo)} to run on {jobs}",
          flush=True)
    failed = []
    with mp.get_context("spawn").Pool(jobs, maxtasksperchild=1) as pool:
        args = [(t, str(prevalidation), str(out_root), rev) for t in todo]
        for k, (name, rows, err) in enumerate(pool.imap_unordered(_worker, args), 1):
            if err:
                failed.append(name)
                print(f"[budget] {time.strftime('%F %T')} FAILED {name}: {err}", flush=True)
            else:
                r30, r60 = rows
                print(f"[budget] {time.strftime('%F %T')} {k}/{len(todo)} {name}: test {r30['test_mse']:.4f} "
                      f"-> {r60['test_mse']:.4f} best@{r60['best_epoch']} {r60['stop_reason']} "
                      f"{r60['wall_s']:.0f}s", flush=True)
    if failed:
        print(f"[budget] {len(failed)} failed: {failed}", file=sys.stderr)
        return 1
    return 0


# ------------------------------------------------------------------------- report
def load_rows(out_root: pathlib.Path) -> list[dict]:
    rows = [r for p in sorted((out_root / "rows").glob("*.json"))
            for r in json.loads(p.read_text(encoding="utf-8"))]
    if len({r["source_revision"] for r in rows}) > 1:
        raise ResultsError(f"rows from more than one source revision in {out_root}")
    return rows


def by_seed(rows, **match) -> dict:
    got = {r["seed"]: r for r in rows if all(r[k] == v for k, v in match.items())}
    if sorted(got) != list(D.SEEDS):
        raise ResultsError(f"{match}: seeds {sorted(got)}, expected {list(D.SEEDS)}")
    return got


def determinism_check(rows: list[dict], design_root: pathlib.Path, winners: list[dict]) -> tuple[list[str], int]:
    """The e30 snapshot must reproduce the design study's row of the same cell and seed.
    Returns (report lines, snapshots compared); refuses when any differs or none can be compared."""
    if not (design_root / "design_trainings.csv").is_file():
        raise ResultsError(f"no {design_root}/design_trainings.csv: the 30-epoch snapshots cannot be checked "
                           "against the design study; run its report first (or pass --design-root)")
    with (design_root / "design_trainings.csv").open(encoding="utf-8") as f:
        design = {r["name"]: r for r in csv.DictReader(f)}
    issues, checked = [], 0
    for r in rows:
        in_design = r["part"] == "E" or r["widths"] == "65x64"  # cells the design study also ran
        if r["budget"] != "e30" or not in_design:
            continue
        if r["family"] == "snn":
            w = next(x for x in winners if x["dataset"] == r["dataset"] and x["run"] == r["run"])
            name = f"{D.winner_key(w)}_lif_lat_mean_lrelu_xavier_s{r['seed']}"
        else:
            name = f"{r['dataset']}_w{r['widths']}_lrelu_xavier_s{r['seed']}"
        d = design.get(name)
        if d is None:
            issues.append(f"{name}: not in the design study")
            continue
        checked += 1
        if abs(float(d["test_mse"]) - r["test_mse"]) > 1e-6 or int(d["best_epoch"]) != r["best_epoch"]:
            issues.append(f"{name}: design test {float(d['test_mse']):.6f} best@{d['best_epoch']}, "
                          f"budget e30 {r['test_mse']:.6f} best@{r['best_epoch']}")
    differ = sum(1 for i in issues if ": design test" in i)
    issues.append(f"determinism: {checked} e30 snapshots compared with the design study, {differ} differ")
    if differ or checked == 0 or len(issues) > 1:
        raise ResultsError("the twin is not deterministic against the design study:\n  " + "\n  ".join(issues))
    return issues, checked


# ------------------------------------------------------------------------- the prose's numbers
# The design section's budget paragraph (v) and limitation (vii) state these numbers and claims;
# like 07_'s findings(), a claim the data no longer support is refused, naming the sentence.
BUDGET_MACROS = ("budgetDataStatus", "budgetEpochs", "budgetProtocolEpochs", "budgetArmGainMax",
                 "budgetGapShiftMax", "budgetGapShiftWidest", "budgetWideTwo", "budgetWideFour",
                 "budgetWideTwoLong", "budgetWideFourLong", "budgetWideBest", "budgetSnapshots")


def _train_fell_after_best(out_root: pathlib.Path, name: str, best_epoch: int) -> bool:
    log = out_root / "logs" / f"{name}.jsonl"
    ep = {e["epoch"]: e["train_loss"] for e in map(json.loads, log.read_text(encoding="utf-8").splitlines())
          if e.get("type") == "epoch"}
    if best_epoch not in ep:
        raise ResultsError(f"{log}: epoch {best_epoch} (the best) is not logged")
    return ep[max(ep)] < ep[best_epoch]


def budget_findings(rows, winners, con, out_root: pathlib.Path, snapshots: int) -> dict[str, str]:
    L = {(c["contrast"], c["field"]): c for c in con}
    d = next(w for w in winners if w["dataset"] == W_DATASET)["widths"][-1]
    if [list(x) for x in W_WIDTHS] != [[d + 1, d], [2 * d, d], [4 * d, d]]:
        raise ResultsError(f"the paper calls W's widths d+1, 2d and 4d (d = {d}); they are {W_WIDTHS}")
    keys = [D.winner_key(w) for w in winners]
    broken = []

    def claim(ok: bool, sentence: str) -> None:
        if not ok:
            broken.append(sentence)

    m = {"budgetEpochs": str(MAX_EPOCHS), "budgetProtocolEpochs": str(PROTOCOL_EPOCHS),
         "budgetSnapshots": str(snapshots)}
    arms = [L[(f"{k} {a} 60 - 30", "test_mse")] for k in keys for a in ("SNN", "ANN")]
    m["budgetArmGainMax"] = D._up(max(-a["mean"] for a in arms), 3)
    claim(all(a["mean"] <= 0 for a in arms), "(v) the longer budget lowered, never raised, either arm's mean error")
    shift = {k: L[(f"{k} gap(60) - gap(30)", "test_mse")] for k in keys}
    m["budgetGapShiftMax"] = D._up(max(abs(s["mean"]) for s in shift.values()), 3)
    claim(not any(D._excludes_zero(s) for s in shift.values()), "(v) no interval of the gap's change excludes zero")
    reach = {k: max(abs(s["lo"]), abs(s["hi"])) for k, s in shift.items()}
    m["budgetGapShiftWidest"] = D._up(max(reach.values()), 3)
    beyond = [k for k, v in reach.items() if v > EQUIVALENCE_MARGIN]
    claim(bool(beyond) and all(k.startswith(f"{W_DATASET}_") for k in beyond),
          f"(v) the widest interval reaches beyond the margin {EQUIVALENCE_MARGIN:g}, and only on EEGMMIDB")
    wide = {(wd, b): L[(f"{b} width {wd} - 65x64", "test_mse")] for wd in ("128x64", "256x64") for b in BUDGETS}
    m["budgetWideTwo"] = D._r(wide[("128x64", "e30")]["mean"])
    m["budgetWideFour"] = D._r(wide[("256x64", "e30")]["mean"])
    m["budgetWideTwoLong"] = D._r(wide[("128x64", "e60")]["mean"])
    m["budgetWideFourLong"] = D._r(wide[("256x64", "e60")]["mean"])
    claim(all(x["lo"] > 0 for x in wide.values()), "(v) the wider controls err more, every interval excluding zero")
    wide_rows = [r for r in rows if r["part"] == "W" and r["widths"] in ("128x64", "256x64") and r["budget"] == "e60"]
    m["budgetWideBest"] = D._span([r["best_epoch"] for r in wide_rows], 0)
    claim(all(_train_fell_after_best(out_root, r["name"], r["best_epoch"]) for r in wide_rows),
          "(v) the wider controls' training error still fell after their best validation epoch")
    if broken:
        raise ResultsError("the budget paragraph's prose no longer matches the data, rewrite: " + "; ".join(broken))
    m["budgetDataStatus"] = "results"
    if set(m) != set(BUDGET_MACROS):
        raise ResultsError(f"budget macros {sorted(set(m) ^ set(BUDGET_MACROS))} out of sync with BUDGET_MACROS")
    return m


def write_budget_tex(data_dir: pathlib.Path, macros: dict[str, str]) -> pathlib.Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    lines = ["% auto-generated by 08_meeting01_budget_test.py -- do not edit"]
    lines += [f"\\newcommand{{\\{k}}}{{{macros[k]}}}" for k in BUDGET_MACROS]
    out = data_dir / "paper_budget.tex"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def write_placeholders(data_dir: pathlib.Path) -> pathlib.Path:
    return write_budget_tex(data_dir, {k: "placeholder" if k == "budgetDataStatus" else "$-$"
                                       for k in BUDGET_MACROS})


def report(prevalidation: pathlib.Path, out_root: pathlib.Path, design_root: pathlib.Path,
           data_dir: pathlib.Path) -> None:
    winners = D.read_winners(prevalidation)
    rows = load_rows(out_root)
    n_tasks = len(build_tasks(winners))
    if len(rows) != 2 * n_tasks:
        raise ResultsError(f"{len(rows)} rows, expected {2 * n_tasks} (2 budgets x {n_tasks} trainings)")
    with (out_root / "budget_trainings.csv").open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=ROW_FIELDS, lineterminator="\n")
        wr.writeheader()
        wr.writerows(sorted(rows, key=lambda r: (r["name"], r["budget"])))
    lines, con = [], []

    def contrast(label, a, b, field="test_mse"):
        c = D.paired_contrast({s: a[s][field] for s in a}, {s: b[s][field] for s in b})
        con.append({"contrast": label, "field": field, **c})
        flag = "" if c["lo"] <= 0 <= c["hi"] else "  *"
        lines.append(f"  {label:58s} {field:9s} {c['mean']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]{flag}")
        return c

    mean = lambda d, f: float(np.mean([r[f] for r in d.values()]))  # noqa: E731
    lines.append("W: EEGMMIDB ANN-AE by hidden width (PCA train 0.105, test 0.192)")
    for budget in BUDGETS:
        # the 65-wide EEG control is one set of runs, shared by W (its base width) and E
        cells = {wd: by_seed(rows, widths=wd, family="ann", dataset=W_DATASET, budget=budget)
                 for wd in (D.widths_key(x) for x in W_WIDTHS)}
        for wd, c in cells.items():
            best = [r["best_epoch"] for r in c.values()]
            lines.append(f"  {budget} width {wd:7s} train {mean(c, 'train_mse'):.4f} test {mean(c, 'test_mse'):.4f} "
                         f"best epochs {best}")
        base = cells["65x64"]
        for wd in ("128x64", "256x64"):
            for field in ("train_mse", "test_mse"):
                c = contrast(f"{budget} width {wd} - 65x64", cells[wd], base, field)
                gap = mean(base, field) - PCA[W_DATASET][field.split('_')[0]]
                lines.append(f"      closes {-c['mean'] / gap:6.1%} of the 65-wide ANN's {field.split('_')[0]} "
                             f"lead lost to PCA ({gap:.4f})")
    lines.append("E: SNN-AE minus ANN-AE control, 30 vs 60 epochs, per winner")
    for w in winners:
        k = D.winner_key(w)
        wk = D.widths_key(w["widths"])
        snn = {b: by_seed(rows, family="snn", dataset=w["dataset"], run=w["run"], budget=b) for b in BUDGETS}
        ann = {b: by_seed(rows, family="ann", dataset=w["dataset"], widths=wk, budget=b) for b in BUDGETS}
        for b in BUDGETS:
            lines.append(f"  {k} {b}: SNN {mean(snn[b], 'test_mse'):.4f} (best epochs "
                         f"{[r['best_epoch'] for r in snn[b].values()]}), ANN {mean(ann[b], 'test_mse'):.4f} "
                         f"(best epochs {[r['best_epoch'] for r in ann[b].values()]})")
        gap = {b: {s: {"test_mse": snn[b][s]["test_mse"] - ann[b][s]["test_mse"]} for s in D.SEEDS} for b in BUDGETS}
        contrast(f"{k} gap(60) - gap(30)", gap["e60"], gap["e30"])
        contrast(f"{k} SNN 60 - 30", snn["e60"], snn["e30"])
        contrast(f"{k} ANN 60 - 30", ann["e60"], ann["e30"])
        contrast(f"{k} SNN - ANN at 60", snn["e60"], ann["e60"])
    det_lines, snapshots = determinism_check(rows, design_root, winners)
    lines += [""] + det_lines
    with (out_root / "budget_contrasts.csv").open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=["contrast", "field", "n", "mean", "lo", "hi"], lineterminator="\n")
        wr.writeheader()
        wr.writerows(con)
    text = "\n".join(lines) + "\n"
    (out_root / "budget_report.txt").write_text(text, encoding="utf-8")
    print(text, end="")
    out = write_budget_tex(data_dir, budget_findings(rows, winners, con, out_root, snapshots))
    print(f"[budget] wrote {out}")



# ------------------------------------------------------------------------- self-test
def self_test() -> int:
    failures = []
    fake = [{"dataset": "fsdd", "run": 1, "widths": [59, 16], "encoding": "direct", "time_steps": 28},
            {"dataset": "fsdd", "run": 2, "widths": [35, 16], "encoding": "poisson", "time_steps": 19},
            {"dataset": "eegmmidb", "run": 1, "widths": [65, 64], "encoding": "poisson", "time_steps": 31},
            {"dataset": "eegmmidb", "run": 2, "widths": [65, 64], "encoding": "poisson", "time_steps": 19}]
    tasks = build_tasks(fake)
    want = 4 * (3 + 4 + 2)  # W: 3 widths; E: 4 SNN winners + 2 FSDD controls (EEG control = W's 65)
    if len(tasks) != want:
        failures.append(f"{len(tasks)} trainings, want {want}")
    names = {t["name"] for t in tasks}
    if "eegmmidb_w65x64_ann_s42" not in names or "eegmmidb_w256x64_ann_s45" not in names:
        failures.append("W cells missing")
    t65 = next(t for t in tasks if t["name"] == "eegmmidb_w65x64_ann_s42")
    if t65["part"] != "W" or t65["steps"] != 1 or t65["encoding"] != "direct":
        failures.append(f"the EEG 65-wide control must be W's analog run: {t65}")
    snn = next(t for t in tasks if t["name"] == "eegmmidb_r1_snn_s43")
    if (snn["steps"], snn["encoding"], snn["neuron"]) != (31, "poisson", "lif"):
        failures.append(f"SNN task: {snn}")
    if tasks[0]["family"] != "snn" or tasks[0]["steps"] != 31:
        failures.append("the longest training must be scheduled first")
    # by_seed refuses a missing seed
    try:
        by_seed([{"seed": 42, "x": 1}], x=1)
        failures.append("by_seed accepted a cell with one seed")
    except ResultsError:
        pass
    # the snapshot rule: the 30-epoch network of a run that continues is its best epoch <= 30;
    # replay a monitor curve through the Stopper exactly as train_budget does
    st = D.Stopper(0.9)
    curve = [1.0, 0.8] + [0.8 - 0.001 * k for k in range(1, 40)]
    snap = None
    for e, v in enumerate(curve[:MAX_EPOCHS], 1):
        st.update(e, v)
        if e == PROTOCOL_EPOCHS and not st.stop:
            snap = st.best_epoch
        if st.stop:
            break
    if snap != 30 or st.best_epoch != len(curve):
        failures.append(f"snapshot best epoch {snap} (want 30), final {st.best_epoch} (want {len(curve)})")
    # the paper's macros: placeholders define every one; the log reader sees a still-falling train loss
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        txt = write_placeholders(tmp).read_text(encoding="utf-8")
        missing = [k for k in BUDGET_MACROS if f"\\newcommand{{\\{k}}}" not in txt]
        if missing or "{placeholder}" not in txt:
            failures.append(f"placeholders: missing {missing}")
        (tmp / "logs").mkdir()
        (tmp / "logs" / "x.jsonl").write_text("\n".join(json.dumps(e) for e in (
            {"type": "start"}, {"type": "epoch", "epoch": 1, "train_loss": 0.30},
            {"type": "epoch", "epoch": 2, "train_loss": 0.25}, {"type": "epoch", "epoch": 3, "train_loss": 0.24})),
            encoding="utf-8")
        if not _train_fell_after_best(tmp, "x", 2) or _train_fell_after_best(tmp, "x", 3):
            failures.append("_train_fell_after_best")
    for f in failures:
        print("[self-test] FAIL:", f)
    if not failures:
        print("[self-test] all budget-test checks passed")
    return 1 if failures else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    for m in ("--plan", "--run", "--report", "--placeholders", "--self-test"):
        mode.add_argument(m, action="store_true")
    ap.add_argument("--prevalidation", default=D.DEFAULT_PREVALIDATION)
    ap.add_argument("--out-root")
    ap.add_argument("--design-root", default="results/meeting01_design")
    ap.add_argument("--data-dir", help="the paper's data/ (paper_budget.tex); --report and --placeholders")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args(argv)
    try:
        if a.self_test:
            return self_test()
        if a.placeholders:
            if not a.data_dir:
                ap.error("--placeholders needs --data-dir")
            print(f"[budget] wrote {write_placeholders(pathlib.Path(a.data_dir))}")
            return 0
        pre = pathlib.Path(a.prevalidation)
        if a.plan:
            tasks = build_tasks(D.read_winners(pre))
            for t in tasks:
                print(t["name"], t["part"], t["widths"], t["encoding"], f"T={t['steps']}")
            print(f"{len(tasks)} trainings")
            return 0
        if not a.out_root:
            ap.error("--run and --report need --out-root")
        if a.run:
            return run(pre, pathlib.Path(a.out_root), a.jobs)
        if not a.data_dir:
            ap.error("--report needs --data-dir")
        report(pre, pathlib.Path(a.out_root), pathlib.Path(a.design_root), pathlib.Path(a.data_dir))
        return 0
    except ResultsError as exc:
        print(f"[budget] REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
