#!/usr/bin/env python3
"""10_meeting01_learning_curves.py -- is the SNN-AE's error a fit gap or a generalization gap?
The paper's learning-curve figure and the numbers its sentence states.

Why it exists. The SNN-AE errs more than its ANN-AE control on held-out groups. That can mean
two different things, and the paper's reading depends on which:

    fit gap             the SNN-AE already errs more on the windows it was trained on; the
                        held-out gap is that same gap carried over
    generalization gap  both fit their training windows equally well, and the SNN-AE loses
                        only on new groups (it learned something that does not transfer)

The visual reviewer (review/ruling_visual.md, F6 and V004) asked for per-arm learning curves and
the training-set error, "the only evidence on V004's split between representation and
training". This script draws the first and states the second.

What it reads, on the headline folds only (is_headline_fold):
    <tag>_<ds>_fold<k>_comparative_metrics.csv  test rows: mse (test windows) and train_mse (the
                                                final network on its training windows, eval mode)
    <tag>_<ds>_fold<k>_events.jsonl             the final fits' per-epoch training loss (the
                                                events are not in git; the figure file is)

What it states, per dataset, over the (fold, run) pairs of SNN-AE and control:
    train gap   mean of (SNN-AE train_mse - control train_mse)
    test gap    mean of (SNN-AE mse - control mse), the same pairs
    share       train gap / test gap: near 1 means the held-out gap is the fit gap
These are pair means, not the group-level estimand of 04_; the share compares like with like.

Usage (from software/nn):
    10_meeting01_learning_curves.py --report --data-dir X [--results-dir R]
        X/paper_loso_learning.tex          macros (claims checked)
        X/paper_loso_learning_figure.tex   the figure (pgfplots groupplot), generated
    10_meeting01_learning_curves.py --placeholders --data-dir X
    10_meeting01_learning_curves.py --self-test

Refuses (no fallbacks): a fold of another results_format, a final fit missing or trained twice,
an epoch sequence with a hole, a pair without its partner, and a sentence the numbers do not
support (a train gap that is not positive on every dataset: the paper's paragraph would then be
wrong, and it is rewritten, not softened).
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent

sys.path.insert(0, str(_SCRIPT_DIR))
from meeting01_results import (
    DATASET_MODALITY,
    DATASET_SHORT,
    ResultsError,
    is_headline_fold,
    paper_datasets,
    run_configuration,
    split_manifest,
)

RUN_TAG = "meeting01_loso"
DEFAULT_RESULTS = pathlib.Path(__file__).resolve().parents[3] / "results" / "meeting01"
ARMS = (("snn_final", "snn-ae", "SNN-AE", "red!75!black"),
        ("ann_final", "ann-ae", "ANN-AE (control)", "blue!70!black"))
MACROS = ("learnDataStatus", "learnGapSpeech", "learnGapEeg", "learnShare", "learnPairs")


# --------------------------------------------------------------------------- reading

def _headline_folds(results: pathlib.Path, ds: str) -> list[int]:
    folds = sorted({int(p.name.split("_fold")[1].split("_")[0])
                    for p in results.glob(f"{RUN_TAG}_{ds}_fold*_split_manifest.json")})
    head = [f for f in folds if is_headline_fold(ds, f)]
    if not head:
        raise ResultsError(f"{ds}: no headline fold under {results}")
    return head


def read_pairs(results: pathlib.Path, ds: str, fold: int) -> list[dict]:
    """One dict per run: the SNN-AE's and the control's test and training MSE."""
    path = results / f"{RUN_TAG}_{ds}_fold{fold}_comparative_metrics.csv"
    with path.open(encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "test"]
    by = {}
    for r in rows:
        if r["model"] in ("snn-ae", "ann-ae"):
            key = (r["model"], int(r["run"]))
            if key in by:
                raise ResultsError(f"{path.name}: {key} appears twice")
            by[key] = r
    runs = sorted({run for _, run in by})
    pairs = []
    for run in runs:
        if ("snn-ae", run) not in by or ("ann-ae", run) not in by:
            raise ResultsError(f"{path.name}: run {run} lacks the SNN-AE or its control")
        s, a = by[("snn-ae", run)], by[("ann-ae", run)]
        pairs.append({"dataset": ds, "fold": fold, "run": run,
                      "snn_test": float(s["mse"]), "ann_test": float(a["mse"]),
                      "snn_train": float(s["train_mse"]), "ann_train": float(a["train_mse"])})
    return pairs


def read_curves(path: pathlib.Path, runs: int) -> dict:
    """{role: [[train_loss at epoch 1, 2, ...] per final fit]} from one fold's events."""
    begins, epochs, ends, role = {}, {}, {}, {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            t, cid = e["type"], e.get("config_id")
            if t == "config_begin":
                begins[cid] = begins.get(cid, 0) + 1
            elif t == "epoch":
                epochs.setdefault(cid, []).append((int(e["epoch"]), float(e["train_loss"])))
            elif t == "train_end":
                ends[cid] = (ends[cid][0] + 1, e) if cid in ends else (1, e)
            elif t == "config_end" and e.get("role") in ("snn_final", "ann_final"):
                if cid in role:
                    raise ResultsError(f"{path.name}: final fit {cid} ends twice")
                role[cid] = e["role"]
    out = {r: [] for r, *_ in ARMS}
    for cid, r in sorted(role.items()):
        if begins.get(cid) != 1 or ends.get(cid, (0,))[0] != 1:
            raise ResultsError(f"{path.name}: final fit {cid} trained {begins.get(cid, 0)} "
                               "times; a resumed task retrained it, and its curve is ambiguous")
        seq = epochs.get(cid, [])
        if [ep for ep, _ in seq] != list(range(1, len(seq) + 1)):
            raise ResultsError(f"{path.name}: {cid} epochs are not 1..n: {[ep for ep, _ in seq]}")
        if len(seq) != int(ends[cid][1]["epochs_run"]):
            raise ResultsError(f"{path.name}: {cid} logged {len(seq)} epochs, train_end says "
                               f"{ends[cid][1]['epochs_run']}")
        out[r].append([loss for _, loss in seq])
    for r, *_ in ARMS:
        if len(out[r]) != runs:
            raise ResultsError(f"{path.name}: {len(out[r])} {r} fits, {runs} runs configured")
    return out


def read_inputs(results: pathlib.Path) -> dict:
    config, _ = run_configuration(results, RUN_TAG)
    runs = int(config["experiment"]["repeats"])
    datasets = paper_datasets(config)
    pairs, curves = {}, {}
    for ds in datasets:
        for fold in _headline_folds(results, ds):
            split_manifest(results, RUN_TAG, ds, fold)  # checks the results_format
            p = read_pairs(results, ds, fold)
            if len(p) != runs:
                raise ResultsError(f"{ds} fold {fold}: {len(p)} pairs, {runs} runs configured")
            pairs.setdefault(ds, []).extend(p)
            c = read_curves(results / f"{RUN_TAG}_{ds}_fold{fold}_events.jsonl", runs)
            for r, v in c.items():
                curves.setdefault(ds, {}).setdefault(r, []).extend(v)
    return {"datasets": datasets, "pairs": pairs, "curves": curves}


# --------------------------------------------------------------------------- claims

def _d3(x: float) -> str:
    return f"{x:.3f}"


def _span(xs: list[float], fmt) -> str:
    lo, hi = fmt(min(xs)), fmt(max(xs))
    return lo if lo == hi else f"{lo} to {hi}"


def gaps(pairs: list[dict]) -> dict:
    n = len(pairs)
    train = sum(p["snn_train"] - p["ann_train"] for p in pairs) / n
    test = sum(p["snn_test"] - p["ann_test"] for p in pairs) / n
    return {"train": train, "test": test, "n": n,
            "positive": sum(p["snn_train"] > p["ann_train"] for p in pairs)}


def findings(inp: dict) -> dict:
    """The macros of the paper's sentence. Refuses when the sentence would be wrong: a train
    gap or a test gap that is not positive on some dataset, or a modality with no dataset."""
    g = {ds: gaps(inp["pairs"][ds]) for ds in inp["datasets"]}
    for ds, v in g.items():
        if v["train"] <= 0 or v["test"] <= 0:
            raise ResultsError(
                f"{ds}: train gap {v['train']:.4f}, test gap {v['test']:.4f}. The paper says the "
                "SNN-AE already errs more on its training windows; that is not so here. Rewrite "
                "the paragraph (and the F6 caption) before regenerating.")
    by_mod = {}
    for ds, v in g.items():
        by_mod.setdefault(DATASET_MODALITY[ds], []).append(v["train"])
    if set(by_mod) != {"speech", "eeg"}:
        raise ResultsError(f"modalities {sorted(by_mod)}: the sentence names speech and EEG")
    pos = sum(v["positive"] for v in g.values())
    n = sum(v["n"] for v in g.values())
    return {"learnDataStatus": "results",
            "learnGapSpeech": _span(by_mod["speech"], _d3),
            "learnGapEeg": _span(by_mod["eeg"], _d3),
            "learnShare": _span([100 * v["train"] / v["test"] for v in g.values()],
                                lambda x: f"{x:.0f}"),
            "learnPairs": f"{pos} of {n}"}


# --------------------------------------------------------------------------- figure

def figure(inp: dict) -> str:
    """A 2 x 2 groupplot, one panel per dataset: every final fit's training loss per epoch,
    thin and translucent, one colour per arm. Each line ends where early stopping ended it."""
    datasets = inp["datasets"]
    xmax = max(len(c) for ds in datasets for r in inp["curves"][ds].values() for c in r)
    lines = ["% auto-generated by 10_meeting01_learning_curves.py -- do not edit",
             r"\begin{tikzpicture}[font=\scriptsize]",
             (rf"\begin{{groupplot}}[group style={{group size=2 by {(len(datasets) + 1) // 2}, "
              r"horizontal sep=1.1cm, vertical sep=1.7cm},"),
             (rf"  width=0.56\columnwidth, height=3.6cm, xmin=1, xmax={xmax}, "
              r"xlabel={epoch}, ylabel={training MSE},"),
             (r"  title style={yshift=-1ex}, ymajorgrids,"
              r" every axis plot/.append style={line width=0.3pt}]")]
    for i, ds in enumerate(datasets):
        opts = f"title={{{DATASET_SHORT.get(ds, ds)}}}"
        if i == 0:
            opts += (r", legend style={font=\tiny, at={(1.1,1.35)}, anchor=south, draw=none},"
                     r" legend columns=2")
        lines.append(rf"\nextgroupplot[{opts}]")
        if i == 0:
            for _, _, name, colour in ARMS:
                lines.append(rf"\addlegendimage{{{colour}, thick}}")
                lines.append(rf"\addlegendentry{{{name}}}")
        for r, _, _, colour in ARMS:
            for c in inp["curves"][ds][r]:
                pts = " ".join(f"({k},{v:.4f})" for k, v in enumerate(c, start=1))
                lines.append(rf"\addplot[{colour}, opacity=0.35, forget plot] "
                             rf"coordinates {{{pts}}};")
    lines += [r"\end{groupplot}", r"\end{tikzpicture}", ""]
    return "\n".join(lines)


def placeholder_figure() -> str:
    return ("% auto-generated by 10_meeting01_learning_curves.py -- do not edit\n"
            "\\fbox{\\parbox[c][3cm][c]{0.9\\columnwidth}{\\centering [learning curves: "
            "run 10\\_meeting01\\_learning\\_curves.py --report]}}\n")


def write(data_dir: pathlib.Path, macros: dict, fig: str) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    lines = ["% auto-generated by 10_meeting01_learning_curves.py -- do not edit"]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()]
    (data_dir / "paper_loso_learning.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (data_dir / "paper_loso_learning_figure.tex").write_text(fig, encoding="utf-8")


def report(results: pathlib.Path, data_dir: pathlib.Path) -> None:
    inp = read_inputs(results)
    macros = findings(inp)  # refuses before any paper file is written
    write(data_dir, macros, figure(inp))
    for ds in inp["datasets"]:
        g = gaps(inp["pairs"][ds])
        print(f"[learning] {ds:10s} pairs {g['n']:3d}  train gap {g['train']:.4f}  test gap "
              f"{g['test']:.4f}  share {100 * g['train'] / g['test']:.0f}%  positive "
              f"{g['positive']}/{g['n']}")
    for k, v in macros.items():
        print(f"  \\{k} = {v}")


# --------------------------------------------------------------------------- self-test

def _write_events(path: pathlib.Path, fits: list[tuple[str, str, list[float]]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for cid, role, losses in fits:
            f.write(json.dumps({"type": "config_begin", "config_id": cid}) + "\n")
            for k, v in enumerate(losses, start=1):
                f.write(json.dumps({"type": "epoch", "config_id": cid, "epoch": k,
                                    "train_loss": v}) + "\n")
            f.write(json.dumps({"type": "train_end", "config_id": cid,
                                "epochs_run": len(losses)}) + "\n")
            f.write(json.dumps({"type": "config_end", "config_id": cid, "role": role}) + "\n")


def self_test() -> int:
    import tempfile

    failures = []

    def expect(ok, what):
        if not ok:
            failures.append(what)

    def refused(fn, needle, what):
        try:
            fn()
            failures.append(f"{what}: not refused")
        except ResultsError as e:
            expect(needle in str(e), f"{what}: message lacks {needle!r}: {e}")

    # Gaps: two pairs, train gaps 0.04 and 0.02 (mean 0.03), test gaps 0.05 and 0.03 (0.04).
    pairs = [{"snn_train": 0.54, "ann_train": 0.50, "snn_test": 0.65, "ann_test": 0.60},
             {"snn_train": 0.42, "ann_train": 0.40, "snn_test": 0.53, "ann_test": 0.50}]
    g = gaps(pairs)
    expect(abs(g["train"] - 0.03) < 1e-12 and abs(g["test"] - 0.04) < 1e-12
           and g["positive"] == 2, f"gaps {g}")
    eeg = [{"snn_train": 0.30, "ann_train": 0.20, "snn_test": 0.40, "ann_test": 0.28}]
    inp = {"datasets": ["fsdd", "eegmmidb"], "pairs": {"fsdd": pairs, "eegmmidb": eeg}}
    m = findings(inp)
    expect(m == {"learnDataStatus": "results", "learnGapSpeech": "0.030", "learnGapEeg": "0.100",
                 "learnShare": "75 to 83", "learnPairs": "3 of 3"}, f"findings {m}")
    # A dataset whose SNN-AE fits its training windows better: the paragraph would be wrong.
    inp["pairs"]["eegmmidb"] = [{"snn_train": 0.19, "ann_train": 0.20, "snn_test": 0.40,
                                 "ann_test": 0.28}]
    refused(lambda: findings(inp), "Rewrite the paragraph", "a negative train gap")

    with tempfile.TemporaryDirectory() as tmp:
        p = pathlib.Path(tmp) / "e.jsonl"
        _write_events(p, [("s1", "snn_final", [1.0, 0.8, 0.7]), ("a1", "ann_final", [0.9, 0.6])])
        c = read_curves(p, 1)
        expect(c == {"snn_final": [[1.0, 0.8, 0.7]], "ann_final": [[0.9, 0.6]]}, f"curves {c}")
        refused(lambda: read_curves(p, 2), "2 runs configured", "a missing final fit")
        # A resumed task that trained a final fit twice.
        _write_events(p, [("s1", "snn_final", [1.0]), ("s1", "snn_final", [1.0])])
        refused(lambda: read_curves(p, 1), "ends twice", "a final fit trained twice")
        # An epoch hole.
        p.write_text("\n".join(json.dumps(e) for e in (
            {"type": "config_begin", "config_id": "s1"},
            {"type": "epoch", "config_id": "s1", "epoch": 1, "train_loss": 1.0},
            {"type": "epoch", "config_id": "s1", "epoch": 3, "train_loss": 0.9},
            {"type": "train_end", "config_id": "s1", "epochs_run": 2},
            {"type": "config_end", "config_id": "s1", "role": "snn_final"})) + "\n")
        refused(lambda: read_curves(p, 0), "not 1..n", "an epoch hole")

    fig = figure({"datasets": ["fsdd", "eegmmidb"],
                  "curves": {"fsdd": {"snn_final": [[1.0, 0.8]], "ann_final": [[0.9, 0.6, 0.5]]},
                             "eegmmidb": {"snn_final": [[1.0]], "ann_final": [[0.9]]}}})
    expect("xmax=3" in fig and fig.count("\\nextgroupplot") == 2
           and fig.count("forget plot") == 4 and fig.count("\\addlegendentry") == 2
           and "(1,1.0000) (2,0.8000)" in fig, f"figure:\n{fig}")
    expect(fig.count("{") == fig.count("}") and fig.count("[") == fig.count("]"),
           f"figure brackets unbalanced:\n{fig}")
    for f in failures:
        print("FAIL", f)
    print(f"[learning] self-test: {'FAILED' if failures else 'ok'}")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    for m in ("--report", "--placeholders", "--self-test"):
        mode.add_argument(m, action="store_true")
    ap.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    ap.add_argument("--data-dir")
    a = ap.parse_args(argv)
    try:
        if a.self_test:
            return self_test()
        if not a.data_dir:
            ap.error("--report and --placeholders need --data-dir")
        if a.placeholders:
            write(pathlib.Path(a.data_dir),
                  {k: "$-$" for k in MACROS} | {"learnDataStatus": "placeholder"},
                  placeholder_figure())
            return 0
        report(pathlib.Path(a.results_dir), pathlib.Path(a.data_dir))
        return 0
    except ResultsError as exc:
        print(f"[learning] REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
