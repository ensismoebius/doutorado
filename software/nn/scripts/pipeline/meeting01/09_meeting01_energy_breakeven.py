#!/usr/bin/env python3
"""09_meeting01_energy_breakeven.py -- when can a spike-coded input save energy at all? The
paper's break-even figure and the numbers its sentence states.

Why it exists. A spiking first layer replaces each multiply-accumulate by one addition per input
spike, which is the usual argument for SNN efficiency. In this paper's energy model
(Meeting01Energy.hpp: Lemaire et al. 2022 at Horowitz 2014 prices) one synaptic event is NOT
cheaper than one MAC: it reads the weight and reads and writes the target's membrane potential,
so it costs about 47 pJ against a MAC's 4.6 pJ plus one ~26 pJ weight read. And the input is
not sparse: Poisson draws one Bernoulli per sample and step with p = the window's min-max scaled
value, and a z-scored window's mean scaled value is near 0.45, so the input spikes per sample
grow as 0.45 x T. The reviewers (review/review_visual.md V006, ruling_visual.md: "satisfied only
with the corrected curve, recurrent marked, and EEG widths shown") asked for this to be drawn,
not argued.

What it computes, at each T of the SNN-AE search's range and each first hidden width the
pre-validation selected (FSDD and EEGMMIDB winners, development folds):

    ref(w)        the ANN-AE's first layer, Linear(256 -> w) once + leaky ReLU
    poisson(w,T)  Poisson generator + Linear on spikes over T steps + the LIF's leak/threshold
    recurrent(w,T) the direct code through the recurrent input transform (one LIF per sample,
                  which emits the spikes) + the same spike-fed Linear and LIF
    break-even    the input spikes per sample and window, theta/M, at which a spike-fed stage
                  costs ref(w); below it the spike-fed stage is cheaper
    codes         where the codes put theta/M: latency exactly 1 (one spike per sample);
                  Poisson p_bar x T, p_bar measured on the fold's test windows as the encoder
                  scales them; the recurrent transform anywhere, set by its own genes

first_stage_energy_nj (meeting01_results.py) is a port of the C++ building blocks, pinned to
them: MemoryAwareEnergy.FirstStageOfTheBreakEvenFigure (parameter_count_gtest.cpp) and this
script's --self-test assert the same three totals. The first hidden LIF's own spikes are not
charged, so every break-even here is an upper bound (a spike-fed stage priced at its cheapest).

Usage (from software/nn):
    09_meeting01_energy_breakeven.py --report --data-dir X [--prevalidation D]
        X/paper_breakeven.tex          macros the paper's sentence states (claims checked)
        X/paper_breakeven_figure.tex   the figure (pgfplots), generated
    09_meeting01_energy_breakeven.py --placeholders --data-dir X
    09_meeting01_energy_breakeven.py --self-test

Refuses (no fallbacks): a pre-validation of a results_format it does not read, a winner it
cannot read (07_'s read_winners), configurations that disagree between the two folds on the
window or the T range, and any claim of the paper's sentence that the numbers do not support.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import pathlib
import sys

import numpy as np

_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("design07", _SCRIPT_DIR / "07_meeting01_design_study.py")
D = importlib.util.module_from_spec(_spec)
sys.modules["design07"] = D
_spec.loader.exec_module(D)

sys.path.insert(0, str(_SCRIPT_DIR))
from meeting01_results import (
    DATASET_SHORT,
    DEVELOPMENT_FOLDS,
    READS_WITHOUT_ANN_SHARE,
    ResultsError,
    break_even_spikes,
    first_stage_energy_nj,
    split_manifest,
)

STAGES = (("poisson_dense", "Poisson"), ("direct_recurrent", "recurrent"))
MACROS = ("beDataStatus", "beTmin", "beTmax", "beRef", "bePoissonHigh", "bePoissonLow",
          "beRecurrentHigh", "beRecurrentLow", "bePbar", "bePoissonAtTmin")


def read_inputs(prevalidation: pathlib.Path) -> dict:
    """Window, T range, the winners' first widths per dataset, and the measured Poisson p_bar."""
    d = prevalidation / "snn-ae"
    winners = D.read_winners(prevalidation)
    windows, tranges, pbar = set(), set(), {}
    for ds in D.DATASETS:
        fold = min(DEVELOPMENT_FOLDS[ds])
        man = split_manifest(d, D.RUN_TAG, ds, fold, accept=READS_WITHOUT_ANN_SHARE)
        cfg = man["config"]
        windows.add(int(man["source"]["window_size"]))
        ga = cfg["evaluation"]["ga"]["snn"]
        tranges.add((int(ga["time_steps_min"]), int(ga["time_steps_max"])))
        x = np.load(d / f"{D.RUN_TAG}_{ds}_fold{fold}_target_test_windows.npy").astype(np.float64)
        lo = x.min(axis=1, keepdims=True)
        span = np.maximum(x.max(axis=1, keepdims=True) - lo, 1e-6)  # encode_sample's guard
        pbar[ds] = float(np.clip((x - lo) / span, 0.0, 1.0).mean())
    if len(windows) != 1 or len(tranges) != 1:
        raise ResultsError(f"the development folds disagree: windows {windows}, T ranges {tranges}")
    widths = {}
    for w in winners:
        widths.setdefault(w["dataset"], set()).add(int(w["widths"][0]))
    return {"window": windows.pop(), "trange": tranges.pop(),
            "widths": {ds: sorted(v) for ds, v in widths.items()}, "pbar": pbar}


def curves(inp: dict) -> dict:
    """(stage, dataset, width) -> [(T, theta/M at break-even)]."""
    m = inp["window"]
    t0, t1 = inp["trange"]
    out = {}
    for stage, _ in STAGES:
        for ds, ws in inp["widths"].items():
            for w in ws:
                out[(stage, ds, w)] = [(t, break_even_spikes(stage, m, w, t) / m) for t in range(t0, t1 + 1)]
    return out


def _r(x: float, nd: int = 2) -> str:
    return f"{x:.{nd}f}"


def findings(inp: dict, cv: dict) -> dict:
    """The macros, after checking every claim the paper's sentence makes about them."""
    t0, t1 = inp["trange"]
    p = {k: v for k, v in cv.items() if k[0] == "poisson_dense"}
    r = {k: v for k, v in cv.items() if k[0] == "direct_recurrent"}
    p_hi = max(v for c in p.values() for _, v in c)
    p_lo = min(v for c in p.values() for _, v in c)
    r_hi = max(v for c in r.values() for _, v in c)
    r_lo = min(v for c in r.values() for _, v in c)
    # Claim 1: latency (theta/M = 1) is above every break-even.
    if max(p_hi, r_hi) >= 1.0:
        raise ResultsError(f"a break-even reaches {max(p_hi, r_hi):.3f} >= 1: the sentence says latency "
                           "never saves energy")
    # Claim 2: each dataset's Poisson code sits above its break-even at every T of the range.
    for (stage, ds, w), c in p.items():
        for t, be in c:
            if inp["pbar"][ds] * t <= be:
                raise ResultsError(f"{ds} w={w} T={t}: Poisson spikes {inp['pbar'][ds] * t:.3f} per sample "
                                   f"are below the break-even {be:.3f}; the sentence says they never are")
    # Claim 3: break-even falls with T (the sentence says "falling with T").
    for c in cv.values():
        vals = [v for _, v in c]
        if any(b > a + 1e-12 for a, b in itertools.pairwise(vals)):
            raise ResultsError("a break-even rises with T somewhere; the sentence says it falls")
    pb = sorted(inp["pbar"].values())
    ref = sorted({first_stage_energy_nj("ann", inp["window"], w) for ws in inp["widths"].values() for w in ws})
    # The sentence reads "between <High> at T=Tmin and <Low> at T=Tmax": with claim 3, the
    # extremes over widths sit at the ends of the T range.
    for name, cs, hi, lo in (("Poisson", p, p_hi, p_lo), ("recurrent", r, r_hi, r_lo)):
        if max(v for c in cs.values() for t, v in c if t == t0) != hi or \
                min(v for c in cs.values() for t, v in c if t == t1) != lo:
            raise ResultsError(f"{name}: the break-even extremes are not at T={t0} and T={t1}")
    if r_lo == 0.0:
        rec_low = (f"to none at $T={t1}$, where for the narrowest winner the transform alone costs "
                   "more than the analog layer")
    else:
        rec_low = f"to {_r(r_lo)} at $T={t1}$"
    m = {"beDataStatus": "results", "beTmin": str(t0), "beTmax": str(t1),
         "beRef": f"{ref[0]:.0f}--{ref[-1]:.0f}" if len(ref) > 1 else f"{ref[0]:.0f}",
         "bePoissonHigh": _r(p_hi), "bePoissonLow": _r(p_lo),
         "beRecurrentHigh": _r(r_hi), "beRecurrentLow": rec_low,
         "bePbar": f"{_r(pb[0])}--{_r(pb[-1])}" if len(pb) > 1 else _r(pb[0]),
         "bePoissonAtTmin": _r(pb[0] * t0)}
    if set(m) != set(MACROS):
        raise ResultsError(f"macros {sorted(set(m) ^ set(MACROS))} out of sync with MACROS")
    return m


_COLORS = {"fsdd": "red!75!black", "eegmmidb": "blue!70!black"}
_MARKS = ("*", "square*", "triangle*")


def figure(inp: dict, cv: dict) -> str:
    t0, t1 = inp["trange"]
    # Log y: the Poisson code's spikes grow as p_bar T, up to ~16 per sample, while the
    # break-evens sit near 0.1-0.6. A break-even of 0 (the stage is dearer even with no spike)
    # has no point on a log axis: the curve stops there, and the caption says so.
    y_hi = 10 ** np.ceil(np.log10(max(inp["pbar"].values()) * t1))
    y_lo = 10 ** np.floor(np.log10(min(v for c in cv.values() for _, v in c if v > 0)))
    lines = ["% auto-generated by 09_meeting01_energy_breakeven.py -- do not edit",
             "\\begin{tikzpicture}[font=\\scriptsize]",
             (f"\\begin{{axis}}[width=0.98\\columnwidth, height=5.6cm, xmin={t0}, xmax={t1}, ymode=log,"
              f" ymin={y_lo:g}, ymax={y_hi:g},"),
             "  xlabel={$T$ (simulation steps)}, ylabel={input spikes per sample, $\\theta/M$},",
             ("  legend style={font=\\tiny, at={(0.5,-0.25)}, anchor=north, draw=none},"
              " legend columns=2, legend cell align=left, ymajorgrids, yminorgrids=false]")]
    for stage, label in STAGES:
        dash = "solid" if stage == "poisson_dense" else "dashed"
        for ds, ws in inp["widths"].items():
            for i, w in enumerate(ws):
                pts = " ".join(f"({t},{v:.4f})" for t, v in cv[(stage, ds, w)] if v > 0)
                lines.append(f"\\addplot[{dash}, thick, color={_COLORS[ds]}, mark={_MARKS[i]}, mark size=1pt,"
                             f" mark repeat=6] coordinates {{{pts}}};")
                lines.append(f"\\addlegendentry{{{label}, {DATASET_SHORT[ds]} $w_1{{=}}{w}$}}")
    for ds, pb in inp["pbar"].items():
        lines.append(f"\\addplot[dotted, thick, color={_COLORS[ds]}, domain={t0}:{t1}, samples=2] {{{pb:.4f}*x}};")
        lines.append(f"\\addlegendentry{{$\\bar p\\,T$, {DATASET_SHORT[ds]}, $\\bar p={pb:.2f}$}}")
    lines.append(f"\\addplot[gray, very thick, domain={t0}:{t1}, samples=2] {{1}};")
    lines.append("\\addlegendentry{latency code}")
    lines += ["\\end{axis}", "\\end{tikzpicture}"]
    return "\n".join(lines) + "\n"


def write(data_dir: pathlib.Path, macros: dict, fig: str) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    lines = ["% auto-generated by 09_meeting01_energy_breakeven.py -- do not edit"]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()]
    (data_dir / "paper_breakeven.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (data_dir / "paper_breakeven_figure.tex").write_text(fig, encoding="utf-8")


def placeholder_figure() -> str:
    return ("% auto-generated by 09_meeting01_energy_breakeven.py -- do not edit\n"
            "\\fbox{\\parbox[c][3cm][c]{0.9\\columnwidth}{\\centering [break-even figure: "
            "run 09\\_meeting01\\_energy\\_breakeven.py --report]}}\n")


def report(prevalidation: pathlib.Path, data_dir: pathlib.Path) -> None:
    inp = read_inputs(prevalidation)
    cv = curves(inp)
    macros = findings(inp, cv)  # refuses before any paper file is written
    write(data_dir, macros, figure(inp, cv))
    print(f"[breakeven] window {inp['window']}, T {inp['trange']}, widths {inp['widths']}, "
          f"p_bar {', '.join(f'{k} {v:.3f}' for k, v in inp['pbar'].items())}")
    for (stage, ds, w), c in cv.items():
        print(f"  {stage:16s} {ds:9s} w1={w:3d}  " + "  ".join(f"T{t}:{v:.3f}" for t, v in c
                                                             if t in (c[0][0], 4, 8, 16, 28, c[-1][0])))
    for k, v in macros.items():
        print(f"  \\{k} = {v}")


def self_test() -> int:
    failures = []

    def check(name, got, want, tol=1e-9):
        if abs(got - want) > tol:
            failures.append(f"{name}: {got!r} != {want!r}")

    # The C++ pins (parameter_count_gtest.cpp, MemoryAwareEnergy.FirstStageOfTheBreakEvenFigure).
    check("ann 256->59", first_stage_energy_nj("ann", 256, 59), 409.5424320690525)
    check("poisson T4 300", first_stage_energy_nj("poisson_dense", 256, 59, 4, 300.0), 796.4763554435484)
    check("recurrent T4 300", first_stage_energy_nj("direct_recurrent", 256, 59, 4, 300.0), 818.6599554435484)
    # Break-even is where the two costs meet.
    for stage in ("poisson_dense", "direct_recurrent"):
        be = break_even_spikes(stage, 256, 59, 4)
        check(f"{stage} meets ann", first_stage_energy_nj(stage, 256, 59, 4, be),
              first_stage_energy_nj("ann", 256, 59), 1e-6)
    # A stage dearer than the reference with no spike gives 0, not a negative count.
    if break_even_spikes("direct_recurrent", 256, 35, 32) != 0.0:
        failures.append("recurrent w=35 T=32 should be dearer than the ann layer at 0 spikes")
    for bad in (lambda: first_stage_energy_nj("rate", 256, 59),
                lambda: first_stage_energy_nj("ann", 256, 59, 4),
                lambda: first_stage_energy_nj("poisson_dense", 4, 2, 2, 9.0)):
        try:
            bad()
            failures.append("a bad stage was priced")
        except ResultsError:
            pass
    # findings refuses a Poisson code below its break-even.
    inp = {"window": 256, "trange": (2, 4), "widths": {"fsdd": [59]}, "pbar": {"fsdd": 0.1}}
    try:
        findings(inp, curves(inp))
        failures.append("findings accepted a Poisson code below break-even")
    except ResultsError:
        pass
    inp["pbar"] = {"fsdd": 0.45}
    m = findings(inp, curves(inp))
    if m["beTmin"] != "2" or m["bePoissonAtTmin"] != "0.90":
        failures.append(f"findings macros {m}")
    fig = figure(inp, curves(inp))
    if "\\begin{tikzpicture}" not in fig or fig.count("\\addplot") != 4:
        failures.append("figure: expected 2 break-even curves, 1 Poisson line and the latency line")
    for f in failures:
        print("FAIL", f)
    print(f"[breakeven] self-test: {'FAILED' if failures else 'ok'}")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    for m in ("--report", "--placeholders", "--self-test"):
        mode.add_argument(m, action="store_true")
    ap.add_argument("--prevalidation", default=D.DEFAULT_PREVALIDATION)
    ap.add_argument("--data-dir")
    a = ap.parse_args(argv)
    try:
        if a.self_test:
            return self_test()
        if not a.data_dir:
            ap.error("--report and --placeholders need --data-dir")
        if a.placeholders:
            write(pathlib.Path(a.data_dir), {k: "$-$" for k in MACROS} | {"beDataStatus": "placeholder"},
                  placeholder_figure())
            return 0
        report(pathlib.Path(a.prevalidation), pathlib.Path(a.data_dir))
        return 0
    except ResultsError as exc:
        print(f"[breakeven] REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
