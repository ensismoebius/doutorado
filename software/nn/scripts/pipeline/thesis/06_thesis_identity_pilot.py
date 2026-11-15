#!/usr/bin/env python3
"""Verdict of the identity pilot: does resynthesis alone change the speaker-verification EER?

THE QUESTION. Every WORLD dysphonia profile is compared with `resynthesis_control`, never with
the raw voice, because resynthesis alone changes the signal. That comparison is only clean if
the control itself leaves the speaker where it was. If the control moved the EER as much as a
mild dysphonia does, "profile minus control" would still be valid, but the thesis could no
longer say the profiles act on a natural-sounding voice of the same speaker.

THE RULE (fixed before any pilot result: proposal section 4.3; plain paired t in d93b25b11,
the corrected t of the thesis since, still before any result).
    pair      = one (repeat, fold): the same split, raw arm and control arm
    dEER      = EER(control) - EER(raw), 15 pairs per scope (3 repeats x 5 folds)
    margin    = 0.01 EER
    interval  corrected resampled t (Nadeau and Bengio 2003), the test the thesis uses for
              repeated k-fold: variance x (1/J + n_test/n_train) = (1/15 + 1/4), df 14
    transparent      95 % interval of mean dEER inside +-margin
                     AND each repeat's mean dEER inside +-margin   (cosine, both scopes)
    not transparent  the interval entirely outside +-margin for a scope
    inconclusive     anything else
`cosine` decides (no training: only the test features differ between arms); `dsnn` is
reported with the same statistics and decides nothing.

Every input file must exist: a missing repeat is an error, not a smaller n.

Run from software/nn after the six pilot profiles:
    python3 scripts/pipeline/thesis/06_thesis_identity_pilot.py \\
        --results results/thesis/pilot --out results/thesis/pilot/pilot_verdict.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics

MARGIN = 0.01
REPEATS = 3
FOLDS = 5
KINDS = ("cosine", "dsnn")
SCOPES = ("probes", "test")
# Two-sided 95 % Student t quantiles (df = n - 1), the only n this pilot has.
T95 = {14: 2.1447866879}
TEST_OVER_TRAIN = 1.0 / (FOLDS - 1)   # one fold tests, the other four train


def fold_eers(results, name):
    out = {}
    for rep in range(REPEATS):
        path = os.path.join(results, f"e05_e05_{name}_rep{rep}_metrics.csv")
        if not os.path.exists(path):
            raise SystemExit(f"[pilot] {path} is missing: run the profile pilot/{name}.json "
                             f"(all {REPEATS} repeats) first")
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                out[(rep, int(row["fold"]))] = float(row["eer"])
    if len(out) != REPEATS * FOLDS:
        raise SystemExit(f"[pilot] {name}: {len(out)} (repeat, fold) EERs, expected "
                         f"{REPEATS * FOLDS}")
    return out


def compare(raw, control):
    keys = sorted(raw)
    if sorted(control) != keys:
        raise SystemExit("[pilot] the raw and control arms do not have the same splits")
    d = [control[k] - raw[k] for k in keys]
    n = len(d)
    mean = statistics.mean(d)
    sd = statistics.stdev(d)
    half = T95[n - 1] * sd * math.sqrt(1.0 / n + TEST_OVER_TRAIN)
    lo, hi = mean - half, mean + half
    per_repeat = [statistics.mean(control[(r, f)] - raw[(r, f)] for f in range(FOLDS))
                  for r in range(REPEATS)]
    if -MARGIN < lo and hi < MARGIN and all(abs(m) < MARGIN for m in per_repeat):
        verdict = "transparent"
    elif hi < -MARGIN or lo > MARGIN:
        verdict = "not transparent"
    else:
        verdict = "inconclusive"
    return {"n_pairs": n, "mean_raw_eer": round(statistics.mean(raw.values()), 4),
            "mean_control_eer": round(statistics.mean(control.values()), 4),
            "mean_delta": round(mean, 4), "sd_delta": round(sd, 4),
            "ci95": [round(lo, 4), round(hi, 4)],
            "per_repeat_mean_delta": [round(m, 4) for m in per_repeat],
            "verdict": verdict}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, help="results/thesis/pilot")
    ap.add_argument("--out", required=True, help="the verdict JSON to write")
    args = ap.parse_args()

    table = {}
    for kind in KINDS:
        raw = fold_eers(args.results, f"pil_{kind}_voice_raw")
        table[kind] = {scope: compare(raw, fold_eers(args.results,
                                                      f"pil_{kind}_voice_control_{scope}"))
                       for scope in SCOPES}
    decisive = [table["cosine"][s]["verdict"] for s in SCOPES]
    if all(v == "transparent" for v in decisive):
        overall = "transparent"
    elif any(v == "not transparent" for v in decisive):
        overall = "not transparent"
    else:
        overall = "inconclusive"
    result = {"margin": MARGIN, "decides": "cosine", "overall": overall, "arms": table}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
