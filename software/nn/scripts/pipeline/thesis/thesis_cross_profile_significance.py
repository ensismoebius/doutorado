#!/usr/bin/env python3
"""Cross-profile significance tests for Experiment 05 — the Guayaquil SNN-vs-LSTM analog.

The Guayaquil (Guayaquil) pipeline reports Cohen's d / t-test / Wilcoxon *within* one run
because it trains two model families (SNN, LSTM) in the same process. An Thesis run
scores exactly one feature set, so the equivalent comparison is *across profiles*:
this script collates every profile's per-fold metrics and tests each condition
against the best one.

Data source: each run writes `results/.../e05_<run_tag>_metrics.csv` with one row per
outer fold (columns `accuracy`, `eer`, ...). Runs with `experiment.repeats > 1` produce
`<run_tag>_rep0/1/2`; folds are pooled across the repeats of the same base run_tag.

Comparison:
  * reference   = the condition with the best mean of the chosen metric
                  (lowest EER, or highest accuracy), or --reference <run_tag>.
  * per condition vs reference: Cohen's d (effect size) plus a significance test.
  * pairing (default auto): when the two conditions have the same fold count the
    folds are treated as paired (same CV seed/splits → paired t + Wilcoxon
    signed-rank); otherwise independent (Welch t + Mann-Whitney U). Force with
    --pairing paired|independent.

NaN folds (closed-set metrics under the verification protocol) are dropped per metric.

Usage:
    python3 scripts/pipeline/thesis/thesis_cross_profile_significance.py \
        --results-dir results/thesis/phase01 \
        --metric eer \
        --out results/thesis/phase01/cross_profile_significance.csv
"""
import argparse
import csv
import glob
import math
import os
import re
import sys
from collections import defaultdict

try:
    import numpy as np
    from scipy import stats
except ImportError as exc:  # pragma: no cover - environment guard
    sys.exit(
        f"missing dependency ({exc}). This script needs numpy + scipy.\n"
        "Run it with the system interpreter (python3), which has both, or "
        "`pip install scipy` into the venv."
    )

_REP_SUFFIX = re.compile(r"_rep\d+$")
_METRIC_FILE = re.compile(r"^e05_(.+)_metrics\.csv$")

# Which direction is "better", used to pick the reference condition.
LOWER_IS_BETTER = {"eer"}


def base_run_tag(run_tag: str) -> str:
    """Strip a trailing _repN so a profile's repeats collapse to one condition."""
    return _REP_SUFFIX.sub("", run_tag)


def collect(results_dir: str, metric: str):
    """condition -> {'values': [per-fold metric], 'label': feature_set, 'files': n}."""
    conds = defaultdict(lambda: {"values": [], "labels": set(), "files": 0})
    pattern = os.path.join(results_dir, "**", "e05_*_metrics.csv")
    for path in sorted(glob.glob(pattern, recursive=True)):
        m = _METRIC_FILE.match(os.path.basename(path))
        if not m:
            continue
        cond = base_run_tag(m.group(1))
        with open(path, newline="") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None or metric not in reader.fieldnames:
                continue
            conds[cond]["files"] += 1
            for row in reader:
                try:
                    v = float(row[metric])
                except (TypeError, ValueError):
                    continue
                if math.isnan(v):
                    continue
                conds[cond]["values"].append(v)
                conds[cond]["labels"].add(row.get("feature_set", cond))
    return conds


def cohens_d(a, b):
    """Effect size a-vs-b with pooled SD (Hedges-style pooled variance)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    sp2 = ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)
    if sp2 == 0.0:
        return 0.0
    return float((a.mean() - b.mean()) / math.sqrt(sp2))


def compare(ref_vals, cur_vals, pairing):
    """Return (test_name, p_value, cohens_d) for cur vs ref."""
    ref, cur = np.asarray(ref_vals, float), np.asarray(cur_vals, float)
    d = cohens_d(cur, ref)
    paired = pairing == "paired" or (pairing == "auto" and len(ref) == len(cur))
    if paired and len(ref) == len(cur) and len(ref) >= 2:
        # Same CV seed/splits → fold i is the same held-out data in both runs.
        if np.allclose(ref, cur):
            return "wilcoxon(paired)", 1.0, d
        try:
            _, p = stats.wilcoxon(cur, ref)
        except ValueError:
            _, p = stats.ttest_rel(cur, ref)
            return "ttest_rel(paired)", float(p), d
        return "wilcoxon(paired)", float(p), d
    if len(ref) >= 1 and len(cur) >= 1:
        _, p = stats.mannwhitneyu(cur, ref, alternative="two-sided")
        return "mannwhitneyu(indep)", float(p), d
    return "n/a", float("nan"), d


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="results/thesis/phase01",
                    help="Directory holding e05_<run_tag>_metrics.csv files (searched recursively).")
    ap.add_argument("--metric", default="eer", choices=["eer", "accuracy", "auc", "f1"],
                    help="Metric to compare on (default: eer).")
    ap.add_argument("--reference", default=None,
                    help="Base run_tag to use as reference (default: best mean metric).")
    ap.add_argument("--pairing", default="auto", choices=["auto", "paired", "independent"],
                    help="Pair folds across profiles (default auto: paired iff equal fold counts).")
    ap.add_argument("--out", default=None, help="Write the comparison table to this CSV.")
    args = ap.parse_args()

    conds = collect(args.results_dir, args.metric)
    conds = {c: v for c, v in conds.items() if len(v["values"]) >= 1}
    if len(conds) < 2:
        print(f"[cross-sig] found {len(conds)} comparable condition(s) under "
              f"{args.results_dir} — need >=2 to compare. Nothing to do.")
        return 0

    lower = args.metric in LOWER_IS_BETTER

    def mean_of(c):
        return float(np.mean(conds[c]["values"]))

    ranked = sorted(conds, key=mean_of, reverse=not lower)
    if args.reference is not None:
        if args.reference not in conds:
            sys.exit(f"--reference '{args.reference}' not among conditions: {sorted(conds)}")
        ref = args.reference
    else:
        ref = ranked[0]

    arrow = "lower=better" if lower else "higher=better"
    print(f"\n=== Thesis cross-profile significance on '{args.metric}' ({arrow}) ===")
    print(f"conditions: {len(conds)}   reference: {ref}\n")
    print(f"{'rank':>4}  {'condition':<44} {'folds':>5} {'mean':>9} {'std':>9}")
    for i, c in enumerate(ranked, 1):
        vals = np.asarray(conds[c]["values"], float)
        tag = " (ref)" if c == ref else ""
        print(f"{i:>4}  {c[:44]:<44} {len(vals):>5} {vals.mean():>9.4f} "
              f"{vals.std(ddof=1) if len(vals) > 1 else 0.0:>9.4f}{tag}")

    ref_vals = conds[ref]["values"]
    rows = []
    print(f"\n{'condition':<44} {'Δmean':>9} {'cohens_d':>9} {'p':>10}  test")
    for c in ranked:
        if c == ref:
            continue
        test, p, d = compare(ref_vals, conds[c]["values"], args.pairing)
        dmean = mean_of(c) - mean_of(ref)
        star = "  *" if (not math.isnan(p) and p < 0.05) else ""
        print(f"{c[:44]:<44} {dmean:>9.4f} {d:>9.3f} {p:>10.4g}  {test}{star}")
        rows.append({
            "reference": ref, "condition": c, "metric": args.metric,
            "n_folds_ref": len(ref_vals), "n_folds_cond": len(conds[c]["values"]),
            "mean_ref": mean_of(ref), "mean_cond": mean_of(c), "delta_mean": dmean,
            "cohens_d": d, "p_value": p, "test": test, "significant_0.05": int(p < 0.05) if not math.isnan(p) else 0,
        })

    print("\n  * p < 0.05.  Δmean = condition − reference.  cohens_d > 0 ⇒ condition's "
          f"{args.metric} is larger than the reference's.")
    print("  Paired tests assume the compared runs share CV seed/splits (same fold i data).\n")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[cross-sig] wrote {len(rows)} comparison rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
