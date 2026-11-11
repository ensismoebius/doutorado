#!/usr/bin/env python3
"""Cross-profile significance tests for the thesis — the Meeting01 SNN-vs-LSTM analog.

Meeting01 reports Cohen's d / t-test / Wilcoxon *within* one run because it trains two
model families in the same process. A thesis run scores exactly one feature set, so the
equivalent comparison is *across profiles*: this script collates each profile's per-fold
metrics and tests each condition against a declared reference.

Data. Each profile declares its run_tag and repeats; its runs write
`e05_<run_tag>[_rep<k>]_metrics.csv`, one row per outer fold. A condition is one profile;
its folds are pooled across its repetitions and keyed by (repetition, fold).

Every choice is declared — nothing has a default:
  --metric      eer | auc | accuracy | f1
  --reference   best (the condition with the best mean: lowest EER, highest otherwise)
                or the run_tag of one profile
  --pairing     paired      Wilcoxon signed-rank on (repetition, fold)-matched values.
                            Refused for any pair whose outer folds are not the same
                            held-out data: the split is fixed by dataset.root,
                            dataset.max_samples, experiment.seed, experiment.repeats,
                            experiment.seed_deterministic, training.k_folds,
                            training.nested_cv and classifier.text_mode, so those must
                            be equal (nested_cv is included conservatively).
                independent Mann-Whitney U, valid for any pair.

Refused rather than skipped: a missing metrics file, a file without the metric column, a
non-numeric or non-finite value (closed-set metrics such as accuracy are NaN under the
verification protocol — compare on eer or auc there), more than one feature set in a
run, fewer than two folds or two conditions. Where a statistic does not exist — Wilcoxon
on identical paired values, Cohen's d with zero pooled variance — the row says so in
`note` and leaves the number empty instead of writing a stand-in (p = 1, d = 0).

Multiplicity. Every condition is compared with the same reference, so the comparisons
are one family: `p_holm` is the Holm-Bonferroni adjusted p over the comparisons that have
a test (a row with no test is not part of the family), and `significant_0.05` reads
`p_holm`, not the raw `p_value`. Holm controls the family-wise error without assuming
independence between the comparisons (they share the reference, so they are not
independent). It does not undo selection: with `--reference best` the reference was
picked from the same data, which biases every comparison towards a difference — name a
reference fixed in advance (`--reference <run_tag>`) for a confirmatory reading.

Usage:
    python3 scripts/pipeline/thesis/thesis_cross_profile_significance.py \
        --profiles-dir src/experiments/thesis/profiles/phase01 \
        --results-dir  results/thesis/phase01 \
        --metric eer --reference best --pairing independent \
        --out results/thesis/phase01/cross_profile_significance.csv
"""
import argparse
import csv
import glob
import json
import math
import os
import sys

try:
    import numpy as np
    from scipy import stats
except ImportError as exc:  # pragma: no cover - environment guard
    sys.exit(
        f"missing dependency ({exc}). This script needs numpy + scipy.\n"
        "Run it with the system interpreter (python3), which has both, or "
        "`pip install scipy` into the venv."
    )

# Which direction is "better", used by --reference best.
LOWER_IS_BETTER = {"eer"}
METRICS = ("eer", "auc", "accuracy", "f1")

# The declared keys that fix which samples outer fold i holds out (GroupKFoldPolicy seeded
# by experiment.seed over the dataset's samples, after the text_mode split).
SPLIT_KEYS = (
    ("dataset", "root"), ("dataset", "max_samples"),
    ("experiment", "seed"), ("experiment", "repeats"), ("experiment", "seed_deterministic"),
    ("training", "k_folds"), ("training", "nested_cv"),
    ("classifier", "text_mode"),
)


def fail(msg):
    sys.exit(f"[cross-sig] {msg}")


def split_signature(prof):
    """The split-defining keys; seed_deterministic exists only for repeats > 1."""
    sig = {}
    for block, key in SPLIT_KEYS:
        if block == "experiment" and key == "seed_deterministic" and prof["experiment"]["repeats"] == 1:
            continue
        sig[f"{block}.{key}"] = prof[block][key]
    return sig


def metric_files(results_dir, prof):
    """(repetition, path) of each metrics file the profile declares."""
    tag = prof["experiment"]["run_tag"]
    repeats = prof["experiment"]["repeats"]
    if repeats == 1:
        return [(0, os.path.join(results_dir, f"e05_{tag}_metrics.csv"))]
    return [(k, os.path.join(results_dir, f"e05_{tag}_rep{k}_metrics.csv")) for k in range(repeats)]


def read_condition(results_dir, prof, metric):
    """{(repetition, fold): value} for one profile. Refuses anything it cannot read exactly."""
    values = {}
    for rep, path in metric_files(results_dir, prof):
        if not os.path.isfile(path):
            fail(f"{path} is missing: run this profile (all {prof['experiment']['repeats']} "
                 f"repetitions) first")
        with open(path, newline="") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None or metric not in reader.fieldnames:
                fail(f"{path}: no {metric!r} column")
            rows = list(reader)
        sets = {r["feature_set"] for r in rows}
        if len(sets) != 1:
            fail(f"{path}: {len(sets)} feature sets {sorted(sets)}; a thesis run classifies "
                 f"exactly one")
        for r in rows:
            try:
                v = float(r[metric])
            except (TypeError, ValueError):
                fail(f"{path}: fold {r['fold']}: {metric} = {r[metric]!r} is not a number")
            if not math.isfinite(v):
                fail(f"{path}: fold {r['fold']}: {metric} = {v}. Closed-set metrics are NaN "
                     f"under the verification protocol; compare on eer or auc.")
            key = (rep, int(r["fold"]))
            if key in values:
                fail(f"{path}: fold {r['fold']} appears twice")
            values[key] = v
    if len(values) < 2:
        fail(f"{prof['experiment']['run_tag']}: {len(values)} fold value(s); a comparison "
             f"needs at least 2")
    return values


def cohens_d(a, b):
    """(d, note): effect size of a vs b with pooled SD; None with a note when undefined."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    # Constant groups: tested on the values themselves, because var() of a constant float
    # array can come out ~1e-34 instead of 0 and would turn 0/0 into a confident 0.0.
    if np.ptp(a) == 0.0 and np.ptp(b) == 0.0:
        return None, "cohens_d undefined: zero pooled variance"
    sp2 = ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)
    return float((a.mean() - b.mean()) / math.sqrt(sp2)), ""


def holm(pvals):
    """Holm-Bonferroni step-down adjusted p-values, {key: p} -> {key: p_adj}.

    Sorted ascending, the i-th smallest p (0-based) is multiplied by (m - i), capped at 1,
    and made monotone with a running max, so a larger raw p never gets a smaller adjusted p.
    Same procedure as meeting01's 04_meeting01_significance_tests.py.
    """
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted, running = {}, 0.0
    for i, (key, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adjusted[key] = running
    return adjusted


def compare(ref, cur, pairing):
    """(test, p or None, d or None, note) for cur vs ref, both {(rep, fold): value}."""
    if pairing == "paired":
        keys = sorted(ref)
        r = np.asarray([ref[k] for k in keys], float)
        c = np.asarray([cur[k] for k in keys], float)
        d, note = cohens_d(c, r)
        if np.array_equal(r, c):
            return "wilcoxon(paired)", None, d, "no test: identical paired values"
        _, p = stats.wilcoxon(c, r)
        return ("wilcoxon(paired)",) + defined_p(p, d, note)
    r = np.asarray(list(ref.values()), float)
    c = np.asarray(list(cur.values()), float)
    d, note = cohens_d(c, r)
    _, p = stats.mannwhitneyu(c, r, alternative="two-sided")
    return ("mannwhitneyu(indep)",) + defined_p(p, d, note)


def defined_p(p, d, note):
    """scipy returns NaN when every value is tied; say so instead of carrying a NaN."""
    p = float(p)
    if math.isfinite(p):
        return p, d, note
    return None, d, "; ".join(x for x in (note, "no test: all values tied") if x)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profiles-dir", required=True)
    ap.add_argument("--results-dir", required=True,
                    help="Directory holding the profiles' e05_<run_tag>[_rep<k>]_metrics.csv.")
    ap.add_argument("--metric", required=True, choices=METRICS)
    ap.add_argument("--reference", required=True,
                    help="'best' (best mean of --metric) or the run_tag of one profile.")
    ap.add_argument("--pairing", required=True, choices=["paired", "independent"])
    ap.add_argument("--out", help="Also write the comparison table to this CSV.")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.profiles_dir, "*.json")))
    if len(paths) < 2:
        fail(f"{len(paths)} profile(s) in {args.profiles_dir}; a comparison needs at least 2")
    profiles = {}
    for path in paths:
        with open(path) as fh:
            prof = json.load(fh)
        tag = prof["experiment"]["run_tag"]
        if tag in profiles:
            fail(f"run_tag {tag!r} is declared by two profiles")
        profiles[tag] = prof
    conds = {tag: read_condition(args.results_dir, prof, args.metric)
             for tag, prof in profiles.items()}

    lower = args.metric in LOWER_IS_BETTER

    def mean_of(c):
        return float(np.mean(list(conds[c].values())))

    ranked = sorted(conds, key=mean_of, reverse=not lower)
    if args.reference == "best":
        ref = ranked[0]
    elif args.reference in conds:
        ref = args.reference
    else:
        fail(f"--reference {args.reference!r} is neither 'best' nor a profile run_tag: "
             f"{sorted(conds)}")

    if args.pairing == "paired":
        ref_sig = split_signature(profiles[ref])
        for c in ranked:
            diff = {k: (ref_sig[k], v) for k, v in split_signature(profiles[c]).items()
                    if ref_sig[k] != v}
            if diff:
                fail(f"--pairing paired: {c} and the reference {ref} do not hold out the same "
                     f"folds ({', '.join(f'{k}: {a!r} vs {b!r}' for k, (a, b) in diff.items())}). "
                     f"Use --pairing independent, or compare profiles that share the split.")
            if set(conds[c]) != set(conds[ref]):
                fail(f"--pairing paired: {c} and {ref} have different (repetition, fold) sets")

    arrow = "lower=better" if lower else "higher=better"
    print(f"\n=== Thesis cross-profile significance on '{args.metric}' ({arrow}) ===")
    print(f"conditions: {len(conds)}   reference: {ref} ({args.reference})   "
          f"pairing: {args.pairing}\n")
    print(f"{'rank':>4}  {'condition':<44} {'folds':>5} {'mean':>9} {'std':>9}")
    for i, c in enumerate(ranked, 1):
        vals = np.asarray(list(conds[c].values()), float)
        tag = " (ref)" if c == ref else ""
        print(f"{i:>4}  {c[:44]:<44} {len(vals):>5} {vals.mean():>9.4f} "
              f"{vals.std(ddof=1):>9.4f}{tag}")

    def fmt(x, spec):
        return "—" if x is None else format(x, spec)

    tested = {c: compare(conds[ref], conds[c], args.pairing) for c in ranked if c != ref}
    p_holm = holm({c: p for c, (_, p, _, _) in tested.items() if p is not None})

    rows = []
    print(f"\n{'condition':<44} {'Δmean':>9} {'cohens_d':>9} {'p':>10} {'p_holm':>10}  test")
    for c, (test, p, d, note) in tested.items():
        dmean = mean_of(c) - mean_of(ref)
        ph = p_holm.get(c)
        star = "  *" if (ph is not None and ph < 0.05) else ""
        print(f"{c[:44]:<44} {dmean:>9.4f} {fmt(d, '9.3f'):>9} {fmt(p, '10.4g'):>10} "
              f"{fmt(ph, '10.4g'):>10}  {test}{star}{'  [' + note + ']' if note else ''}")
        rows.append({
            "reference": ref, "condition": c, "metric": args.metric, "pairing": args.pairing,
            "n_folds_ref": len(conds[ref]), "n_folds_cond": len(conds[c]),
            "mean_ref": mean_of(ref), "mean_cond": mean_of(c), "delta_mean": dmean,
            "cohens_d": "" if d is None else d, "p_value": "" if p is None else p,
            "p_holm": "" if ph is None else ph, "holm_family_size": len(p_holm),
            "test": test, "significant_0.05": "" if ph is None else int(ph < 0.05),
            "note": note,
        })

    print(f"\n  * p_holm < 0.05 (Holm over the {len(p_holm)} tested comparisons).  "
          f"Δmean = condition − reference.  cohens_d > 0 ⇒ condition's {args.metric} is "
          f"larger than the reference's.")
    if args.reference == "best":
        print("  The reference was selected as the best of these same data (--reference best): "
              "every comparison is biased towards a difference. Name a reference fixed in "
              "advance for a confirmatory reading.")

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
