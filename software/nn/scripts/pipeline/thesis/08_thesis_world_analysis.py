#!/usr/bin/env python3
"""Does a WORLD dysphonia profile change speaker identity beyond what resynthesis alone does?

WHY "BEYOND RESYNTHESIS". The powered identity test (proposal section 4.4) found the control
(`resynthesis_control`: WORLD analysis + resynthesis, nothing edited) NOT transparent: it lowers
the per-utterance verification margin 3.6 times as much as the mildest whisper. So a WORLD
profile compared with the original voice would mix two effects, resynthesis and dysphonia. Every
comparison here is profile MINUS CONTROL, both resynthesized by the same WORLD path:

    original --WORLD--> control                    (the resynthesis effect, already measured)
    original --WORLD + edited F0/amplitude--> L    (resynthesis + the dysphonia of level L)
    L - control                                    (the dysphonia of L alone)

THE RULE (fixed before any result of the WORLD grid; proposal section 4.5; decided 2026-11-16).

Part A, CONFIRMATORY: the per-utterance verification margin, source voice, on each base.
    margin(f) = cos(f, own(u)) - max over s' != s of cos(f, other(s'))       (07_..., unchanged)
    own(u): centroid of u's speaker's OTHER clean utterances of the same word (u left out);
    other(s'): centroid of speaker s''s clean utterances of the same word; features
    standardized with the clean features' mean and sd (columns constant on clean dropped).
    d_L(u) = margin(level-L features of u) - margin(control features of u)
    Unit: the speaker mean of d_L (Coretto 15 speakers, BIOMEX-DB 45), two-sided one-sample
    Student t against 0, df = speakers - 1. Family: the 14 levels of one base, Holm-Bonferroni.
    Level L changes identity beyond resynthesis on a base when its Holm p < 0.05; the sign of
    the mean says which way. On BIOMEX-DB (confirmatory, nothing selected there) a level
    REPLICATES when it is significant on both bases with the same sign.
    Features: tools/thesis_features on world/p01w_voice_probes_<level>.json and
    bxworld/bx01w_voice_probes_<level>.json (the scope does not touch the features: the tool
    degrades every voice); the clean features of all 15 runs of a base must be identical
    (checked: same extractor, same base).

Part B, DESCRIPTIVE: the EER of every cell of the grid (both bases, every source, both scopes),
    each level against the control of its own (source, scope), the Nadeau-Bengio corrected
    repeated k-fold t of thesis_cross_profile_significance.py on the 15 (repeat, fold) pairs,
    Holm over the 28 comparisons of one (base, source). Reported, not used for any claim of
    "changes identity": the identity pilot showed its interval to be ~0.07 EER wide.

RESULT (2026-11-17): Coretto, 9 of 14 levels change the margin beyond resynthesis (rough all five,
tremor 10 %, aphonic gaps all three, the last ones upwards); BIOMEX-DB, none; no level
replicates; no EER comparison of Part B is significant. The "exploratory" block of
world_margin.json (each base's clean margin, and control minus clean, the powered test's measure
on both bases) was added after that result and changes nothing in it.

Run from software/nn, after the grid and the feature runs:
    python3 scripts/pipeline/thesis/08_thesis_world_analysis.py \\
        --features results/thesis/world_margin \\
        --profiles-root src/experiments/thesis/profiles --results-root results/thesis \\
        --out results/thesis/world_analysis
Writes <out>/world_margin.json (Part A) and <out>/world_eer.csv (Part B).
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import pathlib
import subprocess
import sys

import numpy as np
from scipy import stats

HERE = pathlib.Path(__file__).resolve().parent
LEVELS = ["rough-j05", "rough-j1", "rough-j2", "rough-j4", "rough-j7",
          "tremor-d2", "tremor-d4", "tremor-d10",
          "jump-r05", "jump-r1", "jump-r2", "aphonic-r05", "aphonic-r1", "aphonic-r2"]
BASES = {"coretto2017": ("world", "p01w", ("voice", "earlypron", "earlyimag", "latepron",
                                            "lateimag")),
         "biomex2021": ("bxworld", "bx01w", ("voice", "earlypron", "latepron"))}
SCOPES = ("probes", "test")
ALPHA = 0.05


def fail(msg: str) -> None:
    sys.exit(f"[world-analysis] ERROR: {msg}")


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M07 = _load("07_thesis_identity_margin")   # read(): the CSV reader of the powered test


# ── Part A ──────────────────────────────────────────────────────────────────────────────────

def standardize(clean: np.ndarray, *others: np.ndarray) -> list[np.ndarray]:
    """As 07's standardize(): mean and population sd of the clean rows, constant columns out."""
    mean, sd = clean.mean(axis=0), clean.std(axis=0)
    keep = sd > 0
    return [(x[:, keep] - mean[keep]) / sd[keep] for x in (clean, *others)]


def _unit(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    if np.any(n == 0):
        fail("a zero vector: cosine undefined")
    return x / n


def margins(meta: list[tuple[int, int, int]], clean_z: np.ndarray,
            probe_z: np.ndarray) -> np.ndarray:
    """07's margins() in numpy: the same numbers (test_thesis_phase_scripts.py checks it)."""
    subj = np.array([m[1] for m in meta])
    stim = np.array([m[2] for m in meta])
    out = np.empty(len(meta))
    probe_u = _unit(probe_z)
    for w in np.unique(stim):
        rows = np.flatnonzero(stim == w)
        speakers = np.unique(subj[rows])
        if len(speakers) < 2:
            fail(f"stimulus {w}: no other speaker")
        sums = {s: clean_z[rows[subj[rows] == s]].sum(axis=0) for s in speakers}
        counts = {s: int(np.sum(subj[rows] == s)) for s in speakers}
        full = {s: sums[s] / counts[s] for s in speakers}
        for i in rows:
            s = subj[i]
            if counts[s] < 2:
                fail(f"sample {meta[i][0]}: speaker {s} has no other utterance of stimulus {w}; "
                     f"the own centroid is undefined")
            own = (sums[s] - clean_z[i]) / (counts[s] - 1)
            rivals = np.array([full[r] for r in speakers if r != s])
            p = probe_u[i]
            out[i] = p @ _unit(own) - np.max(_unit(rivals) @ p)
    return out


def speaker_means(meta, d: np.ndarray) -> dict[int, float]:
    by: dict[int, list[float]] = {}
    for (_, s, _), v in zip(meta, d):
        by.setdefault(s, []).append(float(v))
    return {s: float(np.mean(v)) for s, v in sorted(by.items())}


def t_test(values: list[float]) -> dict:
    x = np.asarray(values)
    n = len(x)
    if n < 3:
        fail(f"{n} speakers: too few for a t interval")
    m, se = float(x.mean()), float(x.std(ddof=1) / math.sqrt(n))
    if se == 0:
        fail("zero variance across speakers: the t statistic does not exist")
    t = m / se
    q = float(stats.t.ppf(0.975, n - 1))
    return {"mean": m, "ci95": [m - q * se, m + q * se], "t": t, "df": n - 1,
            "p": float(2 * stats.t.sf(abs(t), n - 1)), "speakers": n,
            "speakers_below_zero": int(np.sum(x < 0))}


def holm(pvals: dict[str, float]) -> dict[str, float]:
    order = sorted(pvals, key=pvals.get)
    k, running, out = len(order), 0.0, {}
    for i, key in enumerate(order):
        running = max(running, min(1.0, (k - i) * pvals[key]))
        out[key] = running
    return out


def margin_part(features: pathlib.Path) -> dict:
    result = {}
    for base, (_, prefix, _) in BASES.items():
        stem = lambda lv: features / f"e05_{prefix}_voice_probes_{lv}"
        meta, clean = M07.read(str(stem("control")) + "_features_clean.csv")
        _, control = M07.read(str(stem("control")) + "_features_degraded.csv")
        degraded = {}
        for lv in LEVELS:
            m_c, c = M07.read(str(stem(lv)) + "_features_clean.csv")
            m_d, d = M07.read(str(stem(lv)) + "_features_degraded.csv")
            if not (m_c == m_d == meta):
                fail(f"{base} {lv}: the CSVs do not list the same samples as the control's")
            if c != clean:
                fail(f"{base} {lv}: its clean features differ from the control run's")
            degraded[lv] = np.asarray(d)
        z = standardize(np.asarray(clean), np.asarray(control), *degraded.values())
        clean_z, control_z, levels_z = z[0], z[1], dict(zip(LEVELS, z[2:]))
        m_control = margins(meta, clean_z, control_z)
        rows, ps = {}, {}
        for lv in LEVELS:
            d = margins(meta, clean_z, levels_z[lv]) - m_control
            rows[lv] = t_test(list(speaker_means(meta, d).values()))
            rows[lv]["per_speaker"] = {str(s): round(v, 5)
                                       for s, v in speaker_means(meta, d).items()}
            ps[lv] = rows[lv]["p"]
        for lv, ph in holm(ps).items():
            rows[lv]["p_holm"] = ph
            rows[lv]["significant"] = ph < ALPHA
        m_clean = margins(meta, clean_z, clean_z)
        explore = t_test(list(speaker_means(meta, m_control - m_clean).values()))
        result[base] = {"utterances": len(meta), "levels": rows,
                        "control_margin_mean": float(m_control.mean()),
                        "exploratory": {
                            "note": "after the result; not part of the rule",
                            "clean_margin_mean": float(m_clean.mean()),
                            "clean_margin_positive_share": float(np.mean(m_clean > 0)),
                            "control_minus_clean": explore}}
    co, bx = result["coretto2017"]["levels"], result["biomex2021"]["levels"]
    result["replicates"] = {lv: bool(co[lv]["significant"] and bx[lv]["significant"]
                                     and math.copysign(1, co[lv]["mean"])
                                     == math.copysign(1, bx[lv]["mean"])) for lv in LEVELS}
    return result


# ── Part B ──────────────────────────────────────────────────────────────────────────────────

def eer_part(profiles_root: pathlib.Path, results_root: pathlib.Path,
             scratch: pathlib.Path) -> list[dict]:
    sig = HERE / "thesis_cross_profile_significance.py"
    rows = []
    for base, (phase, prefix, sources) in BASES.items():
        for src in sources:
            family = []
            for scope in SCOPES:
                out = scratch / f"sig_{phase}_{src}_{scope}.csv"
                ref = f"e05_{prefix}_{src}_{scope}_control"
                cmd = [sys.executable, str(sig), "--profiles-dir", str(profiles_root / phase),
                       "--results-dir", str(results_root / phase),
                       "--select", f"^e05_{prefix}_{src}_{scope}_", "--metric", "eer",
                       "--reference", ref, "--pairing", "corrected", "--out", str(out)]
                r = subprocess.run(cmd, capture_output=True, text=True, check=False)
                if r.returncode != 0:
                    fail(f"{' '.join(cmd)} failed:\n{r.stdout}\n{r.stderr}")
                with open(out, newline="", encoding="utf-8") as f:
                    got = [row for row in csv.DictReader(f)]
                if len(got) != len(LEVELS):
                    fail(f"{out}: {len(got)} comparisons, expected {len(LEVELS)}")
                for row in got:
                    if row["p_value"] == "":
                        fail(f"{out}: {row['condition']} has no test ({row['note']})")
                    family.append({"base": base, "source": src, "scope": scope,
                                   "level": row["condition"][len(f"e05_{prefix}_{src}_{scope}_"):],
                                   "eer_control": float(row["mean_ref"]),
                                   "eer_level": float(row["mean_cond"]),
                                   "delta": float(row["delta_mean"]),
                                   "p": float(row["p_value"])})
            adj = holm({f"{r['scope']}/{r['level']}": r["p"] for r in family})
            for r in family:
                r["p_holm"] = adj[f"{r['scope']}/{r['level']}"]
                r["significant"] = int(r["p_holm"] < ALPHA)
            rows += family
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", required=True, help="the thesis_features output directory")
    ap.add_argument("--profiles-root", required=True)
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    margin = margin_part(pathlib.Path(args.features))
    with open(out / "world_margin.json", "w", encoding="utf-8") as f:
        json.dump(margin, f, indent=1)
    eer = eer_part(pathlib.Path(args.profiles_root), pathlib.Path(args.results_root), out)
    with open(out / "world_eer.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(eer[0]))
        w.writeheader()
        w.writerows(eer)
    for base in BASES:
        print(f"[world-analysis] {base}: margin, level minus control (Holm over 14)")
        for lv, r in margin[base]["levels"].items():
            print(f"  {lv:12s} {r['mean']:+.4f} [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}] "
                  f"p_holm {r['p_holm']:.4f}{'  *' if r['significant'] else ''}")
    print(f"[world-analysis] replicate on both bases: "
          f"{[lv for lv, ok in margin['replicates'].items() if ok]}")
    print(f"[world-analysis] {len(eer)} EER comparisons -> {out / 'world_eer.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
