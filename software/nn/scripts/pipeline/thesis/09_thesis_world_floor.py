#!/usr/bin/env python3
"""Is the BIOMEX-DB null of the WORLD grid a floor effect?

THE QUESTION. On Coretto, roughness lowers the per-utterance verification margin beyond
resynthesis at all five levels; on BIOMEX-DB no level does (08_thesis_world_analysis.py). An
exploratory look, made AFTER that result, found that on BIOMEX-DB the resynthesis control itself
leaves only 13 % of the clean margin (Coretto: 25 %). The floor hypothesis: with so little speaker
information left after resynthesis, an edit has little room to lower the margin further.

WHAT THE FLOOR PREDICTS, AND WHY THE TEST SPLITS THE DATA. If the floor is the reason, then on
BIOMEX-DB the speakers who keep MORE margin after resynthesis (more "room") should show a
STRONGER (more negative) roughness effect. The naive test regresses the effect
d = margin(level) - margin(control) on margin(control). It is biased: margin(control) appears in
both with opposite signs, so its measurement noise alone makes the slope negative, which is the
predicted direction (regression to the mean). So room and effect are measured on DISJOINT
utterances:

    each (speaker, word), utterances in sample order:  u1  u2  u3  u4  u5  u6 ...
                                                        A   B   A   B   A   B
    half A alone (centroids from A's clean utterances only) -> room_A(s), effect_A(s)
    half B alone (centroids from B's clean utterances only) -> room_B(s), effect_B(s)
    r = ( corr(room_A, effect_B) + corr(room_B, effect_A) ) / 2      over speakers

No utterance, and no centroid, is shared between a room and the effect it is correlated with.

THE RULE (written and committed before this analysis was run; the user chose to test the floor
on 2026-11-17, after seeing the grid result and the exploratory 13 % / 25 %).

    Base: BIOMEX-DB (45 speakers). Features: the Part A features of 08
    (results/thesis/world_margin, source voice), standardized exactly as 08 does.
    margin: 08's margins(), computed within one half (own centroid = the speaker's other clean
    utterances of the same word IN THAT HALF; rivals = other speakers' clean centroids of the
    same word in that half).
    room_H(s)   = speaker mean, over half H, of margin(control features).
    effect_H(s) = speaker mean, over half H, of the mean over the five roughness levels
                  (rough-j05, j1, j2, j4, j7) of margin(level) - margin(control).
    Statistic: r = mean of the two cross-half Pearson correlations above.
    p: two-sided permutation test, the speakers' rooms permuted (one permutation applied to
    room_A and room_B together), 100000 permutations, numpy default_rng(20261117),
    p = (1 + #{|r_perm| >= |r|}) / (1 + 100000).
    Verdict: floor SUPPORTED when p < 0.05 and r < 0; CONTRADICTED when p < 0.05 and r > 0;
    otherwise NOT SUPPORTED by this test.
    Only roughness: it is the family with a clear effect on Coretto (5 of 5 levels), and the
    floor limits decreases; the aphonic gaps raised the margin on Coretto. One test, no family.
    Power, stated before the run: with 45 speakers a two-sided test at 0.05 has 80 % power only
    for |r| >= ~0.41 (Fisher z), and halving the utterances attenuates r. A null result does not
    exclude a weaker floor.
    Coretto (15 speakers) gets the same numbers, DESCRIPTIVE ONLY (no verdict).

Run from software/nn:
    python3 scripts/pipeline/thesis/09_thesis_world_floor.py \\
        --features results/thesis/world_margin --out results/thesis/world_analysis/world_floor.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROUGH = ["rough-j05", "rough-j1", "rough-j2", "rough-j4", "rough-j7"]
PERMUTATIONS = 100000
SEED = 20261117
ALPHA = 0.05
CONFIRMATORY = "biomex2021"


def fail(msg: str) -> None:
    sys.exit(f"[world-floor] ERROR: {msg}")


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M08 = _load("08_thesis_world_analysis")


def halves(meta: list[tuple[int, int, int]]) -> tuple[np.ndarray, np.ndarray]:
    """Row indices of halves A and B: within each (speaker, word), in sample order, alternate.
    Each (speaker, word) needs >= 4 utterances, so each half has >= 2 (the own centroid of a
    left-out utterance needs one more)."""
    groups: dict[tuple[int, int], list[int]] = {}
    for i, (sample, s, w) in enumerate(meta):
        groups.setdefault((s, w), []).append(i)
    a, b = [], []
    for (s, w), rows in sorted(groups.items()):
        if len(rows) < 4:
            fail(f"speaker {s}, word {w}: {len(rows)} utterances; a half needs 2")
        rows = sorted(rows, key=lambda i: meta[i][0])
        a += rows[0::2]
        b += rows[1::2]
    return np.array(sorted(a)), np.array(sorted(b))


def half_measures(meta, clean_z, control_z, rough_z, rows) -> tuple[np.ndarray, np.ndarray, list]:
    """(room per speaker, effect per speaker, speakers) using only the given rows."""
    sub = [meta[i] for i in rows]
    m_control = M08.margins(sub, clean_z[rows], control_z[rows])
    effect = np.mean([M08.margins(sub, clean_z[rows], z[rows]) - m_control for z in rough_z],
                     axis=0)
    room = M08.speaker_means(sub, m_control)
    eff = M08.speaker_means(sub, effect)
    speakers = sorted(room)
    return (np.array([room[s] for s in speakers]), np.array([eff[s] for s in speakers]),
            speakers)


def _corr(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pearson r of each row of x (perm x speakers, or one vector) with the vector y."""
    xc = x - x.mean(axis=-1, keepdims=True)
    yc = y - y.mean()
    den = np.sqrt((xc ** 2).sum(axis=-1) * (yc ** 2).sum())
    if np.any(den == 0):
        fail("a constant room or effect across speakers: the correlation does not exist")
    return (xc @ yc) / den


def cross_half(room_a, eff_a, room_b, eff_b, permutations: int, seed: int) -> dict:
    r_ab, r_ba = float(_corr(room_a, eff_b)), float(_corr(room_b, eff_a))
    r = (r_ab + r_ba) / 2
    rng = np.random.default_rng(seed)
    n = len(room_a)
    perm = np.array([rng.permutation(n) for _ in range(permutations)])
    r_perm = (_corr(room_a[perm], eff_b) + _corr(room_b[perm], eff_a)) / 2
    p = (1 + int(np.sum(np.abs(r_perm) >= abs(r)))) / (1 + permutations)
    slope = lambda x, y: float(np.polyfit(x, y, 1)[0])
    return {"r": r, "r_room_a_effect_b": r_ab, "r_room_b_effect_a": r_ba, "p": p,
            "permutations": permutations, "seed": seed, "speakers": n,
            "slope_mean": (slope(room_a, eff_b) + slope(room_b, eff_a)) / 2,
            "room_mean": float((room_a.mean() + room_b.mean()) / 2),
            "effect_mean": float((eff_a.mean() + eff_b.mean()) / 2)}


def verdict(res: dict) -> str:
    if res["p"] < ALPHA:
        return "supported" if res["r"] < 0 else "contradicted"
    return "not supported"


def analyse(features: pathlib.Path, base: str, permutations: int = PERMUTATIONS) -> dict:
    prefix = M08.BASES[base][1]
    stem = lambda lv: str(features / f"e05_{prefix}_voice_probes_{lv}")
    meta, clean = M08.M07.read(stem("control") + "_features_clean.csv")
    _, control = M08.M07.read(stem("control") + "_features_degraded.csv")
    rough = []
    for lv in ROUGH:
        m_c, c = M08.M07.read(stem(lv) + "_features_clean.csv")
        m_d, d = M08.M07.read(stem(lv) + "_features_degraded.csv")
        if not (m_c == m_d == meta):
            fail(f"{base} {lv}: the CSVs do not list the same samples as the control's")
        if c != clean:
            fail(f"{base} {lv}: its clean features differ from the control run's")
        rough.append(np.asarray(d))
    z = M08.standardize(np.asarray(clean), np.asarray(control), *rough)
    a, b = halves(meta)
    room_a, eff_a, spk_a = half_measures(meta, z[0], z[1], z[2:], a)
    room_b, eff_b, spk_b = half_measures(meta, z[0], z[1], z[2:], b)
    if spk_a != spk_b:
        fail(f"{base}: the halves do not hold the same speakers")
    res = cross_half(room_a, eff_a, room_b, eff_b, permutations, SEED)
    res["utterances_a"], res["utterances_b"] = len(a), len(b)
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--features", required=True, type=pathlib.Path)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    args = ap.parse_args()
    out = {}
    for base in M08.BASES:
        res = analyse(args.features, base)
        res["role"] = "confirmatory" if base == CONFIRMATORY else "descriptive"
        if base == CONFIRMATORY:
            res["verdict"] = verdict(res)
        out[base] = res
        print(f"[world-floor] {base} ({res['role']}): r = {res['r']:+.3f} "
              f"(A->B {res['r_room_a_effect_b']:+.3f}, B->A {res['r_room_b_effect_a']:+.3f}), "
              f"p = {res['p']:.4f}, {res['speakers']} speakers"
              + (f" -> floor {res['verdict']}" if "verdict" in res else ""))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
