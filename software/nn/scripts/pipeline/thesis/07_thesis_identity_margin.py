#!/usr/bin/env python3
"""Does WORLD resynthesis alone move an utterance away from its own speaker? (powered test)

WHY THIS TEST. The identity pilot (proposal section 4.3) compared EERs: 15 per arm, one per
(repeat, fold), dependent, so its corrected interval was ~4x the margin and it could not have
passed. The EER pools every utterance of a fold into one number. This test keeps the
utterance: for each of the 1974 voice samples it asks how much closer the utterance is to its
own speaker than to the most similar other speaker, before and after resynthesis.

THE MEASURE, for utterance u of speaker s saying word w (features standardized with the mean and
sd of the clean features, as the cosine classifier does):

    own(u)      centroid of speaker s's OTHER clean utterances of word w   (u left out)
    other(s')   centroid of speaker s''s clean utterances of word w, s' != s
    margin(f)   cos(f, own(u)) - max over s' of cos(f, other(s'))
                > 0: f sits nearer its own speaker than any impostor

    d_control(u) = margin(control features of u) - margin(clean features of u)
    d_whisper(u) = margin(whisper l25 features of u) - margin(clean features of u)

    utterance 0412, speaker 7, "arriba":  margin clean 0.31, control 0.27, whisper 0.22
                                          d_control -0.04,  d_whisper -0.09

THE UNIT. The 1974 utterances are not independent (each speaker contributes ~130), so the
inference is on the 15 speaker means D(s) = mean of d(u) over s's utterances: Student t, df 14.

THE RULE (fixed before any result of this test; proposal section 4.4).
    yardstick   whisper l25, the mildest degradation of the thesis grid. It must lower the
                margin: its 95 % interval of mean D_whisper must lie below 0. If it does not, the
                yardstick is void and the script stops with an error (no verdict).
    margin      delta = |mean D_whisper| / 2   (half the mildest whisper effect, as the pilot)
    transparent      95 % interval of mean D_control inside (-delta, +delta)
    not transparent  the interval entirely outside (-delta, +delta)
    inconclusive     anything else
The clean features of the two input runs must be identical (same profile extractor): checked.

Inputs: the CSVs of tools/thesis_features (one run on pilot/pil_cosine_voice_control_probes.json,
one on phase01/p01_dysph_voice_probes_l25.json). Run from software/nn:
    python3 scripts/pipeline/thesis/07_thesis_identity_margin.py \\
        --control results/thesis/identity_margin/e05_pil_cosine_voice_control_probes \\
        --whisper results/thesis/identity_margin/e05_p01_dysph_voice_probes_l25 \\
        --out results/thesis/identity_margin/identity_margin_verdict.json
(each path is a stem: <stem>_features_clean.csv and <stem>_features_degraded.csv are read)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics

T95_DF14 = 2.1447866879   # two-sided 95 % Student t, df 14 (15 speakers)


def read(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"[identity-margin] {path} is empty")
    dims = [k for k in rows[0] if k.startswith("f")]
    meta = [(int(r["sample"]), int(r["subject_id"]), int(r["stimulus"])) for r in rows]
    vecs = [[float(r[k]) for k in dims] for r in rows]
    return meta, vecs


def standardize(clean, *others):
    dims = len(clean[0])
    mean = [statistics.fmean(v[d] for v in clean) for d in range(dims)]
    sd = [statistics.pstdev([v[d] for v in clean]) for d in range(dims)]
    keep = [d for d in range(dims) if sd[d] > 0]   # a constant feature carries no speaker

    def z(vs):
        return [[(v[d] - mean[d]) / sd[d] for d in keep] for v in vs]
    return [z(clean)] + [z(o) for o in others]


def cos(a, b):
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        raise SystemExit("[identity-margin] a zero vector: cosine undefined")
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def centroid(vs):
    return [statistics.fmean(c) for c in zip(*vs)]


def margins(meta, clean_z, probe_z):
    """margin(probe of u) against clean centroids, own centroid leaving u out."""
    groups = {}
    for i, (_, subj, stim) in enumerate(meta):
        groups.setdefault((subj, stim), []).append(i)
    full = {k: centroid([clean_z[i] for i in idx]) for k, idx in groups.items()}
    out = []
    for i, (_, subj, stim) in enumerate(meta):
        mates = [j for j in groups[(subj, stim)] if j != i]
        if not mates:
            raise SystemExit(f"[identity-margin] sample {meta[i][0]}: speaker {subj} has no other "
                             f"utterance of stimulus {stim}; the own centroid is undefined")
        own = centroid([clean_z[j] for j in mates])
        rivals = [full[(s, w)] for (s, w) in full if w == stim and s != subj]
        if not rivals:
            raise SystemExit(f"[identity-margin] stimulus {stim}: no other speaker")
        out.append(cos(probe_z[i], own) - max(cos(probe_z[i], c) for c in rivals))
    return out


def speaker_means(meta, d):
    by = {}
    for (_, subj, _), v in zip(meta, d):
        by.setdefault(subj, []).append(v)
    return {s: statistics.fmean(v) for s, v in sorted(by.items())}


def interval(means):
    xs = list(means.values())
    if len(xs) != 15:
        raise SystemExit(f"[identity-margin] {len(xs)} speakers; the rule was fixed for 15 (df 14)")
    m = statistics.fmean(xs)
    half = T95_DF14 * statistics.stdev(xs) / math.sqrt(len(xs))
    return m, m - half, m + half


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--control", required=True, help="stem of the resynthesis_control run")
    ap.add_argument("--whisper", required=True, help="stem of the whisper l25 run")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    meta, clean = read(args.control + "_features_clean.csv")
    meta_c, control = read(args.control + "_features_degraded.csv")
    meta_w0, clean_w = read(args.whisper + "_features_clean.csv")
    meta_w, whisper = read(args.whisper + "_features_degraded.csv")
    if not (meta == meta_c == meta_w0 == meta_w):
        raise SystemExit("[identity-margin] the four CSVs do not list the same samples in order")
    if clean != clean_w:
        raise SystemExit("[identity-margin] the clean features of the two runs differ: the two "
                         "profiles do not use the same extractor")

    clean_z, control_z, whisper_z = standardize(clean, control, whisper)
    m_clean = margins(meta, clean_z, clean_z)
    d_control = [a - b for a, b in zip(margins(meta, clean_z, control_z), m_clean)]
    d_whisper = [a - b for a, b in zip(margins(meta, clean_z, whisper_z), m_clean)]

    sp_control = speaker_means(meta, d_control)
    sp_whisper = speaker_means(meta, d_whisper)
    wm, wlo, whi = interval(sp_whisper)
    if not whi < 0:
        raise SystemExit(f"[identity-margin] the yardstick is void: whisper l25 does not lower the "
                         f"margin (mean {wm:+.4f}, 95 % interval [{wlo:+.4f}, {whi:+.4f}]); "
                         f"no verdict (rule, proposal 4.4)")
    delta = abs(wm) / 2
    cm, clo, chi = interval(sp_control)
    if -delta < clo and chi < delta:
        verdict = "transparent"
    elif chi < -delta or clo > delta:
        verdict = "not transparent"
    else:
        verdict = "inconclusive"
    result = {
        "utterances": len(meta), "speakers": len(sp_control),
        "clean_margin_mean": round(statistics.fmean(m_clean), 4),
        "clean_margin_positive_share": round(sum(m > 0 for m in m_clean) / len(m_clean), 4),
        "whisper_l25": {"mean": round(wm, 4), "ci95": [round(wlo, 4), round(whi, 4)]},
        "delta": round(delta, 4),
        "control": {"mean": round(cm, 4), "ci95": [round(clo, 4), round(chi, 4)],
                    "speakers_below_zero": sum(v < 0 for v in sp_control.values()),
                    "ratio_to_whisper": round(cm / wm, 3)},
        "per_speaker": {str(s): {"control": round(sp_control[s], 4),
                                 "whisper_l25": round(sp_whisper[s], 4)} for s in sp_control},
        "verdict": verdict,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    print(json.dumps({k: v for k, v in result.items() if k != "per_speaker"}, indent=1))


if __name__ == "__main__":
    main()
