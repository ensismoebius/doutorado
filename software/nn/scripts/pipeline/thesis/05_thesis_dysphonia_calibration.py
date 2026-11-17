#!/usr/bin/env python3
"""Calibrate the WORLD dysphonia profiles on real vowels, measured with Praat.

THE PROBLEM. A profile asks for "3 % jitter", but what comes out of a real voice is not 3 %:
the voice has its own jitter, WORLD's analysis smooths part of the request away, and a
female voice (shorter cycles on the same 1 ms frame grid) keeps less of it than a male one.
On the synthetic vowel of thesis_voice_dysphonia_gtest the request and the measure agree;
on real vowels they do not, and the thesis must state what the voices it degraded actually
carry. So the profiles are described by a MEASURED value (proposal section 3.4), and this
script finds, for each target, the request that produces it:

    request 3 % ──► degrade_voice_f0 (the thesis's own C++) ──► Praat: 2.0 % ──► ...
    sweep of requests ──► mean measured curve ──► request whose mean measured = target
                      ──► verify at that request, voice by voice (within +-15 % or not)

THE PIECES.
  * The degradation is NOT reimplemented: the script compiles the thesis's own
    VoiceDysphonia.cpp (with WORLD and the resampler, from the CMake build tree) into a
    shared library and calls thesis::degrade_voice_f0 through ctypes, like
    documentation/00-thesis/monography/images/generate_dysphonia_figure.py does.
  * The measurer is Praat (parselmouth), one program for every voice, raw and degraded:
    CPPS differed by ~15 dB between programs (Skowronski 2015), and jitter, shimmer and HNR
    have the same risk. Praat's standard settings for rough (period 0.1-20 ms, factors 1.3 and
    1.6); for diplophonia the factors are opened (2.0 and 3.0) because the alternation itself
    is what is measured and Praat's 1.6 amplitude factor would drop every pair above 4.1 dB.
  * The voices: the pronounced vowel trials (A, E, I, O, U) of the subjects named, every
    trial, read-only from the Coretto base. The measures use the longest voiced run of the
    RAW trial (Praat pitch), 30 ms trimmed at each end, the same span for every version.

WHAT EACH PROFILE IS MEASURED BY.
  | profile        | measured                                        | Praat                  |
  |----------------|-------------------------------------------------|------------------------|
  | rough          | local jitter %, local shimmer dB, HNR dB         | PointProcess, cc       |
  | tremor         | 5 Hz component of the F0 track / mean F0, %      | Pitch (cc), 5 ms step  |
  | voice_breaks   | breaks per second of the span                    | Pitch (cc), 5 ms step  |
  | diplophonia    | period alternation % = local jitter / 2;         | PointProcess, cc,      |
  |                | amplitude alternation dB = local shimmer dB      | factors 2.0 / 3.0      |
  | resynth. ctrl  | RMS and HNR change against the raw trial         |                        |
Diplophonia: alternating cycles +x % / -x % differ by 2x % from each neighbour, so local
jitter = 2x and x = jitter / 2; the gains +a/2 / -a/2 dB differ by a, so local shimmer = a.
Clinically jitter and shimmer are undefined for a period-doubled voice; here they only read
back the alternation the request put in.

WHAT FAILS LOUDLY. A missing build tree or parselmouth, a degradation the C++ refuses, a
voice whose voiced span is under 0.4 s (two tremor cycles): it is not measured, and it is
LISTED in the output with its reason (never dropped without a trace). A level whose target is
not reached on a voice is reported as such, not clamped.

Run from software/nn (EXPERIMENT_CONFIRMED is not needed: no training, minutes):
    .venv/bin/python scripts/pipeline/thesis/05_thesis_dysphonia_calibration.py \\
        --db ~/Documentos/academico/UNESP/doutorado/databases/database.sqlite \\
        --subjects S10 S03 --jobs 10 --out-dir results/thesis/dysphonia_calibration
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sqlite3
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

try:
    import parselmouth
except ImportError as exc:  # no fallback measurer: another program gives other numbers
    raise SystemExit("05_thesis_dysphonia_calibration: parselmouth (Praat) is required: "
                     ".venv/bin/python -m pip install praat-parselmouth") from exc

HERE = os.path.dirname(os.path.abspath(__file__))
NN = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(NN, "scripts", "lib"))
import thesis_dysphonia_lib as dl  # noqa: E402  the thesis's own C++, compiled for ctypes

FS = 44100.0             # Coretto is recorded and stored at 44.1 kHz: native = stored
VOWELS = ("A", "E", "I", "O", "U")
TOLERANCE = 0.15         # proposal section 4, item 1: within +-15 % of the target
MIN_SPAN_S = 0.4         # two cycles of a 5 Hz tremor
TRIM_S = 0.03
TREMOR_RATE_HZ = 5.0
BREAK_MS = 80.0          # proposal 3.4.1
SEED = 7                 # DEGRADATION_SEED of the profiles; mixed with the voice index

PROFILE_CODE = dl.PROFILE_CODE
BREAK_CODE = dl.BREAK_CODE

# Requests swept, per axis (calibrated independently, then verified together). The other axis
# of the profile is held at OTHER_AXIS: check_dysphonia_params refuses 0 for jitter_pct,
# shimmer_db and period_alt_pct (open interval at 0), and 0.01 % / 0.01 dB is two orders below
# what a natural voice already carries (Praat jitter ~0.3-1 %, shimmer ~0.3 dB).
OTHER_AXIS = {"jitter_pct": ("shimmer_db", 0.01), "shimmer_db": ("jitter_pct", 0.01),
              "period_alt_pct": ("amp_alt_db", 0.0), "amp_alt_db": ("period_alt_pct", 0.01),
              "f0_depth_pct": None}
SWEEPS = {
    "jitter_pct": ("rough", [0.25, 0.5, 1, 2, 3, 4, 6, 8, 10, 12, 15]),
    "shimmer_db": ("rough", [0.25, 0.5, 1, 1.5, 2, 3, 4, 5]),
    "f0_depth_pct": ("tremor", [1, 2, 3, 4, 6, 8, 10, 12, 15]),
    "period_alt_pct": ("diplophonia", [2, 4, 6, 9, 12, 15, 18]),
    "amp_alt_db": ("diplophonia", [1, 2, 3, 4.5, 6, 8, 10]),
}
# Measured key each swept request is read back by.
READ_BACK = {"jitter_pct": "jitter_pct", "shimmer_db": "shimmer_db",
             "f0_depth_pct": "tremor_depth_pct", "period_alt_pct": "alt_period_pct",
             "amp_alt_db": "alt_amp_db"}

# Proposal 3.4.1 (levels, measured targets). Shimmer 1.5 dB, not 2 dB: Praat's standard
# settings cannot read past ~1.95 dB (decision 7). No diplophonia level: not validated, out of
# the thesis (decision 8); its sweep stays, as the evidence for that.
LEVELS = (
    [("rough", {"jitter_pct": j, "shimmer_db": s}) for j, s in
     ((0.5, 0.5), (1, 1), (2, 1.5), (4, 1.5), (7, 1.5))]
    + [("tremor", {"f0_depth_pct": d}) for d in (2, 4, 10)]
)
BREAK_LEVELS = [(kind, rate) for kind in ("pitch_jump", "aphonic") for rate in (0.5, 1, 2)]

build_library = dl.build_library
load_library = dl.load_library


def degrade(x, profile, params, seed, break_kind="pitch_jump"):
    rc, out, msg = dl.call(x, FS, FS, profile, params, seed, break_kind)
    if rc != 0:
        raise RuntimeError(f"degrade_voice_f0 refused {profile} {params}: {msg}")
    return out


# ── Praat measures ──────────────────────────────────────────────────────────────────────
def pitch_track(y):
    snd = parselmouth.Sound(y.astype(np.float64), sampling_frequency=FS)
    pitch = snd.to_pitch_cc(time_step=0.005, pitch_floor=50.0, pitch_ceiling=600.0)
    return pitch.xs(), pitch.selected_array["frequency"]


def voiced_span(x):
    """Longest voiced run of the raw trial (Praat pitch), trimmed; None if too short."""
    t, f = pitch_track(x)
    best, start = (0, 0), None
    for i, v in enumerate(np.append(f, 0.0)):
        if v > 0 and start is None:
            start = i
        elif v <= 0 and start is not None:
            if i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    if best[1] - best[0] < 2:
        return None
    t1, t2 = t[best[0]] + TRIM_S, t[best[1] - 1] - TRIM_S
    return (t1, t2) if t2 - t1 >= MIN_SPAN_S else None


def periodicity(y, span, period_factor, amplitude_factor):
    t1, t2 = span
    snd = parselmouth.Sound(y.astype(np.float64), sampling_frequency=FS)
    pp = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 75.0, 600.0)
    jit = parselmouth.praat.call(pp, "Get jitter (local)", t1, t2, 0.0001, 0.02, period_factor)
    shim = parselmouth.praat.call([snd, pp], "Get shimmer (local_dB)", t1, t2, 0.0001, 0.02,
                                  period_factor, amplitude_factor)
    harm = snd.to_harmonicity_cc(time_step=0.01, minimum_pitch=75.0,
                                 silence_threshold=0.1, periods_per_window=1.0)
    hx, hv = harm.xs(), harm.values[0]
    keep = (hx >= t1) & (hx <= t2) & (hv > -199.0)   # -200 dB marks an unvoiced frame
    hnr = float(hv[keep].mean()) if keep.any() else float("nan")
    return 100.0 * jit, shim, hnr


def tremor_depth(y, span):
    t, f = pitch_track(y)
    keep = (t >= span[0]) & (t <= span[1]) & (f > 0)
    t, f = t[keep], f[keep]
    w = 2 * math.pi * TREMOR_RATE_HZ * t
    design = np.column_stack([np.ones_like(t), t - t.mean(), np.sin(w), np.cos(w)])
    coef, *_ = np.linalg.lstsq(design, f, rcond=None)
    return 100.0 * math.hypot(coef[2], coef[3]) / coef[0]


def breaks(y, span, kind, f0_ref):
    t, f = pitch_track(y)
    keep = (t >= span[0]) & (t <= span[1])
    f = f[keep]
    if kind == "pitch_jump":
        off = (f > 0) & ((f < 0.8 * f0_ref) | (f > 1.25 * f0_ref))
        min_frames = 2                       # 10 ms
    else:
        off = f <= 0
        min_frames = 6                       # 30 ms of no voicing
    runs, n = 0, 0
    for v in np.append(off, False):
        if v:
            n += 1
        else:
            runs += n >= min_frames
            n = 0
    return runs / (span[1] - span[0])


def rms_db(y, span):
    a, b = int(span[0] * FS), int(span[1] * FS)
    seg = y[a:b].astype(np.float64)
    return 10 * math.log10(np.mean(seg ** 2))


# ── one voice ──────────────────────────────────────────────────────────────────────────
def measure_voice(job):
    """Every request of the job on one voice; returns rows."""
    so, voice, x, requests = job
    if not dl.loaded():
        load_library(so)
    span = voiced_span(x)
    if span is None:
        return [{**voice, "status": f"excluded: voiced span under {MIN_SPAN_S} s"}]
    t, f = pitch_track(x)
    f0_ref = float(np.median(f[(t >= span[0]) & (t <= span[1]) & (f > 0)]))
    seed = SEED + voice["index"]
    raw_jit, raw_shim, raw_hnr = periodicity(x, span, 1.3, 1.6)
    base = {**voice, "status": "ok", "span_s": round(span[1] - span[0], 3),
            "f0_hz": round(f0_ref, 1)}
    rows = [{**base, "profile": "raw", "jitter_pct": raw_jit, "shimmer_db": raw_shim,
             "hnr_db": raw_hnr, "tremor_depth_pct": tremor_depth(x, span)}]
    raw_rms = rms_db(x, span)
    for req in requests:
        prof, params = req["profile"], req["params"]
        kind = params.get("break_kind", "pitch_jump")
        y = degrade(x, prof, params, seed, kind)
        row = {**base, "profile": prof, "set": req["set"],
               **{f"req_{k}": v for k, v in params.items()}}
        diplo = prof == "diplophonia"
        jit, shim, hnr = periodicity(y, span, 2.0 if diplo else 1.3, 3.0 if diplo else 1.6)
        row.update({"hnr_db": hnr, "rms_change_db": rms_db(y, span) - raw_rms,
                    "hnr_change_db": hnr - raw_hnr})
        if diplo:
            row.update({"alt_period_pct": jit / 2.0, "alt_amp_db": shim})
        else:
            row.update({"jitter_pct": jit, "shimmer_db": shim})
        if prof in ("tremor", "resynthesis_control"):
            row["tremor_depth_pct"] = tremor_depth(y, span)
        if prof == "voice_breaks":
            row["breaks_per_s"] = breaks(y, span, kind, f0_ref)
        rows.append(row)
    return rows


# ── data ───────────────────────────────────────────────────────────────────────────────
def load_voices(db, subjects):
    con = sqlite3.connect(f"file:{db}?immutable=1", uri=True)   # read-only
    voices = []
    for subj in subjects:
        rows = con.execute(
            "SELECT st.name, t.original_row, a.samples FROM trial t "
            "JOIN subject s ON s.id = t.subject_id JOIN stimulus st ON st.id = t.stimulus_id "
            "JOIN modalities m ON m.id = t.modality_id JOIN audio_samples a ON a.trial_id = t.id "
            "WHERE s.subject_name = ? AND m.name = 'Pronounced' AND st.name IN "
            f"({','.join('?' * len(VOWELS))}) ORDER BY st.name, t.original_row",
            (subj, *VOWELS)).fetchall()
        if not rows:
            raise SystemExit(f"[calibration] {db}: no pronounced vowel of {subj}")
        for vowel, row, blob in rows:
            voices.append(({"subject": subj, "vowel": vowel, "trial": row,
                            "index": len(voices)}, np.frombuffer(blob, dtype=np.float32).copy()))
    con.close()
    return voices


def run(so, voices, requests, jobs):
    rows = []
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        for out in pool.map(measure_voice, [(so, v, x, requests) for v, x in voices]):
            rows.extend(out)
    return rows


# ── calibration ────────────────────────────────────────────────────────────────────────
def mean_curve(rows, axis):
    prof, grid = SWEEPS[axis]
    key = READ_BACK[axis]
    means = []
    for r in grid:
        vals = [row[key] for row in rows if row.get("set") == f"sweep:{axis}"
                and row.get(f"req_{axis}") == r and not math.isnan(row[key])]
        means.append(float(np.mean(vals)))
    return grid, means


def request_for(target, grid, means):
    """Linear interpolation of the request whose MEAN measured value is the target; None when
    the target lies outside the measured range (reported, never clamped)."""
    for (r0, m0), (r1, m1) in zip(zip(grid, means), zip(grid[1:], means[1:])):
        if min(m0, m1) <= target <= max(m0, m1) and m1 != m0:
            return r0 + (target - m0) * (r1 - r0) / (m1 - m0)
    return None


def summarize(rows, levels_rows, calibrated):
    """Per level: per subject mean/sd of each target measure and how many voices are within
    tolerance."""
    out = []
    for lv in calibrated:
        sel = [r for r in levels_rows if r.get("set") == lv["set"]]
        entry = dict(lv)
        for axis, target in lv["targets"].items():
            key = READ_BACK[axis]
            entry[axis] = {}
            for subj in sorted({r["subject"] for r in sel}):
                vals = np.array([r[key] for r in sel if r["subject"] == subj])
                within = int(np.sum(np.abs(vals - target) <= TOLERANCE * target))
                entry[axis][subj] = {"mean": round(float(np.mean(vals)), 3),
                                     "sd": round(float(np.std(vals, ddof=1)), 3),
                                     "within_tolerance": within, "n": int(vals.size)}
        hnr = {}
        for subj in sorted({r["subject"] for r in sel}):
            vals = np.array([r["hnr_db"] for r in sel if r["subject"] == subj])
            hnr[subj] = round(float(np.nanmean(vals)), 2)
        entry["hnr_db"] = hnr
        out.append(entry)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", required=True, help="the Coretto base (database.sqlite)")
    ap.add_argument("--subjects", required=True, nargs="+", help="e.g. S10 S03")
    ap.add_argument("--jobs", required=True, type=int, help="worker processes")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    started = time.time()

    voices = load_voices(os.path.expanduser(args.db), args.subjects)
    print(f"[calibration] {len(voices)} vowel trials of {' '.join(args.subjects)}")
    with tempfile.TemporaryDirectory() as tmp:
        so = build_library(tmp)

        # 1. Sweep (and the control).
        requests = [{"set": "control", "profile": "resynthesis_control", "params": {}}]
        for axis, (prof, grid) in SWEEPS.items():
            for r in grid:
                params = {axis: r}
                if OTHER_AXIS[axis] is not None:
                    params[OTHER_AXIS[axis][0]] = OTHER_AXIS[axis][1]
                if prof == "tremor":
                    params["rate_hz"] = TREMOR_RATE_HZ
                requests.append({"set": f"sweep:{axis}", "profile": prof, "params": params})
        sweep_rows = run(so, voices, requests, args.jobs)
        excluded = [r for r in sweep_rows if r["status"] != "ok"]
        ok = [r for r in sweep_rows if r["status"] == "ok"]

        # 2. Request per target, from the mean curve over every measured voice.
        curves = {axis: mean_curve(ok, axis) for axis in SWEEPS}
        calibrated, verify = [], []
        for i, (prof, targets) in enumerate(LEVELS):
            reqs, unreachable = {}, []
            for axis, target in targets.items():
                r = request_for(target, *curves[axis])
                if r is None:
                    unreachable.append(axis)
                else:
                    reqs[axis] = round(r, 3)
            lv = {"set": f"level{i}", "profile": prof, "targets": targets,
                  "request": reqs, "unreachable_on_mean": unreachable}
            calibrated.append(lv)
            if not unreachable:
                params = dict(reqs)
                if prof == "tremor":
                    params["rate_hz"] = TREMOR_RATE_HZ
                verify.append({"set": lv["set"], "profile": prof, "params": params})
        for kind, rate in BREAK_LEVELS:
            verify.append({"set": f"breaks:{kind}:{rate}", "profile": "voice_breaks",
                           "params": {"break_kind": kind, "breaks_per_second": rate,
                                      "break_ms": BREAK_MS}})
        # The break kind travels in params for the worker; the C++ takes it apart.
        verify_rows = [r for r in run(so, voices, verify, args.jobs) if r["status"] == "ok"
                       and r["profile"] != "raw"]

    reached = [lv for lv in calibrated if not lv["unreachable_on_mean"]]
    summary = summarize(ok, verify_rows, reached)
    raw = [r for r in ok if r["profile"] == "raw"]
    control = [r for r in ok if r.get("set") == "control"]
    breaks_summary = []
    for kind, rate in BREAK_LEVELS:
        sel = [r for r in verify_rows if r["set"] == f"breaks:{kind}:{rate}"]
        for subj in sorted({r["subject"] for r in sel}):
            v = np.array([r["breaks_per_s"] for r in sel if r["subject"] == subj])
            breaks_summary.append({"kind": kind, "requested_per_s": rate, "subject": subj,
                                   "measured_mean_per_s": round(float(v.mean()), 3),
                                   "n": int(v.size)})
    per_subject = {}
    for subj in args.subjects:
        rr = [r for r in raw if r["subject"] == subj]
        cc = [r for r in control if r["subject"] == subj]
        per_subject[subj] = {
            "voices": len(rr), "f0_hz_mean": round(float(np.mean([r["f0_hz"] for r in rr])), 1),
            "raw": {k: round(float(np.nanmean([r[k] for r in rr])), 3)
                    for k in ("jitter_pct", "shimmer_db", "hnr_db", "tremor_depth_pct")},
            "control": {k: round(float(np.nanmean([r[k] for r in cc])), 3)
                        for k in ("jitter_pct", "shimmer_db", "hnr_db", "tremor_depth_pct",
                                  "rms_change_db", "hnr_change_db")},
            "control_within_1dB_rms": int(sum(abs(r["rms_change_db"]) <= 1.0 for r in cc)),
            "control_within_1_5dB_hnr": int(sum(abs(r["hnr_change_db"]) <= 1.5 for r in cc)),
        }

    fields = sorted({k for r in sweep_rows + verify_rows for k in r})
    with open(os.path.join(args.out_dir, "measurements.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in sweep_rows + verify_rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    result = {
        "praat": parselmouth.PRAAT_VERSION, "parselmouth": parselmouth.__version__,
        "subjects": args.subjects, "vowels": list(VOWELS), "tolerance": TOLERANCE,
        "excluded": [{k: r[k] for k in ("subject", "vowel", "trial", "status")}
                     for r in excluded],
        "per_subject": per_subject,
        "curves": {axis: {"request": g, "mean_measured": [round(m, 3) for m in mm]}
                   for axis, (g, mm) in curves.items()},
        "levels": summary,
        "unreachable_levels": [lv for lv in calibrated if lv["unreachable_on_mean"]],
        "breaks": breaks_summary,
        "seconds": round(time.time() - started, 1),
    }
    with open(os.path.join(args.out_dir, "calibration.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    print(json.dumps({k: result[k] for k in ("per_subject", "seconds")}, indent=1))
    print(f"[calibration] {len(excluded)} trials excluded; wrote {args.out_dir}")


if __name__ == "__main__":
    sys.exit(main())
