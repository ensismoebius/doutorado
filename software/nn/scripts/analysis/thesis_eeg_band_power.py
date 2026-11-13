#!/usr/bin/env python3
"""Where the power of the thesis EEG lies, and what that means for its 32 Hz sub-bands.

WHY. The handcrafted EEG features are energies of 16 DTWPT sub-bands of 32 Hz each
(level 4 at 1024 Hz). Whether the 50 Hz mains line, the classic rhythms or nothing at all
sits in each sub-band decides what those features can carry. This script measures it on
the distributed database instead of assuming it (scipy only, none of the C++).

WHAT. Welch spectrum (1 Hz bins) of every trial, mean over the six channels and over all
trials; prints the level at chosen frequencies relative to the 8-12 Hz (alpha) level, the
share of power per range, and the number of trials the database flags with a blink.

Measured 2026-11-12 on the 9315 trials: -30 dB at 46 Hz, -58 dB at 50 Hz, -87 dB at 54 Hz
(consistent with the FIR band-pass of 2-40 Hz the database authors apply,
Pressel Coretto et al., SPIE 2017); 98.1% of the power below
32 Hz (sub-band 0), 1.9% in 32-64 Hz (sub-band 1), ~0 above 64 Hz; 2653 trials flagged
with a blink.
"""
import argparse
import csv
import os
import sqlite3

import numpy as np
from scipy.signal import welch

FS = 1024.0
CHANNELS = ("F3", "F4", "C3", "C4", "P3", "P4")
PROBES = (1, 2, 4, 8, 10, 12, 16, 20, 25, 30, 35, 40, 44, 46, 48, 50, 52, 54, 60, 64, 100, 200, 500)
RANGES = ((0, 4), (4, 8), (8, 13), (13, 30), (0, 32), (32, 64), (64, 512))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--db", required=True, help="the thesis SQLite database")
    ap.add_argument("--out", required=True, help="CSV with the level per probe frequency")
    args = ap.parse_args()

    con = sqlite3.connect(os.path.expanduser(args.db))
    rows = con.execute("select %s from eeg_samples" % ",".join(CHANNELS)).fetchall()
    if not rows:
        raise SystemExit("no EEG rows in %s" % args.db)
    total = None
    for r in rows:
        x = np.stack([np.frombuffer(ch, dtype=np.float32) for ch in r])
        if x.shape != (6, 4096):
            raise SystemExit("unexpected EEG shape %s (expected 6 x 4096 float32)" % (x.shape,))
        f, p = welch(x, fs=FS, nperseg=1024, axis=-1)
        total = p.mean(0) if total is None else total + p.mean(0)
    pm = total / len(rows)
    ref = pm[(f >= 8) & (f <= 12)].mean()
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frequency_hz", "level_db_re_alpha"])
        for hz in PROBES:
            i = int(np.argmin(np.abs(f - hz)))
            level = 10 * np.log10(pm[i] / ref)
            w.writerow([hz, "%.1f" % level])
            print("%5d Hz  %7.1f dB re 8-12 Hz" % (hz, level))
    for a, b in RANGES:
        print("power %3d-%3d Hz: %5.1f%%" % (a, b, 100 * pm[(f >= a) & (f < b)].sum() / pm.sum()))
    blink = con.execute("select blink, count(*) from eeg_samples group by blink").fetchall()
    print("trials:", len(rows), " blink flag counts (1 = no blink, 2 = blink):", dict(blink))


if __name__ == "__main__":
    main()
