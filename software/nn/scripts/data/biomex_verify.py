#!/usr/bin/env python3
"""Check a converted BIOMEX-DB SQLite against the original files and the paper.

WHY. The converter (biomex_to_sqlite.py) refuses what it cannot pair, but a converter can be
wrong in ways it cannot see itself: an EDF read with a wrong offset, a window cut one series
off, speakers numbered in another order. This script re-derives every pair through a second
path and compares, and checks the originals against two sources the converter never reads:

  1. the CSV export of every EEG session (Emotiv TestBench, written separately from the EDF):
     the EDF the converter reads must hold the same values (the CSV stops at a round row
     count, 22000 / 43000, so only the overlap is compared) and the same recording time;
  2. Table 1 of the paper (Moreno-Rodriguez et al., IEEE Access 9:111267, 2021): the digit
     sequence shown in each series, the same for every volunteer. The EEG markers and the
     .lab words of every series must equal it;
  3. every pair in the SQLite must equal the one rebuilt from the originals: the WAV read with
     scipy.io.wavfile (not the wave module), the voice window placed by a re-implementation of
     the rule, the EEG epoch from the EDF checked in (1);
  4. counts: 140 pairs per volunteer, and per digit what Table 1 implies (3: 15, 8: 13, every
     other digit 14), speakers named and numbered as the folders.

Exit status 1 on any difference; every difference is printed.

Usage (from software/nn):
    python3 scripts/data/biomex_verify.py --src ~/biomex/extracted --db ~/biomex.sqlite
"""
import argparse
import hashlib
import os
import sqlite3
import sys
from collections import Counter

import numpy as np
from scipy.io import wavfile
from scipy.signal import filtfilt, resample_poly

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import biomex_to_sqlite as bx  # noqa: E402

# Table 1 of the paper: (a) the 10-digit series, (b) the 4-digit series.
TABLE1 = {
    "01G10": [[7, 9, 0, 2, 1, 5, 8, 6, 4, 3], [1, 7, 0, 3, 8, 4, 6, 5, 2, 9],
              [6, 8, 2, 5, 3, 0, 9, 1, 4, 7], [9, 4, 2, 1, 0, 3, 8, 7, 5, 6],
              [2, 0, 9, 1, 3, 7, 5, 4, 6, 8], [8, 6, 1, 5, 7, 0, 3, 9, 2, 4],
              [3, 5, 6, 8, 1, 2, 4, 7, 9, 0], [4, 3, 5, 6, 9, 7, 0, 8, 2, 1],
              [0, 8, 2, 1, 3, 9, 7, 4, 6, 5], [5, 3, 1, 6, 7, 0, 4, 9, 8, 2]],
    "01G04": [[1, 2, 3, 4], [5, 3, 2, 9], [1, 0, 7, 3], [9, 6, 4, 7], [5, 4, 2, 1],
              [8, 3, 9, 6], [7, 0, 6, 8], [9, 5, 2, 3], [0, 6, 4, 7], [8, 1, 5, 0]],
}
CSV_TOL_UV = 1e-3      # the CSV prints ~1e-4 uV of rounding
VOICE_TOL = 1e-6
EEG_TOL_UV = 1e-3


def code_of(digit):
    return 10 if digit == 0 else digit


def read_csv(path):
    """(header fields, values (rows, 36)) of an Emotiv TestBench CSV export."""
    with open(path) as f:
        head = f.readline()
        body = f.read()
    fields = {}
    for part in head.split(","):
        if ":" in part:
            k, v = part.split(":", 1)
            fields[k.strip()] = v.strip()
    ncol = body.split("\n", 1)[0].count(",") + 1
    vals = np.fromstring(body.replace("\n", ","), sep=",")
    if vals.size % ncol:
        raise ValueError(f"{path}: {vals.size} values do not fill rows of {ncol}")
    return fields, vals.reshape(-1, ncol)


def voice_windows(words, n_samples, win):
    """Window starts by the rule, re-implemented: at the onset, or earlier so as to end at the
    next onset (or the WAV's end); None where no window fits after the previous word."""
    starts = []
    prev_end = 0
    for j, (on, end, _) in enumerate(words):
        nxt = words[j + 1][0] if j + 1 < len(words) else n_samples
        b = on if on + win <= nxt else nxt - win
        starts.append(b if b >= prev_end else None)
        prev_end = end
    return starts


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", required=True, help="the extracted base (F001/, M001/, ...)")
    ap.add_argument("--db", required=True, help="the SQLite written by biomex_to_sqlite.py")
    args = ap.parse_args()
    src, db = os.path.expanduser(args.src), os.path.expanduser(args.db)
    fails = []

    def fail(msg):
        fails.append(msg)
        print(f"  DIFF {msg}", flush=True)

    con = sqlite3.connect(db)
    info = dict(con.execute("SELECT key, value FROM recording"))
    want_info = {"format": bx.FORMAT, "source": bx.SOURCE,
                 "voice_sample_rate": str(bx.OUT_VOICE_RATE),
                 "eeg_sample_rate": str(bx.OUT_EEG_RATE),
                 "voice_window_samples": str(bx.OUT_VOICE_RATE),
                 "eeg_epoch_samples": str(2 * bx.OUT_EEG_RATE),
                 "eeg_channels": ",".join(bx.EEG_CHANNELS)}
    for k, v in want_info.items():
        if info.get(k) != v:
            fail(f"recording.{k} = {info.get(k)!r}, expected {v!r}")

    volunteers = sorted(d for d in os.listdir(src) if bx.VOLUNTEER.match(d))
    speakers = con.execute("SELECT id, name FROM speaker ORDER BY id").fetchall()
    if [n for _, n in speakers] != volunteers or [i for i, _ in speakers] != list(
            range(1, len(volunteers) + 1)):
        fail(f"speakers {speakers[:3]}... do not number the folders {volunteers[:3]}... in order")
    stimuli = dict(con.execute("SELECT id, name FROM stimulus"))
    if stimuli != {c: w.lower() for c, w in bx.DIGIT_WORDS.items()}:
        fail(f"stimulus table {stimuli}")

    stored = {}
    for name, stim, ses, k, j, v, e, on_s, shift_s in con.execute(
            "SELECT s.name, p.stimulus_id, p.session, p.series, p.position, p.voice, p.eeg, "
            "p.voice_onset_s, p.voice_shift_s FROM pair p JOIN speaker s ON s.id = p.speaker_id"):
        stored[(name, ses, k, j)] = (stim, v, e, on_s, shift_s)
    print(f"[verify] {len(volunteers)} volunteers in {src}; {len(stored)} pairs in {db}")

    taps = bx.band_pass()
    win, out_win = bx.VOICE_RATE, bx.OUT_VOICE_RATE
    epoch, up = 2 * bx.EEG_RATE, bx.EEG_UP
    per_digit = Counter(d for rows in TABLE1.values() for r in rows for d in r)
    checked = 0
    for vid in volunteers:
        folder = os.path.join(src, vid)
        digits = Counter()
        n_pairs = 0
        for session, rows in TABLE1.items():
            where = f"{vid} {session}"
            edf = bx.find_edf(folder, vid, session)
            labels, phys, rate, start, raw = bx.read_edf(edf)
            # (1) the EDF against its CSV export
            fields, csv = read_csv(os.path.join(folder, "EEG", f"{vid}-{session}.CSV"))
            if fields.get("recorded") != start:
                fail(f"{where}: EDF recorded {start!r}, CSV {fields.get('recorded')!r}")
            n = min(csv.shape[0], phys.shape[1])
            csv_labels = fields.get("labels", "").split()
            for ch in bx.EEG_CHANNELS + ("MARKER",):
                d = np.abs(csv[:n, csv_labels.index(ch)] - phys[labels.index(ch), :n]).max()
                if d > (0 if ch == "MARKER" else CSV_TOL_UV):
                    fail(f"{where}: {ch} differs from the CSV by up to {d:.4g}")
            # (2) the series against Table 1
            runs = bx.digit_runs(raw[labels.index("MARKER")], where)
            if len(runs) != len(rows):
                fail(f"{where}: {len(runs)} digit runs, Table 1 has {len(rows)}")
                continue
            eeg = filtfilt(taps, [1.0], np.stack([phys[labels.index(c)]
                                                  for c in bx.EEG_CHANNELS]), axis=-1)
            eeg_up = resample_poly(eeg, up, 1, axis=-1)
            for k, (row, run) in enumerate(zip(rows, runs), 1):
                stem = os.path.join(folder, "Voice", f"{vid}_{session}_{k}")
                words = bx.read_lab(stem + ".lab", bx.LAB_MISSING_WORDS.get((vid, session, k)))
                said = [0 if w == "CERO" else [c for c, x in bx.DIGIT_WORDS.items()
                                               if x == w][0] for _, _, w in words]
                shown = [0 if c == 10 else c for _, c in run]
                if shown != row or said != row:
                    fail(f"{where} series {k}: Table 1 {row}, markers {shown}, .lab {said}")
                    continue
                rate_w, pcm = wavfile.read(stem + ".wav")
                if rate_w != bx.VOICE_RATE or pcm.dtype != np.int16 or pcm.ndim != 1:
                    fail(f"{stem}.wav: {rate_w} Hz {pcm.dtype} {pcm.shape}")
                    continue
                voice_up = resample_poly(pcm.astype(np.float32) / 32768.0,
                                         bx.VOICE_UP, bx.VOICE_DOWN)
                starts = voice_windows(words, len(pcm), win)
                # (3) every pair against the one rebuilt here
                for j, ((m, code), (on, _, _), b) in enumerate(zip(run, words, starts), 1):
                    key = (vid, session, k, j)
                    if key not in stored:
                        fail(f"{key}: missing from the SQLite")
                        continue
                    stim, v, e, on_s, shift_s = stored.pop(key)
                    n_pairs += 1
                    fails_before = len(fails)
                    digits[0 if code == 10 else code] += 1
                    if stim != code:
                        fail(f"{key}: stimulus {stim}, the marker says {code}")
                    if b is None:
                        fail(f"{key}: no window fits, yet the pair exists")
                        continue
                    v0 = int(round(b * bx.VOICE_UP / bx.VOICE_DOWN))
                    want_v = voice_up[v0:v0 + out_win]
                    want_e = eeg_up[:, m * up:(m + epoch) * up]
                    got_v = np.frombuffer(v, np.float32)
                    got_e = np.frombuffer(e, np.float32).reshape(len(bx.EEG_CHANNELS), -1)
                    if got_v.shape != want_v.shape or np.abs(got_v - want_v).max() > VOICE_TOL:
                        fail(f"{key}: voice differs from the rebuilt window")
                    if got_e.shape != want_e.shape or np.abs(got_e - want_e).max() > EEG_TOL_UV:
                        fail(f"{key}: EEG differs from the rebuilt epoch")
                    if abs(on_s - on / bx.VOICE_RATE) > 1e-6 or abs(
                            shift_s - (on - b) / bx.VOICE_RATE) > 1e-6:
                        fail(f"{key}: onset/shift {on_s}/{shift_s} differ from the .lab")
                    checked += len(fails) == fails_before
        # (4) counts per volunteer
        if n_pairs != 140:
            fail(f"{vid}: {n_pairs} pairs, expected 140")
        if digits != per_digit:
            fail(f"{vid}: per-digit counts {dict(sorted(digits.items()))}, Table 1 implies "
                 f"{dict(sorted(per_digit.items()))}")
        print(f"[verify] {vid}: {n_pairs} pairs checked", flush=True)
    for key in stored:
        fail(f"{key}: in the SQLite but not in the originals")
    con.close()
    print(f"[verify] {checked} pairs equal to the originals; {len(fails)} difference(s)")
    stamp = db + ".verified"
    if os.path.exists(stamp):
        os.remove(stamp)
    if fails:
        return 1
    # The deploy (gridunesp_deploy.sh) sends the base only with this stamp, and only if the
    # file's sha256 is still the one checked here.
    h = hashlib.sha256()
    with open(db, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    with open(stamp, "w") as f:
        f.write(f"{h.hexdigest()}\n{len(volunteers)} volunteers, {checked} pairs equal to "
                f"{src} (EDF = CSV export, series = Table 1 of the paper)\n")
    print(f"[verify] stamp written: {stamp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
