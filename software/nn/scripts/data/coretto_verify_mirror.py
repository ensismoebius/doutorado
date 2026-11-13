#!/usr/bin/env python3
"""Check the Coretto SQLite (~/database.sqlite) against an independent copy of the EEG .mat files.

WHY. The SQLite the thesis reads was built from the authors' .mat files by
scripts/data/import_mat_dataset_to_sqlite.py, and those files are no longer here; the authors'
Google Drive folder now asks for a sign-in. The EEG half of the base is re-hosted on Zenodo
(record 19502780, uploaded 2026-04-11 for the MOABB framework; the uploader states the .mat
files are bit-identical to the originals, renamed S{NN}_EEG.mat -> sub-{NN}_eeg.mat). That
copy is not the authors', so agreement here means two independent derivations of the same
files agree, not that either is certified.

What is compared, per subject, row by row of the .mat matrix (Pressel Coretto et al. 2017,
sec. 2.3: each row is the six channels F3 F4 C3 C4 P3 P4 of one trial, 4096 samples each,
concatenated in that order, then three labels: modality, stimulus code, blink artifact):
  - every row is in the SQLite exactly once (trial.original_row), and nothing else is;
  - the six channels equal the stored blobs exactly;
  - modality (1 imagined, 2 pronounced), stimulus (1-5 /a e i o u/, 6-11 arriba, abajo,
    adelante, atras, derecha, izquierda) and the blink label equal the stored ones.
The audio is not in the mirror and is not checked here.

Exit status 1 on any difference.

Usage (from software/nn):
    python3 scripts/data/coretto_verify_mirror.py --db ~/database.sqlite --mirror ~/coretto_zenodo
"""
import argparse
import io
import os
import sqlite3
import sys
import zipfile

import numpy as np
from scipy.io import loadmat

CHANNELS = ("F3", "F4", "C3", "C4", "P3", "P4")
N = 4096
MODALITY = {1: "Imagined", 2: "Pronounced"}
STIMULUS = {1: "A", 2: "E", 3: "I", 4: "O", 5: "U", 6: "Arriba", 7: "Abajo", 8: "Adelante",
            9: "Atras", 10: "Derecha", 11: "Izquierda"}


def blob(b):
    """The stored channel in its own precision. The importer wrote the .mat's float64 values
    as float32, so the comparison rounds the .mat to the stored precision and then requires
    exact equality: any difference beyond float32 rounding is a different value."""
    if len(b) == 4 * N:
        return np.frombuffer(b, np.float32)
    if len(b) == 8 * N:
        return np.frombuffer(b, np.float64)
    raise ValueError(f"EEG blob of {len(b)} bytes")


def mat_matrix(zip_path):
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if n.lower().endswith(".mat")]
        if len(names) != 1:
            raise ValueError(f"{zip_path}: {names}")
        mat = loadmat(io.BytesIO(z.read(names[0])))
    arrays = {k: v for k, v in mat.items()
              if not k.startswith("__") and isinstance(v, np.ndarray) and v.ndim == 2
              and v.shape[1] == 6 * N + 3}
    if len(arrays) != 1:
        raise ValueError(f"{zip_path}: expected one (trials x {6 * N + 3}) matrix, found "
                         f"{ {k: v.shape for k, v in mat.items() if not k.startswith('__')} }")
    return names[0], next(iter(arrays.values()))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--db", required=True)
    ap.add_argument("--mirror", required=True, help="folder with S01.zip ... S15.zip")
    args = ap.parse_args()
    db, mirror = os.path.expanduser(args.db), os.path.expanduser(args.mirror)
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    fails = []

    def fail(msg):
        fails.append(msg)
        print(f"  DIFF {msg}", flush=True)

    total = equal = 0
    for sid, subject in con.execute("SELECT id, subject_name FROM subject ORDER BY id").fetchall():
        name, m = mat_matrix(os.path.join(mirror, f"{subject}.zip"))
        stored = {}
        for row in con.execute(
                "SELECT t.original_row, mo.name, st.name, e.blink, e.F3, e.F4, e.C3, e.C4, "
                "e.P3, e.P4 FROM trial t JOIN eeg_samples e ON e.trial_id = t.id "
                "JOIN modalities mo ON mo.id = t.modality_id "
                "JOIN stimulus st ON st.id = t.stimulus_id WHERE t.subject_id = ?", (sid,)):
            if row[0] in stored:
                fail(f"{subject}: row {row[0]} stored twice")
            stored[row[0]] = row[1:]
        for r in range(m.shape[0]):
            total += 1
            if r not in stored:
                fail(f"{subject}: .mat row {r} is not in the SQLite")
                continue
            mod, stim, blink, *chans = stored.pop(r)
            ok = True
            for c, ch in enumerate(CHANNELS):
                have = blob(chans[c])
                want = m[r, c * N:(c + 1) * N].astype(have.dtype)
                if not np.array_equal(have, want):
                    d = np.abs(have.astype(np.float64) - want).max()
                    fail(f"{subject} row {r}: {ch} differs beyond {have.dtype} rounding "
                         f"(max {d:.3g})")
                    ok = False
            labels = (MODALITY.get(int(m[r, 6 * N])), STIMULUS.get(int(m[r, 6 * N + 1])),
                      int(m[r, 6 * N + 2]))
            if (mod, stim, blink) != labels:
                fail(f"{subject} row {r}: SQLite {mod}/{stim}/blink {blink}, .mat {labels}")
                ok = False
            equal += ok
        for r in stored:
            fail(f"{subject}: SQLite row {r} has no .mat row")
        print(f"[coretto] {subject} ({name}): {m.shape[0]} rows", flush=True)
    con.close()
    print(f"[coretto] {equal} of {total} trials equal to the mirror (EEG and labels); "
          f"{len(fails)} difference(s)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
