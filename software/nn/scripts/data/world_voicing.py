#!/usr/bin/env python3
"""Which voices can the thesis's WORLD dysphonia path not analyse?

WHY. Every WORLD profile (degradation.profile other than whisper_mix) starts by asking
WORLD's Harvest for the F0 contour of the voice. A voice in which Harvest finds no voiced frame
has no pitch to edit, and VoiceDysphonia.cpp refuses it (no fallback: the run stops, naming the
sample). On BIOMEX-DB three windows do that (found 2026-11-16, when the WORLD grid stopped on
sample 3919): a window holding only a faint hum (M006 "siete") and two very short, barely voiced
"seis" (M014). The user decided they leave the base (biomex_to_sqlite.py UNANALYSABLE_VOICES);
biomex_verify.py proves the list with this module: each listed voice, rebuilt from the
originals, is refused, and no stored voice is.

HOW. It does not re-implement Harvest: it compiles the thesis's own VoiceDysphonia.cpp with the
vendored WORLD (scripts/lib/thesis_dysphonia_lib.py, the build the Praat calibration uses too)
and asks for `resynthesis_control` on each voice, at the rates the thesis uses (the
stored rate and the base's native rate), exactly as thesis::degrade_voices does. A refusal is
the answer; its message is kept.

Usage (from software/nn; needs g++ and a configured build tree for the WORLD sources):
    python3 scripts/data/world_voicing.py --db <databases>/biomex.sqlite   # scan a base
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import tempfile
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))
import thesis_dysphonia_lib as dl  # noqa: E402

SEED = 7                # the thesis profiles' degradation.seed (irrelevant to the refusal)


def _one(job):
    i, x, fs, native = job
    seed = SEED ^ ((i * 2654435761) & 0xFFFFFFFF)   # as thesis::degrade_voices
    rc, _, msg = dl.call(x, fs, native, "resynthesis_control", {}, seed)
    return i, rc, msg


_SO = None   # the compiled library, once per process (the verifier asks twice)


def refused(voices, fs: float, native: float, jobs: int) -> dict[int, str]:
    """{index: VoiceDysphonia's message} for every voice the WORLD path refuses."""
    global _SO
    if _SO is None:
        _SO = dl.build_library(tempfile.mkdtemp(prefix="world_voicing_"))
    so = _SO
    with Pool(jobs, initializer=dl.load_library, initargs=(so,)) as pool:
        res = pool.map(_one, [(i, v, fs, native) for i, v in enumerate(voices)], chunksize=20)
    return {i: msg for i, rc, msg in res if rc != 0}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--db", required=True, help="a thesis-paired base (biomex.sqlite)")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()
    con = sqlite3.connect(os.path.expanduser(args.db))
    info = dict(con.execute("SELECT key, value FROM recording"))
    rows = con.execute(
        "SELECT p.id, s.name, p.session, p.series, p.position, p.voice FROM pair p "
        "JOIN speaker s ON s.id = p.speaker_id WHERE s.excluded = 0 ORDER BY p.id").fetchall()
    con.close()
    bad = refused([np.frombuffer(r[5], dtype=np.float32) for r in rows],
                  float(info["voice_sample_rate"]), float(info["voice_native_rate"]), args.jobs)
    print(f"[world-voicing] {len(rows)} voices, {len(bad)} refused by the WORLD path")
    for i, msg in sorted(bad.items()):
        pid, name, ses, k, j, _ = rows[i]
        print(f"  pair {pid}: {name} {ses} series {k} digit {j}: {msg[:110]}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
