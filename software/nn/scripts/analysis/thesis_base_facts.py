#!/usr/bin/env python3
"""The facts about the two bases that the thesis prose states, as a CSV of key, kind, value.

WHY. Chapter 8 says how many volunteers BIOMEX-DB keeps, which were excluded and why, how many
pairs survive, that every pair equals the originals, that the Coretto EEG equals an independent
copy, and how the two bases differ below 2 Hz. Typed by hand, any of these goes stale silently
the next time a base is rebuilt (`.wiki/Experiments/Thesis.md` sec. 10). So they are read here
from the bases themselves and from the verifiers' stamps, and the thesis quotes them through
thesis_build_text_numbers.py (keys base.*).

Sources, and why each is trusted:
  ~/biomex.sqlite            the converted base (scripts/data/biomex_to_sqlite.py)
  ~/biomex.sqlite.verified   scripts/data/biomex_verify.py's stamp: line 1 the sha256 of the
                             base it passed, line 2 its counts. Refused unless the sha256 is
                             the current file's: a fact about a base nobody checked is not one.
  ~/database.sqlite          the Coretto base
  ~/database.sqlite.mirror-verified
                             scripts/data/coretto_verify_mirror.py's stamp (same layout)

One constant is not read from a file: the paper's own count of S01's pronounced /o/ trials
(Pressel Coretto et al. 2017, Table 4: 10), quoted next to the base's count.

Power below 2 Hz: every pronounced epoch of each base (no sample: an 80-epoch draw moved the
BIOMEX value from 3.3% to 2.5% between two seeds), the periodogram of each channel over the
epoch's own length, power below 2 Hz over the total, averaged over epochs.

Usage (from software/nn; reads both bases, about a minute):
    python3 scripts/analysis/thesis_base_facts.py --biomex ~/biomex.sqlite \\
        --coretto ~/database.sqlite --out results/thesis/base_facts.csv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sqlite3
import sys

import numpy as np

PAPER_S01_O_PRONOUNCED = 10    # Pressel Coretto et al. (2017), Table 4
PROTOCOL_PAIRS = 140           # Moreno-Rodriguez et al. (2021), Table 1: 10x10 + 10x4 digits


def fail(msg: str) -> None:
    sys.exit(f"[base-facts] ERROR: {msg}")


def stamp(db: str, suffix: str) -> dict:
    path = db + suffix
    if not os.path.isfile(path):
        fail(f"{path} is missing: run the verifier that writes it first")
    with open(path) as f:
        lines = f.read().splitlines()
    if len(lines) != 2:
        fail(f"{path}: expected a sha256 line and a JSON line, found {len(lines)} lines "
             f"(a stamp from before 2026-11-13: run the verifier again)")
    h = hashlib.sha256()
    with open(db, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    if lines[0] != h.hexdigest():
        fail(f"{path} was written for another version of {db}: verify it again")
    facts = json.loads(lines[1])
    if facts.get("differences") != 0:
        fail(f"{path} records differences: {facts.get('differences')}")
    return facts


def below_2hz_share(x: np.ndarray, fs: int) -> float:
    """Share of one epoch's power (channels x samples) below 2 Hz."""
    p = np.abs(np.fft.rfft(x, axis=-1)) ** 2
    f = np.fft.rfftfreq(x.shape[-1], 1 / fs)
    return float(p[..., f < 2].sum() / p.sum())


def blob(b: bytes, n: int) -> np.ndarray:
    if len(b) == 4 * n:
        return np.frombuffer(b, np.float32).astype(np.float64)
    if len(b) == 8 * n:
        return np.frombuffer(b, np.float64)
    fail(f"EEG blob of {len(b)} bytes for {n} samples")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--biomex", required=True)
    ap.add_argument("--coretto", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    bx_db, co_db = os.path.expanduser(args.biomex), os.path.expanduser(args.coretto)
    rows: list[tuple[str, str, str]] = []

    def put(key: str, kind: str, value) -> None:
        if any(k == key for k, _, _ in rows):
            fail(f"duplicate key {key}")
        rows.append((key, kind, str(value)))

    # ── BIOMEX-DB ──────────────────────────────────────────────────────────────────────
    vf = stamp(bx_db, ".verified")
    con = sqlite3.connect(f"file:{bx_db}?mode=ro", uri=True)
    info = dict(con.execute("SELECT key, value FROM recording"))
    kept = [n for (n,) in con.execute("SELECT name FROM speaker ORDER BY id")]
    excluded = sorted(e.split(":", 1)[0] for e in info["excluded_volunteers"].split(" | ")
                      if ":" in e)
    if excluded != sorted(vf["excluded"]):
        fail(f"excluded volunteers: base {excluded}, verifier {vf['excluded']}")
    pairs = con.execute("SELECT count(*) FROM pair").fetchone()[0]
    if pairs != vf["pairs_equal"] or len(kept) != vf["volunteers"]:
        fail(f"base holds {len(kept)} volunteers / {pairs} pairs; the verifier passed "
             f"{vf['volunteers']} / {vf['pairs_equal']}")
    edge = [] if info["edge_dropped_pairs"] == "none" else info["edge_dropped_pairs"].split(" | ")
    if edge != vf["edge_dropped"]:
        fail(f"edge-dropped pairs: base {edge}, verifier {vf['edge_dropped']}")
    unlabelled = [p for p in info["excluded_pairs"].split("; ") if p]
    if PROTOCOL_PAIRS * len(kept) - pairs != len(edge) + len(unlabelled):
        fail(f"{len(kept)} x {PROTOCOL_PAIRS} - {pairs} pairs is not the {len(edge)} edge + "
             f"{len(unlabelled)} unlabelled pairs the base names")
    recorded = kept + excluded
    put("base.biomex.recorded", "count", len(recorded))
    put("base.biomex.recorded.f", "count", sum(v.startswith("F") for v in recorded))
    put("base.biomex.recorded.m", "count", sum(v.startswith("M") for v in recorded))
    put("base.biomex.kept", "count", len(kept))
    put("base.biomex.kept.f", "count", sum(v.startswith("F") for v in kept))
    put("base.biomex.kept.m", "count", sum(v.startswith("M") for v in kept))
    put("base.biomex.excluded", "count", len(excluded))
    put("base.biomex.excluded.list", "text", ", ".join(excluded))
    put("base.biomex.pairs", "count", pairs)
    put("base.biomex.pairs.protocol", "count", PROTOCOL_PAIRS)
    put("base.biomex.pairs.dropped", "count", PROTOCOL_PAIRS * len(kept) - pairs)
    put("base.biomex.pairs.edge", "count", len(edge))
    put("base.biomex.pairs.edge.list", "text", ", ".join(sorted({e.split()[0] for e in edge})))
    put("base.biomex.pairs.unlabelled", "count", len(unlabelled))
    put("base.biomex.voice.shifted", "count", int(info["voice_windows_shifted"]))
    put("base.biomex.voice.cut", "count", int(info["voice_words_cut"]))
    put("base.biomex.csv.missing", "count", len(vf["sessions_without_csv"]))
    for k in ("voice_native_rate", "eeg_native_rate", "voice_sample_rate", "eeg_sample_rate"):
        put(f"base.biomex.{k.replace('_', '.')}", "count", int(info[k]))
    n_bx, fs_bx = int(info["eeg_epoch_samples"]), int(info["eeg_sample_rate"])
    bx_low = [below_2hz_share(np.frombuffer(e, np.float32).astype(np.float64).reshape(-1, n_bx),
                              fs_bx) for (e,) in con.execute("SELECT eeg FROM pair")]
    if len(bx_low) != pairs:
        fail(f"read {len(bx_low)} BIOMEX epochs of {pairs}")
    con.close()

    # ── Coretto ────────────────────────────────────────────────────────────────────────
    mf = stamp(co_db, ".mirror-verified")
    con = sqlite3.connect(f"file:{co_db}?mode=ro", uri=True)
    subjects = con.execute("SELECT count(*) FROM subject").fetchone()[0]
    by_mod = dict(con.execute(
        "SELECT mo.name, count(*) FROM trial t JOIN modalities mo ON mo.id = t.modality_id "
        "GROUP BY mo.name"))
    if set(by_mod) != {"Imagined", "Pronounced"}:
        fail(f"modalities {by_mod}")
    if mf["trials_equal"] != sum(by_mod.values()) or mf["subjects"] != subjects:
        fail(f"the mirror stamp covers {mf['trials_equal']} trials of {mf['subjects']} "
             f"subjects; the base has {sum(by_mod.values())} of {subjects}")
    s01o = con.execute(
        "SELECT count(*) FROM trial t JOIN subject s ON s.id = t.subject_id "
        "JOIN modalities mo ON mo.id = t.modality_id JOIN stimulus st ON st.id = t.stimulus_id "
        "WHERE s.subject_name = 'S01' AND mo.name = 'Pronounced' AND st.name = 'O'").fetchone()[0]
    put("base.coretto.subjects", "count", subjects)
    put("base.coretto.trials.imagined", "count", by_mod["Imagined"])
    put("base.coretto.trials.pronounced", "count", by_mod["Pronounced"])
    put("base.coretto.trials", "count", sum(by_mod.values()))
    put("base.coretto.mirror.equal", "count", mf["trials_equal"])
    put("base.coretto.s01o.base", "count", s01o)
    put("base.coretto.s01o.paper", "count", PAPER_S01_O_PRONOUNCED)
    co_low = [below_2hz_share(np.array([blob(b, 4096) for b in r]), 1024) for r in con.execute(
        "SELECT e.F3, e.F4, e.C3, e.C4, e.P3, e.P4 FROM eeg_samples e "
        "JOIN trial t ON t.id = e.trial_id JOIN modalities mo ON mo.id = t.modality_id "
        "WHERE mo.name = 'Pronounced'")]
    if len(co_low) != by_mod["Pronounced"]:
        fail(f"read {len(co_low)} pronounced Coretto epochs of {by_mod['Pronounced']}")
    con.close()
    put("base.coretto.below2hz.pct", "num1", f"{100 * np.mean(co_low):.1f}")
    put("base.biomex.below2hz.pct", "num1", f"{100 * np.mean(bx_low):.1f}")
    put("base.biomex.vs.coretto.speakers", "num1", f"{len(kept) / subjects:.1f}")

    with open(os.path.expanduser(args.out), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "kind", "value"])
        w.writerows(rows)
    print(f"[base-facts] {len(rows)} facts -> {args.out}")
    for k, _, v in rows:
        print(f"  {k} = {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
