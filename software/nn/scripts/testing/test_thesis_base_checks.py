#!/usr/bin/env python3
"""Tests for the checks of the two thesis bases:

  scripts/data/coretto_verify_mirror.py     the Coretto SQLite against the Zenodo EEG copy
  scripts/analysis/thesis_base_facts.py      the facts chapter 8 quotes, read from the bases
                                             and the verifiers' stamps

Both are trusted by the thesis text, so both must fail loudly: a verifier that cannot fail
proves nothing, and a fact read from a base nobody verified is not a fact. The fixtures are
small bases in the real schemas (2 Coretto trials, a few BIOMEX-DB pairs).

Run:  python3 scripts/testing/test_thesis_base_checks.py
"""
import csv
import hashlib
import io
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unittest
import zipfile

import numpy as np
from scipy.io import savemat

HERE = os.path.dirname(os.path.abspath(__file__))
NN = os.path.dirname(os.path.dirname(HERE))
MIRROR = os.path.join(NN, "scripts", "data", "coretto_verify_mirror.py")
FACTS = os.path.join(NN, "scripts", "analysis", "thesis_base_facts.py")
N = 4096
CHANNELS = ("F3", "F4", "C3", "C4", "P3", "P4")
STIMULI = ("A", "E", "I", "O", "U", "Arriba", "Abajo", "Adelante", "Atras", "Derecha",
           "Izquierda")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def run(script, *args):
    return subprocess.run([sys.executable, script, *args], capture_output=True, text=True)


def coretto_db(path, trials):
    """trials: [(subject, original_row, modality 1|2, stimulus 1..11, blink, eeg (6, N))]."""
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE subject (id INTEGER PRIMARY KEY, subject_name TEXT);
        CREATE TABLE modalities (id INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE stimulus (id INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE trial (id INTEGER PRIMARY KEY, subject_id INT, modality_id INT,
                            stimulus_id INT, original_row INT);
        CREATE TABLE eeg_samples (trial_id INT, blink INT, F3 BLOB, F4 BLOB, C3 BLOB, C4 BLOB,
                                  P3 BLOB, P4 BLOB);""")
    con.executemany("INSERT INTO modalities VALUES (?, ?)", [(1, "Imagined"), (2, "Pronounced")])
    con.executemany("INSERT INTO stimulus VALUES (?, ?)", list(enumerate(STIMULI, 1)))
    subjects = sorted({t[0] for t in trials})
    con.executemany("INSERT INTO subject VALUES (?, ?)", list(enumerate(subjects, 1)))
    for i, (subj, row, mod, stim, blink, eeg) in enumerate(trials, 1):
        con.execute("INSERT INTO trial VALUES (?, ?, ?, ?, ?)",
                    (i, subjects.index(subj) + 1, mod, stim, row))
        con.execute("INSERT INTO eeg_samples VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (i, blink, *[eeg[c].astype(np.float32).tobytes() for c in range(6)]))
    con.commit()
    con.close()


def mirror_zip(folder, subject, rows):
    """rows: [(modality, stimulus, blink, eeg (6, N) float64)] in original_row order."""
    m = np.array([np.concatenate([eeg.reshape(-1), [mod, stim, blink]])
                  for mod, stim, blink, eeg in rows])
    buf = io.BytesIO()
    savemat(buf, {"EEG": m})
    with zipfile.ZipFile(os.path.join(folder, f"{subject}.zip"), "w") as z:
        z.writestr(f"sub-{subject[1:]}_eeg.mat", buf.getvalue())


class CorettoMirror(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db = os.path.join(self.tmp, "database.sqlite")
        self.mirror = os.path.join(self.tmp, "mirror")
        os.makedirs(self.mirror)
        rng = np.random.default_rng(1)
        # The .mat holds float64; the importer stored float32. Values chosen so the two
        # differ (as on the real base): the check must compare at the stored precision.
        self.eeg = [rng.normal(0, 20, (6, N)) for _ in range(3)]
        self.rows = [(2, 4, 0, self.eeg[0]), (1, 7, 1, self.eeg[1]), (2, 1, 0, self.eeg[2])]
        self.write()

    def write(self, db_rows=None):
        rows = self.rows if db_rows is None else db_rows
        if os.path.exists(self.db):
            os.remove(self.db)
        coretto_db(self.db, [("S01", i, mod, stim, blink, eeg)
                             for i, (mod, stim, blink, eeg) in enumerate(rows)])
        mirror_zip(self.mirror, "S01", self.rows)

    def check(self):
        return run(MIRROR, "--db", self.db, "--mirror", self.mirror)

    def test_equal_at_the_stored_precision_passes_and_stamps(self):
        self.assertFalse(np.array_equal(self.eeg[0].astype(np.float32).astype(np.float64),
                                        self.eeg[0]))   # the fixture really differs in float64
        r = self.check()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("3 of 3 trials equal", r.stdout)
        with open(self.db + ".mirror-verified") as f:
            lines = f.read().splitlines()
        self.assertEqual(lines[0], sha256(self.db))
        facts = json.loads(lines[1])
        self.assertEqual((facts["trials_equal"], facts["differences"], facts["per_subject"]),
                         (3, 0, {"S01": 3}))

    def test_a_value_beyond_float32_rounding_is_a_difference(self):
        changed = [r[:3] + (r[3].copy(),) for r in self.rows]
        changed[1][3][2, 100] += 1e-3
        self.write(changed)
        r = self.check()
        self.assertEqual(r.returncode, 1)
        self.assertIn("row 1: C3 differs", r.stdout)
        self.assertFalse(os.path.exists(self.db + ".mirror-verified"))

    def test_a_label_difference_is_reported(self):
        changed = list(self.rows)
        changed[2] = (2, 5) + changed[2][2:]          # stimulus /u/ instead of /a/
        self.write(changed)
        r = self.check()
        self.assertEqual(r.returncode, 1)
        self.assertIn("row 2: SQLite Pronounced/U", r.stdout)

    def test_a_row_missing_from_the_base_is_reported(self):
        self.write(self.rows[:2])
        r = self.check()
        self.assertEqual(r.returncode, 1)
        self.assertIn(".mat row 2 is not in the SQLite", r.stdout)

    def test_an_old_stamp_is_removed_when_the_check_fails(self):
        self.assertEqual(self.check().returncode, 0)
        self.write(self.rows[:2])
        self.check()
        self.assertFalse(os.path.exists(self.db + ".mirror-verified"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


EPOCH = 256          # BIOMEX-DB epoch length in this fixture (the real base: 2048)


def biomex_db(path, speakers, pairs, info):
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE recording (key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE speaker (id INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE pair (speaker_id INT, eeg BLOB);""")
    con.executemany("INSERT INTO recording VALUES (?, ?)", info.items())
    con.executemany("INSERT INTO speaker VALUES (?, ?)", list(enumerate(speakers, 1)))
    con.executemany("INSERT INTO pair VALUES (?, ?)", [(1, e.astype(np.float32).tobytes())
                                                       for e in pairs])
    con.commit()
    con.close()


def stamp(db, facts):
    with open(db + ".verified", "w") as f:
        f.write(sha256(db) + "\n" + json.dumps(facts) + "\n")


class BaseFacts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.bx = os.path.join(self.tmp, "biomex.sqlite")
        self.co = os.path.join(self.tmp, "database.sqlite")
        self.out = os.path.join(self.tmp, "facts.csv")
        t = np.arange(EPOCH) / 1024
        # One kept volunteer of the protocol's 140 pairs, 138 stored: one at the filter's
        # edge and one unlabelled digit. EEG: a 12 Hz tone, 3 whole cycles per 0.25 s epoch (no
        # leakage), so almost nothing below 2 Hz.
        self.pairs = [np.tile(np.sin(2 * np.pi * 12 * t), (14, 1)) for _ in range(138)]
        self.info = {"excluded_volunteers": "F007: 01G04: 0 complete recordings | "
                                            "M001: 01G10: 2 complete recordings",
                     "edge_dropped_pairs": "F001 01G04 series 10 digit 4",
                     "excluded_pairs": "F001 01G10 series 8 digit 7: no .lab line",
                     "voice_windows_shifted": "3", "voice_words_cut": "2",
                     "voice_native_rate": "16000", "eeg_native_rate": "128",
                     "voice_sample_rate": "44100", "eeg_sample_rate": "1024",
                     "eeg_epoch_samples": str(EPOCH)}
        self.bx_facts = {"volunteers": 1, "excluded": ["F007", "M001"], "pairs_equal": 138,
                         "differences": 0, "sessions_without_csv": ["F001 01G10"],
                         "edge_dropped": ["F001 01G04 series 10 digit 4"]}
        biomex_db(self.bx, ["F001"], self.pairs, self.info)
        stamp(self.bx, self.bx_facts)
        # Coretto: 2 subjects, S01 with 2 pronounced /o/ trials; a slow drift in the EEG.
        slow = np.tile(np.sin(2 * np.pi * 0.5 * np.arange(N) / 1024), (6, 1))
        coretto_db(self.co, [("S01", 0, 2, 4, 0, slow), ("S01", 1, 2, 4, 0, slow),
                             ("S01", 2, 1, 4, 0, slow), ("S02", 0, 2, 1, 0, slow)])
        with open(self.co + ".mirror-verified", "w") as f:
            f.write(sha256(self.co) + "\n" + json.dumps(
                {"trials_equal": 4, "differences": 0, "subjects": 2}) + "\n")

    def facts(self):
        return run(FACTS, "--biomex", self.bx, "--coretto", self.co, "--out", self.out)

    def read(self):
        with open(self.out, newline="") as f:
            return {r["key"]: r["value"] for r in csv.DictReader(f)}

    def test_the_facts_are_read_from_the_bases(self):
        r = self.facts()
        self.assertEqual(r.returncode, 0, r.stderr)
        got = self.read()
        want = {"base.biomex.recorded": "3", "base.biomex.kept": "1", "base.biomex.kept.f": "1",
                "base.biomex.excluded.list": "F007, M001", "base.biomex.pairs": "138",
                "base.biomex.pairs.dropped": "2", "base.biomex.pairs.edge": "1",
                "base.biomex.pairs.edge.list": "F001", "base.biomex.pairs.unlabelled": "1",
                "base.biomex.csv.missing": "1", "base.biomex.voice.shifted": "3",
                "base.coretto.subjects": "2", "base.coretto.trials": "4",
                "base.coretto.trials.imagined": "1", "base.coretto.s01o.base": "2",
                "base.coretto.mirror.equal": "4"}
        self.assertEqual({k: got[k] for k in want}, want)
        # 12 Hz tone vs 0.5 Hz drift: the shares say which is which.
        self.assertLess(float(got["base.biomex.below2hz.pct"]), 5.0)
        self.assertGreater(float(got["base.coretto.below2hz.pct"]), 95.0)

    def test_a_base_changed_after_its_check_is_refused(self):
        con = sqlite3.connect(self.bx)
        con.execute("UPDATE recording SET value = '4' WHERE key = 'voice_windows_shifted'")
        con.commit()
        con.close()
        r = self.facts()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("written for another version", r.stderr)
        self.assertFalse(os.path.exists(self.out))

    def test_a_stamp_without_its_counts_is_refused(self):
        with open(self.co + ".mirror-verified", "w") as f:
            f.write(sha256(self.co) + "\n")
        r = self.facts()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("expected a sha256 line and a JSON line", r.stderr)

    def test_counts_the_verifier_did_not_pass_are_refused(self):
        stamp(self.bx, dict(self.bx_facts, pairs_equal=137))
        r = self.facts()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("the verifier passed", r.stderr)

    def test_dropped_pairs_the_base_does_not_name_are_refused(self):
        con = sqlite3.connect(self.bx)
        con.execute("UPDATE recording SET value = 'none' WHERE key = 'edge_dropped_pairs'")
        con.commit()
        con.close()
        stamp(self.bx, dict(self.bx_facts, edge_dropped=[]))
        r = self.facts()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("is not the 0 edge + 1 unlabelled pairs", r.stderr)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
