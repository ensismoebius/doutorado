#!/usr/bin/env python3
"""Tests for scripts/data/biomex_to_sqlite.py on a synthetic BIOMEX-DB volunteer.

The synthetic volunteer has the real layout (EDF at 128 Hz with COUNTER, INTERPOLATED, the
14 channels and MARKER; one WAV + .lab per series) and a signature in every digit: in the
EEG, a 10 Hz burst of amplitude 10 x code uV after the digit's marker; in the voice, a
200 Hz tone of amplitude code/20 at the word's onset. A pair is right only if both
signatures carry the same code, so these tests check the pairing itself, not just counts.

Run:  python3 scripts/testing/test_biomex_convert.py
"""
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
import wave
from unittest import mock

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CONVERT = os.path.join(os.path.dirname(HERE), "data", "biomex_to_sqlite.py")
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "data"))
import biomex_to_sqlite as bx  # noqa: E402

FS = 128
VFS = 16000
OUT_FS = bx.OUT_EEG_RATE      # the rates the converter writes (the Coretto base's)
OUT_VFS = bx.OUT_VOICE_RATE
DIGIT_GAP_S = 2.6     # EEG clock
WORD_GAP_S = 2.0      # voice clock (deliberately different, as in the real base)
WORD_ON_S = 1.1
WORD_LEN_S = 0.35
RNG = np.random.default_rng(3)


def series_codes(per_series):
    return [list(RNG.permutation(np.arange(1, 11))[:per_series]) for _ in range(10)]


def write_edf(path, signals, labels, phys, dig):
    ns, n = len(labels), signals.shape[1]
    n_rec = n // FS
    h = b"0       " + b" " * 80 + b" " * 80 + b"18.07.18" + b"12.29.06"
    h += f"{256 * (ns + 1):<8}".encode() + b" " * 44 + f"{n_rec:<8}".encode()
    h += f"{1:<8}".encode() + f"{ns:<4}".encode()
    h += b"".join(f"{lb:<16}".encode() for lb in labels) + b" " * 80 * ns + b"uV      " * ns
    h += b"".join(f"{p[0]:<8}".encode() for p in phys)
    h += b"".join(f"{p[1]:<8}".encode() for p in phys)
    h += b"".join(f"{d[0]:<8}".encode() for d in dig)
    h += b"".join(f"{d[1]:<8}".encode() for d in dig)
    h += b" " * 80 * ns + f"{FS:<8}".encode() * ns + b" " * 32 * ns
    gain = np.array([(p[1] - p[0]) / (d[1] - d[0]) for p, d in zip(phys, dig)])
    digital = np.round((signals - np.array([p[0] for p in phys])[:, None]) / gain[:, None]
                       + np.array([d[0] for d in dig])[:, None]).astype("<i2")
    body = digital[:, :n_rec * FS].reshape(ns, n_rec, FS).transpose(1, 0, 2).tobytes()
    with open(path, "wb") as f:
        f.write(h + body)


def make_volunteer(root, vid="F999", typo=True, break_series=None, bad_marker=False,
                   word_len_s=WORD_LEN_S):
    for sub in ("EEG", "Voice"):
        os.makedirs(os.path.join(root, vid, sub))
    truth = []
    for session, per in bx.SESSIONS.items():
        codes = series_codes(per)
        n = int((25 + 10 * (per * DIGIT_GAP_S + 5)) * FS)
        eeg = np.zeros((14, n)) + 4200.0  # the Emotiv DC offset the filter must remove
        marker = np.zeros(n)
        t = 22.0
        marker[int(1 * FS)] = 99
        marker[int(10 * FS)] = 89
        for k, series in enumerate(codes, 1):
            for j, c in enumerate(series):
                m = int((t + j * DIGIT_GAP_S) * FS)
                marker[m] = c
                tt = np.arange(int(2.0 * FS)) / FS
                eeg[:, m:m + len(tt)] += 10.0 * c * np.sin(2 * np.pi * 10 * tt)
                truth.append((session, k, j + 1, c))
            t += per * DIGIT_GAP_S
            marker[int(t * FS)] = 69
            t += 5
        if bad_marker and session == "01G10":
            marker[int(5 * FS)] = 42
        labels = ["COUNTER", "INTERPOLATED"] + list(bx.EEG_CHANNELS) + ["MARKER"]
        sig = np.vstack([np.arange(n) % 128, np.zeros(n), eeg, marker])
        phys = [(0, 16000)] * len(labels)
        dig = [(0, 31200)] * (len(labels) - 1) + [(0, 16000)]
        write_edf(os.path.join(root, vid, "EEG", f"{vid}-{session}-18.07.18.12.29.05.edf"),
                  sig, labels, phys, dig)
        for k, series in enumerate(codes, 1):
            vn = int((WORD_ON_S + per * WORD_GAP_S + 0.5) * VFS)
            v = np.zeros(vn)
            lab, prev = [], 0
            words = [bx.DIGIT_WORDS[c] for c in series]
            if break_series == (session, k):
                words = words[1:] + words[:1]
            for j, (c, w) in enumerate(zip(series, words)):
                on = int((WORD_ON_S + j * WORD_GAP_S) * VFS)
                end = on + int(word_len_s * VFS)
                tt = np.arange(end - on) / VFS
                v[on:end] = (c / 20) * np.sin(2 * np.pi * 200 * tt)
                spelled = "UN0" if (typo and w == "UNO") else w
                lab += [(prev, on - 1, "SILENCIO"), (on, end, spelled)]
                prev = end + 1
            lab.append((prev, vn - 1, "SILENCIO"))
            stem = os.path.join(root, vid, "Voice", f"{vid}_{session}_{k}")
            with wave.open(stem + ".wav", "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(VFS)
                w.writeframes((np.clip(v, -1, 1) * 32767).astype("<i2").tobytes())
            with open(stem + ".lab", "w") as f:
                for a, b, word in lab:
                    # 100 ns units; one line in scientific notation, as in the real labels
                    a_txt = f"{a / VFS * 1e7:.6e}" if word == "SILENCIO" else str(int(a / VFS * 1e7))
                    f.write(f"{a_txt} {int(b / VFS * 1e7)} {word}\n")
    return truth


def read(path):
    with open(path) as f:
        return f.read()


def write(path, text):
    with open(path, "w") as f:
        f.write(text)


def run(src, out):
    return subprocess.run([sys.executable, CONVERT, "--src", src, "--out", out],
                          capture_output=True, text=True)


class BiomexConvert(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.src = os.path.join(self.tmp.name, "src")
        self.out = os.path.join(self.tmp.name, "b.sqlite")

    def tearDown(self):
        self.tmp.cleanup()

    def test_pairs_carry_the_same_digit_in_eeg_and_voice(self):
        truth = make_volunteer(self.src)
        r = run(self.src, self.out)
        self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
        con = sqlite3.connect(self.out)
        rows = con.execute("SELECT session, series, position, stimulus_id, voice, eeg "
                           "FROM pair ORDER BY session DESC, series, position").fetchall()
        self.assertEqual(len(rows), 140)
        self.assertEqual(sorted((s, k, j, c) for s, k, j, c, _, _ in rows), sorted(truth))
        for s, k, j, c, voice, eeg in rows:
            v = np.frombuffer(voice, dtype=np.float32)
            e = np.frombuffer(eeg, dtype=np.float32).reshape(14, 2 * OUT_FS)
            self.assertEqual(v.size, OUT_VFS)  # 1 s, written at the Coretto rate
            # voice: the tone of this code fills the first WORD_LEN_S of the window
            rms_v = np.sqrt(np.mean(v[int(0.01 * OUT_VFS):int(0.3 * OUT_VFS)] ** 2))
            self.assertAlmostEqual(rms_v, (c / 20) / np.sqrt(2), delta=0.01)
            # EEG: the burst of this code, DC removed by the band-pass (0.5-1.5 s)
            mid = e[:, OUT_FS // 2:3 * OUT_FS // 2]
            self.assertAlmostEqual(float(np.sqrt(np.mean(mid ** 2))), 10 * c / np.sqrt(2),
                                   delta=0.1 * 10 * c)
            self.assertLess(abs(float(e.mean())), 1.0)
        info = dict(con.execute("SELECT key, value FROM recording"))
        con.close()
        self.assertEqual(info["source"], "biomex2021")
        self.assertEqual(info["eeg_sample_rate"], "1024")
        self.assertEqual(info["voice_sample_rate"], "44100")
        self.assertEqual(info["eeg_epoch_samples"], "2048")
        self.assertEqual(info["voice_window_samples"], "44100")
        self.assertEqual((info["voice_native_rate"], info["eeg_native_rate"]), ("16000", "128"))

    def test_the_upsampled_signals_hold_nothing_above_the_native_nyquist(self):
        # Upsampling must not invent content: the band a 16 kHz / 128 Hz recording cannot
        # hold (above 8 kHz / 64 Hz) stays empty at the Coretto rates.
        make_volunteer(self.src)
        r = run(self.src, self.out)
        self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
        con = sqlite3.connect(self.out)
        voice, eeg = con.execute("SELECT voice, eeg FROM pair LIMIT 1").fetchone()
        con.close()
        for x, fs, nyq in ((np.frombuffer(voice, np.float32), OUT_VFS, VFS / 2),
                           (np.frombuffer(eeg, np.float32).reshape(14, -1)[0], OUT_FS, FS / 2)):
            # Hann-tapered so the cut edges of the window do not leak across the spectrum;
            # -50 dB is far below any band the extractor reads (Coretto: 0.1% voice > 8 kHz).
            p = np.abs(np.fft.rfft(x * np.hanning(x.size))) ** 2
            f = np.fft.rfftfreq(x.size, 1 / fs)
            self.assertLess(p[f > 1.05 * nyq].sum() / p.sum(), 1e-5)

    def test_a_series_whose_words_differ_from_its_markers_is_refused(self):
        make_volunteer(self.src, break_series=("01G04", 3))
        r = run(self.src, self.out)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("01G04 series 3", r.stderr)
        self.assertFalse(os.path.exists(self.out))

    def test_an_unknown_marker_is_refused(self):
        make_volunteer(self.src, bad_marker=True)
        r = run(self.src, self.out)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("unknown MARKER code 42", r.stderr)

    def test_an_unknown_label_is_refused_and_the_known_typo_is_not(self):
        # Every 01G10 series holds all ten digits, so its .lab spells "UN0"; the pairing test
        # above converts that volunteer, so the typo map is exercised there.
        make_volunteer(self.src, typo=True)
        lab = os.path.join(self.src, "F999", "Voice", "F999_01G10_1.lab")
        text = read(lab)
        self.assertIn("UN0", text)
        write(lab, text.replace("SILENCIO", "SILENSIO", 1))
        r = run(self.src, self.out)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("unknown label 'SILENSIO'", r.stderr)

    def _move_words(self, series, word1, word2):
        """Relabel the first two words of a 01G10 series to (onset, end) in seconds. Only the
        .lab moves: the window is cut where the label says, which is what is under test."""
        lab = os.path.join(self.src, "F999", "Voice", f"F999_01G10_{series}.lab")
        lines = read(lab).splitlines()
        for i, (a, b) in ((1, word1), (3, word2)):
            lines[i] = f"{int(a * 1e7)} {int(b * 1e7)} {lines[i].split()[2]}"
        write(lab, "\n".join(lines) + "\n")

    def test_a_voice_window_that_reaches_the_next_word_moves_earlier(self):
        make_volunteer(self.src)
        # Word 2 relabelled to start 0.5 s after word 1 (1.1 s): word 1's 1 s window must end
        # at 1.6 s, so it starts at 0.6 s and holds word 1's tone from 0.5 s on.
        self._move_words(2, (WORD_ON_S, WORD_ON_S + WORD_LEN_S), (1.6, 1.9))
        r = run(self.src, self.out)
        self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
        con = sqlite3.connect(self.out)
        rows = con.execute("SELECT stimulus_id, voice, voice_shift_s, voice_cut FROM pair "
                           "WHERE session='01G10' AND series=2 ORDER BY position").fetchall()
        info = dict(con.execute("SELECT key, value FROM recording"))
        con.close()
        code, voice, shift, cut = rows[0]
        self.assertAlmostEqual(shift, 0.5, places=3)
        self.assertEqual(cut, 0)
        v = np.frombuffer(voice, dtype=np.float32)
        self.assertLess(float(np.abs(v[:int(0.45 * OUT_VFS)]).max()), 1e-3)  # silence before
        rms = np.sqrt(np.mean(v[int(0.51 * OUT_VFS):int(0.8 * OUT_VFS)] ** 2))
        self.assertAlmostEqual(rms, (code / 20) / np.sqrt(2), delta=0.01)  # word 1's own tone
        self.assertEqual([r[2] for r in rows[1:]], [0.0] * 9)
        self.assertEqual(info["voice_windows_shifted"], "1")

    def test_a_voice_window_with_no_room_is_refused(self):
        make_volunteer(self.src)
        # Word 1 at 0.2 s, word 2 at 0.7 s: a 1 s window ending at 0.7 s would start before
        # the WAV does.
        self._move_words(2, (0.2, 0.4), (0.7, 0.9))
        r = run(self.src, self.out)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("01G10 series 2 digit 1: no 1.0 s voice window fits", r.stderr)
        self.assertFalse(os.path.exists(self.out))

    # ── errata: applied only to a file exactly as described, refused otherwise ──────────
    def _convert_in_process(self):
        return bx.convert_volunteer(os.path.join(self.src, "F999"), "F999", bx.band_pass())

    def test_a_missing_lab_word_is_inserted_only_into_a_free_gap(self):
        truth = make_volunteer(self.src)
        lab = os.path.join(self.src, "F999", "Voice", "F999_01G10_1.lab")
        lines = read(lab).splitlines()
        a, b, word = lines[5].split()          # the third word of series 1
        write(lab, "\n".join(lines[:5] + lines[6:]) + "\n")
        with self.assertRaises(bx.ConversionError) as e:
            self._convert_in_process()
        self.assertIn("01G10 series 1", str(e.exception))
        word = bx.LABEL_TYPOS.get(word, word)
        with mock.patch.dict(bx.LAB_MISSING_WORDS, {("F999", "01G10", 1): (int(a), int(b), word)}):
            pairs = self._convert_in_process()
        self.assertEqual(sorted((p["session"], p["series"], p["position"], p["code"])
                                for p in pairs), sorted(truth))
        # An erratum that overlaps a line the file has is not the file it describes.
        prev_word_end = int(lines[3].split()[1])
        with mock.patch.dict(bx.LAB_MISSING_WORDS,
                             {("F999", "01G10", 1): (prev_word_end - 10, int(b), word)}):
            with self.assertRaises(bx.ConversionError) as e:
                self._convert_in_process()
        self.assertIn("overlaps", str(e.exception))

    def test_a_misnamed_edf_is_used_only_through_its_erratum(self):
        make_volunteer(self.src)
        eeg_dir = os.path.join(self.src, "F999", "EEG")
        old = [f for f in os.listdir(eeg_dir) if f.startswith("F999-01G04-")][0]
        os.rename(os.path.join(eeg_dir, old), os.path.join(eeg_dir, "F998" + old[4:]))
        with self.assertRaises(bx.ConversionError) as e:
            self._convert_in_process()
        self.assertIn("expected one EDF, found 0", str(e.exception))
        with mock.patch.dict(bx.EDF_FILE_ERRATA, {("F999", "01G04"): "F998" + old[4:]}):
            self.assertEqual(len(self._convert_in_process()), 140)

    def test_an_aborted_edf_is_set_aside_only_if_it_holds_no_digit(self):
        make_volunteer(self.src)
        eeg_dir = os.path.join(self.src, "F999", "EEG")
        n = 6 * FS                                   # a few seconds, start marker only
        labels = ["COUNTER", "INTERPOLATED"] + list(bx.EEG_CHANNELS) + ["MARKER"]
        sig = np.zeros((len(labels), n))
        sig[-1, FS] = 99
        aborted = "F999-01G10-18.07.18.12.00.00.edf"
        write_edf(os.path.join(eeg_dir, aborted), sig, labels, [(0, 16000)] * len(labels),
                  [(0, 31200)] * (len(labels) - 1) + [(0, 16000)])
        with self.assertRaises(bx.ConversionError) as e:
            self._convert_in_process()
        self.assertIn("expected one EDF, found 2", str(e.exception))
        with mock.patch.dict(bx.EDF_ABORTED, {("F999", "01G10"): aborted}):
            self.assertEqual(len(self._convert_in_process()), 140)
        # Named as aborted but holding digits: refused.
        real = [f for f in os.listdir(eeg_dir) if f.startswith("F999-01G10-18.07.18.12.29")][0]
        with mock.patch.dict(bx.EDF_ABORTED, {("F999", "01G10"): real}):
            with self.assertRaises(bx.ConversionError) as e:
                self._convert_in_process()
        self.assertIn("holds digit markers", str(e.exception))

    def test_an_existing_output_is_never_overwritten(self):
        make_volunteer(self.src)
        write(self.out, "")
        r = run(self.src, self.out)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("exists", r.stderr)


if __name__ == "__main__":
    unittest.main()
