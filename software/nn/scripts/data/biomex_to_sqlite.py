#!/usr/bin/env python3
"""Convert BIOMEX-DB (EEG + voice, pronounced digits) into the thesis paired-trial SQLite.

WHY. The thesis replicates its pronounced-speech results on a second, independent base:
BIOMEX-DB (Moreno-Rodriguez et al., IEEE Access 9:111267, 2021; data 10.17632/s7chktmb6x.1,
CC BY 4.0). Its files are per volunteer and per session; the thesis reads one sample per
spoken digit with the voice of that digit and the EEG recorded while it was spoken. This
script builds exactly those pairs, once, and refuses anything it cannot pair for certain.

THE BASE, AS MEASURED (volunteer F001, 2026-11-13).
  <ID>/EEG/<ID>-<session>-<timestamp>.edf   Emotiv Epoc, 36 signals at 128 Hz: COUNTER,
                                            INTERPOLATED, 14 EEG channels, contact quality,
                                            gyroscope, MARKER. EEG physical = digital x
                                            16000/31200 uV (the CSV export, to 1e-4 uV).
  <ID>/Voice/<ID>_<session>_<k>.wav + .lab  one WAV per digit series (16 kHz, mono, 16 bit)
                                            and its word labels in 100 ns units.
  session 01G10: 10 series of 10 digits; session 01G04: 10 series of 4 digits.

  MARKER codes: 99 start, 89 eyes open, 69 rest between series, 79 end, and one code per
  digit shown: 1..9 for the digit, 10 for zero.

HOW A PAIR IS MADE.

    EEG session (one clock)                    voice series k (another clock)
    ... 69 | 7   9   10  2  ...  3 | 69 ...    SILENCIO SIETE SILENCIO NUEVE ... TRES
           |<- series k: run of    |           <- the k-th WAV, words in order
           |   digit markers       |

  The k-th run of digit markers is series k. Its digits must equal the words of the k-th
  WAV, in order; the pair is (marker j, word j). The two clocks do NOT agree (in F001 a
  series lasts 26.3 s in the EEG and 20.0 s in the WAV), so each signal is cut on its own
  clock: the EEG from the digit's marker, the voice from the word's onset in the .lab.

  EEG   EEG_EPOCH_S after the marker, the 14 channels, after a zero-phase FIR band-pass of
        2-40 Hz on the whole session. Why the filter: the Emotiv EEG carries a DC offset of
        ~4000 uV that differs per electrode and per headset fit; unfiltered, energy and
        spectrum features would read the fit of the headset, not the person, and identify
        speakers by it. The Coretto base used for the rest of the thesis is distributed
        already filtered 2-40 Hz by its authors, so the same band keeps the two comparable.
  voice VOICE_WINDOW_S from the word's onset (the word plus the silence after it; the
        Coretto voice trials are fixed 4 s windows that also hold silence). Some speakers
        say the first digits of a series faster than the window (F004: UNO then DOS 0.526 s
        later), and some WAVs end less than a window after the last word (F006: 0.996 s).
        Such a window is moved earlier, by the least amount that makes it end at the next
        word's onset (or the WAV's end); the silence before the word fills its start. The
        move is stored per pair (voice_shift_s). If the moved window would reach back into
        the previous word (or before the WAV's start), the volunteer is refused.
        Shrinking the window for everyone instead would cut the longer words (p99 0.74 s
        over F001-F006) to fit one fast speaker; dropping the pairs would be a silent
        selection.

Nothing is guessed: an unknown marker or label, a series whose digits differ from its
words, a missing or duplicated file, another sample rate or channel set, a window that does
not fit, all raise and name the volunteer, session and series.

Usage (from software/nn):
    python3 scripts/data/biomex_to_sqlite.py --src <dir holding F001/, M001/, ...> \\
        --out ~/biomex.sqlite
"""
import argparse
import glob
import os
import re
import sqlite3
import sys
import wave

import numpy as np
from scipy.signal import filtfilt, firwin

FORMAT = "thesis-paired-v1"
SOURCE = "biomex2021"
EEG_RATE = 128
EEG_CHANNELS = ("AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4",
                "F8", "AF4")
EEG_EPOCH_S = 2.0           # shortest gap between two digit markers in F001: 2.42 s
VOICE_RATE = 16000
VOICE_WINDOW_S = 1.0        # F001-F006: word p99 0.74 s; 7 of 840 onset gaps/tails < 1 s
BAND_HZ = (2.0, 40.0)
FIR_TAPS = 423              # Hamming, ~1 Hz transition at 128 Hz (3.3 x 128 / 1)
SESSIONS = {"01G10": 10, "01G04": 4}  # digits per series
SERIES_PER_SESSION = 10

# marker code -> digit word (the .lab spelling); 10 is zero.
DIGIT_WORDS = {1: "UNO", 2: "DOS", 3: "TRES", 4: "CUATRO", 5: "CINCO", 6: "SEIS",
               7: "SIETE", 8: "OCHO", 9: "NUEVE", 10: "CERO"}
NON_DIGIT_MARKERS = {99: "start", 89: "eyes open", 69: "rest", 79: "end"}
# .lab spellings that are typing errors in the distributed labels, mapped explicitly.
# Anything else that is not a digit word or SILENCIO is refused.
LABEL_TYPOS = {"UN0": "UNO"}
SILENCE = "SILENCIO"
VOLUNTEER = re.compile(r"^[FM]\d{3}$")


class ConversionError(Exception):
    pass


# ── EDF ────────────────────────────────────────────────────────────────────────────────
def read_edf(path):
    """(labels, physical signals (n_signals, n_samples) float64, rates, start) of an EDF."""
    with open(path, "rb") as f:
        b = f.read()

    def field(off, n):
        return b[off:off + n].decode("ascii").strip()

    hdr = int(field(184, 8))
    n_rec = int(field(236, 8))
    rec_s = float(field(244, 8))
    ns = int(field(252, 4))
    start = field(168, 8) + " " + field(176, 8)
    o = 256
    labels = [field(o + i * 16, 16) for i in range(ns)]
    o += ns * (16 + 80 + 8)  # labels, transducer, physical dimension
    pmin = [float(field(o + i * 8, 8)) for i in range(ns)]
    o += ns * 8
    pmax = [float(field(o + i * 8, 8)) for i in range(ns)]
    o += ns * 8
    dmin = [int(field(o + i * 8, 8)) for i in range(ns)]
    o += ns * 8
    dmax = [int(field(o + i * 8, 8)) for i in range(ns)]
    o += ns * 8 + ns * 80  # prefiltering
    spr = [int(field(o + i * 8, 8)) for i in range(ns)]
    if len(set(spr)) != 1:
        raise ConversionError(f"{path}: signals with different sample counts per record {spr}")
    n = spr[0]
    need = hdr + n_rec * ns * n * 2
    if len(b) != need:
        raise ConversionError(f"{path}: {len(b)} bytes, the header implies {need}")
    raw = np.frombuffer(b[hdr:need], dtype="<i2").reshape(n_rec, ns, n)
    raw = raw.transpose(1, 0, 2).reshape(ns, n_rec * n).astype(np.float64)
    gain = (np.array(pmax) - np.array(pmin)) / (np.array(dmax) - np.array(dmin))
    phys = (raw - np.array(dmin)[:, None]) * gain[:, None] + np.array(pmin)[:, None]
    return labels, phys, n / rec_s, start, raw


def digit_runs(marker, where):
    """The series of the session: runs of digit markers, as lists of (sample, code)."""
    codes = marker.astype(np.int64)
    onsets = [i for i in range(len(codes)) if codes[i] != 0 and (i == 0 or codes[i - 1] == 0)]
    runs, run = [], []
    for i in onsets:
        c = int(codes[i])
        if c in DIGIT_WORDS:
            run.append((i, c))
        elif c in NON_DIGIT_MARKERS:
            if run:
                runs.append(run)
                run = []
        else:
            raise ConversionError(f"{where}: unknown MARKER code {c} at sample {i}")
    if run:
        runs.append(run)
    return runs


# ── voice ──────────────────────────────────────────────────────────────────────────────
def read_wav(path):
    with wave.open(path) as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (VOICE_RATE, 1, 2):
            raise ConversionError(f"{path}: {w.getframerate()} Hz, {w.getnchannels()} "
                                  f"channel(s), {8 * w.getsampwidth()} bit; expected "
                                  f"{VOICE_RATE} Hz mono 16 bit")
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    return x.astype(np.float32) / 32768.0


def read_lab(path):
    """The spoken words of a series, in order, as (onset sample, end sample, word)."""
    words = []
    with open(path, encoding="ascii") as f:
        lines = f.read().splitlines()
    for n, line in enumerate(lines, 1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 3:
            raise ConversionError(f"{path}:{n}: expected '<start> <end> <label>', got {line!r}")
        a, b, label = float(parts[0]), float(parts[1]), parts[2]
        label = LABEL_TYPOS.get(label, label)
        if label == SILENCE:
            continue
        if label not in DIGIT_WORDS.values():
            raise ConversionError(f"{path}:{n}: unknown label {parts[2]!r}")
        # 100 ns units -> samples
        words.append((int(round(a * 1e-7 * VOICE_RATE)), int(round(b * 1e-7 * VOICE_RATE)), label))
    return words


# ── one volunteer ──────────────────────────────────────────────────────────────────────
def band_pass():
    return firwin(FIR_TAPS, BAND_HZ, pass_zero=False, fs=EEG_RATE, window="hamming")


def convert_volunteer(folder, vid, taps):
    """Every (session, series, position, digit, voice, eeg, interpolated) of a volunteer."""
    epoch = int(round(EEG_EPOCH_S * EEG_RATE))
    win = int(round(VOICE_WINDOW_S * VOICE_RATE))
    pairs = []
    for session, per_series in SESSIONS.items():
        where = f"{vid} {session}"
        edfs = glob.glob(os.path.join(folder, "EEG", f"{vid}-{session}-*.edf"))
        if len(edfs) != 1:
            raise ConversionError(f"{where}: expected one EDF, found {len(edfs)}")
        labels, phys, rate, start, raw = read_edf(edfs[0])
        if rate != EEG_RATE:
            raise ConversionError(f"{where}: EEG at {rate} Hz, expected {EEG_RATE}")
        missing = [c for c in EEG_CHANNELS + ("MARKER", "INTERPOLATED") if c not in labels]
        if missing:
            raise ConversionError(f"{where}: EDF lacks {missing}")
        eeg = np.stack([phys[labels.index(c)] for c in EEG_CHANNELS])
        eeg = filtfilt(taps, [1.0], eeg, axis=-1)
        interp = raw[labels.index("INTERPOLATED")] != 0
        runs = digit_runs(raw[labels.index("MARKER")], where)
        if len(runs) != SERIES_PER_SESSION or any(len(r) != per_series for r in runs):
            raise ConversionError(f"{where}: digit runs of lengths {[len(r) for r in runs]}; "
                                  f"expected {SERIES_PER_SESSION} of {per_series}")
        for k, run in enumerate(runs, 1):
            stem = os.path.join(folder, "Voice", f"{vid}_{session}_{k}")
            if not (os.path.isfile(stem + ".wav") and os.path.isfile(stem + ".lab")):
                raise ConversionError(f"{where} series {k}: {stem}.wav/.lab missing")
            voice = read_wav(stem + ".wav")
            words = read_lab(stem + ".lab")
            shown = [DIGIT_WORDS[c] for _, c in run]
            said = [w for _, _, w in words]
            if shown != said:
                raise ConversionError(f"{where} series {k}: EEG markers {shown}, "
                                      f".lab words {said}")
            prev_end = 0
            for j, ((m, code), (on, end, word)) in enumerate(zip(run, words), 1):
                if j < len(run) and run[j][0] - m < epoch:
                    raise ConversionError(f"{where} series {k} digit {j}: next marker after "
                                          f"{(run[j][0] - m) / EEG_RATE:.2f} s < {EEG_EPOCH_S} s")
                if m + epoch > eeg.shape[1]:
                    raise ConversionError(f"{where} series {k} digit {j}: EEG ends")
                nxt = words[j][0] if j < len(words) else len(voice)
                begin = min(on, nxt - win)
                if begin < prev_end:
                    raise ConversionError(
                        f"{where} series {k} digit {j}: no {VOICE_WINDOW_S} s voice window fits "
                        f"between {'the previous word' if j > 1 else 'the start'} "
                        f"({prev_end / VOICE_RATE:.3f} s) and "
                        f"{'the next word' if j < len(words) else 'the end'} "
                        f"({nxt / VOICE_RATE:.3f} s)")
                prev_end = end
                pairs.append({"session": session, "series": k, "position": j, "code": code,
                              "voice": voice[begin:begin + win].astype(np.float32),
                              "eeg": eeg[:, m:m + epoch].astype(np.float32),
                              "interpolated": int(interp[m:m + epoch].sum()),
                              "voice_onset_s": on / VOICE_RATE,
                              "voice_shift_s": (on - begin) / VOICE_RATE,
                              "voice_cut": int(end > begin + win),
                              "eeg_onset_s": m / EEG_RATE, "eeg_start": start})
    return pairs


# ── output ─────────────────────────────────────────────────────────────────────────────
SCHEMA = """
CREATE TABLE recording (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE speaker (id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL);
CREATE TABLE stimulus (id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL);
CREATE TABLE pair (
    id INTEGER PRIMARY KEY,
    speaker_id INTEGER NOT NULL REFERENCES speaker(id),
    stimulus_id INTEGER NOT NULL REFERENCES stimulus(id),
    session TEXT NOT NULL, series INTEGER NOT NULL, position INTEGER NOT NULL,
    voice BLOB NOT NULL,            -- float32, voice_window_samples values
    eeg BLOB NOT NULL,              -- float32, channel-major: eeg_channels x eeg_epoch_samples
    eeg_interpolated INTEGER NOT NULL,  -- samples the headset interpolated (kept, counted)
    voice_onset_s REAL NOT NULL,    -- the word's onset in its series WAV (.lab)
    voice_shift_s REAL NOT NULL,    -- how far before the onset the window starts (0 = at it)
    voice_cut INTEGER NOT NULL,     -- 1 if the labelled word runs past the window's end
    eeg_onset_s REAL NOT NULL,
    UNIQUE (speaker_id, session, series, position));
"""


def recording_info():
    return {
        "format": FORMAT, "source": SOURCE,
        "voice_sample_rate": str(VOICE_RATE), "voice_window_samples":
            str(int(round(VOICE_WINDOW_S * VOICE_RATE))),
        "voice_window": f"{VOICE_WINDOW_S:g} s from the word's onset; moved earlier to end at "
                        f"the next word's onset or the WAV's end when it would reach them "
                        f"(pair.voice_shift_s)",
        "eeg_sample_rate": str(EEG_RATE), "eeg_channels": ",".join(EEG_CHANNELS),
        "eeg_epoch_samples": str(int(round(EEG_EPOCH_S * EEG_RATE))),
        "eeg_filter": f"FIR band-pass {BAND_HZ[0]:g}-{BAND_HZ[1]:g} Hz, {FIR_TAPS} taps, "
                      f"Hamming, zero-phase (filtfilt), whole session",
        "citation": "Moreno-Rodriguez et al., IEEE Access 9:111267 (2021); "
                    "data 10.17632/s7chktmb6x.1, CC BY 4.0",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", required=True, help="directory holding one folder per volunteer")
    ap.add_argument("--out", required=True, help="the SQLite file to write (must not exist)")
    args = ap.parse_args()
    src = os.path.expanduser(args.src)
    out = os.path.expanduser(args.out)
    if os.path.exists(out):
        sys.exit(f"{out} exists; remove it first (this script never overwrites a base)")
    volunteers = sorted(d for d in os.listdir(src) if VOLUNTEER.match(d)
                        and os.path.isdir(os.path.join(src, d)))
    if not volunteers:
        sys.exit(f"no volunteer folders (F001, M001, ...) in {src}")
    taps = band_pass()
    tmp = out + ".partial"
    if os.path.exists(tmp):
        os.remove(tmp)
    con = sqlite3.connect(tmp)
    con.executescript(SCHEMA)
    con.executemany("INSERT INTO recording VALUES (?, ?)", recording_info().items())
    for code, word in sorted(DIGIT_WORDS.items()):
        con.execute("INSERT INTO stimulus VALUES (?, ?)", (code, word.lower()))
    total = shifted = cut = 0
    for sid, vid in enumerate(volunteers, 1):
        try:
            pairs = convert_volunteer(os.path.join(src, vid), vid, taps)
        except ConversionError as e:
            con.close()
            os.remove(tmp)
            sys.exit(f"[biomex] refused: {e}")
        con.execute("INSERT INTO speaker VALUES (?, ?)", (sid, vid))
        con.executemany(
            "INSERT INTO pair (speaker_id, stimulus_id, session, series, position, voice, eeg, "
            "eeg_interpolated, voice_onset_s, voice_shift_s, voice_cut, eeg_onset_s) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            [(sid, p["code"], p["session"], p["series"], p["position"], p["voice"].tobytes(),
              p["eeg"].tobytes(), p["interpolated"], p["voice_onset_s"], p["voice_shift_s"],
              p["voice_cut"], p["eeg_onset_s"])
             for p in pairs])
        n_shift = sum(p["voice_shift_s"] > 0 for p in pairs)
        n_cut = sum(p["voice_cut"] for p in pairs)
        total += len(pairs)
        shifted += n_shift
        cut += n_cut
        print(f"[biomex] {vid}: {len(pairs)} pairs, {n_shift} voice windows moved earlier, "
              f"{n_cut} words longer than the window "
              f"(EEG sessions start {sorted({p['eeg_start'] for p in pairs})})")
    con.executemany("INSERT INTO recording VALUES (?, ?)",
                    [("voice_windows_shifted", str(shifted)), ("voice_words_cut", str(cut))])
    con.commit()
    con.close()
    os.replace(tmp, out)
    print(f"[biomex] {len(volunteers)} volunteers, {total} pairs ({shifted} voice windows "
          f"moved earlier, {cut} words longer than the window) -> {out}")


if __name__ == "__main__":
    main()
