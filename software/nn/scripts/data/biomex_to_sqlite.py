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

  Both are written at the Coretto base's rates, 44.1 kHz voice and 1024 Hz EEG (see
  OUT_VOICE_RATE below for why), resampled whole before the windows are cut.

  EEG   EEG_EPOCH_S after the marker, the 14 channels, through the Coretto base's own EEG
        filter (see eeg_chain): the same treatment the rest of the thesis's EEG had. First
        the Emotiv's ADC offset (~4200 uV, different per electrode and per headset fit) is
        removed; left in, energy and spectrum features would read the fit of the headset,
        not the person, and identify speakers by it.
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
from scipy.signal import fftconvolve, firwin, resample_poly

FORMAT = "thesis-paired-v1"
SOURCE = "biomex2021"
EEG_RATE = 128
EEG_CHANNELS = ("AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4",
                "F8", "AF4")
EEG_EPOCH_S = 2.0           # shortest gap between two digit markers in F001: 2.42 s
VOICE_RATE = 16000
VOICE_WINDOW_S = 1.0        # F001-F006: word p99 0.74 s; 7 of 840 onset gaps/tails < 1 s
# The rates the pairs are WRITTEN at: the Coretto base's (44.1 kHz voice, 1024 Hz EEG). The
# frozen Coretto pipeline splits 0..Nyquist into 2^level equal DTWPT sub-bands, so a column of
# its feature vector is a band in Hz only at the rate it was selected at. Measured at the
# native rates (2026-11-13): Coretto voice holds 99.2% of its power in sub-band 0 of 16 and
# Coretto EEG 97.2%, while BIOMEX at 16 kHz / 128 Hz spreads over 9-10 sub-bands; the same
# column measured different Hz in the two bases. Upsampled to the Coretto rates, BIOMEX puts
# ~90-95% in sub-band 0 too. Upsampling adds no content above the native Nyquist (8 kHz /
# 64 Hz); Coretto holds 0.1% of its voice power above 8 kHz and 0.0% of its EEG above 64 Hz.
# resample_poly low-passes at the native Nyquist, so no images are added.
OUT_VOICE_RATE = 44100      # 16000 * 441 / 160
OUT_EEG_RATE = 1024         # 128 * 8
VOICE_UP, VOICE_DOWN = 441, 160
EEG_UP = 8
# The EEG filter of the Coretto base as its paper gives it (Pressel Coretto et al., SPIE
# 10160, 2017, sec. 2.3): a 2-40 Hz band-pass made of a low-pass FIR of order 372 and a
# high-pass FIR of order 1204 at 1024 Hz, applied once to the whole recording, its 788-sample
# group delay removed by shifting. The window is not stated; Hamming is MATLAB fir1's default
# (and firwin's). Designed so, the group delay is (372 + 1204) / 2 = 788 samples, as the
# paper states; -6 dB at 2 and 40 Hz, flat 4-30 Hz, ~ -53 dB stopband.
BAND_HZ = (2.0, 40.0)
CORETTO_LP_ORDER = 372
CORETTO_HP_ORDER = 1204
FILTER_EDGE = (CORETTO_LP_ORDER + CORETTO_HP_ORDER) // 2   # 788 samples at 1024 Hz
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

# ── what the distributed files get wrong, and what is done about it ───────────────────
# Found by checking the base against itself, its CSV exports and Table 1 of the paper
# (2026-11-13). Affected data are EXCLUDED, not corrected (the user's decisions, 2026-11-13).
#
# THE STRICT RULE (own_recordings). The paper describes one 10-digit (01G10) and one 4-digit
# (01G04) EEG recording per volunteer. A volunteer is kept only if its folder holds exactly
# one COMPLETE recording of each (10 runs of 10 / of 4 digits) named "<id>-<session>-...".
# The folders that break it, over the first 36 volunteers: F007 (its 01G04 is filed under
# F006's name), M007 ("M007-1G10"), M010 ("M10-01G04"), M001 (two complete 01G10 and two
# 01G04 the same day; which one its voice was recorded with cannot be told). Excluded whole,
# so every kept speaker has the same design: one session, 140 pairs. Not breaking it:
#   - recordings that hold no digit marker at all (starts aborted after a few seconds:
#     F008, M006, and M004's "M1G10"): no data, set aside;
#   - second sessions on another day (02G10 / 02G04: M001, M003, M004), which the paper does
#     not describe: not read.
# F017 01G10 series 8: the .lab has no line for the 7th digit (0, CERO): no line covers
# 126715626..131048750 (100 ns), where there is speech and where the EEG marker and Table 1
# put the 0. That one pair is left out; the other 9 digits of the series pair as usual.
# (Listed by name: a word missing from a .lab is otherwise refused.)
UNLABELLED_DIGITS = {("F017", "01G10", 8): (7, 126715626, 131048750)}


class ConversionError(Exception):
    pass


class Excluded(Exception):
    """A volunteer the strict rule leaves out; the message is the reason."""


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


def lab_entries(path):
    """Every line of a .lab as (start, end, label), in its 100 ns units."""
    entries = []
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
        if label != SILENCE and label not in DIGIT_WORDS.values():
            raise ConversionError(f"{path}:{n}: unknown label {parts[2]!r}")
        entries.append((a, b, label))
    return entries


def lab_samples(t):
    """A .lab time (100 ns units) in voice samples."""
    return int(round(t * 1e-7 * VOICE_RATE))


def read_lab(path):
    """The spoken words of a series, in order, as (onset sample, end sample, word)."""
    return [(lab_samples(a), lab_samples(b), label)
            for a, b, label in lab_entries(path) if label != SILENCE]


def unlabelled_span(path, a, b):
    """(start, end) in samples of a span no line of the .lab covers; refused otherwise."""
    overlap = [e for e in lab_entries(path) if e[0] <= b and a <= e[1]]
    if overlap:
        raise ConversionError(f"{path}: lines {overlap} cover {a}..{b}; the file is not the "
                              f"one UNLABELLED_DIGITS describes")
    return lab_samples(a), lab_samples(b)


def own_recordings(folder, vid):
    """{session: EDF path} under the strict rule (see the comment above UNLABELLED_DIGITS);
    Excluded, with the reason, when the folder breaks it."""
    chosen = {}
    for session, per in SESSIONS.items():
        complete, partial = [], []
        for path in sorted(glob.glob(os.path.join(folder, "EEG", f"{vid}-{session}-*.edf"))):
            labels, _, _, _, raw = read_edf(path)
            runs = digit_runs(raw[labels.index("MARKER")], path)
            if not runs:
                continue                     # an aborted start: no digit, no data
            if len(runs) == SERIES_PER_SESSION and all(len(r) == per for r in runs):
                complete.append(path)
            else:
                partial.append(path)
        if partial:
            raise Excluded(f"{session}: an incomplete recording with digits in it "
                           f"({', '.join(os.path.basename(p) for p in partial)})")
        if len(complete) != 1:
            held = sorted(os.listdir(os.path.join(folder, "EEG")))
            raise Excluded(f"{session}: {len(complete)} complete recordings named "
                           f"{vid}-{session}-*.edf (EEG/ holds {', '.join(held)})")
        chosen[session] = complete[0]
    return chosen


# ── one volunteer ──────────────────────────────────────────────────────────────────────
def coretto_filters():
    """The Coretto paper's low-pass (order 372, 40 Hz) and high-pass (order 1204, 2 Hz)."""
    lp = firwin(CORETTO_LP_ORDER + 1, BAND_HZ[1], fs=OUT_EEG_RATE, window="hamming")
    hp = firwin(CORETTO_HP_ORDER + 1, BAND_HZ[0], pass_zero=False, fs=OUT_EEG_RATE,
                window="hamming")
    return lp, hp


def eeg_chain(eeg_native, filters):
    """A session's EEG (channels, samples at 128 Hz) -> the 1024 Hz signal epochs are cut from.

    1. The Emotiv's ADC offset, one constant per channel (the session mean), is removed. The
       Coretto amplifier was AC-coupled (analog 0.3-35 Hz), so its FIR never saw an offset;
       on the Emotiv's ~4200 uV, the FIR's -53 dB stopband alone would leave ~9 uV per
       electrode (measured: epoch means -12..-7 uV; with the offset removed, -2.8..2.7 uV,
       as on Coretto, -3.1..2.3 uV).
    2. Upsampled x8 to 1024 Hz, the rate the Coretto filter was designed for.
    3. The Coretto filter, applied once: mode="same" with an odd, symmetric kernel is the
       convolution shifted left by its group delay, as the paper did ("left-shifting")."""
    lp, hp = filters
    x = eeg_native - eeg_native.mean(axis=1, keepdims=True)
    x = resample_poly(x, EEG_UP, 1, axis=-1)
    x = fftconvolve(x, lp[None, :], mode="same", axes=-1)
    return fftconvolve(x, hp[None, :], mode="same", axes=-1)


def convert_volunteer(folder, vid, filters, edge_dropped):
    """Every (session, series, position, digit, voice, eeg, interpolated) of a volunteer.

    A pair whose EEG epoch lies within FILTER_EDGE samples of either end of its session is
    left out and named in `edge_dropped` (decided 2026-11-13): there the Coretto filter reads
    zeros instead of EEG on one side, so the epoch would differ from every other one with
    nothing in the result to show it. A rule, not a per-file exception: it applies to every
    volunteer (M015 and M019 end their 01G04 60 and 220 samples short after the last digit)."""
    epoch = int(round(EEG_EPOCH_S * EEG_RATE))
    win = int(round(VOICE_WINDOW_S * VOICE_RATE))
    out_epoch = epoch * EEG_UP
    out_win = int(round(VOICE_WINDOW_S * OUT_VOICE_RATE))
    pairs = []
    recordings = own_recordings(folder, vid)
    for session, per_series in SESSIONS.items():
        where = f"{vid} {session}"
        labels, phys, rate, start, raw = read_edf(recordings[session])
        if rate != EEG_RATE:
            raise ConversionError(f"{where}: EEG at {rate} Hz, expected {EEG_RATE}")
        missing = [c for c in EEG_CHANNELS + ("MARKER", "INTERPOLATED") if c not in labels]
        if missing:
            raise ConversionError(f"{where}: EDF lacks {missing}")
        # The whole session at once, so no epoch has a filter or resampling edge of its own.
        eeg_out = eeg_chain(np.stack([phys[labels.index(c)] for c in EEG_CHANNELS]), filters)
        interp = raw[labels.index("INTERPOLATED")] != 0
        runs = digit_runs(raw[labels.index("MARKER")], where)
        if len(runs) != SERIES_PER_SESSION or any(len(r) != per_series for r in runs):
            raise ConversionError(f"{where}: digit runs of lengths {[len(r) for r in runs]}; "
                                  f"expected {SERIES_PER_SESSION} of {per_series}")
        for k, run in enumerate(runs, 1):
            stem = os.path.join(folder, "Voice", f"{vid}_{session}_{k}")
            if not (os.path.isfile(stem + ".wav") and os.path.isfile(stem + ".lab")):
                raise ConversionError(f"{where} series {k}: {stem}.wav/.lab missing")
            for i in range(len(run) - 1):
                if run[i + 1][0] - run[i][0] < epoch:
                    raise ConversionError(
                        f"{where} series {k} digit {i + 1}: next marker after "
                        f"{(run[i + 1][0] - run[i][0]) / EEG_RATE:.2f} s < {EEG_EPOCH_S} s")
            voice = read_wav(stem + ".wav")
            voice_out = resample_poly(voice, VOICE_UP, VOICE_DOWN)  # the whole series WAV
            words = read_lab(stem + ".lab")
            positions = list(range(1, len(run) + 1))
            kept = list(run)
            # Spans a voice window must not reach: every labelled word, and an unlabelled one.
            spoken = [(on, end) for on, end, _ in words]
            if (vid, session, k) in UNLABELLED_DIGITS:
                pos, a, b = UNLABELLED_DIGITS[(vid, session, k)]
                spoken = sorted(spoken + [unlabelled_span(stem + ".lab", a, b)])
                del kept[pos - 1]
                del positions[pos - 1]
            shown = [DIGIT_WORDS[c] for _, c in kept]
            said = [w for _, _, w in words]
            if shown != said:
                raise ConversionError(f"{where} series {k}: EEG markers {shown}, "
                                      f".lab words {said}")
            for j, (m, code), (on, end, word) in zip(positions, kept, words):
                nxt = min([s for s, _ in spoken if s > on] + [len(voice)])
                prev_end = max([e for _, e in spoken if e < on] + [0])
                begin = min(on, nxt - win)
                if begin < prev_end:
                    raise ConversionError(
                        f"{where} series {k} digit {j}: no {VOICE_WINDOW_S} s voice window fits "
                        f"between {prev_end / VOICE_RATE:.3f} s (the previous word or the "
                        f"start) and {nxt / VOICE_RATE:.3f} s (the next word or the end)")
                # The window is chosen on the native clock (the .lab and markers are in
                # native samples) and cut from the upsampled signal at the same instant.
                v0 = int(round(begin * VOICE_UP / VOICE_DOWN))
                e0 = m * EEG_UP
                if v0 + out_win > len(voice_out):
                    raise ConversionError(f"{where} series {k} digit {j}: the upsampled "
                                          f"voice window passes the end of its WAV")
                if e0 < FILTER_EDGE or e0 + out_epoch + FILTER_EDGE > eeg_out.shape[1]:
                    edge_dropped.append(f"{where} series {k} digit {j}")
                    continue
                pairs.append({"session": session, "series": k, "position": j, "code": code,
                              "voice": voice_out[v0:v0 + out_win].astype(np.float32),
                              "eeg": eeg_out[:, e0:e0 + out_epoch].astype(np.float32),
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
        "voice_sample_rate": str(OUT_VOICE_RATE), "voice_window_samples":
            str(int(round(VOICE_WINDOW_S * OUT_VOICE_RATE))),
        "voice_native_rate": str(VOICE_RATE), "eeg_native_rate": str(EEG_RATE),
        "resampling": f"scipy resample_poly of the whole series WAV ({VOICE_UP}/{VOICE_DOWN}) "
                      f"and of the whole EEG session (x{EEG_UP}), to the Coretto rates "
                      f"{OUT_VOICE_RATE} / {OUT_EEG_RATE} Hz, before cutting",
        "voice_window": f"{VOICE_WINDOW_S:g} s from the word's onset; moved earlier to end at "
                        f"the next spoken word's onset or the WAV's end when it would reach "
                        f"them (pair.voice_shift_s)",
        "eeg_sample_rate": str(OUT_EEG_RATE), "eeg_channels": ",".join(EEG_CHANNELS),
        "eeg_epoch_samples": str(int(round(EEG_EPOCH_S * EEG_RATE)) * EEG_UP),
        "eeg_filter": f"Emotiv ADC offset removed (per channel, session mean); upsampled to "
                      f"{OUT_EEG_RATE} Hz; the Coretto paper's filter: FIR low-pass order "
                      f"{CORETTO_LP_ORDER} ({BAND_HZ[1]:g} Hz) and high-pass order "
                      f"{CORETTO_HP_ORDER} ({BAND_HZ[0]:g} Hz), Hamming, applied once, "
                      f"{FILTER_EDGE}-sample group delay removed, whole session",
        "excluded_pairs": "; ".join(f"{v} {s} series {k} digit {p}: no .lab line"
                                    for (v, s, k), (p, _, _) in sorted(UNLABELLED_DIGITS.items())),
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
    filters = coretto_filters()
    tmp = out + ".partial"
    if os.path.exists(tmp):
        os.remove(tmp)
    con = sqlite3.connect(tmp)
    con.executescript(SCHEMA)
    con.executemany("INSERT INTO recording VALUES (?, ?)", recording_info().items())
    for code, word in sorted(DIGIT_WORDS.items()):
        con.execute("INSERT INTO stimulus VALUES (?, ?)", (code, word.lower()))
    total = shifted = cut = 0
    kept, excluded, edge_dropped = [], [], []
    for vid in volunteers:
        try:
            own_recordings(os.path.join(src, vid), vid)
        except Excluded as e:
            excluded.append(f"{vid}: {e}")
            print(f"[biomex] {vid}: EXCLUDED by the strict rule - {e}")
            continue
        except ConversionError as e:
            con.close()
            os.remove(tmp)
            sys.exit(f"[biomex] refused: {e}")
        kept.append(vid)
    con.execute("INSERT INTO recording VALUES (?, ?)",
                ("excluded_volunteers", " | ".join(excluded) or "none"))
    for sid, vid in enumerate(kept, 1):
        try:
            pairs = convert_volunteer(os.path.join(src, vid), vid, filters, edge_dropped)
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
                    [("voice_windows_shifted", str(shifted)), ("voice_words_cut", str(cut)),
                     ("edge_dropped_pairs", " | ".join(edge_dropped) or "none")])
    for d in edge_dropped:
        print(f"[biomex] {d}: left out, its EEG epoch is within {FILTER_EDGE} samples of the "
              f"session's edge")
    con.commit()
    con.close()
    os.replace(tmp, out)
    print(f"[biomex] {len(kept)} volunteers ({len(volunteers) - len(kept)} excluded), "
          f"{total} pairs ({shifted} voice windows moved earlier, {cut} words longer than "
          f"the window) -> {out}")


if __name__ == "__main__":
    main()
