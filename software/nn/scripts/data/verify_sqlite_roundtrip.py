#!/usr/bin/env python3
"""
verify_sqlite_roundtrip.py

- Fetch a random trial from the SQLite DB
- Deserialize EEG and audio BLOBs
- Compare to original .mat file for round-trip verification

Usage:
    python scripts/verify_sqlite_roundtrip.py <db_path> <dataset_root>
"""
import sys
import os
import sqlite3
import numpy as np
from scipy.io import loadmat
from sqlite_reader import connect, fetch_trial, random_trial_ids

CHANNELS = ['F3', 'F4', 'C3', 'C4', 'P3', 'P4']
EEG_SR = 1024
EEG_DURATION_S = 4
EEG_CHANNELS = 6
EEG_SAMPLES_PER_CHANNEL = EEG_SR * EEG_DURATION_S  # 4096
EEG_SIGNAL_COLUMNS = EEG_SAMPLES_PER_CHANNEL * EEG_CHANNELS  # 24576

AUDIO_SR = 44100
AUDIO_DURATION_S = 4
AUDIO_SAMPLES = AUDIO_SR * AUDIO_DURATION_S  # 176400

def deserialize_blob(blob, dtype, count):
    arr = np.frombuffer(blob, dtype=dtype)
    if arr.size != count:
        raise ValueError(f"Blob size {arr.size} does not match expected {count}")
    return arr

def _verify_multiple_trials(conn, dataset_root, ids):
    """Round-trip check each of `ids` via `fetch_trial()`.

    Returns `(passed, failed, last_trial_id, last_subject_name)` -- the
    last two are the LOOP's final iteration values, kept because `main()`
    feeds them into the second, independent verification pass below.
    """
    passed = 0
    failed = 0
    subject_name = None
    trial_id = None
    for trial_id in ids:
        print(f"Verifying trial {trial_id}...")
        info = fetch_trial(conn, trial_id, dataset_root=dataset_root)
        subject_name = info['metadata']['subject_name']
        # DB data
        eeg_db = info['eeg']
        audio_db = info['audio']
        orig_row = info['original_mat_row']
        if orig_row is None:
            print(f"No original_row stored for trial {trial_id}; skipping")
            failed += 1
            continue
        # compare per-channel
        all_match = True
        for ch_idx, ch_name in enumerate(CHANNELS):
            db_arr = eeg_db.get(ch_name)
            mat_arr = orig_row[ch_idx*EEG_SAMPLES_PER_CHANNEL:(ch_idx+1)*EEG_SAMPLES_PER_CHANNEL]
            if not np.allclose(db_arr, mat_arr):
                print(f"Mismatch in channel {ch_name} for trial {trial_id}")
                all_match = False
                break
        # audio
        if audio_db is not None:
            # find audio row index stored
            cursor = conn.cursor()
            cursor.execute('SELECT audio_row FROM audio_samples WHERE trial_id=? LIMIT 1', (trial_id,))
            ares = cursor.fetchone()
            if ares and ares[0] is not None:
                audio_row_idx = int(ares[0])
                mat_audio = loadmat(os.path.join(dataset_root, subject_name, f"{subject_name}_Audio.mat"), squeeze_me=True)['Audio']
                mat_audio_row = mat_audio[audio_row_idx][:AUDIO_SAMPLES]
                if not np.allclose(audio_db, mat_audio_row):
                    print(f"Mismatch in audio for trial {trial_id}")
                    all_match = False
            else:
                print(f"No audio_row stored for trial {trial_id}; skipping audio check")
                all_match = False

        if all_match:
            print(f"Trial {trial_id} PASSED")
            passed += 1
        else:
            print(f"Trial {trial_id} FAILED")
            failed += 1
    return passed, failed, trial_id, subject_name


def _resolve_original_mat_row_index(c, trial_id, eeg_arr):
    """`trial_id`'s row index into `eeg_arr` -- the stored `trial.original_row` when present,
    else a fallback match by stimulus code + blink flag."""
    # Prefer exact original row index stored in trial.original_row
    c.execute("SELECT original_row FROM trial WHERE id=?", (trial_id,))
    res = c.fetchone()
    if res and res[0] is not None:
        match_idx = int(res[0])
        print(f"Using stored original_row index: {match_idx}")
        return match_idx

    # Fallback: find by stimulus and blink
    c.execute("SELECT t.stimulus_id, e.blink FROM trial t JOIN eeg_samples e ON e.trial_id = t.id WHERE t.id=?", (trial_id,))
    db_stimulus_id, db_blink = c.fetchone()
    c.execute("SELECT name FROM stimulus WHERE id=?", (db_stimulus_id,))
    db_stimulus_name = c.fetchone()[0]
    db_stim_code = int(db_stimulus_name.split('_')[-1])
    for idx, row in enumerate(eeg_arr):
        stim_code = int(row[EEG_SIGNAL_COLUMNS + 1])
        blink = int(row[EEG_SIGNAL_COLUMNS + 2])
        if stim_code == db_stim_code and blink == db_blink:
            print(f"Fallback matched .mat row index: {idx}")
            return idx
    print("No matching .mat row found for this trial's stimulus and blink.")
    sys.exit(1)


def _verify_single_trial_raw_blobs(conn, dataset_root, trial_id, subject_name):
    """A second, independent round-trip check: re-fetches `trial_id`'s raw blobs directly via
    SQL (rather than through `fetch_trial()`, like `_verify_multiple_trials` above) and compares
    them against its resolved `.mat` row.

    `trial_id`/`subject_name` are `_verify_multiple_trials`'s last-loop-iteration values, not a
    fresh selection -- preserved as-is; this refactor only splits the function, it does not
    change what gets verified.
    """
    c = conn.cursor()
    # Fetch EEG blobs
    c.execute("SELECT F3, F4, C3, C4, P3, P4, blink FROM eeg_samples WHERE trial_id=?", (trial_id,))
    eeg_row = c.fetchone()
    # Fetch audio blob
    c.execute("SELECT samples FROM audio_samples WHERE trial_id=?", (trial_id,))
    audio_row = c.fetchone()
    # Get sample counts
    c.execute("SELECT eeg_sample_count, audio_sample_count FROM trial WHERE id=?", (trial_id,))
    eeg_count, audio_count = c.fetchone()
    # Find subject folder
    subj_path = os.path.join(dataset_root, subject_name)
    eeg_file = [f for f in os.listdir(subj_path) if f.endswith('_EEG.mat')][0]
    audio_file = [f for f in os.listdir(subj_path) if f.endswith('_Audio.mat')][0]
    eeg_mat = loadmat(os.path.join(subj_path, eeg_file), squeeze_me=True)
    audio_mat = loadmat(os.path.join(subj_path, audio_file), squeeze_me=True)
    eeg_arr = eeg_mat['EEG']
    audio_arr = audio_mat['Audio']

    match_idx = _resolve_original_mat_row_index(c, trial_id, eeg_arr)

    # Compare EEG
    all_match = True
    for ch_idx, ch_name in enumerate(CHANNELS):
        blob = eeg_row[ch_idx]
        arr_db = deserialize_blob(blob, np.float64, EEG_SAMPLES_PER_CHANNEL)
        arr_mat = eeg_arr[match_idx][ch_idx*EEG_SAMPLES_PER_CHANNEL:(ch_idx+1)*EEG_SAMPLES_PER_CHANNEL]
        if not np.allclose(arr_db, arr_mat):
            print(f"Mismatch in channel {ch_name}")
            all_match = False
        else:
            print(f"Channel {ch_name} matches.")
    # Find matching audio row by stored audio_row in audio_samples
    c.execute("SELECT audio_row FROM audio_samples WHERE trial_id=? LIMIT 1", (trial_id,))
    ares = c.fetchone()
    if ares and ares[0] is not None:
        match_audio_idx = int(ares[0])
        arr_db = deserialize_blob(audio_row[0], np.float64, AUDIO_SAMPLES)
        arr_mat = audio_arr[match_audio_idx][:AUDIO_SAMPLES]
        if not np.allclose(arr_db, arr_mat):
            print("Mismatch in audio samples.")
            all_match = False
        else:
            print("Audio samples match.")
    else:
        print("No stored audio_row for this trial; cannot verify audio.")
    if all_match:
        print("Round-trip verification PASSED.")
    else:
        print("Round-trip verification FAILED.")


def main():
    if len(sys.argv) < 3:
        print("Usage: python scripts/verify_sqlite_roundtrip.py <db_path> <dataset_root> [num_checks]")
        sys.exit(1)
    db_path = sys.argv[1]
    dataset_root = sys.argv[2]
    num_checks = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    conn = connect(db_path)
    ids = random_trial_ids(conn, n=num_checks, require_audio=True)
    if not ids:
        print('No trials with audio available for verification')
        sys.exit(1)

    passed, failed, last_trial_id, last_subject_name = _verify_multiple_trials(conn, dataset_root, ids)
    print(f"Verification complete: passed={passed}, failed={failed}, total={len(ids)}")

    _verify_single_trial_raw_blobs(conn, dataset_root, last_trial_id, last_subject_name)

if __name__ == "__main__":
    main()
