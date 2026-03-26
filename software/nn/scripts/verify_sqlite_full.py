#!/usr/bin/env python3
"""
verify_sqlite_full.py

Deterministic full verification: iterate all trials in the DB (ordered by id)
and compare stored EEG/audio blobs against the original .mat rows.

Usage:
    python scripts/verify_sqlite_full.py <db_path> <dataset_root>
"""
import sys
import os
import sqlite3
import numpy as np
from scipy.io import loadmat
from sqlite_reader import connect, fetch_trial

CHANNELS = ['F3', 'F4', 'C3', 'C4', 'P3', 'P4']
EEG_SR = 1024
EEG_DURATION_S = 4
EEG_CHANNELS = 6
EEG_SAMPLES_PER_CHANNEL = EEG_SR * EEG_DURATION_S  # 4096
EEG_SIGNAL_COLUMNS = EEG_SAMPLES_PER_CHANNEL * EEG_CHANNELS  # 24576

AUDIO_SR = 44100
AUDIO_DURATION_S = 4
AUDIO_SAMPLES = AUDIO_SR * AUDIO_DURATION_S  # 176400


def main():
    if len(sys.argv) < 3:
        print("Usage: python scripts/verify_sqlite_full.py <db_path> <dataset_root>")
        sys.exit(1)
    db_path = sys.argv[1]
    dataset_root = sys.argv[2]

    conn = connect(db_path)
    c = conn.cursor()
    c.execute('SELECT id FROM trial ORDER BY id')
    ids = [r[0] for r in c.fetchall()]
    if not ids:
        print('No trials found in DB')
        sys.exit(1)

    passed = 0
    failed = 0
    skipped = 0

    for trial_id in ids:
        print(f"Verifying trial {trial_id}...")
        info = fetch_trial(conn, trial_id, dataset_root=dataset_root)
        subject_name = info['metadata']['subject_name']
        eeg_db = info['eeg']
        audio_db = info['audio']
        orig_row = info['original_mat_row']

        # If no EEG and no audio, skip
        if (not eeg_db or all(v is None for v in eeg_db.values())) and audio_db is None:
            print(f"Trial {trial_id} has no EEG or audio; skipping")
            skipped += 1
            continue

        # Determine original .mat row index
        match_idx = None
        if orig_row is not None:
            # fetch original_row stored in metadata
            c.execute("SELECT original_row FROM trial WHERE id=?", (trial_id,))
            res = c.fetchone()
            if res and res[0] is not None:
                match_idx = int(res[0])
                print(f"Using stored original_row index: {match_idx}")

        # Load subject EEG mat once
        subj_path = os.path.join(dataset_root, subject_name)
        eeg_files = [f for f in os.listdir(subj_path) if f.endswith('_EEG.mat')]
        audio_files = [f for f in os.listdir(subj_path) if f.endswith('_Audio.mat')]
        eeg_mat = None
        audio_mat = None
        if eeg_files:
            eeg_mat = loadmat(os.path.join(subj_path, eeg_files[0]), squeeze_me=True)['EEG']
        if audio_files:
            audio_mat = loadmat(os.path.join(subj_path, audio_files[0]), squeeze_me=True)['Audio']

        all_match = True

        # If we don't have match_idx, try fallback matching using stimulus+blink
        if match_idx is None and eeg_mat is not None:
            c.execute("SELECT t.stimulus_id, e.blink FROM trial t JOIN eeg_samples e ON e.trial_id = t.id WHERE t.id=?", (trial_id,))
            row = c.fetchone()
            if row:
                db_stimulus_id, db_blink = row
                c.execute("SELECT name FROM stimulus WHERE id=?", (db_stimulus_id,))
                res = c.fetchone()
                db_stimulus_name = res[0] if res else None
                match_idx = None
                # try to find a row matching stimulus code and blink
                for idx, row in enumerate(eeg_mat):
                    stim_code = int(row[EEG_SIGNAL_COLUMNS + 1])
                    blink = int(row[EEG_SIGNAL_COLUMNS + 2])
                    if db_stimulus_name and db_stimulus_name.startswith('stimulus_'):
                        try:
                            db_code = int(db_stimulus_name.split('_')[-1])
                        except Exception:
                            db_code = None
                        if db_code == stim_code and blink == db_blink:
                            match_idx = idx
                            break
                if match_idx is None:
                    print(f"No matching .mat row found for trial {trial_id}; will attempt partial checks")

        # Compare EEG per-channel if available
        if eeg_db and eeg_mat is not None and match_idx is not None:
            for ch_idx, ch_name in enumerate(CHANNELS):
                db_arr = eeg_db.get(ch_name)
                mat_arr = eeg_mat[match_idx][ch_idx*EEG_SAMPLES_PER_CHANNEL:(ch_idx+1)*EEG_SAMPLES_PER_CHANNEL]
                if not np.allclose(db_arr, mat_arr):
                    print(f"Mismatch in channel {ch_name} for trial {trial_id}")
                    all_match = False
                    break
        elif eeg_db and eeg_mat is None:
            print(f"No EEG .mat file available for subject {subject_name}; skipping EEG check for trial {trial_id}")
        else:
            if eeg_db and match_idx is None:
                print(f"Could not determine original EEG row for trial {trial_id}; skipping EEG check")

        # Compare audio if present
        if audio_db is not None and audio_mat is not None:
            cursor = conn.cursor()
            cursor.execute('SELECT audio_row FROM audio_samples WHERE trial_id=? LIMIT 1', (trial_id,))
            ares = cursor.fetchone()
            if ares and ares[0] is not None:
                audio_row_idx = int(ares[0])
                mat_audio_row = audio_mat[audio_row_idx][:AUDIO_SAMPLES]
                if not np.allclose(audio_db, mat_audio_row):
                    print(f"Mismatch in audio for trial {trial_id}")
                    all_match = False
            else:
                print(f"No audio_row stored for trial {trial_id}; skipping audio check")

        if all_match:
            print(f"Trial {trial_id} PASSED")
            passed += 1
        else:
            print(f"Trial {trial_id} FAILED")
            failed += 1

    total = len(ids)
    print(f"Full verification complete: passed={passed}, failed={failed}, skipped={skipped}, total={total}")


if __name__ == '__main__':
    main()
