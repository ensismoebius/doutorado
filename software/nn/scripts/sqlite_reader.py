#!/usr/bin/env python3
"""
sqlite_reader.py

Small stable API to fetch and deserialize trials from the sqlite DB produced by
`mat_to_sqlite_redo.py`.

Functions:
 - connect(db_path) -> sqlite3.Connection
 - fetch_trial(conn, trial_id, dataset_root) -> dict with subject, eeg_channels, audio, metadata
 - random_trial_ids(conn, n, require_audio=True) -> list of trial ids

"""
import sqlite3
import numpy as np
import os
from scipy.io import loadmat

CHANNELS = ['F3', 'F4', 'C3', 'C4', 'P3', 'P4']
EEG_SAMPLES_PER_CHANNEL = 1024 * 4
AUDIO_SAMPLES = 44100 * 4


def connect(db_path):
    return sqlite3.connect(db_path)


def _deserialize(blob, dtype=np.float64, count=None):
    if blob is None:
        return None
    arr = np.frombuffer(blob, dtype=dtype)
    if count is not None and arr.size != count:
        raise ValueError(f"Deserialized size {arr.size} != expected {count}")
    return arr


def fetch_trial(conn, trial_id, dataset_root=None):
    c = conn.cursor()
    c.execute('SELECT subject_id, modality_id, stimulus_id, eeg_sample_count, audio_sample_count, original_row FROM trial WHERE id=?', (trial_id,))
    t = c.fetchone()
    if not t:
        raise KeyError(f'trial {trial_id} not found')
    subject_id, modality_id, stimulus_id, eeg_count, audio_count, original_row = t
    c.execute('SELECT subject_name FROM subject WHERE id=?', (subject_id,))
    subject_name = c.fetchone()[0]

    # EEG
    c.execute('SELECT F3,F4,C3,C4,P3,P4,blink FROM eeg_samples WHERE trial_id=?', (trial_id,))
    eeg_row = c.fetchone()
    eeg = {}
    if eeg_row:
        for i, ch in enumerate(CHANNELS):
            eeg[ch] = _deserialize(eeg_row[i], count=EEG_SAMPLES_PER_CHANNEL)
        blink = eeg_row[-1]
    else:
        blink = None

    # Audio
    c.execute('SELECT samples FROM audio_samples WHERE trial_id=? LIMIT 1', (trial_id,))
    audio_row = c.fetchone()
    audio = None
    if audio_row and audio_row[0] is not None:
        audio = _deserialize(audio_row[0], count=AUDIO_SAMPLES)

    metadata = {
        'subject_name': subject_name,
        'modality_id': modality_id,
        'stimulus_id': stimulus_id,
        'eeg_sample_count': eeg_count,
        'audio_sample_count': audio_count,
        'original_row': original_row,
        'blink': blink,
    }

    # If dataset_root provided, also return original .mat row (for verification)
    original_mat_row = None
    if dataset_root and original_row is not None:
        subj_path = os.path.join(dataset_root, subject_name)
        eeg_files = [f for f in os.listdir(subj_path) if f.endswith('_EEG.mat')]
        if eeg_files:
            mat = loadmat(os.path.join(subj_path, eeg_files[0]), squeeze_me=True)
            original_mat_row = mat['EEG'][int(original_row)]

    return {
        'trial_id': trial_id,
        'eeg': eeg,
        'audio': audio,
        'metadata': metadata,
        'original_mat_row': original_mat_row,
    }


def random_trial_ids(conn, n=10, require_audio=True):
    c = conn.cursor()
    if require_audio:
        c.execute('SELECT id FROM trial WHERE audio_sample_count IS NOT NULL ORDER BY RANDOM() LIMIT ?', (n,))
    else:
        c.execute('SELECT id FROM trial ORDER BY RANDOM() LIMIT ?', (n,))
    return [r[0] for r in c.fetchall()]
