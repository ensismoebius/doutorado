#!/usr/bin/env python3
"""
mat_to_sqlite_redo.py

Import EEG and Audio data from the Imagined Speech dataset into SQLite
according to the requested schema (per-channel EEG BLOBs, audio BLOBs).

Usage:
    python scripts/mat_to_sqlite_redo.py <dataset_root> <output_db>

This script expects the .mat files organized per-subject under folders
S01..S15 with files named `SXX_EEG.mat` and `SXX_Audio.mat`.

Notes:
- Uses METADATA constants: 6 channels, 4096 EEG samples per channel,
  audio 176400 samples per trial.
- Stores per-channel arrays as raw float64 BLOBs (no compression).
"""
import os
import sys
import sqlite3
from scipy.io import loadmat
import numpy as np
from tqdm import tqdm

# Channel order from METADATA
CHANNELS = ['F3', 'F4', 'C3', 'C4', 'P3', 'P4']
EEG_SR = 1024
EEG_DURATION_S = 4
EEG_CHANNELS = 6
EEG_SAMPLES_PER_CHANNEL = EEG_SR * EEG_DURATION_S  # 4096
EEG_SIGNAL_COLUMNS = EEG_SAMPLES_PER_CHANNEL * EEG_CHANNELS  # 24576

AUDIO_SR = 44100
AUDIO_DURATION_S = 4
AUDIO_SAMPLES = AUDIO_SR * AUDIO_DURATION_S  # 176400


# Allowed names (from include/nn/dataLoaders/10.1117/schema/NAMES.hpp)
ALLOWED_MODALITIES = {"Imagined", "Pronounced"}
# stimulus code -> name map (1-based codes)
STIMULUS_CODE_MAP = {
    1: "A",
    2: "E",
    3: "I",
    4: "O",
    5: "U",
    6: "Arriba",
    7: "Abajo",
    8: "Adelante",
    9: "Atras",
    10: "Derecha",
    11: "Izquierda",
}
ALLOWED_STIMULI = set(STIMULUS_CODE_MAP.values())


def create_schema(conn):
    c = conn.cursor()
    # subjects
    c.execute("""
    CREATE TABLE IF NOT EXISTS subject (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        subject_name TEXT UNIQUE
    );
    """)
    # modalities
    c.execute("""
    CREATE TABLE IF NOT EXISTS modalities (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE
    );
    """)
    # stimulus
    c.execute("""
    CREATE TABLE IF NOT EXISTS stimulus (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE
    );
    """)
    # trial
    c.execute("""
    CREATE TABLE IF NOT EXISTS trial (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        subject_id INTEGER,
        modality_id INTEGER,
        stimulus_id INTEGER,
        eeg_sample_count INTEGER,
        audio_sample_count INTEGER,
        original_row INTEGER,
        FOREIGN KEY(subject_id) REFERENCES subject(id),
        FOREIGN KEY(modality_id) REFERENCES modalities(id),
        FOREIGN KEY(stimulus_id) REFERENCES stimulus(id)
    );
    """)
    # eeg_samples: per-trial, per-channel blobs
    c.execute("""
    CREATE TABLE IF NOT EXISTS eeg_samples (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        trial_id INTEGER,
        F3 BLOB,
        F4 BLOB,
        C3 BLOB,
        C4 BLOB,
        P3 BLOB,
        P4 BLOB,
        blink INTEGER,
        FOREIGN KEY(trial_id) REFERENCES trial(id)
    );
    """)
    # audio_samples
    c.execute("""
    CREATE TABLE IF NOT EXISTS audio_samples (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        trial_id INTEGER,
        samples BLOB,
        audio_row INTEGER,
        FOREIGN KEY(trial_id) REFERENCES trial(id)
    );
    """)
    # simple indexes
    c.execute("CREATE INDEX IF NOT EXISTS idx_trial_subject ON trial(subject_id);")
    c.execute("CREATE INDEX IF NOT EXISTS idx_eeg_trial ON eeg_samples(trial_id);")
    c.execute("CREATE INDEX IF NOT EXISTS idx_audio_trial ON audio_samples(trial_id);")
    conn.commit()


def get_or_create_subject(conn, name):
    c = conn.cursor()
    c.execute("INSERT OR IGNORE INTO subject(subject_name) VALUES (?)", (name,))
    conn.commit()
    c.execute("SELECT id FROM subject WHERE subject_name=?", (name,))
    return c.fetchone()[0]


def get_or_create_modality(conn, name):
    if name is None:
        return None
    if name not in ALLOWED_MODALITIES:
        raise ValueError(f"Modality '{name}' is not allowed")
    c = conn.cursor()
    c.execute("INSERT OR IGNORE INTO modalities(name) VALUES (?)", (name,))
    conn.commit()
    c.execute("SELECT id FROM modalities WHERE name=?", (name,))
    return c.fetchone()[0]


def get_or_create_stimulus(conn, name):
    if name is None:
        return None
    if name not in ALLOWED_STIMULI:
        raise ValueError(f"Stimulus '{name}' is not allowed")
    c = conn.cursor()
    c.execute("INSERT OR IGNORE INTO stimulus(name) VALUES (?)", (name,))
    conn.commit()
    c.execute("SELECT id FROM stimulus WHERE name=?", (name,))
    return c.fetchone()[0]


def prepopulate_modalities_and_stimuli(conn, subject_list):
    # ensure subject rows, modality rows and stimulus rows exist
    for s in subject_list:
        get_or_create_subject(conn, s)
    for m in sorted(ALLOWED_MODALITIES):
        get_or_create_modality(conn, m)
    for name in sorted(ALLOWED_STIMULI):
        get_or_create_stimulus(conn, name)


def serialize_array(arr: np.ndarray) -> sqlite3.Binary:
    # Store numeric arrays as float32 to match C++ `float` (4 bytes)
    # and reduce DB size compared to float64.
    a = np.asarray(arr, dtype=np.float32)
    return sqlite3.Binary(a.tobytes())


def process_subject(conn, subject_path, subject_name):
    # Load EEG and Audio .mat files
    eeg_file = None
    audio_file = None
    for f in os.listdir(subject_path):
        if f.endswith('_EEG.mat'):
            eeg_file = os.path.join(subject_path, f)
        if f.endswith('_Audio.mat'):
            audio_file = os.path.join(subject_path, f)
    if not eeg_file:
        print(f"Skipping {subject_name}: no EEG file")
        return 0
    # load
    eeg_mat = loadmat(eeg_file, squeeze_me=True)
    eeg_arr = eeg_mat.get('EEG')
    if eeg_arr is None:
        raise ValueError(f"EEG variable not found in {eeg_file}")

    # Try to extract human-readable modality/stimulus names from the .mat file if present
    def _make_list(obj):
        if obj is None:
            return None
        try:
            arr = np.asarray(obj)
        except Exception:
            return None
        # flatten and coerce to str list when possible
        try:
            flattened = arr.tolist()
        except Exception:
            return None
        # Ensure list of strings
        out = []
        if isinstance(flattened, (list, tuple)):
            for v in flattened:
                if v is None:
                    out.append(None)
                else:
                    out.append(str(v))
            return out
        return None

    modality_map = None
    stimulus_map = None
    # common variable names used in matlab exports / dataset descriptions
    cand_mod_keys = ['modality_names', 'mode_names', 'modes', 'Mode', 'ModeNames', 'modality']
    cand_stim_keys = ['stimulus_names', 'stimulus_name', 'Stimulus_names', 'StimulusNames', 'stimuli', 'Stimuli', 'words', 'WordList']
    for k in cand_mod_keys:
        if k in eeg_mat:
            modality_map = _make_list(eeg_mat[k])
            break
    for k in cand_stim_keys:
        if k in eeg_mat:
            stimulus_map = _make_list(eeg_mat[k])
            break

    def map_code(mapping, code, prefix):
        if mapping is None:
            return None
        try:
            code_i = int(code)
        except Exception:
            return None
        # try zero-based then one-based indexing
        if 0 <= code_i < len(mapping):
            return mapping[code_i]
        if 1 <= code_i <= len(mapping):
            return mapping[code_i - 1]
        return None

    # eeg_arr expected shape (n_rows, eegTotalColumns)
    n_eeg_rows = eeg_arr.shape[0]

    # mapping from eeg row index -> trial_id
    eeg_row_to_trial = {}
    subj_id = get_or_create_subject(conn, subject_name)

    cur = conn.cursor()
    # First pass: insert trial rows only (map row_idx -> trial_id)
    for row_idx in range(n_eeg_rows):
        row = eeg_arr[row_idx]
        mode_code = int(row[EEG_SIGNAL_COLUMNS])
        stim_code = int(row[EEG_SIGNAL_COLUMNS + 1])

        # determine modality name (prefer .mat mapping; otherwise accept common codes)
        mod_name_from_mat = map_code(modality_map, mode_code, 'mode')
        if mod_name_from_mat:
            modality_name = mod_name_from_mat
        else:
            # support either 0/1 or 1/2 conventions, map to allowed names
            if mode_code == 0 or mode_code == 1:
                modality_name = "Imagined" if mode_code == 0 else "Pronounced"
            elif mode_code == 1 or mode_code == 2:
                modality_name = "Imagined" if mode_code == 1 else "Pronounced"
            else:
                raise ValueError(f"Unrecognized modality code {mode_code} for subject {subject_name} row {row_idx}")

        if modality_name not in ALLOWED_MODALITIES:
            raise ValueError(f"Modality '{modality_name}' is not allowed (subject {subject_name} row {row_idx})")

        stim_name_from_mat = map_code(stimulus_map, stim_code, 'stimulus')
        if stim_name_from_mat:
            stimulus_name = stim_name_from_mat
        else:
            try:
                stim_int = int(stim_code)
            except Exception:
                raise ValueError(f"Invalid stimulus code {stim_code} (subject {subject_name} row {row_idx})")
            if stim_int in STIMULUS_CODE_MAP:
                stimulus_name = STIMULUS_CODE_MAP[stim_int]
            else:
                raise ValueError(f"Unrecognized stimulus code {stim_code} for subject {subject_name} row {row_idx}")

        if stimulus_name not in ALLOWED_STIMULI:
            raise ValueError(f"Stimulus '{stimulus_name}' is not allowed (subject {subject_name} row {row_idx})")

        modality_id = get_or_create_modality(conn, modality_name)
        stimulus_id = get_or_create_stimulus(conn, stimulus_name)

        # insert trial row only (eeg samples and blink will be inserted in second pass)
        cur.execute(
            "INSERT INTO trial(subject_id, modality_id, stimulus_id, eeg_sample_count, audio_sample_count, original_row) VALUES (?, ?, ?, ?, ?, ?)",
            (subj_id, modality_id, stimulus_id, EEG_SAMPLES_PER_CHANNEL, None, row_idx)
        )
        trial_id = cur.lastrowid
        eeg_row_to_trial[row_idx] = trial_id
    conn.commit()

    # Second pass: insert eeg_samples rows using existing trial ids
    for row_idx in range(n_eeg_rows):
        row = eeg_arr[row_idx]
        signal = row[:EEG_SIGNAL_COLUMNS]
        blink_flag = int(row[EEG_SIGNAL_COLUMNS + 2])
        trial_id = eeg_row_to_trial[row_idx]

        # extract per-channel signals — channels concatenated sequentially
        channel_blobs = {}
        for ch_idx, ch_name in enumerate(CHANNELS):
            start = ch_idx * EEG_SAMPLES_PER_CHANNEL
            stop = start + EEG_SAMPLES_PER_CHANNEL
            ch_arr = signal[start:stop]
            channel_blobs[ch_name] = serialize_array(ch_arr)

        cur.execute(
            "INSERT INTO eeg_samples(trial_id, F3, F4, C3, C4, P3, P4, blink) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                trial_id,
                channel_blobs['F3'],
                channel_blobs['F4'],
                channel_blobs['C3'],
                channel_blobs['C4'],
                channel_blobs['P3'],
                channel_blobs['P4'],
                blink_flag,
            ),
        )
    conn.commit()

    # Process Audio if present
    if audio_file and os.path.exists(audio_file):
        audio_mat = loadmat(audio_file, squeeze_me=True)
        audio_arr = audio_mat.get('Audio')
        if audio_arr is None:
            print(f"Warning: Audio variable not found in {audio_file}")
            return n_eeg_rows
        n_audio_rows = audio_arr.shape[0]
        for arow_idx in range(n_audio_rows):
            arow = audio_arr[arow_idx]
            samples = arow[:AUDIO_SAMPLES]
            stim_code = int(arow[AUDIO_SAMPLES])
            eeg_ref = int(arow[AUDIO_SAMPLES + 1])

            # prefer stimulus mapping from EEG .mat if available
            stim_name_from_mat = map_code(stimulus_map, stim_code, 'stimulus')
            if stim_name_from_mat:
                stimulus_name = stim_name_from_mat
            else:
                try:
                    stim_int = int(stim_code)
                except Exception:
                    raise ValueError(f"Invalid audio stimulus code {stim_code} (subject {subject_name} audio row {arow_idx})")
                if stim_int in STIMULUS_CODE_MAP:
                    stimulus_name = STIMULUS_CODE_MAP[stim_int]
                else:
                    raise ValueError(f"Unrecognized audio stimulus code {stim_code} (subject {subject_name} audio row {arow_idx})")

            if stimulus_name not in ALLOWED_STIMULI:
                raise ValueError(f"Stimulus '{stimulus_name}' is not allowed (subject {subject_name} audio row {arow_idx})")

            stimulus_id = get_or_create_stimulus(conn, stimulus_name)

            # find corresponding EEG trial if referenced
            trial_id = eeg_row_to_trial.get(eeg_ref)
            if trial_id is None:
                # create a new trial row linked to subject
                cur.execute(
                    "INSERT INTO trial(subject_id, modality_id, stimulus_id, eeg_sample_count, audio_sample_count, original_row) VALUES (?, ?, ?, ?, ?, ?)",
                    (subj_id, None, stimulus_id, None, AUDIO_SAMPLES, None),
                )
                trial_id = cur.lastrowid
            else:
                # update audio_sample_count on existing trial
                cur.execute("UPDATE trial SET audio_sample_count=? WHERE id=?", (AUDIO_SAMPLES, trial_id))

            # insert audio blob and record audio row index
            cur.execute(
                "INSERT INTO audio_samples(trial_id, samples, audio_row) VALUES (?, ?, ?)",
                (trial_id, serialize_array(samples), arow_idx),
            )
        conn.commit()

    return n_eeg_rows


def main():
    if len(sys.argv) != 3:
        print("Usage: python scripts/mat_to_sqlite_redo.py <dataset_root> <output_db>")
        sys.exit(1)
    root, db_path = sys.argv[1:3]
    conn = sqlite3.connect(db_path)
    create_schema(conn)

    subject_dirs = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)) and d.upper().startswith('S')])
    # Prepopulate subjects, modalities and stimulus tables to enforce ordering
    prepopulate_modalities_and_stimuli(conn, subject_dirs)
    total_subjects = len(subject_dirs)
    processed = 0
    for subj in tqdm(subject_dirs, desc='Subjects'):
        subj_path = os.path.join(root, subj)
        try:
            rows = process_subject(conn, subj_path, subj)
            processed += 1
        except Exception as e:
            print(f"Error processing {subj}: {e}")
    conn.close()
    print(f"Finished. Processed {processed}/{total_subjects} subjects.")


if __name__ == '__main__':
    main()
