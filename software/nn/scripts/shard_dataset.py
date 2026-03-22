#!/usr/bin/env python3
"""
Simple dataset sharder: split per-subject Audio/EEG .npz files into small shard .npz files
and emit a shard index JSON in the subject directory.

Usage: python3 scripts/shard_dataset.py <dataset_root> --rows-per-shard 8
"""
import argparse
import json
import os
import numpy as np


def shard_array(arr, rows_per_shard):
    nrows = arr.shape[0]
    for start in range(0, nrows, rows_per_shard):
        end = min(start + rows_per_shard, nrows)
        yield start, end - start, arr[start:end]


def process_subject(subject_dir, rows_per_shard):
    # Expect existing SXX_Audio.npz and SXX_EEG.npz (or .mat converted beforehand)
    audio_path = None
    eeg_path = None
    for fname in os.listdir(subject_dir):
        if fname.endswith('_Audio.npz'):
            audio_path = os.path.join(subject_dir, fname)
        if fname.endswith('_EEG.npz'):
            eeg_path = os.path.join(subject_dir, fname)

    index = {'audio': [], 'eeg': []}

    if audio_path and os.path.exists(audio_path):
        data = np.load(audio_path, allow_pickle=True)
        arr = data['Audio']
        for i, (start, count, block) in enumerate(shard_array(arr, rows_per_shard)):
            shard_name = f"{os.path.splitext(os.path.basename(audio_path))[0]}_shard_{i:04d}.npz"
            shard_path = os.path.join(subject_dir, shard_name)
            np.savez_compressed(shard_path, Audio=block)
            index['audio'].append({'file': shard_name, 'start': start, 'count': int(count)})

    if eeg_path and os.path.exists(eeg_path):
        data = np.load(eeg_path, allow_pickle=True)
        arr = data['EEG']
        for i, (start, count, block) in enumerate(shard_array(arr, rows_per_shard)):
            shard_name = f"{os.path.splitext(os.path.basename(eeg_path))[0]}_shard_{i:04d}.npz"
            shard_path = os.path.join(subject_dir, shard_name)
            np.savez_compressed(shard_path, EEG=block)
            index['eeg'].append({'file': shard_name, 'start': start, 'count': int(count)})

    # write shard index
    if index['audio'] or index['eeg']:
        idx_path = os.path.join(subject_dir, os.path.basename(subject_dir) + '_shards.json')
        with open(idx_path, 'w') as f:
            json.dump(index, f)
        print('Wrote', idx_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_root')
    parser.add_argument('--rows-per-shard', type=int, default=8)
    args = parser.parse_args()

    for entry in os.listdir(args.dataset_root):
        subdir = os.path.join(args.dataset_root, entry)
        if os.path.isdir(subdir):
            print('Processing', subdir)
            process_subject(subdir, args.rows_per_shard)


if __name__ == '__main__':
    main()
