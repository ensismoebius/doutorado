# experiment03 — Autoencoder training runner

Overview
--------
`experiment03` is a training runner for ANN/SNN autoencoders over the 10.1117
imagined-speech EEG+Audio dataset variants. It supports profile-based runtime
configuration and writes a per-run JSON summary.

The pipeline implemented by `experiment03`:

- Subject directories discovered by regex
- Dataset selection: protocol, eeg-window, audio-window, fused-window
- Autoencoder selection: ANN and SNN variants per dataset
- `DataLoader` + `BatchPrefetcher` training pipeline
- Profile loading from `src/experiments/03/profiles/*.json`
- Result summary output to `src/experiments/03/results/*.json`

Key features
------------
- Discover subjects automatically under a dataset root (subject directory names match a regex, default `^S(\\d+)$`).
- Print a concise dataset summary showing per-subject counts and an estimated
  number of audio rows that have corresponding EEG records.
- Configurable batching and prefetching (`--batch-size`, `--max-batches`, `--lookahead`).
- In-place single-line progress bar with correct handling when runs are capped
  by `--max-batches`.

Profiles and results
--------------------
- Profiles: `src/experiments/03/profiles/<name>.json`
- Results: `src/experiments/03/results/<timestamp>_<profile>.json`

Result fields include:
- `profile`, `dataset_type`, `autoencoder_type`
- `exit_code`, `total_samples`, `processed_samples`, `seen_batches`
- `epoch_mean_losses`
- `error`

Quick start — build and run
--------------------------
From the project root (where CMakeLists.txt lives):

```bash
mkdir -p build && cd build
cmake -S .. -B . -DCMAKE_BUILD_TYPE=Debug
cmake --build . -- -j$(nproc)

# Run experiment03 with defaults
./src/experiments/03/experiment03

# Example: lightweight fused ANN smoke run
./src/experiments/03/experiment03 --profile fused-window-ann-lightweight --epochs 1 --max-batches 2
```

Command-line options (high-level)
---------------------------------
- `--profile NAME` — profile JSON stem (for example `default`, `lightweight`, `fused-window-snn-lightweight`).
- `--dataset-root PATH` — dataset root folder containing subject dirs `S01/ S02/...`.
- `--subject-regex PATTERN` — regex used to select subject directories (default `^S(\\d+)$`).
- `--batch-size N` — batch size used by `DataLoader` (default 5).
- `--max-batches N` — cap number of produced batches for demo runs (default 1000).
- `--lookahead N` — prefetch queue depth (default 4). Larger values increase I/O overlap but may interact with I/O thread-safety of underlying MAT readers in some setups; the implementation uses a single producer thread to avoid concurrent MAT reads.
- `--shuffle/--no-shuffle`, `--seed` — sampling options.
- `--input-mode` — one of `Concatenated`, `EegOnly`, or `AudioOnly` (controls what `get_item()` returns).

Data format and conventions
---------------------------
This project expects the 10.1117 dataset layout (per-subject directories):

    dataset_root/
      S01/
        S01_EEG.mat
        S01_Audio.mat
      S02/
        S02_EEG.mat
        S02_Audio.mat
      ...

- Each `SXX_EEG.mat` is expected to contain a matrix (rows = trials / captures,
  columns = EEG signal samples + label columns). The repository provides utilities
  and a schema describing exact column positions; the code validates expected
  column counts at runtime.
- Each `SXX_Audio.mat` contains audio rows as long vector samples plus label
  columns (including an EEG index that maps this audio row to the corresponding
  EEG row in the subject's EEG MAT).

Important dataset specifics
---------------------------
- The dataset code builds a `prefix_audio_row_offsets` table to map global
  dataset indices to (subject, audio_row) pairs. `dataset.size()` returns the
  total number of synchronized audio rows across all discovered subjects.
- For performance and to avoid unsafe concurrent MAT I/O, the prefetcher uses a
  single producer thread which reads dataset rows sequentially and pushes
  `Batch` objects into a bounded queue. If you see `Stimulus mismatch` errors or
  MAT-related crashes when using high lookahead values, reduce `--lookahead` to
  1 while debugging file integrity.
- The `printDatasetSummary` helper prints per-subject counts and an estimated
  `AudioWithEEG` count (a fast conservative estimate equal to
  `min(audio_rows, eeg_rows)` per subject). Computing the exact count requires
  reading every audio MAT row and is therefore slower; the helper uses the
  estimate by default for speed.

Progress, logging and validation
-------------------------------
- An in-place progress bar is printed to the console and correctly shows 100%
  when `--max-batches` truncates the run early (it computes an effective
  denominator).
- If the run produces zero batches, the program prints a helpful message and
  exits; check `dataset_root`, subject naming, and MAT file row counts.

Troubleshooting
---------------
- Stimulus mismatch errors: usually indicate misaligned audio/EEG labels or
  corrupted MAT files. Run the dataset discovery and small `DataLoader` runs
  with `--max-batches 1 --lookahead 1` to isolate the failing subject.
- Memory/ASAN leaks during concurrent access: ensure you are running the
  single-producer prefetcher (default) and try reducing `--lookahead`.

Extending or reusing
--------------------
- `experiment03` is intended as a runnable training scaffold. You can
  reuse `Protocol101117Dataset` and `BatchPrefetcher` from other binaries to
  build full training or evaluation pipelines.
- The repo contains unit tests for dataset modes and loader utilities under
  `src/core/dataLoaders/10.1117/tests` — run them with `ctest` from the build
  directory.

Contact / authorship
--------------------
This README was generated as part of the experiment utilities in this repo.
If you need the exact MAT variable schema or column indices, see the
`nn/dataLoaders/10.1117` headers and the `ImaginedSpeechSchema_10_1117` object
used throughout the code.
