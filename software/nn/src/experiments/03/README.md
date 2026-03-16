# experiment04 — Dataset loading & demo runner

Overview
--------
`experiment04` is a small demonstration executable that implements a PyTorch-style
data pipeline for the 10.1117 imagined-speech EEG+Audio dataset used in this
repository. Its purpose is to show how `Protocol101117Dataset`, `DataLoader`, and
the prefetcher interact, to exercise I/O, and to provide a convenient demo for
downstream model development.

The pipeline implemented by `experiment04`:

- Subject directories (S01, S02, ...) discovered by filename regex
- `Protocol101117Dataset` (synchronized audio + EEG samples)
- `DataLoader` (batching + sampling)
- `BatchPrefetcher` (producer thread, bounded queue)
- `DemoProbeModel` (dummy model to exercise forward pass)

Key features
------------
- Discover subjects automatically under a dataset root (subject directory names match a regex, default `^S(\\d+)$`).
- Print a concise dataset summary showing per-subject counts and an estimated
  number of audio rows that have corresponding EEG records.
- Configurable batching and prefetching (`--batch-size`, `--max-batches`, `--lookahead`).
- In-place single-line progress bar with correct handling when runs are capped
  by `--max-batches`.

Quick start — build and run
--------------------------
From the project root (where CMakeLists.txt lives):

```bash
mkdir -p build && cd build
cmake -S .. -B . -DCMAKE_BUILD_TYPE=Debug
cmake --build . -- -j$(nproc)

# Run experiment04 with defaults
./src/experiments/03/experiment04

# Example: small smoke run with lookahead 4, only 5 batches
./src/experiments/03/experiment04 --dataset-root /path/to/dataset --max-batches 5 --lookahead 4
```

Command-line options (high-level)
---------------------------------
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
- `experiment04` is intended as a demonstration and integration test. You can
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
