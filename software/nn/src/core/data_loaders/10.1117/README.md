# 10.1117 — the Coretto imagined-speech base

Loader for the base of Coretto *et al.* (SPIE 10160, DOI 10.1117/12.2255697): 15 Spanish
speakers, each trial a pronounced or imagined word or vowel, with simultaneous 6-channel EEG
(1024 Hz) and audio (44.1 kHz). The folder is named after the DOI prefix so a second base never
collides with it. The thesis experiment reads this base from the SQLite file made by
`scripts/data/import_mat_dataset_to_sqlite.py` (see
[Data Bases](../../../../.wiki/Guides/Data-Bases.md)). BIOMEX-DB, the thesis's second base, has
its own paired schema and is read by the thesis experiment itself (`dataset.source =
"biomex2021"`, in `src/experiments/thesis/lib/src/ThesisDataset.cpp`), not by this loader.

| What | Where |
|---|---|
| Public headers | [`include/data_loaders/10.1117/`](../../../../include/data_loaders/10.1117/) (same subfolders) |
| Sources | this folder |
| CMake targets | `dataLoaders_10_1117`, `dataLoaders_10_1117_windowing` |
| Tests | `tests/` → `dataLoaders_10_1117_gtest` (+ `tests/windowing/`, `tests/fuzz/`) |

## Layout

| Subfolder | Holds | Files |
|---|---|---|
| `loaders/` | readers of one trial's rows, per modality | `AudioLoader`, `EEGLoader` (+ `AudioData`, `EEGData` headers) |
| `schema/` | what the base contains and how to find it | `SqliteDataset`, `SubjectDiscovery`, `Metadata.hpp`, `Names.hpp`, `SchemaIndexing.hpp` |
| `codec/` | label/target conversion | `InputModeCodec`, `BatchTargetFormatter` |
| `datasets/raw/` | whole-trial dataset and batch assembly | `Dataset101117`, `SamplePacking`, `SynchronizedBatchAssembler`, `Dataset101117Printer`, `RowRequest.hpp` |
| `datasets/windowed/` | sliding-window views of the trials | `AudioWindowDataset`, `EEGWindowDataset`, `FusedWindowDataset`, `WindowingDatasetPrinter` |

Generic window arithmetic (`WindowSpec`, `WindowingEngine`) lives in
[`include/windowing/`](../../../../include/windowing/), not here: keep only what is specific to
this base in this folder.

## Changing it

- New source: add it to the matching subfolder **and** to `CMakeLists.txt` here; its header goes
  in the same subfolder under `include/data_loaders/10.1117/`.
- New test: put it in the matching `tests/<subfolder>/` and add it to `tests/CMakeLists.txt`.
  Run it by suite name, e.g.
  `ctest --test-dir out/build/max-performance -R 'BatchTargetFormatterTest' --output-on-failure`.
- The fuzz targets in `tests/fuzz/` are described in their own README.
