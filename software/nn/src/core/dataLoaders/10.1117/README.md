# 10.1117 Data Loaders Layout

This folder contains the 10.1117 imagined-speech data loading pipeline and related utilities.

## Structure

- `AudioLoader.cpp`, `EEGLoader.cpp`
  - MAT row readers for each modality.
- `Protocol101117Dataset.cpp`
  - Main dataset integration for the baseline protocol.
- `SamplePacking.cpp`, `SynchronizedBatchAssembler.cpp`
  - Packing and batch assembly helpers.
- `InputModeCodec.cpp`, `BatchTargetFormatter.cpp`
  - Label/target conversion helpers.
- `SubjectDiscovery.cpp`
  - Subject file discovery and metadata.
- `windowing/`
  - Sliding-window datasets for multimodal strategies:
    - `EEGWindowDataset`
    - `AudioWindowDataset`
    - `FusedWindowDataset`
- `tests/`
  - Unit tests grouped by concern (`loaders`, `schema`, `protocol`, `codec`, `windowing`, `fuzz`).

## Notes

- Keep schema-specific behavior here (10.1117 ownership).
- Keep generic window math in `include/nn/windowing`.
- Avoid introducing 10.1117-specific code in higher-level generic dataLoader modules.
