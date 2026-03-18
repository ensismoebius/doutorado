# 10.1117 dataset module

This folder exposes the public headers and documents the source layout for the 10.1117 dataset support.

Layout (public headers)
- `loaders/` — `AudioLoader.h`, `EEGLoader.h`, `AudioData.h`, `EEGData.h`
- `schema/` — `METADATA.hpp`, `NAMES.hpp`, `SchemaIndexing.hpp`, `SubjectDiscovery.hpp`
- `codec/` — `InputModeCodec.hpp`, `BatchTargetFormatter.hpp`
- `protocol/` — `Protocol101117Dataset.hpp`, `SamplePacking.hpp`, `SynchronizedBatchAssembler.hpp`
- `windowing/` — windowing dataset headers (`AudioWindowDataset.hpp`, `EEGWindowDataset.hpp`, `FusedWindowDataset.hpp`)

Notes
- Sources are under `src/core/dataLoaders/10.1117/` and mirror the header subfolders (`loaders/`, `schema/`, `codec/`, `protocol/`, `windowing/`).
- When adding new files, update `src/core/dataLoaders/10.1117/CMakeLists.txt` to include the new source paths.
- Public include paths follow: `#include "nn/dataLoaders/10.1117/<subfolder>/<HeaderName>.hpp"`.

Quick build & test
```bash
cmake -S . -B build
cmake --build build -j$(nproc)
ctest --test-dir build -j4 --output-on-failure
```

Tests for the 10.1117 module live in `src/core/dataLoaders/10.1117/tests/` and should pass after changes.
