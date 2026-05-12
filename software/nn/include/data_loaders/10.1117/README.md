# 10.1117 dataset module

This folder exposes the public headers and documents the source layout for the 10.1117 dataset support.

Layout (public headers)
- `loaders/` — `AudioLoader.h`, `EEGLoader.h`, `AudioData.h`, `EEGData.h`
- `schema/` — `METADATA.hpp`, `NAMES.hpp`, `SchemaIndexing.hpp`, `SubjectDiscovery.hpp`
- `codec/` — `InputModeCodec.hpp`, `BatchTargetFormatter.hpp`
- `protocol/` — `Protocol101117Dataset.hpp`, `SamplePacking.hpp`, `SynchronizedBatchAssembler.hpp`
- `windowing/` — windowing dataset headers (`AudioWindowDataset.hpp`, `EEGWindowDataset.hpp`, `FusedWindowDataset.hpp`)

Recent reorganization (2026-04-08)
- Public headers for protocol and windowing were moved into a `datasets/` subfolder
	to improve discoverability and to group dataset variants and dataset-specific
	utilities together. New public header paths are now:

	- `include/nn/data_loaders/10.1117/datasets/raw/Protocol101117Dataset.hpp`
	- `include/nn/data_loaders/10.1117/datasets/raw/SamplePacking.hpp`
	- `include/nn/data_loaders/10.1117/datasets/raw/SynchronizedBatchAssembler.hpp`
	- `include/nn/data_loaders/10.1117/datasets/windowed/AudioWindowDataset.hpp`
	- `include/nn/data_loaders/10.1117/datasets/windowed/EEGWindowDataset.hpp`
	- `include/nn/data_loaders/10.1117/datasets/windowed/FusedWindowDataset.hpp`

	Additionally, dataset printers were extracted into dedicated headers:

	- `include/nn/data_loaders/10.1117/datasets/raw/Protocol101117DatasetPrinter.hpp`
	- `include/nn/data_loaders/10.1117/datasets/windowed/WindowingDatasetPrinter.hpp`

Migration notes
- The codebase has been updated to reference the new `datasets/` paths. Old
	forwarding headers were removed and the header contents were moved in-place.
	If you maintain external code that included the old headers, update includes
	To the new `nn/data_loaders/10.1117/datasets/...` paths. Example replacements:

	- `nn/data_loaders/10.1117/protocol/Protocol101117Dataset.hpp` →
		`nn/data_loaders/10.1117/datasets/raw/Protocol101117Dataset.hpp`
	- `nn/data_loaders/10.1117/windowing/AudioWindowDataset.hpp` →
		`nn/data_loaders/10.1117/datasets/windowed/AudioWindowDataset.hpp`

API notes
- Datasets expose a polymorphic printing API via `Dataset::print(IDatasetPrinter&)`.
	Use the new printer types above (or `IDatasetPrinter`/`WindowedDatasetPrinter`) to
	format dataset summaries in tools and experiments.

Notes
- Sources are under `src/core/data_loaders/10.1117/` and mirror the header subfolders (`loaders/`, `schema/`, `codec/`, `protocol/`, `windowing/`).
- When adding new files, update `src/core/data_loaders/10.1117/CMakeLists.txt` to include the new source paths.
- Public include paths follow: `#include "nn/data_loaders/10.1117/<subfolder>/<HeaderName>.hpp"`.

Quick build & test
```bash
cmake -S . -B build
cmake --build build -j$(nproc)
ctest --test-dir build -j4 --output-on-failure
```

Tests for the 10.1117 module live in `src/core/data_loaders/10.1117/tests/` and should pass after changes.
