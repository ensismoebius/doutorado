# utility

Small shared helpers: batching, signal preprocessing (`read_csv_signal`, `zscore_inplace`), synthetic data, file and string utilities, logging glue.

**Full documentation:** [`.wiki/Core/Logging.md`](../../../.wiki/Core/Logging.md). This README only says where things are and how to change them.

| What | Where |
|---|---|
| Public headers | [`include/utility`](../../../include/utility), [`include/logging`](../../../include/logging) |
| Implementation (`.cpp`) | 8 file(s) in this folder |
| CMake library target | `util` |
| Test executables | `util_gtest` |

## Header map

- `include/logging/`: `Logger.hpp`, `StreamRedirector.hpp`
- `include/utility/`: `AudioMeanStdNormalize.hpp`, `BandpassNotchFilter.hpp`, `BufferPool.hpp`, `Compose.hpp`, `EEGWindowZScore.hpp`, `FusedModalityTransform.hpp`, `GaussianNoise.hpp`, `GradClip.hpp`, `HighPerfSpscQueue.hpp`, `ITransform.hpp`, `Normalization.hpp`, `ParameterCount.hpp`, `RandomCrop.hpp`, `RandomIndexCrop.hpp`, `Resample.hpp`, `SeedDerivation.hpp`, `SignalPreprocessing.hpp`, `SpscRingBuffer.hpp`, `Transforms.hpp`, `WindowZScore.hpp`, `Windowing.hpp`, `XtensorParallel.hpp`, `batching.hpp`, `comparison.hpp`, `parallel_batching.hpp`, `path_expand.hpp`, `printTensor.hpp`, `progress.hpp`, `reset.hpp`, `synthetic_spike_data.hpp`, `vectorizationCheck.hpp`

## Changing it

- A new header in `include/...` needs nothing else unless it has a `.cpp`; a new `.cpp` must be added to this folder's `CMakeLists.txt`.
- A new test goes in `tests/`, named `*_gtest.cpp`, and is registered in `tests/CMakeLists.txt` (`add_executable` + `target_link_libraries(... google_test ...)` + `gtest_discover_tests(<exe>)`; copy a neighbour) ([Build System](../../../.wiki/Guides/Build-System.md#running-tests)).
- Build and run (from `software/nn`):

  ```bash
  cmake --build --preset=max-performance --target util_gtest
  ctest --test-dir out/build/max-performance -R 'BandpassNotchFilter' --output-on-failure
  ```

  `ctest -R` matches the GoogleTest **suite** name, not the executable: `-R util_gtest` silently matches nothing.

  Suites here: `BandpassNotchFilter`, `PrintProgress`, `ResampleBandlimited`, `EEGWindowZScore`, `AudioMeanStdNormalize`, `Compose`, `FusedModalityTransform`, `RandomCrop`, `RandomIndexCrop`, `WindowZScore`, `GaussianNoise`, `UtilTest` (and 11 more).
- After a behaviour change, update [`.wiki/Core/Logging.md`](../../../.wiki/Core/Logging.md) (and `.wiki/References.md` if a paper is involved).
