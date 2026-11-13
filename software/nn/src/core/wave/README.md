# wave

WAV input/output and audio feature helpers (FFT, filter banks, LFCC).

**Full documentation:** [`.wiki/Core/Wave.md`](../../../.wiki/Core/Wave.md). This README only says where things are and how to change them.

| What | Where |
|---|---|
| Public headers | [`include/wave`](../../../include/wave) |
| Implementation (`.cpp`) | 6 file(s) in this folder |
| CMake library target | `waveCoreLib` |
| Test executables | `create_wav_demo`, `wave_gtest` |

## Header map

- `include/wave/`: `Wav.hpp`, `audioFeatureExtraction.hpp`, `audioTypes.hpp`, `filter_operations.hpp`, `lfcc_pipeline_utils.hpp`, `signal_operations.hpp`

## Changing it

- A new header in `include/...` needs nothing else unless it has a `.cpp`; a new `.cpp` must be added to this folder's `CMakeLists.txt`.
- A new test goes in `tests/`, named `*_gtest.cpp`, and is registered in `tests/CMakeLists.txt` (`add_executable` + `target_link_libraries(... google_test ...)` + `gtest_discover_tests(<exe>)`; copy a neighbour) ([Build System](../../../.wiki/Guides/Build-System.md#running-tests)).
- Build and run (from `software/nn`):

  ```bash
  cmake --build --preset=max-performance --target wave_gtest
  ctest --test-dir out/build/max-performance -R 'SimpleSignalOperationsTest' --output-on-failure
  ```

  `ctest -R` matches the GoogleTest **suite** name, not the executable: `-R wave_gtest` silently matches nothing.

  Suites here: `SimpleSignalOperationsTest`, `FilterOperationsTest`, `WavFileTest`, `AudioFeatureExtractionTest`.
- After a behaviour change, update [`.wiki/Core/Wave.md`](../../../.wiki/Core/Wave.md) (and `.wiki/References.md` if a paper is involved).
