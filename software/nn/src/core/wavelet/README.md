# wavelet

Discrete wavelet transform and wavelet packet transform (DWT, DTWPT), checked against PyWavelets by `pywt_parity_gtest`.

**Full documentation:** [`.wiki/Core/Wavelet.md`](../../../.wiki/Core/Wavelet.md). This README only says where things are and how to change them.

| What | Where |
|---|---|
| Public headers | [`include/wavelet`](../../../include/wavelet) |
| Implementation (`.cpp`) | 2 file(s) in this folder |
| CMake library target | `wavelet` |
| Test executables | `pywt_parity_gtest`, `wavelet_gtest` |

## Header map

- `include/wavelet/`: `Types.hpp`, `WaveletCoefficients.hpp`, `WaveletTraits.hpp`, `WaveletTransformResults.hpp`, `waveletOperations.hpp`

## Changing it

- A new header in `include/...` needs nothing else unless it has a `.cpp`; a new `.cpp` must be added to this folder's `CMakeLists.txt`.
- A new test goes in `tests/`, named `*_gtest.cpp`, and is registered in `tests/CMakeLists.txt` (`add_executable` + `target_link_libraries(... google_test ...)` + `gtest_discover_tests(<exe>)`; copy a neighbour) ([Build System](../../../.wiki/Guides/Build-System.md#running-tests)).
- Build and run (from `software/nn`):

  ```bash
  cmake --build --preset=max-performance --target pywt_parity_gtest
  ctest --test-dir out/build/max-performance -R 'PyWtParity' --output-on-failure
  ```

  `ctest -R` matches the GoogleTest **suite** name, not the executable: `-R pywt_parity_gtest` silently matches nothing.

  Suites here: `PyWtParity`, `WaveletTypesTest`, `WaveletOperationsTest`, `WaveletTransformResultsTest`.
- After a behaviour change, update [`.wiki/Core/Wavelet.md`](../../../.wiki/Core/Wavelet.md) (and `.wiki/References.md` if a paper is involved).
