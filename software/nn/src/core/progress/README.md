# progress

Terminal progress bars (`ProgressBar`, `ProgressManager`) shared by the experiment runners.

| What | Where |
|---|---|
| Public headers | [`include/progress`](../../../include/progress) |
| Implementation (`.cpp`) | 1 file(s) in this folder |
| CMake library target | `nn_progress` |
| Test executables | `progress_gtest` |

## Header map

- `include/progress/`: `ProgressBar.hpp`, `ProgressManager.hpp`

## Changing it

- A new header in `include/...` needs nothing else unless it has a `.cpp`; a new `.cpp` must be added to this folder's `CMakeLists.txt`.
- A new test goes in `tests/`, named `*_gtest.cpp`, and is registered in `tests/CMakeLists.txt` (`add_executable` + `target_link_libraries(... google_test ...)` + `gtest_discover_tests(<exe>)`; copy a neighbour) ([Build System](../../../.wiki/Guides/Build-System.md#running-tests)).
- Build and run (from `software/nn`):

  ```bash
  cmake --build --preset=max-performance --target progress_gtest
  ctest --test-dir out/build/max-performance -R 'ProgressTest' --output-on-failure
  ```

  `ctest -R` matches the GoogleTest **suite** name, not the executable: `-R progress_gtest` silently matches nothing.

  Suites here: `ProgressTest`, `LoggerTest`.
