# paraconsistent

Paraconsistent annotated logic: the certainty and contradiction degrees and `d_penalized`, the class-separability score the thesis Phase 00 uses to rank feature extractors without a classifier.

**Full documentation:** [`.wiki/Core/Paraconsistent.md`](../../../.wiki/Core/Paraconsistent.md). This README only says where things are and how to change them.

| What | Where |
|---|---|
| Public headers | [`include/paraconsistent`](../../../include/paraconsistent) |
| Implementation (`.cpp`) | 1 file(s) in this folder |
| CMake library target | `paraconsistent` |
| Test executables | `paraconsistent_gtest` |

## Header map

- `include/paraconsistent/`: `paraconsistent.hpp`

## Changing it

- A new header in `include/...` needs nothing else unless it has a `.cpp`; a new `.cpp` must be added to this folder's `CMakeLists.txt`.
- A new test goes in `tests/`, named `*_gtest.cpp`, and is registered in `tests/CMakeLists.txt` (`add_executable` + `target_link_libraries(... google_test ...)` + `gtest_discover_tests(<exe>)`; copy a neighbour) ([Build System](../../../.wiki/Guides/Build-System.md#running-tests)).
- Build and run (from `software/nn`):

  ```bash
  cmake --build --preset=max-performance --target paraconsistent_gtest
  ctest --test-dir out/build/max-performance -R 'paraconsistentTest' --output-on-failure
  ```

  `ctest -R` matches the GoogleTest **suite** name, not the executable: `-R paraconsistent_gtest` silently matches nothing.

  Suites here: `paraconsistentTest`.
- After a behaviour change, update [`.wiki/Core/Paraconsistent.md`](../../../.wiki/Core/Paraconsistent.md) (and `.wiki/References.md` if a paper is involved).
