# initializers

Weight initialisers (Xavier/Glorot, He/Kaiming, constant, ...) used by layer constructors.

**Full documentation:** [`.wiki/Core/Initializers.md`](../../../.wiki/Core/Initializers.md). This README only says where things are and how to change them.

| What | Where |
|---|---|
| Public headers | [`include/initializers`](../../../include/initializers) |
| Implementation (`.cpp`) | none — header-only; this folder holds the tests and the CMake target |
| CMake library target | `initializers` |
| Test executables | `initializers_gtest` |

## Header map

- `include/initializers/`: `kaiming_snn.hpp`, `xavier.hpp`

## Changing it

- A new header in `include/...` needs nothing else unless it has a `.cpp`; a new `.cpp` must be added to this folder's `CMakeLists.txt`.
- A new test goes in `tests/`, named `*_gtest.cpp`, and is registered in `tests/CMakeLists.txt` (`add_executable` + `target_link_libraries(... google_test ...)` + `gtest_discover_tests(<exe>)`; copy a neighbour) ([Build System](../../../.wiki/Guides/Build-System.md#running-tests)).
- Build and run (from `software/nn`):

  ```bash
  cmake --build --preset=max-performance --target initializers_gtest
  ctest --test-dir out/build/max-performance -R 'InitializerTest' --output-on-failure
  ```

  `ctest -R` matches the GoogleTest **suite** name, not the executable: `-R initializers_gtest` silently matches nothing.

  Suites here: `InitializerTest`.
- After a behaviour change, update [`.wiki/Core/Initializers.md`](../../../.wiki/Core/Initializers.md) (and `.wiki/References.md` if a paper is involved).
