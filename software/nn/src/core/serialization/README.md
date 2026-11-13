# serialization

Saving and loading models: `NnSaver` (state dicts) and `NetworkSerializer` (`.npz` network files), plus the report/state I/O helpers in `include/io/`.

**Full documentation:** [`.wiki/Core/Saver.md`](../../../.wiki/Core/Saver.md). This README only says where things are and how to change them.

| What | Where |
|---|---|
| Public headers | [`include/serialization`](../../../include/serialization), [`include/io`](../../../include/io) |
| Implementation (`.cpp`) | none — header-only; this folder holds the tests and the CMake target |
| CMake library target | `serialization` |
| Test executables | `core_gtest` |

## Header map

- `include/io/`: `ReportIO.hpp`, `StateIO.hpp`
- `include/serialization/`: `NetworkSerializer.hpp`, `NnSaver.hpp`

## Changing it

- A new header in `include/...` needs nothing else unless it has a `.cpp`; a new `.cpp` must be added to this folder's `CMakeLists.txt`.
- A new test goes in `tests/`, named `*_gtest.cpp`, and is registered in `tests/CMakeLists.txt` (`add_executable` + `target_link_libraries(... google_test ...)` + `gtest_discover_tests(<exe>)`; copy a neighbour) ([Build System](../../../.wiki/Guides/Build-System.md#running-tests)).
- Build and run (from `software/nn`):

  ```bash
  cmake --build --preset=max-performance --target core_gtest
  ctest --test-dir out/build/max-performance -R 'NetworkSerializerTest' --output-on-failure
  ```

  `ctest -R` matches the GoogleTest **suite** name, not the executable: `-R core_gtest` silently matches nothing.

  Suites here: `NetworkSerializerTest`, `StateIO`.
- After a behaviour change, update [`.wiki/Core/Saver.md`](../../../.wiki/Core/Saver.md) (and `.wiki/References.md` if a paper is involved).
