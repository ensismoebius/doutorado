# linear_algebra

Linear-algebra helpers between `nn::Tensor` and BLAS/LAPACK (Eigen).

**Full documentation:** [`.wiki/Core/LinearAlgebra.md`](../../../.wiki/Core/LinearAlgebra.md). This README only says where things are and how to change them.

| What | Where |
|---|---|
| Public headers | [`include/linear_algebra`](../../../include/linear_algebra) |
| Implementation (`.cpp`) | 1 file(s) in this folder |
| CMake library target | `linear_algebra` |
| Test executables | `linearAlgebra_gtest` |

## Header map

- `include/linear_algebra/`: `linear_algebra.hpp`

## Changing it

- A new header in `include/...` needs nothing else unless it has a `.cpp`; a new `.cpp` must be added to this folder's `CMakeLists.txt`.
- A new test goes in `tests/`, named `*_gtest.cpp`, and is registered in `tests/CMakeLists.txt` (`add_executable` + `target_link_libraries(... google_test ...)` + `gtest_discover_tests(<exe>)`; copy a neighbour) ([Build System](../../../.wiki/Guides/Build-System.md#running-tests)).
- Build and run (from `software/nn`):

  ```bash
  cmake --build --preset=max-performance --target linearAlgebra_gtest
  ctest --test-dir out/build/max-performance -R 'LinearAlgebraTest' --output-on-failure
  ```

  `ctest -R` matches the GoogleTest **suite** name, not the executable: `-R linearAlgebra_gtest` silently matches nothing.

  Suites here: `LinearAlgebraTest`.
- After a behaviour change, update [`.wiki/Core/LinearAlgebra.md`](../../../.wiki/Core/LinearAlgebra.md) (and `.wiki/References.md` if a paper is involved).
