# optimizers

Optimisers (Adam, SGD, ...) and `OptimizerFactory`, which builds one from a configuration token such as `"adam"`. Header-only; this folder holds the tests and the CMake target.

**Full documentation:** [`.wiki/Core/Optimizers.md`](../../../.wiki/Core/Optimizers.md). This README only says where things are and how to change them.

| What | Where |
|---|---|
| Public headers | [`include/optimizers`](../../../include/optimizers) |
| Implementation (`.cpp`) | none — header-only; this folder holds the tests and the CMake target |
| CMake library target | `optimizers` |
| Test executables | `optimizer_parity_gtest`, `optimizers_gtest` |

## Header map

- `include/optimizers/`: `Adam.hpp`, `Lion.hpp`, `Optimizer.hpp`, `OptimizerEvalScope.hpp`, `OptimizerFactory.hpp`, `OptimizerFactoryConfig.hpp`, `SGD.hpp`, `SGDMinimal.hpp`, `ScheduleFreeAdamW.hpp`

## Changing it

- A new header in `include/...` needs nothing else unless it has a `.cpp`; a new `.cpp` must be added to this folder's `CMakeLists.txt`.
- A new test goes in `tests/`, named `*_gtest.cpp`, and is registered in `tests/CMakeLists.txt` (`add_executable` + `target_link_libraries(... google_test ...)` + `gtest_discover_tests(<exe>)`; copy a neighbour) ([Build System](../../../.wiki/Guides/Build-System.md#running-tests)).
- Build and run (from `software/nn`):

  ```bash
  cmake --build --preset=max-performance --target optimizer_parity_gtest
  ctest --test-dir out/build/max-performance -R 'OptimizerParity' --output-on-failure
  ```

  `ctest -R` matches the GoogleTest **suite** name, not the executable: `-R optimizer_parity_gtest` silently matches nothing.

  Suites here: `AdamModelState`, `OptimizerParity`, `OptimizerTest`, `OptimizerEdgeCases`, `OptimizerExceptionTest`, `OptimizerFactoryTest`, `OptimizerNumericalEdgeTest`, `OptimizerThreadSafetyTest`, `OptimizerComprehensiveTest`, `SGDNullParamTest`, `SGDMinimalTest`, `OptimizerBaseTest` (and 4 more).
- After a behaviour change, update [`.wiki/Core/Optimizers.md`](../../../.wiki/Core/Optimizers.md) (and `.wiki/References.md` if a paper is involved).
