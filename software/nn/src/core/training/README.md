# training

`Trainer`: the generic autoencoder and supervised training loops, with callbacks (`ITrainingCallback`) for progress and early stopping. Header-only.

**Full documentation:** [`.wiki/Core/Training.md`](../../../.wiki/Core/Training.md). This README only says where things are and how to change them.

| What | Where |
|---|---|
| Public headers | [`include/training`](../../../include/training) |
| Headers kept in this folder | `EpochResult.hpp`, `Trainer.hpp`, `TrainerConfig.hpp` — include them as `"core/training/EpochResult.hpp"` |
| Implementation (`.cpp`) | none — header-only; this folder holds the tests and the CMake target |
| CMake library target | — (header-only) |
| Test executables | `trainer_genericity_gtest`, `trainer_gtest` |

## Header map

- `include/training/`: `BestWeightsCallback.hpp`, `EarlyStoppingCallback.hpp`, `ITrainingCallback.hpp`, `ProgressCallback.hpp`, `TrainingState.hpp`

## Changing it

- A new header in `include/...` needs nothing else unless it has a `.cpp`; a new `.cpp` must be added to this folder's `CMakeLists.txt`.
- A new test goes in `tests/`, named `*_gtest.cpp`, and is registered in `tests/CMakeLists.txt` (`add_executable` + `target_link_libraries(... google_test ...)` + `gtest_discover_tests(<exe>)`; copy a neighbour) ([Build System](../../../.wiki/Guides/Build-System.md#running-tests)).
- Build and run (from `software/nn`):

  ```bash
  cmake --build --preset=max-performance --target trainer_genericity_gtest
  ctest --test-dir out/build/max-performance -R 'TrainerGenericity' --output-on-failure
  ```

  `ctest -R` matches the GoogleTest **suite** name, not the executable: `-R trainer_genericity_gtest` silently matches nothing.

  Suites here: `TrainerGenericity`, `TrainerConfigTest`, `EpochResultTest`, `GradClipTest`, `SpikeCountLossTest`, `ITrainingCallbackTest`, `EarlyStoppingCallback`, `BestWeightsCallback`.
- After a behaviour change, update [`.wiki/Core/Training.md`](../../../.wiki/Core/Training.md) (and `.wiki/References.md` if a paper is involved).
