# statistics

Metrics and statistical tools: confusion matrices, classification metrics, EER scoring, (stratified) k-fold splitters, effect sizes and the approximate inference tests.

**Full documentation:** [`.wiki/Core/Statistics.md`](../../../.wiki/Core/Statistics.md). This README only says where things are and how to change them.

| What | Where |
|---|---|
| Public headers | [`include/statistics`](../../../include/statistics) |
| Implementation (`.cpp`) | 6 file(s) in this folder |
| CMake library target | `statistics` |
| Test executables | `confusionMatrix_gtest`, `eer_scorer_gtest`, `kfold_gtest`, `statistics_gtest` |

## Header map

- `include/statistics/`: `ClassificationEERScorer.hpp`, `GenuineImpostorEERScorer.hpp`, `GroupKFoldPolicy.hpp`, `IStatistic.hpp`, `NestedKFold.hpp`, `RunningMean.hpp`, `SampleKFoldPolicy.hpp`, `StratifiedKFold.hpp`, `binary_classification_metrics.hpp`, `confusion_matrix.hpp`, `eer_scorer.hpp`, `inference_tests.hpp`, `kfold.hpp`, `multi_class_metrics.hpp`, `reconstruction_metrics.hpp`, `statistics.hpp`

## Changing it

- A new header in `include/...` needs nothing else unless it has a `.cpp`; a new `.cpp` must be added to this folder's `CMakeLists.txt`.
- A new test goes in `tests/`, named `*_gtest.cpp`, and is registered in `tests/CMakeLists.txt` (`add_executable` + `target_link_libraries(... google_test ...)` + `gtest_discover_tests(<exe>)`; copy a neighbour) ([Build System](../../../.wiki/Guides/Build-System.md#running-tests)).
- Build and run (from `software/nn`):

  ```bash
  cmake --build --preset=max-performance --target confusionMatrix_gtest
  ctest --test-dir out/build/max-performance -R 'ConfusionMatrixTest' --output-on-failure
  ```

  `ctest -R` matches the GoogleTest **suite** name, not the executable: `-R confusionMatrix_gtest` silently matches nothing.

  Suites here: `ConfusionMatrixTest`, `GenuineImpostorEERScorerTest`, `ClassificationEERScorerTest`, `KFoldTest`, `StratifiedKFoldTest`, `KFoldCompatibilityTest`, `NestedKFoldTest`, `SampleKFoldPolicyTest`, `GroupKFoldPolicyTest`, `NestedKFoldPolicyTest`, `StatisticsTest`, `MultiClassMetricsTest` (and 6 more).
- After a behaviour change, update [`.wiki/Core/Statistics.md`](../../../.wiki/Core/Statistics.md) (and `.wiki/References.md` if a paper is involved).
