## Plan: Behavior-Preserving Refactor Roadmap

Refactor in small, verifiable slices that preserve APIs and runtime characteristics while reducing monolithic files, extracting nested helpers where beneficial, and centralizing schema/loader responsibilities. The priority path is: split oversized orchestration files, extract pure helpers into dedicated modules, and keep all external contracts (`Tensor`, `Module`, `Dataset`, `DataLoader`, optimizer interfaces, CMake targets) stable.

**Current problems detected**
1. Large monolithic files with mixed responsibilities increase cognitive load and change risk:
- `/home/ensismoebius/Repos/doutorado/software/nn/src/demos/exec_loadingData/loadingData.cpp`
- `/home/ensismoebius/Repos/doutorado/software/nn/src/experiments/02/experiment_02.cpp`
- `/home/ensismoebius/Repos/doutorado/software/nn/src/demos/voice_biometrics_cpp/main.cpp`
- `/home/ensismoebius/Repos/doutorado/software/nn/src/core/wave/Wav.cpp`
2. Data/schema knowledge (10.1117 indexing, label columns, row mapping) is scattered across demo and loader logic, making evolution error-prone.
3. Internal algorithmic logic is embedded in orchestration code (`collate` grouping/runs, sampler selection branches, async prefetch control flow).
4. Some nested type declarations reduce discoverability and testability when they represent reusable behavior (not all should be extracted).
5. Include boundaries are mostly good, but some source files still behave as “god files” with parsing + orchestration + business logic combined.

**Global architecture improvements**
1. Define and enforce module boundaries:
- `data/` for dataset schemas, row mapping, and batch assembly helpers.
- `training/` for experiment orchestration and model training workflows.
- `signal/` for audio/EEG preprocessing and feature extraction.
- `utils/` for small, stateless helpers.
2. Keep public contracts in `include/nn/**` stable; move complexity behind private implementation files in `src/core/**` and demo-local libraries.
3. Isolate policies from mechanisms:
- Policy: sampler mode, distributed options, experiment parameters.
- Mechanism: row reads, run coalescing, collation assembly.
4. Standardize file-level responsibility: one major class/struct per pair of `.hpp/.cpp` where reuse or independent testing is expected.

**Refactoring strategy**
1. Start with non-behavioral structure changes (extract helpers and move code) with zero logic edits.
2. Introduce characterization tests/snapshots before each medium/high-risk split.
3. Refactor by seams already present (existing helper functions, private methods, session objects).
4. Keep API compatibility by preserving signatures and adding adapter aliases where needed.
5. Validate continuously (build + targeted tests + selected demo runs) after each step.

**List of files that should be split**
1. `/home/ensismoebius/Repos/doutorado/software/nn/src/demos/exec_loadingData/loadingData.cpp`
- Split dataset implementation, batch assembly logic, and demo runner.
2. `/home/ensismoebius/Repos/doutorado/software/nn/src/experiments/02/experiment_02.cpp`
- Split config loading, preprocessing, training loop, evaluation/reporting.
3. `/home/ensismoebius/Repos/doutorado/software/nn/src/demos/voice_biometrics_cpp/main.cpp`
- Split I/O, feature pipeline, training/inference orchestration.
4. `/home/ensismoebius/Repos/doutorado/software/nn/src/core/wave/Wav.cpp`
- Split WAV parsing/serialization from DSP transforms.
5. `/home/ensismoebius/Repos/doutorado/software/nn/src/core/dataLoaders/10.1117/AudioLoader.cpp`
- Split MAT row-read/cache mechanics from schema-specific mapping.
6. `/home/ensismoebius/Repos/doutorado/software/nn/src/core/dataLoaders/10.1117/EEGLoader.cpp`
- Same split pattern as AudioLoader for symmetry and maintainability.

**List of nested classes/structs that should be extracted**
1. Extract:
- `DataLoader::Iterator` from `/home/ensismoebius/Repos/doutorado/software/nn/include/nn/dataLoaders/DataLoader.hpp` to dedicated files for clarity and independent testing.
2. Keep nested/local (do not extract now):
- `TensorImpl<Backend>::CommaInitializer` in `/home/ensismoebius/Repos/doutorado/software/nn/include/nn/tensor/Tensor.hpp` (tight template coupling).
- Demo-local POD/helpers when single-purpose and non-reusable.

**Proposed new file structure**
1. Data loader and schema extraction (private internals)
- `/home/ensismoebius/Repos/doutorado/software/nn/include/nn/dataLoaders/DataLoaderIterator.hpp`
- `/home/ensismoebius/Repos/doutorado/software/nn/src/core/dataLoaders/DataLoaderIterator.cpp`
- `/home/ensismoebius/Repos/doutorado/software/nn/src/core/dataLoaders/10.1117/SchemaColumnIndexer.hpp`
- `/home/ensismoebius/Repos/doutorado/software/nn/src/core/dataLoaders/10.1117/SchemaColumnIndexer.cpp`
- `/home/ensismoebius/Repos/doutorado/software/nn/src/core/dataLoaders/10.1117/SynchronizedBatchAssembler.hpp`
- `/home/ensismoebius/Repos/doutorado/software/nn/src/core/dataLoaders/10.1117/SynchronizedBatchAssembler.cpp`
2. Demo loading pipeline decomposition
- `/home/ensismoebius/Repos/doutorado/software/nn/src/demos/exec_loadingData/lib/include/Protocol101117Dataset.hpp`
- `/home/ensismoebius/Repos/doutorado/software/nn/src/demos/exec_loadingData/lib/src/Protocol101117Dataset.cpp`
- `/home/ensismoebius/Repos/doutorado/software/nn/src/demos/exec_loadingData/lib/include/BatchPrefetcher.hpp`
- `/home/ensismoebius/Repos/doutorado/software/nn/src/demos/exec_loadingData/lib/src/BatchPrefetcher.cpp`
- `/home/ensismoebius/Repos/doutorado/software/nn/src/demos/exec_loadingData/lib/include/DemoProbeModel.hpp`
3. Experiment orchestration decomposition
- `/home/ensismoebius/Repos/doutorado/software/nn/src/experiments/02/Experiment02Config.hpp`
- `/home/ensismoebius/Repos/doutorado/software/nn/src/experiments/02/Experiment02Config.cpp`
- `/home/ensismoebius/Repos/doutorado/software/nn/src/experiments/02/Experiment02Pipeline.hpp`
- `/home/ensismoebius/Repos/doutorado/software/nn/src/experiments/02/Experiment02Pipeline.cpp`
- `/home/ensismoebius/Repos/doutorado/software/nn/src/experiments/02/Experiment02Evaluation.hpp`
- `/home/ensismoebius/Repos/doutorado/software/nn/src/experiments/02/Experiment02Evaluation.cpp`
4. Wave module cleanup
- `/home/ensismoebius/Repos/doutorado/software/nn/src/core/wave/WavReader.hpp`
- `/home/ensismoebius/Repos/doutorado/software/nn/src/core/wave/WavReader.cpp`
- `/home/ensismoebius/Repos/doutorado/software/nn/src/core/wave/WavWriter.hpp`
- `/home/ensismoebius/Repos/doutorado/software/nn/src/core/wave/WavWriter.cpp`
- `/home/ensismoebius/Repos/doutorado/software/nn/src/core/wave/AudioResampler.hpp`
- `/home/ensismoebius/Repos/doutorado/software/nn/src/core/wave/AudioResampler.cpp`

**Refactoring steps in safe incremental order**
1. Baseline freeze and safety net
- Tag baseline commit and record benchmark numbers/tests currently passing.
- Add/confirm characterization tests for loader outputs and experiment summary metrics.
2. Extract `DataLoader::Iterator` with compatibility shim
- Introduce `DataLoaderIterator.hpp/.cpp` and keep public type alias/usage identical.
- Verify all `DataLoader` tests.
3. Split `loadingData` demo without changing behavior
- Move `Protocol101117Dataset`, `DemoProbeModel`, and prefetch control to `lib/` files.
- Keep `loadingData.cpp` as thin composition root.
4. Centralize 10.1117 schema column mapping
- Create `SchemaColumnIndexer` and replace duplicated magic-index calculations in Audio/EEG loaders and demo dataset internals.
5. Extract synchronized batch assembly mechanism
- Move grouping/sorting/run coalescing logic from demo dataset `collate` into `SynchronizedBatchAssembler`.
- Preserve algorithm and data flow exactly.
6. Split `experiment_02.cpp` into config, pipeline, evaluation modules
- First move-only refactor; then local naming cleanup and function decomposition.
7. Split `Wav.cpp` responsibilities
- Isolate binary format handling from DSP utilities while preserving all call paths.
8. Header hygiene and dependency minimization pass
- Forward declarations where safe, remove redundant includes, keep include order consistent.
9. Documentation pass
- Add concise Doxygen comments for new public/internal interfaces and module responsibility notes.
10. Final parity and performance gate
- Run full build/test matrix and focused perf checks versus baseline.

**Potential risks and how to mitigate them**
1. Risk: accidental behavior drift during extraction.
- Mitigation: extraction-only commits first (no logic edits), characterization tests, golden output comparisons for demo pipelines.
2. Risk: API or ABI breakage from moved declarations.
- Mitigation: preserve public headers/signatures; use forwarding headers or aliases during migration.
3. Risk: performance regressions from extra abstraction/allocation.
- Mitigation: keep data paths by reference/span, preallocate vectors, avoid virtual dispatch in hot loops unless already present.
4. Risk: CMake target/link breakage after file moves.
- Mitigation: adjust targets in small steps and build after each phase.
5. Risk: include cycles after decomposition.
- Mitigation: use forward declarations and one-directional dependency rules (`core` -> `utils`, not inverse).

**Performance considerations**
1. Preserve algorithmic complexity in collation/grouping and MAT row access paths.
2. Avoid additional copies for large tensors/signal buffers; prefer move semantics and existing block writes.
3. Keep hot-path helpers eligible for inlining (small headers where appropriate).
4. Retain contiguous memory access patterns in batch assembly and signal transforms.
5. Benchmark checkpoints after Phases 3, 5, 7 against baseline for runtime and memory.
6. Compile with current optimization flags and verify no new sanitizer/analysis warnings.

**Verification**
1. Build:
- `cmake --build build -j`
2. Core data loader tests:
- `ctest --test-dir build -R "AudioLoaderTest|AudioLoaderArticleSpecTest|AudioLoaderPropertyTest|EEGLoaderTest|EEGLoaderArticleSpecTest|EEGLoaderPropertyTest|dataLoaders_gtest|dataLoaders_samplers_gtest" --output-on-failure`
3. Demo parity:
- Run `/home/ensismoebius/Repos/doutorado/software/nn/build/src/demos/exec_loadingData/loadingData` with fixed seed and compare representative output fields.
4. Experiment parity:
- Run existing experiment entry points and compare aggregate metrics files.
5. Performance parity:
- Compare baseline vs refactor timings for representative loader and experiment runs.

**Decisions**
- Included: structural refactoring, file decomposition, naming cleanup, documentation improvement.
- Excluded: algorithm redesign, protocol/schema semantic changes, public API signature changes.
- Constraint: preserve backend-agnostic boundaries for demos/experiments composition logic.

**Final expected architecture**
1. Thin entry points (`main`/experiment runners) orchestrating dedicated modules.
2. Data loading logic split into schema mapping, session I/O, and batch assembly components.
3. Experiment code split into config/pipeline/evaluation units with clear boundaries.
4. Public headers stable and concise; implementation complexity shifted to internal `.cpp` files.
5. Easier unit testing at module level and safer future optimization work without behavioral risk.
