Ad-hoc Console/File Debug Prints — Findings & Suggested Replacements

Scope
- Scanned `src/` and `include/` for direct console and file debug usage (`std::cout`, `std::cerr`, `ofstream`, `/tmp/*` writes).
- Excluded build/_deps and generated files from edits; recommended focus: `src/` and `include/`.

Suggested policy
- Non-error informational prints -> replace with `NN_LOG_DEBUG` or equivalent logger info macro.
- Error/exception prints -> emit via the project logger at ERROR level (do not use plain `std::cerr`).
- Test-only and demo prints: skip by default (or convert on request).
- File-based debug logs under `/tmp` -> remove or route to test-only temporary logger.

Sample files with occurrences (representative; not exhaustive)
- src/core/dataLoaders/SqliteBatchSource.cpp
  - Many `std::cerr` diagnostic prints and exception messages.
  - Suggestion: convert to logger.error(...) or NN_LOG_ERROR and include contextual tags (e.g., `SqliteBatchSource`).

- src/core/dataLoaders/BatchPrefetcher.cpp
  - Producer thread exception prints to `std::cerr`.
  - Suggestion: `NN_LOG_ERROR("Producer thread exception: " + std::string(e.what()))`.

- include/nn/tensor/EigenTensorBackend.hpp
  - `std::cerr` on shape mismatch and writes to `/tmp/mse_diag.log`.
  - Suggestion: remove `/tmp` writes; use logger at `Error` or `Debug` depending on severity.

- include/nn/layers/MSELoss.hpp
  - Warning printed with `std::cerr` for non-finite gradients.
  - Suggestion: `NN_LOG_ERROR("Warning: Non-finite gradients detected in MSE backward pass")`.

- src/experiments/03/lib/src/experiment03.cpp and src/experiments/03/experiment03.cpp
  - `using std::cout; using std::cerr;` and occasional direct prints.
  - These are top-level experiment drivers; prefer logger.info/error for user-facing messages, but keep small run summaries as info-level logger output.

- src/core/utility/vectorizationCheck.cpp
  - Prints feature detection to stdout. Suggestion: convert to `NN_LOG_INFO` or keep for interactive builds only.

- src/core/wave/lfcc_pipeline_utils.cpp
  - `std::cout` status messages during subject processing. Suggestion: logger.info.

- include/nn/utility/printTensor.hpp and demos
  - Small debug helpers that intentionally print to stdout. Suggest leaving as-is OR guard with `#ifdef DEBUG`.

- Various demos and examples (many `std::cout` / `std::cerr` usages)
  - Recommendation: skip by default; these are user-facing demos and can remain, or be converted later.

Plan for automated patch (recommended)
1. Create a focused patch that replaces non-test, non-demo occurrences in `src/` and `include/` only.
2. For each replacement: use `NN_LOG_DEBUG(...)` for debug/verbose prints; use logger error/info macros for error and info messages.
3. Rebuild and run smoke tests for `experiment03`.

Next steps
- Option A (Apply changes): I can generate and apply a patch converting the most important runtime prints (SqliteBatchSource, BatchPrefetcher, EigenTensorBackend, MSELoss, Experiment drivers). Then I'll rebuild and run a smoke run.
- Option B (Review report): I can expand this report into a complete file-by-file diff suggestion for review before applying.

Tell me: apply the focused patch now (A) or produce a full review list first (B).