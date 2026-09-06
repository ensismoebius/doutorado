---
description: "Migrate ad-hoc prints to nn::logging::Logger and enforce consistent log level discipline."
---

# logging

Centralize runtime output and remove ad-hoc console/file diagnostics.

## Project Context (nn framework)

**Logger header:** `include/logging/Logger.hpp` — macros: `NN_LOG_ERROR`, `NN_LOG_WARN`, `NN_LOG_INFO`, `NN_LOG_DEBUG`

**Key log points in `Trainer.hpp`:**
- `INFO`: epoch start/end with loss values
- `DEBUG` (gated, not in hot path): per-batch loss when debug level enabled
- `ERROR`: NaN loss detected — log and abort training
- `WARN`: SNN biophysical param (R, C) hit clamp boundary

**Never log inside:**
- `LifBPTT` inner time loop — called `time_steps × batch_size` times per forward
- `matmul` inner K-loop — called `rows × cols × K` times
- Any loop with >1000 iterations in typical workload

## Rules

- **LOGGER_ONLY**: Use `NN_LOG_ERROR` / `NN_LOG_WARN` / `NN_LOG_INFO` / `NN_LOG_DEBUG`. No new `std::cout`, `std::cerr`, or `/tmp/*.log` in core paths.
- **LEVEL_DISCIPLINE**: Map severity correctly: `ERROR` for failures, `WARN` for degraded state, `INFO` for milestones, `DEBUG` for trace. No high-volume info logs in hot loops.
- **CAPTURE_CONSISTENCY**: Use `StreamRedirector` where unified output is required. No mixed direct console and logger writes in the same path.
- **PERF_GATE**: Keep logging out of tight inner loops unless gated by debug level. No high-frequency logging in hot paths.

## Workflow

1. Replace ad-hoc prints (`std::cout`, `std::cerr`, `printf`) in target files.
2. Add `#include "nn/logging/Logger.hpp"` where needed.
3. Build the affected target and run smoke path to validate output.

## Validation

- No new ad-hoc prints in modified core files.
- Output remains readable with progress UI.
- Build passes after migration.
