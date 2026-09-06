
# one-definition-per-file

Goal
- Keep the codebase modular: each type in its own file. Improves build parallelism and reduces header dependency chains.

Rules

- RULE: ONE_CLASS_PER_FILE
  DO: Each class must reside in its own header file. Exception: tightly coupled RAII handle + manager pairs may stay together
- RULE: ONE_ENUM_PER_FILE
  DO: Each `enum class` must reside in its own header file
- RULE: ONE_INTERFACE_PER_FILE
  DO: Each interface (abstract class with pure virtual methods) must have its own header
- RULE: AGGREGATE_HEADERS
  DO: Create aggregation headers that `#include` all related type headers. This maintains backward compatibility for existing includes while enabling granular includes

Validation

- Build passes after refactoring (`run_build` MCP, or raw `cmake --build`).
- All tests pass (`run_tests` MCP, or raw `ctest`).
- No existing `#include` path breaks (aggregation header in place) — confirmed via `find_references` on the moved type(s), not assumed.

Project Context (nn framework)

**Reference layout** (one class per file, followed correctly):
- `include/nn/layers/spiking/` — one LIF variant per file: `Leaky.hpp`, `LeakyBPTT.hpp`, `ThresholdDependentBatchNorm.hpp`, `PoissonLatentLayer.hpp`
- `include/nn/layers/activations/` — one activation per file: `ReLU.hpp`, `Sigmoid.hpp`, `Tanh.hpp`
- `include/nn/layers/spiking/ExponentialSurrogate.hpp`, `BoxcarSurrogate.hpp` — one surrogate per file

**Anti-pattern (already fixed):** Multiple surrogate gradient types in a single header. This was previously the case — do not revert.

**Exception:** `include/nn/layers/Layers.hpp` is auto-generated and aggregates all layers; it is `#include`-only, not for editing. It is `.gitignore`d.

**Code intelligence (MCP `code_intelligence`) — prefer over grep/manual commands for anything about the code itself:**
- `find_symbol` / `search_text` / `list_symbols` — resolve/search/enumerate symbols in indexed files, each hit tagged with its enclosing symbol (replaces `rg`/`grep`/`find` for anything already indexed)
- `get_source_range` / `symbol_source` / `outline_symbol` — exact, budget-checked source instead of a full-file read (`{"truncated": true, "recommended_ranges": [...]}` on overflow — read what it recommends, don't guess smaller)
- `find_references` / `find_dependencies` — callers/callees marked `"exact"` (real compiler) or `"heuristic"` (name-matching) — never read a heuristic "0 callers" as dead code
- `get_violations` / `rank_symbols` / `rename_symbol` — structural findings, complexity hotspots, and gated multi-site renames
- `ast_search` / `ast_replace` — AST-pattern structural search and rewrite (`foo($A, $B)` matches a 2-arg call to `foo` regardless of formatting/argument names) — prefer over a regex `search_text`/`rg` for anything shaped like code structure rather than text
- `run_build` / `run_tests` / `run_lint` / `run_format` / `detect_toolchain` — structured build/test/lint output, not raw logs (`run_lint`/`run_format` cover Python only; C++ still goes through `analysis-all`/`clang-format-changed.sh`)
- `git_status` / `git_log` / `git_blame` / `git_diff_stat` / `compare_baseline` — repo state/history/diff without shelling out to `git`

Naming Conventions

| Kind | Pattern | Example |
|------|---------|---------|
| Interface | `I<Name>.hpp` | `ISurrogateGradient.hpp` |
| Aggregation | Plural/collective name | `Transforms.hpp` |
| Concrete class | Original name | `BoxcarSurrogate.hpp` |

Refactoring Pattern

1. `get_file_structure`/`list_symbols` (MCP) on the source header first —
   confirms exactly which types it currently holds before splitting it.
2. Create a new separate header for each type.
3. Move the type definition to its own file.
4. Update the aggregation header to `#include` all new files.
5. Ensure all existing includes still resolve (backward compatibility) —
   `find_references`/`find_dependencies` (MCP) on each moved type
   confirms nothing depends on a path this split changes.
6. Build and run all tests — `run_build`/`run_tests` (MCP), or
   `cmake --build ... --target core_gtest` + `ctest -R core` directly.
7. `list_symbols`/`file_report` (MCP) on the new files — confirms each
   holds exactly one class/enum/interface, without re-reading them by eye.

Past Refactoring Examples (reference)

- `SurrogateGradient.hpp` → `ISurrogateGradient.hpp`, `ExponentialSurrogate.hpp`, `BoxcarSurrogate.hpp`
- `Regularization.hpp` → `IRegularization.hpp`, `L1Regularization.hpp`, `L2Regularization.hpp`
- `Transforms.hpp` → `ITransform.hpp`, `Compose.hpp`, `AudioMeanStdNormalize.hpp`, `EEGWindowZScore.hpp`, `FusedModalityTransform.hpp`
- `Device.hpp` → `DeviceType.hpp`, `Device.hpp`, `DeviceRuntime.hpp`
- `Trainer.hpp` → `EpochResult.hpp`, `TrainerConfig.hpp`, `Trainer.hpp`
