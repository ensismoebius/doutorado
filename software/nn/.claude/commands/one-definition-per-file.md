---
description: "Enforce one enum/class/interface per header file for better modularity and compile-time isolation."
---

# one-definition-per-file

Keep the codebase modular: each type in its own file. Improves build parallelism and reduces header dependency chains.

## Code intelligence (MCP `code_intelligence`)

Prefer over grep/manual git/cmake for anything about the code itself:
- `find_symbol` / `search_text` / `list_symbols` — resolve/search/enumerate symbols in indexed files, each hit tagged with its enclosing symbol (replaces `rg`/`grep`/`find` for anything already indexed)
- `get_source_range` / `symbol_source` / `outline_symbol` — exact, budget-checked source instead of a full-file read (`{"truncated": true, "recommended_ranges": [...]}` on overflow — read what it recommends, don't guess smaller)
- `ast_search` / `ast_replace` — AST-pattern structural search and rewrite (`foo($A, $B)` matches a 2-arg call to `foo` regardless of formatting/argument names) — prefer over a regex `search_text`/`rg` for anything shaped like code structure rather than text
- `find_references` / `find_dependencies` — callers/callees marked `"exact"` (real compiler) or `"heuristic"` (name-matching) — never read a heuristic "0 callers" as dead code
- `get_violations` / `rank_symbols` / `rename_symbol` / `replace_symbol` — structural findings, complexity hotspots, and hash-gated multi-site renames/edits
- `run_build` / `run_tests` / `run_lint` / `run_format` — structured build/test/lint output, not raw logs
- `git_status` / `git_log` / `git_blame` / `git_diff_stat` / `compare_baseline` — repo state/history/diff without shelling out to `git`

## Project Context (nn framework)

**Reference layout** (one class per file, followed correctly):
- `include/layers/spiking/` — one LIF variant per file: `Lif.hpp`, `LifBPTT.hpp`, `ThresholdDependentBatchNorm.hpp`, `PoissonLatentLayer.hpp`
- `include/layers/activations/` — one activation per file: `ReLU.hpp`, `Sigmoid.hpp`, `Tanh.hpp`
- `include/layers/spiking/ExponentialSurrogate.hpp`, `BoxcarSurrogate.hpp` — one surrogate per file

**Anti-pattern (already fixed):** Multiple surrogate gradient types in a single header. This was previously the case — do not revert.

**Exception:** `include/layers/Layers.hpp` is auto-generated and aggregates all layers; it is `#include`-only, not for editing. It is `.gitignore`d.

## Rules

- **ONE_CLASS_PER_FILE**: Each class must reside in its own header file. Exception: tightly coupled RAII handle + manager pairs may stay together.
- **ONE_ENUM_PER_FILE**: Each `enum class` must reside in its own header file.
- **ONE_INTERFACE_PER_FILE**: Each interface (abstract class with pure virtual methods) must have its own header.
- **AGGREGATE_HEADERS**: Create aggregation headers that `#include` all related type headers. This maintains backward compatibility for existing includes while enabling granular includes.

## Naming Conventions

| Kind | Pattern | Example |
|------|---------|---------|
| Interface | `I<Name>.hpp` | `ISurrogateGradient.hpp` |
| Aggregation | Plural/collective name | `Transforms.hpp` |
| Concrete class | Original name | `BoxcarSurrogate.hpp` |

## Refactoring Pattern

1. Create a new separate header for each type.
2. Move the type definition to its own file.
3. Update the aggregation header to `#include` all new files.
4. Ensure all existing includes still resolve (backward compatibility).
5. Build and run all tests.

## Past Refactoring Examples (reference)

- `SurrogateGradient.hpp` → `ISurrogateGradient.hpp`, `ExponentialSurrogate.hpp`, `BoxcarSurrogate.hpp`
- `Regularization.hpp` → `IRegularization.hpp`, `L1Regularization.hpp`, `L2Regularization.hpp`
- `Transforms.hpp` → `ITransform.hpp`, `Compose.hpp`, `AudioMeanStdNormalize.hpp`, `EEGWindowZScore.hpp`, `FusedModalityTransform.hpp`
- `Device.hpp` → `DeviceType.hpp`, `Device.hpp`, `DeviceRuntime.hpp`
- `Trainer.hpp` → `EpochResult.hpp`, `TrainerConfig.hpp`, `Trainer.hpp`

## Validation

- Build passes after refactoring.
- All tests pass.
- No existing `#include` path breaks (aggregation header in place).
