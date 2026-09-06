---
name: patching
description: "Focused minimal diffs with compile/test validation before reporting completion."
---

# patching

Produce minimal, reviewable diffs with preserved behavior.

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

**Module<Backend> contract** — every layer must implement:
- `forward(input, requires_grad)` — caches state for backward; call before `backward()`
- `backward(grad_output)` — returns grad w.r.t. input; grad shape == forward input shape
- `params()` → `std::span<nn::Tensor*>` — raw pointers to member tensors (not temporaries)
- `reset_state()` — stateful layers (SNN/LSTM) must clear ALL hidden state between sequences

**SNN invariants** (must not break when patching spiking layers):
- Time-major layout: input shape `(T*B, F)`, not `(B, T, F)`
- R, C clamped to `≥1e-6` in forward; grad zeroed in clamped region
- β = exp(−Δt/(R·C)) recomputed each forward step

**Build targets by patch type:**
```bash
# Core layer patch
cmake --build out/build/max-performance --target core_gtest -j$(nproc)

# Trainer or training loop patch
cmake --build out/build/max-performance --target trainer_gtest -j$(nproc)

# Profile JSON or Exp04 config patch
cmake --build out/build/max-performance --target profile_audit_gtest -j$(nproc)
```

## Rules

- **PATCH_SMALL**: Change only files required by the issue. No opportunistic refactors.
- **API_STABILITY**: Preserve public contracts unless change is explicitly requested. No silent API drift.
- **BUILD_AFTER_EDIT**: Build affected targets after every edit. Never return unverified code changes.
- **TEST_NEARBY**: Run related tests/smoke checks. Don't run the full test suite when targeted tests suffice.
- **PERF_GATE**: For hot-path patches, include allocation/locality impacts in the rationale. No unmeasured optimization diffs.

## Workflow

1. Gather file + symbol context (use `navigation` skill if needed).
2. Apply the patch.
3. Build the affected target: `cmake --build build --target <target> -j4`.
4. Run targeted tests: `ctest --test-dir build -R <pattern> --output-on-failure`.
5. Report diff + verification results.

## Validation

- Diff is focused on the stated issue only.
- Build passes for touched targets.
- Relevant tests executed or explicitly deferred with justification.
