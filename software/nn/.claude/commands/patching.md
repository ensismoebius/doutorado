---
description: "Focused minimal diffs with compile/test validation before reporting completion."
---

# patching

Produce minimal, reviewable diffs with preserved behavior.

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
