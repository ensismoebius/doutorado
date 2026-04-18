---
name: patching
description: "Skill for focused apply_patch edits with compile/test validation."
---

# patching

Goal
- Produce minimal, reviewable diffs with preserved behavior.

Rules
- RULE: NO_LEGACY_FORWARDERS
  DO: Migrate call sites to canonical headers and update include paths directly.
  AVOID: Creating legacy forwarding headers, compatibility wrapper headers, or shim include files.
- RULE: PATCH_SMALL
  DO: Change only files required by the issue.
  AVOID: Opportunistic refactors.
- RULE: API_STABILITY
  DO: Preserve public contracts unless change is requested.
  AVOID: Silent API drift.
- RULE: BUILD_AFTER_EDIT
  DO: Build affected targets after edits.
  AVOID: Returning unverified code changes.
- RULE: TEST_NEARBY
  DO: Run related tests/smoke checks.
  AVOID: Full test suite when targeted tests suffice.
- RULE: PERF_GATE
  DO: For hot-path patches, include allocation/locality impacts in the patch rationale.
  AVOID: Merging unmeasured optimization diffs.

Workflow
1. Gather file+symbol context.
2. Apply patch.
3. Build target.
4. Run targeted tests.
5. Report diff + verification.

Validation
- Diff is focused.
- Build passes for touched targets.
- Relevant tests executed or explicitly deferred.
