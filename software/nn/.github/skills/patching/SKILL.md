---
name: patching
description: "Skill for focused apply_patch edits with compile/test validation."
---

# patching

Goal
- Produce minimal, reviewable diffs with preserved behavior.

Rules
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

Project Context (nn framework)
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

**Wiki & knowledge graph:**
- Documentation at `.wiki/` — theory, guides, experiment pages, concept definitions
- Graph output at `.wiki/graphify-out/` — 1926 nodes, 4987 edges, 203 communities
- Find any symbol/concept:
```bash
python3 -c "
import json,sys
with open('.wiki/graphify-out/graph.json') as f: g=json.load(f)
q=sys.argv[1].lower()
for n in g['nodes']:
    if q in n['id'].lower() or q in n.get('label','').lower():
        print(n['id'],'|',n.get('source_file',''),'|',n.get('source_location',''))
" <QUERY>
```
- Workflow: `GRAPH_REPORT.md` → community → node → `source_file` → read → follow edges
