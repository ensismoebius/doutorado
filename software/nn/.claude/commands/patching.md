---
description: "Focused minimal diffs with compile/test validation before reporting completion."
---

# patching

Produce minimal, reviewable diffs with preserved behavior.

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
