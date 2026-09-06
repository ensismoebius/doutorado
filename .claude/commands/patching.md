
# patching

Goal
- Produce minimal, reviewable diffs with preserved behavior.

Rules

- RULE: PATCH_SMALL
  DO: Change only files required by the issue
  AVOID: No opportunistic refactors
- RULE: API_STABILITY
  DO: Preserve public contracts unless change is explicitly requested
  AVOID: No silent API drift
- RULE: BUILD_AFTER_EDIT
  DO: Build affected targets after every edit
  AVOID: Never return unverified code changes
- RULE: TEST_NEARBY
  DO: Run related tests/smoke checks
  AVOID: Don't run the full test suite when targeted tests suffice
- RULE: RENAME_GATED
  DO: For a symbol rename touching call sites, use `rename_symbol` (MCP) — `dry_run=True` first, apply only when `blocked: false`
  AVOID: Never hand-roll a multi-file find/replace for a rename; it can't tell a real call site from an unrelated same-named token the way `rename_symbol`'s exact-coverage gate does
- RULE: PERF_GATE
  DO: For hot-path patches, include allocation/locality impacts in the rationale
  AVOID: No unmeasured optimization diffs

Workflow

1. Gather file + symbol context (use `navigation` skill if needed).
2. Apply the patch — `replace_symbol`/`insert_lines` (MCP) for a structural
   edit gated on the target being unchanged, or `rename_symbol` for a
   rename that must also update call sites.
3. Build the affected target: `run_build(target=...)` (MCP), or
   `cmake --build build --target <target> -j4` when a specific preset
   other than the most-recently-built one is required (see `build-test`).
4. Run targeted tests: `run_tests(filter=<pattern>)` (MCP), or
   `ctest --test-dir build -R <pattern> --output-on-failure`.
5. Before reporting, check `git_diff_stat`/`git_status` (MCP) — confirms
   the diff only touches what the patch intended, catching an
   accidentally-modified unrelated file before it reaches review.
6. Report diff + verification results.

Validation

- Diff is focused on the stated issue only — confirmed via `git_diff_stat`, not just memory of what was edited.
- Build passes for touched targets.
- Relevant tests executed or explicitly deferred with justification.

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

**Code intelligence (MCP `code_intelligence`) — prefer over grep/manual commands for anything about the code itself:**
- `find_symbol` / `search_text` / `list_symbols` — resolve/search/enumerate symbols in indexed files, each hit tagged with its enclosing symbol (replaces `rg`/`grep`/`find` for anything already indexed)
- `get_source_range` / `symbol_source` / `outline_symbol` — exact, budget-checked source instead of a full-file read (`{"truncated": true, "recommended_ranges": [...]}` on overflow — read what it recommends, don't guess smaller)
- `find_references` / `find_dependencies` — callers/callees marked `"exact"` (real compiler) or `"heuristic"` (name-matching) — never read a heuristic "0 callers" as dead code
- `get_violations` / `rank_symbols` / `rename_symbol` — structural findings, complexity hotspots, and gated multi-site renames
- `ast_search` / `ast_replace` — AST-pattern structural search and rewrite (`foo($A, $B)` matches a 2-arg call to `foo` regardless of formatting/argument names) — prefer over a regex `search_text`/`rg` for anything shaped like code structure rather than text
- `run_build` / `run_tests` / `run_lint` / `run_format` / `detect_toolchain` — structured build/test/lint output, not raw logs (`run_lint`/`run_format` cover Python only; C++ still goes through `analysis-all`/`clang-format-changed.sh`)
- `git_status` / `git_log` / `git_blame` / `git_diff_stat` / `compare_baseline` — repo state/history/diff without shelling out to `git`

**Wiki & knowledge graph** (concepts, papers, docs — not code symbols; use the MCP tools above for those):
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
