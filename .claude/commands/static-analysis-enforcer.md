
# static-analysis-enforcer

Goal
- Keep code quality and safety regressions visible and controlled.

Rules

- RULE: STRUCTURAL_FIRST
  DO: Run `get_violations`/`summarize_violations` (MCP) on touched files first — cross-language, no compile step, catches naming/missing-docs/LOC/duplication `cppcheck` doesn't look for
  AVOID: Don't treat it as a substitute for `cppcheck`/`clang-tidy` — it's structural, not semantic; run both
- RULE: ANALYZE_ALWAYS
  DO: Run `cppcheck` and relevant analyzer tasks on touched areas
  AVOID: No shipping unreviewed static-analysis findings
- RULE: SECURITY_SCAN
  DO: Run `flawfinder` for security-sensitive paths
  AVOID: No ignoring high-risk warnings
- RULE: WARNING_POLICY
  DO: Fix critical warnings and undefined-behavior risks before marking a task complete
  AVOID: No suppressing issues without justification

Workflow

1. `get_violations`/`summarize_violations` (MCP) on touched files — cheapest pass, no build required, gives `symbol_id`/location directly.
2. Run `cppcheck` on all files touched by the current change.
3. Run `flawfinder` on security-sensitive paths (I/O, parsing, network).
4. Triage findings: fix critical/undefined-behavior issues, document suppressed low-impact ones.
5. Re-run analyzers to confirm zero critical findings.

Validation

- Critical analyzer findings are resolved or explicitly justified with inline comments.
- No new `cppcheck` errors in modified files.

Project Context (nn framework)

**Script location (after reorg):** `scripts/ci/validate_static_analysis.py`

**Config files:**
- `.cppcheck` — cppcheck configuration at repo root
- `.clang-tidy` — clang-tidy checks at repo root

**CI integration:** `.github/workflows/ci.yml:126` — runs `validate_static_analysis.py` on cppcheck XML output

**CMake targets:**
- `analysis-cppcheck` — run cppcheck only
- `analysis-clang-tidy` — run clang-tidy only
- `analysis-all` — run both

**Suppressions allowlist:** `scripts/ci/validate_static_analysis.py::APPROVED_SUPPRESSIONS`
```bash
python3 scripts/ci/validate_static_analysis.py --list-approved
```

**Code intelligence (MCP `code_intelligence`) — prefer over grep/manual commands for anything about the code itself:**
- `find_symbol` / `search_text` / `list_symbols` — resolve/search/enumerate symbols in indexed files, each hit tagged with its enclosing symbol (replaces `rg`/`grep`/`find` for anything already indexed)
- `get_source_range` / `symbol_source` / `outline_symbol` — exact, budget-checked source instead of a full-file read (`{"truncated": true, "recommended_ranges": [...]}` on overflow — read what it recommends, don't guess smaller)
- `find_references` / `find_dependencies` — callers/callees marked `"exact"` (real compiler) or `"heuristic"` (name-matching) — never read a heuristic "0 callers" as dead code
- `get_violations` / `rank_symbols` / `rename_symbol` — structural findings, complexity hotspots, and gated multi-site renames
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

Commands

```bash
# cppcheck on modified files
cppcheck --enable=all --std=c++20 --suppress=missingInclude \
  --error-exitcode=1 <file-or-dir>

# flawfinder security scan
flawfinder --minlevel=3 src/ include/

# clang-tidy (if configured)
clang-tidy <file> -- -std=c++20 -I include/
```
