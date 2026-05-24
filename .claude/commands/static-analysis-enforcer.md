
# static-analysis-enforcer

Goal
- Keep code quality and safety regressions visible and controlled.

Rules

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

1. Run `cppcheck` on all files touched by the current change.
2. Run `flawfinder` on security-sensitive paths (I/O, parsing, network).
3. Triage findings: fix critical/undefined-behavior issues, document suppressed low-impact ones.
4. Re-run analyzers to confirm zero critical findings.

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
