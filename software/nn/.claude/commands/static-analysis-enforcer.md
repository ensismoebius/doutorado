---
description: "Enforce cppcheck and flawfinder static analysis checks; remediate critical warnings before completion."
---

# static-analysis-enforcer

Keep code quality and safety regressions visible and controlled.

## Project Context (nn framework)

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

## Rules

- **ANALYZE_ALWAYS**: Run `cppcheck` and relevant analyzer tasks on touched areas. No shipping unreviewed static-analysis findings.
- **SECURITY_SCAN**: Run `flawfinder` for security-sensitive paths. No ignoring high-risk warnings.
- **WARNING_POLICY**: Fix critical warnings and undefined-behavior risks before marking a task complete. No suppressing issues without justification.

## Commands

```bash
# cppcheck on modified files
cppcheck --enable=all --std=c++20 --suppress=missingInclude \
  --error-exitcode=1 <file-or-dir>

# flawfinder security scan
flawfinder --minlevel=3 src/ include/

# clang-tidy (if configured)
clang-tidy <file> -- -std=c++20 -I include/
```

## Workflow

1. Run `cppcheck` on all files touched by the current change.
2. Run `flawfinder` on security-sensitive paths (I/O, parsing, network).
3. Triage findings: fix critical/undefined-behavior issues, document suppressed low-impact ones.
4. Re-run analyzers to confirm zero critical findings.

## Validation

- Critical analyzer findings are resolved or explicitly justified with inline comments.
- No new `cppcheck` errors in modified files.
