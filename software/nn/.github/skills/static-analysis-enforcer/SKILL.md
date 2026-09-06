---
name: static-analysis-enforcer
description: "Enforce static analysis checks and critical warning remediation."
---

# static-analysis-enforcer

Goal
- Keep code quality and safety regressions visible and controlled.

Rules
- RULE: ANALYZE_ALWAYS
  DO: Run `cppcheck` and relevant analyzer tasks on touched areas.
  AVOID: Shipping unreviewed static-analysis findings.
- RULE: SECURITY_SCAN
  DO: Run `flawfinder` for security-sensitive paths.
  AVOID: Ignoring high-risk warnings.
- RULE: WARNING_POLICY
  DO: Fix critical warnings and undefined-behavior risks before completion.
  AVOID: Suppressing issues without justification.

Validation
- Critical analyzer findings are resolved or explicitly justified.

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
