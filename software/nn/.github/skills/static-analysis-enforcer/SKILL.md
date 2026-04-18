---
name: static-analysis-enforcer
description: "Enforce static analysis checks and critical warning remediation."
---

# static-analysis-enforcer

Goal
- Keep code quality and safety regressions visible and controlled.

Rules
- RULE: NO_LEGACY_FORWARDERS
  DO: Migrate call sites to canonical headers and update include paths directly.
  AVOID: Creating legacy forwarding headers, compatibility wrapper headers, or shim include files.
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
