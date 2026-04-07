---
name: cli-contract-enforcer
description: "Enforce deterministic, side-effect-safe CLI behavior and argument validation."
---

# cli-contract-enforcer

Goal
- Keep CLI interfaces deterministic and safe.

Rules
- RULE: HELP_IS_PURE
  DO: Make `--help` exit immediately with status 0.
  AVOID: Side effects on help output.
- RULE: ARG_VALIDATION
  DO: Validate all required and range-constrained arguments.
  AVOID: Silent invalid-argument fallthrough.
- RULE: OVERRIDE_SUPPORT
  DO: Allow profile/config override flags explicitly.
  AVOID: Hidden precedence rules.

Validation
- Help path performs no file writes or side effects.
- Invalid inputs fail fast with clear diagnostics.