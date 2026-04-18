---
name: cli-contract-enforcer
description: "Enforce deterministic, side-effect-safe CLI behavior and argument validation."
---

# cli-contract-enforcer

Goal
- Keep CLI interfaces deterministic and safe.

Rules
- RULE: NO_LEGACY_FORWARDERS
  DO: Migrate call sites to canonical headers and update include paths directly.
  AVOID: Creating legacy forwarding headers, compatibility wrapper headers, or shim include files.
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
