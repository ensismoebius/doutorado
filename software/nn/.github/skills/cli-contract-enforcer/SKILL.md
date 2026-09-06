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

Project Context (nn framework)
**Main CLI entry points:**
- `guayaquil --comparative-config <profile.json>` — primary paper pipeline entry point
- `guayaquil --help` — must list all flags, exit 0, no side effects
- `autoencoderRunner --help` — same requirement
- `paraconsistentBaseline [config.json]` — optional positional arg for config path

**Profile JSON is the primary config surface** for Exp04. All training parameters (model, data, folds, hyperparams) come from the profile — no hidden defaults that differ between `--help` output and actual behavior.

**No side effects on `--help`:** `--help` must not write files, touch the filesystem, or open OpenCL devices. Use `nn::logging::StreamRedirector` only after help check.
