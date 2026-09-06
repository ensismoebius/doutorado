---
name: cli-contract-enforcer
description: "Enforce deterministic, side-effect-safe CLI argument validation and help behavior."
---

# cli-contract-enforcer

Goal
- Keep CLI interfaces deterministic and safe across all entry points.

Rules

- RULE: HELP_IS_PURE
  DO: Make `--help` exit immediately with status 0
  AVOID: No side effects (file writes, network calls) on the help path
- RULE: ARG_VALIDATION
  DO: Validate all required and range-constrained arguments at startup
  AVOID: No silent invalid-argument fallthrough
- RULE: OVERRIDE_SUPPORT
  DO: Allow profile/config override flags explicitly
  AVOID: No hidden precedence rules between config file and CLI flags

Validation

- `--help` path performs no file writes or side effects.
- Invalid inputs fail fast with clear diagnostics and non-zero exit code.
- Argument precedence is documented and deterministic.

Project Context (nn framework)

**Main CLI entry points:**
- `guayaquil --comparative-config <profile.json>` — primary paper pipeline entry point
- `guayaquil --help` — must list all flags, exit 0, no side effects
- `autoencoderRunner --help` — same requirement
- `paraconsistentBaseline [config.json]` — optional positional arg for config path

**Profile JSON is the primary config surface** for Exp04. All training parameters (model, data, folds, hyperparams) come from the profile — no hidden defaults that differ between `--help` output and actual behavior.

**No side effects on `--help`:** `--help` must not write files, touch the filesystem, or open OpenCL devices. Use `nn::logging::StreamRedirector` only after help check.

Checklist

1. Trace the `--help` code path — does it touch any file I/O or side-effectful code?
2. List all required arguments and verify each is validated before use.
3. Confirm that CLI flag values override config-file values (or the opposite) with documented, explicit precedence.
4. Test invalid inputs: missing required args, out-of-range values — verify each produces a clear diagnostic and non-zero exit.
