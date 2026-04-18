---
name: logging
description: "Skill for consistent use of nn::logging::Logger and migration from ad-hoc prints."
---

# logging

Goal
- Centralize runtime output and remove ad-hoc console/file diagnostics.

Rules
- RULE: NO_LEGACY_FORWARDERS
  DO: Migrate call sites to canonical headers and update include paths directly.
  AVOID: Creating legacy forwarding headers, compatibility wrapper headers, or shim include files.
- RULE: LOGGER_ONLY
  DO: Use `NN_LOG_ERROR/WARN/INFO/DEBUG`.
  AVOID: New `std::cout`, `std::cerr`, `/tmp/*.log` diagnostics in core paths.
- RULE: LEVEL_DISCIPLINE
  DO: Map severity correctly (`ERROR`/`WARN`/`INFO`/`DEBUG`).
  AVOID: High-volume info logs in hot loops.
- RULE: CAPTURE_CONSISTENCY
  DO: Use `StreamRedirector` where unified output is required.
  AVOID: Mixed direct console and logger writes.
- RULE: PERF_GATE
  DO: Keep logging out of tight inner loops unless gated by debug level.
  AVOID: High-frequency logging in hot paths.

Workflow
1. Replace ad-hoc prints in target files.
2. Add `Logger.hpp` include where needed.
3. Build and run smoke path to validate output.

Validation
- No new ad-hoc prints in modified core files.
- Output remains readable with progress UI.
