---
name: dev-tooling-integrator
description: "Standardize use of local CLI tools to reduce manual complexity and token-heavy workflows."
---

# dev-tooling-integrator

Goal
- Use installed tooling to minimize repetitive manual steps.

Rules
- RULE: NO_LEGACY_FORWARDERS
  DO: Migrate call sites to canonical headers and update include paths directly.
  AVOID: Creating legacy forwarding headers, compatibility wrapper headers, or shim include files.
- RULE: SEARCH_TOOLING
  DO: Use `rg` for code/text search and file discovery.
  AVOID: Slow broad scans.
- RULE: JSON_TOOLING
  DO: Use `jq` to validate/query JSON configs.
  AVOID: Manual JSON parsing.
- RULE: LOOP_AUTOMATION
  DO: Use `entr` or equivalent for watch/rebuild loops.
  AVOID: Re-running repetitive commands manually.
- RULE: INDEX_SUPPORT
  DO: Use `ctags`/symbol indexes for fast navigation where useful.
  AVOID: Repeated whole-file reads.

Validation
- Proposed workflows use existing local tools first.
