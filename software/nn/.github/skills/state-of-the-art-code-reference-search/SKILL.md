---
name: state-of-the-art-code-reference-search
description: "Find and summarize state-of-the-art code references and implementation patterns for a target problem."
---

# state-of-the-art-code-reference-search

Goal
- Retrieve high-quality, recent, and reproducible code references for a specific technical task.

Rules
- RULE: NO_LEGACY_FORWARDERS
  DO: Migrate call sites to canonical headers and update include paths directly.
  AVOID: Creating legacy forwarding headers, compatibility wrapper headers, or shim include files.
- RULE: QUERY_PRECISION
  DO: Build narrow queries with domain, method, metric, and constraints.
  AVOID: Broad keyword-only searches.
- RULE: SOURCE_QUALITY
  DO: Prioritize peer-reviewed papers, official docs, maintained repositories, and benchmark leaders.
  AVOID: Unverified blog snippets as primary references.
- RULE: RECENCY_BALANCE
  DO: Prefer recent references while preserving foundational canonical methods.
  AVOID: Recency-only selection bias.
- RULE: CODE_VERIFIABILITY
  DO: Prefer references with runnable code, license clarity, and dependency transparency.
  AVOID: Non-reproducible pseudo-code-only sources.
- RULE: COMPARATIVE_SUMMARY
  DO: Return concise comparison tables (method, complexity, memory, constraints, links).
  AVOID: Long narrative dumps.
- RULE: TOKEN_EFFICIENCY
  DO: Summarize findings in compact bullets and keep only actionable excerpts.
  AVOID: Copying large raw passages.

Workflow
1. Define target problem and constraints (hardware, latency, memory, dataset type).
2. Build query set (canonical terms + modern variants).
3. Collect candidate references from authoritative sources.
4. Filter by quality, recency, reproducibility, and relevance.
5. Produce a compact shortlist with implementation guidance.

Validation
- At least 3 high-quality references returned for non-trivial topics.
- Each reference includes why it matters and where it applies.
- Output includes trade-offs (speed, memory, complexity, reproducibility).
