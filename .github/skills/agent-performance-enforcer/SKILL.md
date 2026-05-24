---
name: agent-performance-enforcer
description: "Skill synchronized from Claude command /agent-performance-enforcer."
---

# agent-performance-enforcer

Converted from: .claude/commands/agent-performance-enforcer.md

# Agent Performance Enforcer

Enforce high-performance agent execution standards across planning, implementation, validation, and handoff.

Use this skill whenever work quality, speed, reliability, or context efficiency matters.

## Objectives

1. Improve first-pass success rate
2. Reduce unnecessary tool calls and token waste
3. Keep fixes deterministic and verifiable
4. Preserve repository safety and conventions

## Required input contract

Before implementation, ensure these are explicit:

1. Goal: one clear target outcome
2. Done criteria: concrete checks that define completion
3. Constraints: safety, compatibility, performance, style
4. Scope boundaries: what is in scope and out of scope

If any are missing, derive from repo docs and existing behavior. If still ambiguous, ask concise clarifying questions.

## Enforced workflow

### 1) Problem framing

- Restate the failure in one sentence
- Define expected behavior and measurable success
- Note assumptions and risks before edits

### 2) High-signal context only

- Read only relevant files and nearby references
- Prefer narrow, parallel searches over broad scans
- Avoid dumping large outputs unless needed for a decision

### 3) Task decomposition

- Split into small, testable deltas
- Apply one behavioral change at a time
- Validate after each delta where feasible

### 4) Deterministic implementation

- Prefer existing wrappers/scripts over ad-hoc commands
- Preserve established APIs, paths, and conventions
- Avoid hidden side effects

### 5) Fast feedback loop

- Run the smallest meaningful validation first
- Escalate to broader checks only as needed
- Stop and report clearly when validation cannot run

### 6) Repository hygiene

- Keep edits focused and minimal
- Do not reformat unrelated code
- Update docs for non-obvious behavior changes

### 7) Memory and recurrence control

- Record concise repo memory for root causes likely to recur
- Capture canonical fix patterns, not long narratives

### 8) Completion gate (must pass)

Do not declare done unless all apply:

- Behavior fixed against stated done criteria
- Relevant checks executed or blocked with reason
- Documentation updated when behavior changed
- Cross-agent skill sync completed when source skills changed
- Risks and next steps communicated clearly

## Anti-patterns to block

1. Silent fallback to alternate data source without stating limitation
2. Large speculative edits before reproducing the issue
3. Running expensive global checks before targeted validation
4. Relying on manual, non-repeatable steps when wrappers exist
5. Closing task without verification evidence

## Output standard

When reporting completion, include:

1. What changed
2. Why it changed
3. How it was validated
4. Residual risks or follow-up actions
