# Dual-Agent Consensus: Proposer vs Antagonist

Goal: solution implemented only when both agents agree on same canonical solution.

## Roles

1. Proposer: Copilot (GPT-5.3-Codex)
2. Antagonist: Claude Code

Proposer designs plan and patch.
Antagonist tries to break plan, find flaws, propose counterexample.
Implementation allowed only after explicit consensus.

## Hard Gate Rules

1. Both reports must exist.
2. Both must set `AGREE: yes`.
3. Both must reference same `TASK_ID`.
4. Both must reference same `SOLUTION_SHA256`.
5. Validation checks must pass.

Any mismatch blocks implementation.

## Report Format

Use template at:
- `docs/consensus/TEMPLATE.report.md`

Create files per task:
- `docs/consensus/TASK_XXX.proposer.md`
- `docs/consensus/TASK_XXX.antagonist.md`

## Canonical Solution Hash

Build stable canonical text from final agreed solution.

Example:

```bash
cat agreed_solution.txt | sha256sum
```

Copy hash into both reports as `SOLUTION_SHA256`.

## Gate Command

```bash
bash scripts/dev/dual_agent_consensus.sh \
  --task-id TASK_001 \
  --proposer docs/consensus/TASK_001.proposer.md \
  --antagonist docs/consensus/TASK_001.antagonist.md \
  --checks "cmake --build --preset=max-performance -j4;;ctest --test-dir out/build/max-performance --output-on-failure -R "Thesis.*""
```

If command fails, no merge.
If command passes, implementation and merge allowed.

## Chat-Only Mode (No Script)

If you want zero terminal commands, run gate directly in chat.

### Required payload pasted in chat

Paste exactly these 6 fields from both sides:

1. `TASK_ID`
2. `SOLUTION_CANONICAL_TEXT`
3. `SOLUTION_SHA256`
4. `AGREE`
5. `BLOCKERS`
6. `TEST_PLAN`

### Decision rule applied by Copilot in chat

Implementation is blocked unless all true:

1. Proposer `AGREE: yes`
2. Antagonist `AGREE: yes`
3. Same `TASK_ID`
4. Same `SOLUTION_SHA256`
5. `BLOCKERS` empty on both sides or explicitly resolved in same turn

If any item fails, Copilot returns `CONSENSUS: BLOCKED` and does not implement.
If all items pass, Copilot returns `CONSENSUS: APPROVED` and may implement.

### Minimal chat template

```text
TASK_ID: TASK_001

PROPOSER:
SOLUTION_CANONICAL_TEXT: ...
SOLUTION_SHA256: ...
AGREE: yes
BLOCKERS: none
TEST_PLAN: ...

ANTAGONIST:
SOLUTION_CANONICAL_TEXT: ...
SOLUTION_SHA256: ...
AGREE: yes
BLOCKERS: none
TEST_PLAN: ...
```

### Optional strict mode

Add this line in chat:

`MODE: STRICT_CONSENSUS`

With strict mode, Copilot also requires:

1. Antagonist lists at least one attempted counterexample
2. Proposer includes explicit mitigation for counterexample
3. Both include same acceptance test list

## Operating Loop

1. Proposer draft.
2. Antagonist critique.
3. Revise proposal.
4. Repeat until both set `AGREE: yes` and same hash.
5. Run gate script.
6. Merge only if gate passes.

## Suggested Branch Policy

1. Feature branch per task.
2. Commit message includes `TASK_ID`.
3. Merge request must attach both reports.
4. Reviewer verifies gate output in CI log.
