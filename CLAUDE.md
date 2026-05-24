# doutorado — Claude Code instructions

## Caveman mode

Caveman active (full). Terse responses. Drop articles, filler, hedging. Technical substance exact.
`/caveman lite|full|ultra` to change.

## Working directory

Most work happens in `software/nn/`. Detailed CLAUDE.md there covers: build system, module contract, SNN invariants, tensor shapes, serialization, test conventions, wiki maintenance.

**Always `cd software/nn` before cmake/ctest commands** — presets and paths are relative to that dir.

## Scope constraints (permanent)

Do NOT treat these as active goals or mention as deliverables:
- Own database creation — removed from scope 2026-05-12
- FAMERP partnership — removed from scope 2026-05-12

## Expensive experiment guard

Long-running commands (full pipeline ~2.5h, `timeout 600+`, `run_article_profiles.sh`) require explicit confirmation. Add `# CONFIRMED` to the command or set `EXPERIMENT_CONFIRMED=1`. The `expensive-experiment-guard` hook blocks unlabeled runs.

## Repo structure

```
software/nn/         C++20 SNN/ML framework — primary codebase
  CLAUDE.md          Full technical rules (build, layers, SNN, tests, wiki)
  AGENTS.md          OpenCode guidance (build commands, conventions)
  .wiki/             Documentation wiki + graphify knowledge graph
  include/           Backend-agnostic public headers
  src/               Implementation + experiments (00–05)
  results/           Experiment outputs (JSON, CSV, .npy)
  scripts/           Python + bash analysis scripts
notebooks/           Jupyter notebooks (EEG/signal processing, prototyping)
documentation/       Research papers, books, project proposal
hardware/            Hardware-related files
results/             Top-level experiment results
.claude/
  commands/          33 slash commands (ML, C++, SNN, research skills)
  hooks/             PreToolUse: expensive-experiment-guard
                     PostToolUse: wiki-sync-reminder, completion-gate-reminder
```

## Key skills available

| Skill | Use when |
|-------|----------|
| `/build-test` | CMake build + test patterns |
| `/snn-efficiency-optimizer` | SNN temporal overhead |
| `/snn-parameter-bounds-enforcer` | LIF/BPTT param safety |
| `/nn-core-usage-enforcer` | Reuse existing abstractions |
| `/gradient-flow-validator` | Backward pass correctness |
| `/checkpoint-versioning-enforcer` | Save/load compatibility |
| `/experiment-reproducibility` | Seed, config, output tracing |
| `/bibliography-verifier` | .bib entry correctness |
| `/state-of-the-art-code-reference-search` | Paper/method lookup |
| `/wiki` | Generate/update wiki |
| `/agent-performance-enforcer` | End-of-task gates |
| `/validation-sequencing-enforcer` | Fastest-first validation |

## Wiki maintenance rule

After any layer, loss, optimizer, or training feature change: update `.wiki/Core/` and `.wiki/References.md`. The `wiki-sync-reminder` hook prompts this after every source edit.

## CI

`.github/workflows/ci.yml` — runs on push/PR. Uses cmake + ninja + clang-tidy + cppcheck. Check CI before marking work done.
