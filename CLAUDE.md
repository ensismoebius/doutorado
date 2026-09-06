# doutorado — Claude Code instructions

## Caveman mode

Caveman active (full). Terse responses. Drop articles, filler, hedging. Technical substance exact.
`/caveman lite|full|ultra` to change.

## Code intelligence MCP (default, not skill-gated)

`code_intelligence` (auto-loaded skill at `.claude/skills/code-intelligence/`) is
persistent and incremental — prefer it over `Read`/`grep`/raw `git`/raw
`cmake`/`ctest` for anything about the code itself, whenever it saves tokens,
in ANY task, not only when a specific skill below is invoked:
- Locate: `find_symbol` / `search_text` / `list_symbols` (each hit tagged with its enclosing symbol — replaces `grep`/`find` for indexed files)
- Read narrow: `get_source_range` / `symbol_source` / `outline_symbol` — never a full-file `Read` when a symbol or range answers it
- Structural pattern (shape, not text): `ast_search` / `ast_replace` — `foo($A, $B)` matches a 2-arg call regardless of formatting; a regex asked for the same distinction either over- or under-matches
- Impact: `find_references` / `find_dependencies` — tagged `"exact"`/`"heuristic"`; never read a heuristic "0 callers" as dead code
- Quality/edit: `get_violations` / `rank_symbols` / `rename_symbol` / `replace_symbol` / `insert_lines` — hash-gated, refuses on a stale target instead of guessing
- Build/test/repo state: `run_build` / `run_tests` / `run_lint` / `run_format` / `git_status` / `git_log` / `git_blame` / `git_diff_stat` / `compare_baseline` — structured results instead of raw log/diff text

## Working directory

Most work happens in `software/nn/`. Detailed CLAUDE.md there covers: build system, module contract, SNN invariants, tensor shapes, serialization, test conventions, wiki maintenance.

**Always `cd software/nn` before cmake/ctest commands** — presets and paths are relative to that dir.

## Scope constraints (permanent)

Do NOT treat these as active goals or mention as deliverables:
- Own database creation — removed from scope 2026-05-12
- FAMERP partnership — removed from scope 2026-05-12

## Expensive experiment guard

Long-running commands (full pipeline ~2.5h, `timeout 600+`, `01_guayaquil_run_article_profiles.sh`) require explicit confirmation. Add `# CONFIRMED` to the command or set `EXPERIMENT_CONFIRMED=1`. The `expensive-experiment-guard` hook blocks unlabeled runs.

## Repo structure

```
software/nn/         C++20 SNN/ML framework — primary codebase
  CLAUDE.md          Full technical rules (build, layers, SNN, tests, wiki)
  AGENTS.md          OpenCode guidance (build commands, conventions)
  .wiki/             Documentation wiki
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

## Explanations must be didactic (PERMANENT)

Any answer to "what does X mean / why is this like this", any wiki page, and any rationale
comment follows `/didactic-explanation`: problem before definition, one concrete example
with real numbers, the structure drawn rather than described, confusable pairs contrasted
side by side, and the failure mode named as loud or silent. If the user asks twice, the
first answer failed — rebuild it from first principles instead of rephrasing.

Reference page: `software/nn/.wiki/Concepts/Time-Steps.md`.

---

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
| `/didactic-explanation` | **Always on** for any explanation, wiki page, or "why" comment |
| `/agent-performance-enforcer` | End-of-task gates |
| `/validation-sequencing-enforcer` | Fastest-first validation |

## Wiki maintenance rule

After any layer, loss, optimizer, or training feature change: update `.wiki/Core/` and `.wiki/References.md`. The `wiki-sync-reminder` hook prompts this after every source edit.

## CI

`.github/workflows/ci.yml` — runs on push/PR. Uses cmake + ninja + clang-tidy + cppcheck. Check CI before marking work done.
