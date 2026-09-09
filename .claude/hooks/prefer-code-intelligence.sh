#!/usr/bin/env bash
# prefer-code-intelligence.sh — PreToolUse hook.
#
# Both CLAUDE.md files mandate the `code_intelligence` MCP over raw
# Read/grep/find/cmake/ctest/git for anything about the code itself. This hook
# enforces it and points at the *right* MCP tool for the job.
#
#   BLOCK (exit 2)  — Grep tool; repo-tree grep/rg/egrep/find; ctest;
#                     cmake --build / --preset  (no `# raw-*-ok` marker)
#   NUDGE (exit 0 + additionalContext) — git log/blame/diff; whole-file Read /
#                     cat / sed / head / tail of a source file
#
# NEVER touches: running built binaries, python scripts, git commit/status/add,
# pkill, file writes, or any command carrying `# raw-cmake-ok` / `# raw-grep-ok`.
#
# The MCP surface is large — the messages below are a map, not the whole list.
# Full reference: ~/dotfiles/code_intelligence/docs/mcp.md

set -euo pipefail

input="$(cat)"
tool="$(printf '%s' "$input" | jq -r '.tool_name // empty')"

# ---- shared guidance map (kept short; grouped by what you are trying to do) ---
CI_MAP='code_intelligence MCP tools by intent:
  orient   workspace_snapshot | repository_summary | workspace_status
  locate   search_text | find_symbol | list_symbols | list_files | ast_search ("foo($A,$B)")
  read     symbol_source | get_source_range | outline_symbol | get_file_structure | block_range | inspect
  impact   find_references | find_dependencies | impact | affected_tests | context_for_task | change_summary | clang_references (exact C++)
  build    run_build | run_tests | diagnose | compare_baseline | run_lint | run_format
  quality  get_violations | summarize_violations | rank_symbols | file_report
  edit     edit_and_validate | transaction | replace_symbol | insert_lines | ast_replace | rename_symbol | task/plan/execute_plan
  repo     git_status | git_log | git_blame | git_diff_stat'

case "$tool" in
  Bash) cmd="$(printf '%s' "$input" | jq -r '.tool_input.command // empty')" ;;
  Grep)
    { printf 'code_intelligence MCP preferred over Grep — hits carry their enclosing symbol and skip build/out/_deps:\n'
      printf '  text -> mcp__code_intelligence__search_text     structural -> mcp__code_intelligence__ast_search\n'
      printf '  symbol by name -> mcp__code_intelligence__find_symbol\n\n%s\n' "$CI_MAP"
      printf '\nIf the index genuinely cannot serve this, rerun as Bash `rg ...` with `# raw-grep-ok`.\n'
    } >&2
    exit 2 ;;
  Read)
    fp="$(printf '%s' "$input" | jq -r '.tool_input.file_path // empty')"
    off="$(printf '%s' "$input" | jq -r '.tool_input.offset // empty')"
    lim="$(printf '%s' "$input" | jq -r '.tool_input.limit // empty')"
    case "$fp" in
      *.cpp|*.cc|*.cxx|*.hpp|*.hh|*.h|*.py|*.js|*.ts|*.jsx|*.tsx|*.java|*.php|*.go|*.rs)
        if [ -z "$off" ] && [ -z "$lim" ]; then
          printf '{"hookSpecificOutput":{"hookEventName":"PreToolUse","additionalContext":"Whole-file Read of a source file. Prefer code_intelligence: get_file_structure / outline_symbol for the map, then symbol_source (by name) or get_source_range / block_range for the part you need. inspect(path|symbol) is the one-call adaptive read. Full-file Read is fine when you truly need every line."}}\n'
          exit 0
        fi ;;
    esac
    exit 0 ;;
  *) exit 0 ;;
esac

[ -z "${cmd:-}" ] && exit 0

case "$cmd" in
  *"# raw-cmake-ok"*|*"# raw-grep-ok"*) exit 0 ;;
esac

first="$(printf '%s' "$cmd" | sed -E 's/^[[:space:]]*(sudo[[:space:]]+)?(env[[:space:]]+[A-Za-z_]+=[^[:space:]]*[[:space:]]+)*//' | awk '{print $1}')"
base="${first##*/}"

block() { { printf '%s\n\n' "$1"; printf '%s\n' "$CI_MAP"; } >&2; exit 2; }

case "$base" in
  grep|rg|egrep|fgrep)
    block "code_intelligence MCP preferred: search_text (text, enclosing-symbol tagged) / find_symbol (symbols) / ast_search (structural). Genuinely need raw grep? add \`# raw-grep-ok\`." ;;
  find)
    case "$cmd" in
      *" -name "*|*" -iname "*|*" -path "*|*" -regex "*)
        block "code_intelligence MCP preferred: list_files / find_symbol / search_text over the source tree. Genuinely need raw find? add \`# raw-grep-ok\`." ;;
    esac ;;
  ctest)
    block "code_intelligence MCP preferred: run_tests (structured {status,counts,failed_tests}) or diagnose (failures mapped to enclosing symbol). Genuinely need raw ctest? add \`# raw-cmake-ok\`." ;;
  cmake)
    case "$cmd" in
      *"--build"*|*"--preset"*)
        block "code_intelligence MCP preferred: run_build (structured errors) or diagnose. Genuinely need raw cmake? add \`# raw-cmake-ok\`." ;;
    esac ;;
  git)
    case "$cmd" in
      "git log"*|"git blame"*|"git diff"*|*" git log "*|*" git blame "*|*" git diff "*)
        printf '{"hookSpecificOutput":{"hookEventName":"PreToolUse","additionalContext":"Prefer code_intelligence: git_log / git_blame / git_diff_stat / git_status, or change_summary / compare_baseline for a semantic diff vs a ref. Raw git is fine when you need exact formatting."}}\n'
        exit 0 ;;
    esac ;;
  cat|sed|head|tail|less|more|bat)
    case "$cmd" in
      *.cpp*|*.cc*|*.cxx*|*.hpp*|*.hh*|*.h\ *|*.h\"*|*.py*|*.js*|*.ts*|*.java*|*.php*)
        printf '{"hookSpecificOutput":{"hookEventName":"PreToolUse","additionalContext":"Reading a source file through the shell. Prefer code_intelligence: get_file_structure then symbol_source / get_source_range / block_range, or inspect(path|symbol). Shell text tools are fine for non-source files, logs, and results/."}}\n'
        exit 0 ;;
    esac ;;
esac

exit 0
