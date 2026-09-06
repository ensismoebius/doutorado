
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

**Code intelligence (MCP `code_intelligence`) — prefer over grep/manual commands for anything about the code itself:**
- `find_symbol` / `search_text` / `list_symbols` — resolve/search/enumerate symbols in indexed files, each hit tagged with its enclosing symbol (replaces `rg`/`grep`/`find` for anything already indexed)
- `get_source_range` / `symbol_source` / `outline_symbol` — exact, budget-checked source instead of a full-file read (`{"truncated": true, "recommended_ranges": [...]}` on overflow — read what it recommends, don't guess smaller)
- `find_references` / `find_dependencies` — callers/callees marked `"exact"` (real compiler) or `"heuristic"` (name-matching) — never read a heuristic "0 callers" as dead code
- `get_violations` / `rank_symbols` / `rename_symbol` — structural findings, complexity hotspots, and gated multi-site renames
- `ast_search` / `ast_replace` — AST-pattern structural search and rewrite (`foo($A, $B)` matches a 2-arg call to `foo` regardless of formatting/argument names) — prefer over a regex `search_text`/`rg` for anything shaped like code structure rather than text
- `run_build` / `run_tests` / `run_lint` / `run_format` / `detect_toolchain` — structured build/test/lint output, not raw logs (`run_lint`/`run_format` cover Python only; C++ still goes through `analysis-all`/`clang-format-changed.sh`)
- `git_status` / `git_log` / `git_blame` / `git_diff_stat` / `compare_baseline` — repo state/history/diff without shelling out to `git`

**Wiki & knowledge graph** (concepts, papers, docs — not code symbols; use the MCP tools above for those):
- Documentation at `.wiki/` — theory, guides, experiment pages, concept definitions
- Graph output at `.wiki/graphify-out/` — 1926 nodes, 4987 edges, 203 communities
- Find any symbol/concept:
```bash
python3 -c "
import json,sys
with open('.wiki/graphify-out/graph.json') as f: g=json.load(f)
q=sys.argv[1].lower()
for n in g['nodes']:
    if q in n['id'].lower() or q in n.get('label','').lower():
        print(n['id'],'|',n.get('source_file',''),'|',n.get('source_location',''))
" <QUERY>
```
- Workflow: `GRAPH_REPORT.md` → community → node → `source_file` → read → follow edges

Checklist

1. Trace the `--help` code path — does it touch any file I/O or side-effectful code?
2. List all required arguments and verify each is validated before use.
3. Confirm that CLI flag values override config-file values (or the opposite) with documented, explicit precedence.
4. Test invalid inputs: missing required args, out-of-range values — verify each produces a clear diagnostic and non-zero exit.
