
# dev-tooling-integrator

Goal
- Use installed tooling to minimize repetitive manual steps and avoid token-heavy workarounds.

Rules

- RULE: SEARCH_TOOLING
  DO: Use `rg` for code/text search and file discovery
  AVOID: No slow broad `find` + `grep` combos when `rg` suffices
- RULE: JSON_TOOLING
  DO: Use `jq` to validate/query JSON configs
  AVOID: No manual JSON parsing or string matching
- RULE: LOOP_AUTOMATION
  DO: Use `entr` or equivalent for watch/rebuild loops
  AVOID: No re-running repetitive commands manually
- RULE: INDEX_SUPPORT
  DO: Use `ctags`/symbol indexes for fast navigation where useful
  AVOID: No repeated whole-file reads when a symbol index is available

Validation

- Proposed workflows use existing local tools first before reaching for heavier alternatives.
- No manual JSON parsing where `jq` applies.

Project Context (nn framework)

**Installed tools and their roles:**
- `rg` (ripgrep) — primary code/text search; use instead of `find + grep`
- `clang-format` — via `scripts/dev/clang-format-changed.sh`; staged files only
- `ccache` — wired in CMake presets; clear with `cmake --build ... --target clean-cache`
- `ctest` — test runner; use `-R <pattern>` to target specific tests
- `graphify` — knowledge graph generation; see `.opencode/plugins/graphify.js`

**Static analysis:**
- `cmake --build ... --target analysis-all` — runs cppcheck + clang-tidy
- `python3 scripts/ci/validate_static_analysis.py --list-approved` — show suppression allowlist

Common Invocations

```bash
# Fast symbol search
rg "class Tensor" include/ src/

# Query a JSON config field
jq '.batch_size' results/guayaquil/config.json

# Watch and rebuild on change
find src/ include/ -name "*.cpp" -o -name "*.hpp" | entr cmake --build build --target nn -j4

# Regenerate ctags index
ctags -R --c++-kinds=+p --fields=+iaS --extras=+q src/ include/
```
