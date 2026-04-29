---
description: "Standardize use of local CLI tools (rg, jq, entr, ctags) to reduce manual complexity and token-heavy workflows."
---

# dev-tooling-integrator

Use installed tooling to minimize repetitive manual steps and avoid token-heavy workarounds.

## Rules

- **SEARCH_TOOLING**: Use `rg` for code/text search and file discovery. No slow broad `find` + `grep` combos when `rg` suffices.
- **JSON_TOOLING**: Use `jq` to validate/query JSON configs. No manual JSON parsing or string matching.
- **LOOP_AUTOMATION**: Use `entr` or equivalent for watch/rebuild loops. No re-running repetitive commands manually.
- **INDEX_SUPPORT**: Use `ctags`/symbol indexes for fast navigation where useful. No repeated whole-file reads when a symbol index is available.

## Common Invocations

```bash
# Fast symbol search
rg "class Tensor" include/ src/

# Query a JSON config field
jq '.batch_size' results/experiment04/config.json

# Watch and rebuild on change
find src/ include/ -name "*.cpp" -o -name "*.hpp" | entr cmake --build build --target nn -j4

# Regenerate ctags index
ctags -R --c++-kinds=+p --fields=+iaS --extras=+q src/ include/
```

## Validation

- Proposed workflows use existing local tools first before reaching for heavier alternatives.
- No manual JSON parsing where `jq` applies.
