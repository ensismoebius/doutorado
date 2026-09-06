
# state-of-the-art-code-reference-search

Goal
- Retrieve high-quality, recent, and reproducible code references for a specific technical task.

Rules

- RULE: QUERY_PRECISION
  DO: Build narrow queries with domain, method, metric, and constraints
  AVOID: No broad keyword-only searches
- RULE: SOURCE_QUALITY
  DO: Prioritize peer-reviewed papers, official docs, maintained repositories, and benchmark leaders
  AVOID: No unverified blog snippets as primary references
- RULE: RECENCY_BALANCE
  DO: Prefer recent references while preserving foundational canonical methods
  AVOID: No recency-only selection bias
- RULE: CODE_VERIFIABILITY
  DO: Prefer references with runnable code, license clarity, and dependency transparency
  AVOID: No non-reproducible pseudo-code-only sources
- RULE: COMPARATIVE_SUMMARY
  DO: Return concise comparison tables (method, complexity, memory, constraints, links)
  AVOID: No long narrative dumps
- RULE: TOKEN_EFFICIENCY
  DO: Summarize findings in compact bullets. Keep only actionable excerpts
  AVOID: No copying large raw passages

Workflow

1. Define target problem and constraints (hardware, latency, memory, dataset type).
2. Build query set (canonical terms + modern variants).
3. Collect candidate references from authoritative sources (arXiv, IEEE, ACM, NeurIPS, ICML, ICLR, GitHub).
4. Filter by quality, recency, reproducibility, and relevance.
5. Produce a compact shortlist with implementation guidance.

Validation

- At least 3 high-quality references returned for non-trivial topics.
- Each reference includes why it matters and where it applies.
- Output includes trade-offs (speed, memory, complexity, reproducibility).

Project Context (nn framework)

**Domain:** neuromorphic computing + SNN autoencoders for EEG/BCI signal reconstruction

**Key venues for relevant literature:**
- NeurIPS, ICLR — surrogate gradients, SNN training methods
- IEEE TNNLS, IEEE TNN — spiking neural network applications
- Frontiers in Neuroscience — BCI, EEG decoding
- SPIE proceedings — the 10.1117 dataset paper

**Key foundational papers to verify are cited correctly:**
- Surrogate gradients: Neftci, Mostafa, Zenke (2019) — IEEE Signal Processing Magazine
- LSNN (adaptive SNNs): Bellec et al. (2018) — NeurIPS
- Spike encoding (rate/latency): review papers in IEEE TNNLS
- Paraconsistent logic: Da Costa foundational work (mathematical logic journals)

**Code intelligence (MCP `code_intelligence`) — prefer over grep/manual commands for anything about the code itself:**
- `find_symbol` / `search_text` / `list_symbols` — resolve/search/enumerate symbols in indexed files, each hit tagged with its enclosing symbol (replaces `rg`/`grep`/`find` for anything already indexed)
- `get_source_range` / `symbol_source` / `outline_symbol` — exact, budget-checked source instead of a full-file read (`{"truncated": true, "recommended_ranges": [...]}` on overflow — read what it recommends, don't guess smaller)
- `find_references` / `find_dependencies` — callers/callees marked `"exact"` (real compiler) or `"heuristic"` (name-matching) — never read a heuristic "0 callers" as dead code
- `get_violations` / `rank_symbols` / `rename_symbol` — structural findings, complexity hotspots, and gated multi-site renames
- `ast_search` / `ast_replace` — AST-pattern structural search and rewrite (`foo($A, $B)` matches a 2-arg call to `foo` regardless of formatting/argument names) — prefer over a regex `search_text`/`rg` for anything shaped like code structure rather than text
- `run_build` / `run_tests` / `run_lint` / `run_format` / `detect_toolchain` — structured build/test/lint output, not raw logs (`run_lint`/`run_format` cover Python only; C++ still goes through `analysis-all`/`clang-format-changed.sh`)
- `git_status` / `git_log` / `git_blame` / `git_diff_stat` / `compare_baseline` — repo state/history/diff without shelling out to `git`

Output Format

```
| Method | Complexity | Memory | Constraints | Link |
|--------|-----------|--------|-------------|------|
| ...    | ...       | ...    | ...         | ...  |
```

Then for each top pick: 2–3 bullet points on why it matters and where it applies.
