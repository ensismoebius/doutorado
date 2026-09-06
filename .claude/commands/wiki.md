
## MANDATORY FIRST STEP — Web search (automatic, no exceptions)

**Do this before anything else. Do NOT ask the user. Do NOT skip.**

1. Search official docs for every tool/API/component you will touch
2. Search for known bugs, changelogs, breaking changes
3. Search GitHub issues / forums for the exact error or behavior
4. Find working real-world examples

Training-data knowledge is outdated. Search first, implement second. Always.



# wiki

Goal
- Create a complete, self-contained wiki documenting the nn project from theory to implementation. A reader with no source access must be able to understand the entire system.

Project Context (nn framework)

**Required Experiment pages** (`.wiki/Experiments/`): Experiment00 through Experiment04. All must be linked from `Home.md`.

**Required Concepts pages** (`.wiki/Concepts/`):
- `Paraconsistent-Logic.md` — Da Costa framework (novel thesis contribution)
- `Time-Major-Layout.md` — `(T*B, F)` tensor convention
- `Surrogate-Gradients.md` — exponential/boxcar surrogate for SNN backward
- `Membrane-Dynamics.md` — LIF RC circuit, β = exp(−Δt/(RC))
- `Wavelet-Decomposition.md` — wavelet packet feature extraction

**Staleness check** — before editing a page that documents a specific source file, compare `git_log(path=<source file>)` against `git_log(path=<wiki page>)` (MCP): if the source's latest commit is newer than the wiki page's, the page's code snippets/behavior claims may already be stale — re-verify against current source rather than trusting what the page says. Cheaper than diffing the whole file by eye.

**Orphan check** — every wiki page must have ≥1 backlink (except `Home.md`):
```bash
cd .wiki && python3 -c "
import os, re
pages = []
for root, dirs, files in os.walk('.'):
    for f in files:
        if f.endswith('.md'): pages.append(os.path.join(root,f).lstrip('./'))
link_pat = re.compile(r'\[.*?\]\(([^)]+\.md[^)]*)\)')
backlinks = {p: set() for p in pages}
for src in pages:
    for m in link_pat.finditer(open(src).read()):
        href = m.group(1).split('#')[0]
        tgt = os.path.normpath(os.path.join(os.path.dirname(src), href))
        if tgt in backlinks: backlinks[tgt].add(src)
orphans = [p for p in pages if not backlinks[p] and p != 'Home.md']
print('Orphans:', orphans or 'none')
"
```

**Code intelligence (MCP `code_intelligence`) — prefer over grep/manual commands for anything about the code itself:**
- `find_symbol` / `search_text` / `list_symbols` — resolve/search/enumerate symbols in indexed files, each hit tagged with its enclosing symbol (replaces `rg`/`grep`/`find` for anything already indexed)
- `get_source_range` / `symbol_source` / `outline_symbol` — exact, budget-checked source instead of a full-file read (`{"truncated": true, "recommended_ranges": [...]}` on overflow — read what it recommends, don't guess smaller)
- `find_references` / `find_dependencies` — callers/callees marked `"exact"` (real compiler) or `"heuristic"` (name-matching) — never read a heuristic "0 callers" as dead code
- `get_violations` / `rank_symbols` / `rename_symbol` — structural findings, complexity hotspots, and gated multi-site renames
- `ast_search` / `ast_replace` — AST-pattern structural search and rewrite (`foo($A, $B)` matches a 2-arg call to `foo` regardless of formatting/argument names) — prefer over a regex `search_text`/`rg` for anything shaped like code structure rather than text; useful here to confirm a code snippet's exact shape still exists in source before pasting it into a page (the "compile-verified" quality gate)
- `run_build` / `run_tests` / `run_lint` / `run_format` / `detect_toolchain` — structured build/test/lint output, not raw logs (`run_lint`/`run_format` cover Python only; C++ still goes through `analysis-all`/`clang-format-changed.sh`)
- `git_status` / `git_log` / `git_blame` / `git_diff_stat` / `compare_baseline` — repo state/history/diff without shelling out to `git`

Phase 1: Directory Layout

Create this structure under `.wiki/`:

```
.wiki/
├── Home.md                  ← project overview, table of contents, quick-start
├── Architecture.md          ← high-level system diagram and module map
├── Core/
│   ├── Tensor.md
│   ├── Layers.md
│   ├── Optimizers.md
│   ├── DataLoaders.md
│   └── ...one file per src/core module
├── Experiments/
│   ├── AutoencoderRunner.md
│   ├── Guayaquil.md
│   └── ...one file per experiment
├── Concepts/
│   ├── LSTM-and-BPTT.md
│   ├── SNN-and-Surrogate-Gradients.md
│   ├── Autoencoders.md
│   └── ...one file per major concept
└── References.md            ← all bibliographic citations in IEEE format
```

Phase 2: Article Template

Every article MUST contain, in order:

a) **Title (H1)** and one-paragraph summary
b) **"Theoretical Background"** — first-principles explanation, ≥2 peer-reviewed citations using [Author, Year]
c) **"How It Is Implemented Here"** — maps theory to source files/functions with code snippets ≤30 lines
d) **"Data Flow"** — Mermaid diagram showing inputs → processing → outputs with tensor shapes
e) **"Usage Example"** — minimal runnable code with comments
f) **"Common Pitfalls"** — ≥3 known failure modes and how to avoid them
g) **"See Also"** — links to related wiki articles and external refs
h) **"References"** — IEEE format citations

Phase 3: Citation Rules

- RULE: DIDACTIC
  DO: Every page follows `/didactic-explanation` — open with the problem the thing solves,
      one concrete example with real project numbers carried throughout, the structure
      DRAWN (ASCII/mermaid), confusable pairs contrasted in a table, and the failure mode
      named as loud or silent. `.wiki/Concepts/Time-Steps.md` is the reference.
  AVOID: Never open a page with a definition or a signature; never describe a data layout
      in prose alone.

- Use web search to find canonical paper for every major algorithm.
- Every claim about algorithm behavior MUST cite its source.
- Prefer: arXiv, IEEE, ACM, NeurIPS, ICML, ICLR.
- Do NOT cite Wikipedia as a primary source.

Phase 4: Writing Standards

- Explain as if teaching a first-year grad student new to neural networks.
- Use analogies for abstract concepts.
- Define every symbol on first use.
- Equations: LaTeX inside `$...$` or `$$...$$`.
- Code snippets: specify file path as comment on first line.
- Use Mermaid for diagrams (`flowchart TD` or `sequenceDiagram`).
- Keep each article 600–2000 words; split longer content.

Phase 5: Execution Steps

### STEP 1 — RECONNAISSANCE
Enumerate with `list_files`/`get_workspace_structure` (MCP) — already
excludes `build/`/`_deps/` and matches this project's own file inventory,
no manual `! -path` filtering needed:
```bash
find . -type f \( -name "*.hpp" -o -name "*.cpp" -o -name "*.md" \) \
  ! -path "./build/*" ! -path "./_deps/*" | sort
```
`get_file_structure`/`list_symbols` (MCP) gives every file's symbols
(kind, location, LOC, has_doc) without opening it — use it to triage which
files are substantial enough to need a full read first. Read every
README.md and public header under `include/nn/`. Read full source of
every file under `src/experiments/`. Do NOT start writing until the full
codebase is read — the triage above orders the reading, it doesn't
shorten it.

### STEP 2 — WEB RESEARCH
Collect canonical citations for: LSTM, BPTT, SNNs + surrogate gradients, Autoencoders, Adam, Xavier/Glorot init, Kaiming/He init, ResNet skip connections, k-fold cross-validation, ReduceLROnPlateau, EEG/BCI signal processing, imagined-speech EEG decoding.

### STEP 3 — WRITE IN THIS ORDER
1. `.wiki/References.md` — populate from research
2. `.wiki/Home.md` — project overview + TOC
3. `.wiki/Architecture.md` — system-wide diagram
4. Core articles (Tensor, Layers, Optimizers, DataLoaders, ...)
5. Concept articles (Autoencoders, LSTM-and-BPTT, SNN-and-Surrogate-Gradients, ...)
6. Experiment articles (AutoencoderRunner, Experiment04, ...)
7. `.wiki/Home.md` — final pass to add all links

### STEP 4 — CROSS-LINK
Every article must link to at least 2 other wiki articles using relative Markdown links: `[Tensor](../Core/Tensor.md)`.

### STEP 5 — VALIDATE
- List all `.wiki/` files, verify layout matches the structure above.
- Confirm all 7 required sections in each article.
- Confirm `References.md` has an entry for every citation used.
- Fix missing sections or broken links before finishing.

Quality Gates (check before saving each file)

- [ ] Every H2 section is present
- [ ] At least 2 citations per article
- [ ] All code snippets copied from actual source (compile-verified)
- [ ] Mermaid diagram has at least 4 nodes
- [ ] No article references a non-existent file in the repository
- [ ] References section uses IEEE format
