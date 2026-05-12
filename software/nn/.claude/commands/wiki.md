---
description: "Create a comprehensive codebase wiki under .wiki/ with theory, implementation, data flow, and IEEE-cited references."
---

# wiki

Create a complete, self-contained wiki documenting the nn project from theory to implementation. A reader with no source access must be able to understand the entire system.

## Project Context (nn framework)

**Required Experiment pages** (`.wiki/Experiments/`): Experiment00 through Experiment04. All must be linked from `Home.md`.

**Required Concepts pages** (`.wiki/Concepts/`):
- `Paraconsistent-Logic.md` — Da Costa framework (novel thesis contribution)
- `Time-Major-Layout.md` — `(T*B, F)` tensor convention
- `Surrogate-Gradients.md` — exponential/boxcar surrogate for SNN backward
- `Membrane-Dynamics.md` — LIF RC circuit, β = exp(−Δt/(RC))
- `Wavelet-Decomposition.md` — wavelet packet feature extraction

**Knowledge graph:** `.wiki/graphify-out/` — 1851 nodes, 4205 edges, 174 communities. Auto-generated; do not manually edit. Re-run graphify when structure changes significantly.

**Orphan check** — every wiki page must have ≥1 backlink (except `Home.md`):
```bash
cd .wiki && python3 -c "
import os, re
pages = []
for root, dirs, files in os.walk('.'):
    dirs[:] = [d for d in dirs if d != 'graphify-out']
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

## Phase 1: Directory Layout

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
│   ├── Experiment03.md
│   ├── Experiment04.md
│   └── ...one file per experiment
├── Concepts/
│   ├── LSTM-and-BPTT.md
│   ├── SNN-and-Surrogate-Gradients.md
│   ├── Autoencoders.md
│   └── ...one file per major concept
└── References.md            ← all bibliographic citations in IEEE format
```

## Phase 2: Article Template

Every article MUST contain, in order:

a) **Title (H1)** and one-paragraph summary  
b) **"Theoretical Background"** — first-principles explanation, ≥2 peer-reviewed citations using [Author, Year]  
c) **"How It Is Implemented Here"** — maps theory to source files/functions with code snippets ≤30 lines  
d) **"Data Flow"** — Mermaid diagram showing inputs → processing → outputs with tensor shapes  
e) **"Usage Example"** — minimal runnable code with comments  
f) **"Common Pitfalls"** — ≥3 known failure modes and how to avoid them  
g) **"See Also"** — links to related wiki articles and external refs  
h) **"References"** — IEEE format citations

## Phase 3: Citation Rules

- Use web search to find canonical paper for every major algorithm.
- Every claim about algorithm behavior MUST cite its source.
- Prefer: arXiv, IEEE, ACM, NeurIPS, ICML, ICLR.
- Do NOT cite Wikipedia as a primary source.

## Phase 4: Writing Standards

- Explain as if teaching a first-year grad student new to neural networks.
- Use analogies for abstract concepts.
- Define every symbol on first use.
- Equations: LaTeX inside `$...$` or `$$...$$`.
- Code snippets: specify file path as comment on first line.
- Use Mermaid for diagrams (`flowchart TD` or `sequenceDiagram`).
- Keep each article 600–2000 words; split longer content.

## Phase 5: Execution Steps

### STEP 1 — RECONNAISSANCE
```bash
find . -type f \( -name "*.hpp" -o -name "*.cpp" -o -name "*.md" \) \
  ! -path "./build/*" ! -path "./_deps/*" | sort
```
Read every README.md and public header under `include/nn/`. Read full source of every file under `src/experiments/`. Do NOT start writing until the full codebase is read.

### STEP 2 — WEB RESEARCH
Collect canonical citations for: LSTM, BPTT, SNNs + surrogate gradients, Autoencoders, Adam, Xavier/Glorot init, Kaiming/He init, ResNet skip connections, k-fold cross-validation, ReduceLROnPlateau, EEG/BCI signal processing, imagined-speech EEG decoding.

### STEP 3 — WRITE IN THIS ORDER
1. `.wiki/References.md` — populate from research
2. `.wiki/Home.md` — project overview + TOC
3. `.wiki/Architecture.md` — system-wide diagram
4. Core articles (Tensor, Layers, Optimizers, DataLoaders, ...)
5. Concept articles (Autoencoders, LSTM-and-BPTT, SNN-and-Surrogate-Gradients, ...)
6. Experiment articles (Experiment03, Experiment04, ...)
7. `.wiki/Home.md` — final pass to add all links

### STEP 4 — CROSS-LINK
Every article must link to at least 2 other wiki articles using relative Markdown links: `[Tensor](../Core/Tensor.md)`.

### STEP 5 — VALIDATE
- List all `.wiki/` files, verify layout matches the structure above.
- Confirm all 7 required sections in each article.
- Confirm `References.md` has an entry for every citation used.
- Fix missing sections or broken links before finishing.

## Quality Gates (check before saving each file)

- [ ] Every H2 section is present
- [ ] At least 2 citations per article
- [ ] All code snippets copied from actual source (compile-verified)
- [ ] Mermaid diagram has at least 4 nodes
- [ ] No article references a non-existent file in the repository
- [ ] References section uses IEEE format
