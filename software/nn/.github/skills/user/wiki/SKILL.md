---
name: user/wiki
description: "Create comprehensive codebase wiki with theory, implementation, and usage documentation."
---

# user/wiki

Goal
- Create a complete, self-contained wiki that documents the nn project from theory to implementation.
- Ensure a reader with no source code access can understand the entire system.

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

## Phase 2: Article Template

Every article MUST contain, in order:

a) **Title (H1)** and one-paragraph summary
b) **"Theoretical Background"** section — explains concept from first principles, assumes undergraduate maths; cites ≥2 peer-reviewed papers using [Author, Year] inline
c) **"How It Is Implemented Here"** section — maps theory to source files/functions with code snippets ≤30 lines
d) **"Data Flow"** section — Mermaid diagram showing inputs → processing → outputs with tensor shapes
e) **"Usage Example"** section — minimal runnable code with comments
f) **"Common Pitfalls"** section — ≥3 known failure modes and how to avoid
g) **"See Also"** section — links to related wiki articles and external refs
h) **"References"** section at bottom in IEEE format

## Phase 3: Citation Rules

- Use web_search to find canonical paper for every major algorithm
- Every claim about algorithm behavior MUST cite its source
- Prefer: arXiv, IEEE, ACM, NeurIPS, ICML, ICLR proceedings
- Do NOT cite Wikipedia as primary source

## Phase 4: Writing Standards

- Explain as if teaching first-year grad student new to neural networks
- Use analogies for abstract concepts
- Avoid jargon without definition; define every symbol on first use
- Equations: LaTeX inside $...$ or $$...$$
- Code snippets: specify file path as comment on first line
- Use Mermaid for diagrams (flowchart TD or sequenceDiagram)
- Keep each article 600–2000 words; split longer content

## Phase 5: Execution Steps

### STEP 1 — RECONNAISSANCE
- Run: `find . -type f \( -name "*.hpp" -o -name "*.cpp" -o -name "*.md" \) | sort`
- Read every README.md and public header under include/
- Read full source of every file under src/experiments/
- Do NOT start writing until you have read the full codebase

### STEP 2 — WEB RESEARCH
Search for and collect canonical citation for each:
- LSTM (Long Short-Term Memory)
- Backpropagation Through Time (BPTT)
- Spiking Neural Networks + surrogate gradients
- Autoencoders (denoising, variational, sparse)
- Adam optimizer
- Xavier / Glorot weight initialisation
- Kaiming / He weight initialisation
- Residual / skip connections (ResNet)
- k-fold cross-validation
- ReduceLROnPlateau scheduler
- EEG signal processing for BCI
- Imagined speech EEG decoding (10.1117 dataset)

### STEP 3 — WRITE IN THIS ORDER
1. .wiki/References.md — populate from research
2. .wiki/Home.md — project overview + TOC
3. .wiki/Architecture.md — system-wide diagram
4. Core articles (Tensor, Layers, Optimizers, DataLoaders, Statistics, Initializers, LinearAlgebra, Wave, Wavelet, Paraconsistent, Saver)
5. Concept articles (Autoencoders, LSTM-and-BPTT, SNN-and-Surrogate-Gradients, Residual-Blocks, Weight-Initialisation, Adam-Optimiser, Data-Normalisation, K-Fold-Cross-Validation)
6. Experiment articles (AutoencoderRunner, Experiment04)
7. .wiki/Home.md — final pass to add links

### STEP 4 — CROSS-LINK
Every article must link to at least 2 other wiki articles using relative Markdown links: [Tensor](../Core/Tensor.md)

### STEP 5 — VALIDATE
- List all .wiki/ files, verify layout matches
- Confirm all 7 required sections in each article
- Confirm References.md has entry for every citation used
- Fix missing sections or broken links before finishing

## Phase 6: Graphify Integration

The wiki is integrated with graphify for knowledge graph navigation:

### Graphify Outputs
Graphify outputs are stored in `.wiki/graphify-out/`:
- `graph.html` - Interactive knowledge graph visualization
- `GRAPH_REPORT.md` - Audit report with god nodes, communities, surprising connections
- `graph.json` - Raw GraphRAG-ready JSON

### Running Graphify
To rebuild the graph:
```bash
cd .wiki
python -m graphify.serve graphify-out/graph.json
```

Or run full pipeline:
```bash
graphify .wiki
```

### MCP Server
For agent access to the knowledge graph, start the MCP server:
```bash
python -m graphify.serve .wiki/graphify-out/graph.json
```

This exposes tools: `query_graph`, `get_node`, `get_neighbors`, `get_community`, `god_nodes`, `graph_stats`, `shortest_path`

### Using Graphify Before Writing
Before creating or updating wiki articles:
1. Query the graph for related concepts: `graphify query "tensor operations"`
2. Find god nodes (most connected): check GRAPH_REPORT.md
3. Discover surprising connections between modules

## Quality Gates (check before saving each file)
- [ ] Every H2 section is present
- [ ] At least 2 citations per article
- [ ] All code snippets compile (copy-paste from actual source)
- [ ] Mermaid diagram has at least 4 nodes
- [ ] No article references non-existent file in repository
- [ ] References section uses IEEE format
