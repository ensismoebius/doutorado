---
description: "Find and summarize state-of-the-art code references and implementation patterns for a target problem."
---

# state-of-the-art-code-reference-search

Retrieve high-quality, recent, and reproducible code references for a specific technical task.

## Project Context (nn framework)

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

**Wiki & knowledge graph:**
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

## Rules

- **QUERY_PRECISION**: Build narrow queries with domain, method, metric, and constraints. No broad keyword-only searches.
- **SOURCE_QUALITY**: Prioritize peer-reviewed papers, official docs, maintained repositories, and benchmark leaders. No unverified blog snippets as primary references.
- **RECENCY_BALANCE**: Prefer recent references while preserving foundational canonical methods. No recency-only selection bias.
- **CODE_VERIFIABILITY**: Prefer references with runnable code, license clarity, and dependency transparency. No non-reproducible pseudo-code-only sources.
- **COMPARATIVE_SUMMARY**: Return concise comparison tables (method, complexity, memory, constraints, links). No long narrative dumps.
- **TOKEN_EFFICIENCY**: Summarize findings in compact bullets. Keep only actionable excerpts. No copying large raw passages.

## Workflow

1. Define target problem and constraints (hardware, latency, memory, dataset type).
2. Build query set (canonical terms + modern variants).
3. Collect candidate references from authoritative sources (arXiv, IEEE, ACM, NeurIPS, ICML, ICLR, GitHub).
4. Filter by quality, recency, reproducibility, and relevance.
5. Produce a compact shortlist with implementation guidance.

## Output Format

```
| Method | Complexity | Memory | Constraints | Link |
|--------|-----------|--------|-------------|------|
| ...    | ...       | ...    | ...         | ...  |
```

Then for each top pick: 2–3 bullet points on why it matters and where it applies.

## Validation

- At least 3 high-quality references returned for non-trivial topics.
- Each reference includes why it matters and where it applies.
- Output includes trade-offs (speed, memory, complexity, reproducibility).
