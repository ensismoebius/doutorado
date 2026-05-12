---
description: "Locate files, symbols, tests, and targets quickly with minimal tool calls."
---

# navigation

Find the right file/symbol fast, with reproducible search steps.

## Project Context (nn framework)

**Key file index:**

| What | Where |
|---|---|
| Module base | `include/layers/base/Module.hpp` |
| LIF single-step | `include/layers/spiking/Lif.hpp` |
| LIF BPTT | `include/layers/spiking/LifBPTT.hpp` |
| Trainer | `src/core/training/Trainer.hpp` |
| Exp04 profile parser | `src/experiments/04/lib/include/ComparativeConfig.hpp` |
| Exp04 encoding transforms | `src/experiments/04/lib/src/ComparativeEncoding.cpp` |
| KFold / NestedKFold | `include/statistics/kfold.hpp` |
| OpenCL context | `include/tensor/opencl/OpenCLContext.hpp` |

**SNN search anchors** — grep for these to find the relevant code:
- `v_mem_history`, `spike_history` — membrane and spike recording in LifBPTT
- `time_steps`, `surrogate_grad`, `readout_mode` — SNN config fields
- `R_min`, `C_min` — biophysical parameter clamp sites

**Experiment paths:**
- `src/experiments/00/` — wavelet + paraconsistent baseline
- `src/experiments/02/` — wavelet autoencoder
- `src/experiments/03/` — multimodal autoencoder
- `src/experiments/04/` — LSTM vs SNN comparative

**Article pipeline chain:**
`scripts/pipeline/run_article_profiles.sh` → CSVs → `scripts/pipeline/build_paper_data.py` → DAT → `pdflatex paper.tex`

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

- **SEARCH_NARROWLY**: Start in `src/`, `include/`, `cmake/`, `scripts/`. Avoid `build/`, `_deps/`, generated outputs unless requested.
- **CHEAP_TO_DEEP**: Use file search → grep search → read file. Never read large files before locating symbol anchors.
- **USAGE_CONFIRMATION**: Use symbol usage tools for cross-file impact. Never assume from a single occurrence.
- **PERF_GATE**: When locating hot paths, include allocation and loop hotspots explicitly. Don't produce search plans that ignore memory/CPU cost centers.

## Workflow

1. Locate filenames with `find src/ include/ cmake/ scripts/ -name "*.hpp" -o -name "*.cpp"` or `rg --files`.
2. Locate symbols/text with `rg <symbol>`.
3. Read only the needed line ranges.
4. Confirm references across all usages and related tests.

## Validation

- Candidate list includes implementation + header + tests.
- Search scope excludes irrelevant directories (`build/`, `_deps/`).
