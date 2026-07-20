---
name: navigation
description: "Skill to locate files, symbols, tests, and targets quickly with minimal tool calls."
---

# navigation

Goal
- Find the right file/symbol fast, with reproducible search steps.

Rules
- RULE: SEARCH_NARROWLY
  DO: Start in `src/`, `include/`, `cmake/`, `scripts/`.
  AVOID: `build/`, `_deps/`, generated outputs unless requested.
- RULE: CHEAP_TO_DEEP
  DO: `file_search` -> `grep_search` -> `read_file`.
  AVOID: Reading large files before locating symbol anchors.
- RULE: USAGE_CONFIRMATION
  DO: Use symbol usage tools for cross-file impact.
  AVOID: Assumptions from a single occurrence.
- RULE: PERF_GATE
  DO: When locating hot paths, include allocation and loop hotspots explicitly.
  AVOID: Search plans that ignore memory/CPU cost centers.

Workflow
1. Locate filenames.
2. Locate symbols/text.
3. Read only needed ranges.
4. Confirm references and tests.

Validation
- Candidate list includes implementation + header + tests.
- Search scope excludes irrelevant directories.

Project Context (nn framework)
**Key file index:**

| What | Where |
|---|---|
| Module base | `include/layers/base/Module.hpp` |
| LIF single-step | `include/layers/spiking/Leaky.hpp` |
| LIF BPTT | `include/layers/spiking/LeakyBPTT.hpp` |
| Trainer | `src/core/training/Trainer.hpp` |
| Exp04 profile parser | `src/experiments/guayaquil/lib/include/ComparativeConfig.hpp` |
| Exp04 encoding transforms | `src/experiments/guayaquil/lib/src/ComparativeEncoding.cpp` |
| KFold / NestedKFold | `include/statistics/kfold.hpp` |
| OpenCL context | `include/tensor/opencl/OpenCLContext.hpp` |

**SNN search anchors** — grep for these to find the relevant code:
- `v_mem_history`, `spike_history` — membrane and spike recording in LeakyBPTT
- `time_steps`, `surrogate_grad`, `readout_mode` — SNN config fields
- `R_min`, `C_min` — biophysical parameter clamp sites

**Experiment paths:**
- `src/experiments/paraconsistentBaseline/` — wavelet + paraconsistent baseline
- `src/experiments/waveletAE/` — wavelet autoencoder
- `src/experiments/autoencoderRunner/` — multimodal autoencoder
- `src/experiments/guayaquil/` — LSTM vs SNN comparative

**Article pipeline chain:**
`scripts/pipeline/guayaquil/01_guayaquil_run_article_profiles.sh` → CSVs → `scripts/pipeline/guayaquil/02_guayaquil_build_lstm_vs_snn_paper_data.py` → DAT → `pdflatex paper.tex`

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
