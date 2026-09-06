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
| Exp04 profile parser | `src/experiments/guayaquil/lib/include/ComparativeConfig.hpp` |
| Exp04 encoding transforms | `src/experiments/guayaquil/lib/src/ComparativeEncoding.cpp` |
| KFold / NestedKFold | `include/statistics/kfold.hpp` |
| OpenCL context | `include/tensor/opencl/OpenCLContext.hpp` |

**SNN search anchors** — grep for these to find the relevant code:
- `v_mem_history`, `spike_history` — membrane and spike recording in LifBPTT
- `time_steps`, `surrogate_grad`, `readout_mode` — SNN config fields
- `R_min`, `C_min` — biophysical parameter clamp sites

**Experiment paths:**
- `src/experiments/paraconsistentBaseline/` — wavelet + paraconsistent baseline
- `src/experiments/waveletAE/` — wavelet autoencoder
- `src/experiments/autoencoderRunner/` — multimodal autoencoder
- `src/experiments/guayaquil/` — LSTM vs SNN comparative

**Article pipeline chain:**
`scripts/pipeline/guayaquil/01_guayaquil_run_article_profiles.sh` → CSVs → `scripts/pipeline/guayaquil/02_guayaquil_build_lstm_vs_snn_paper_data.py` → DAT → `pdflatex paper.tex`

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
