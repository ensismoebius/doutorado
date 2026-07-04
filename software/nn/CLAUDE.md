# nn project — Claude Code rules

## OpenCode Duo Directives (PERMANENT)

**Role split:**
- Claude Code = **Planner + Reviewer**. Designs solutions, writes plans, reviews output, synthesizes.
- OpenCode = **Executor**. Runs tasks via MCP tools (`mcp__opencode__*`). Handles file edits, shell, search.

**Model priority for OpenCode tasks** (free/low-cost first):
1. Gemini / Gemma (Google) — prefer for code generation and analysis
2. Minmax (OpenCode Zen) — prefer for reasoning and debate
3. Other free models available in OpenCode
4. Paid models only as last resort

**Workflow:**
1. Claude plans. Uses free OpenCode models for debate/second-opinion during planning if useful.
2. Claude delegates execution to OpenCode via MCP (`mcp__opencode__run_task` or equivalent).
3. Claude reviews OpenCode output, synthesizes, reports to user.
4. If OpenCode warns about token exhaustion → immediately warn user: "⚠️ OpenCode token limit approaching — reduce scope or start new session."

**Token warning rule:** Any response from OpenCode containing "token", "limit", "context", "quota", or "exhausted" → surface warning to user before continuing.

**MCP not available:** If `mcp__opencode__*` tools absent (OpenCode not running) → warn user: "OpenCode MCP offline. Start OpenCode first, then restart Claude Code." Fall back to solo execution.

---

## Caveman mode

Caveman active (full). Terse responses. Drop articles, filler, hedging. Technical substance exact.
`/caveman lite|full|ultra` to change.

---

## Graphify knowledge graph

Output: `.wiki/graphify-out/` — 1926 nodes, 4987 edges, 203 communities.

```bash
# Find nodes by label
python3 -c "
import json,sys
with open('.wiki/graphify-out/graph.json') as f: g=json.load(f)
q=sys.argv[1].lower()
for n in g['nodes']:
    if q in n['id'].lower() or q in n.get('label','').lower():
        print(n['id'],'|',n.get('source_file',''),'|',n.get('source_location',''))
" <QUERY>

# Find edges from/to node
python3 -c "
import json,sys
with open('.wiki/graphify-out/graph.json') as f: g=json.load(f)
q=sys.argv[1].lower()
for e in g['links']:
    if q in e['source'].lower() or q in e['target'].lower():
        print(e['source'],'--['+e.get('type','?')+']-->',e['target'])
" <NODE_ID>
```

Workflow: `GRAPH_REPORT.md` → community → node → `source_file` → read → follow edges.

---

## Build & test

**Always use presets** — never invent raw cmake flags:

```bash
# Configure (first time or after CMakeLists change)
cmake --preset=max-performance

# Build a specific target
cmake --build out/build/max-performance --target <target> -j$(nproc)

# Run all tests
ctest --test-dir out/build/max-performance --output-on-failure -j4

# Run one test binary directly (fastest iteration)
./out/build/max-performance/path/to/test_binary --gtest_filter="SuiteName.*"
```

**Named build targets** (all phony, use with `--target`):

| Target | What |
|---|---|
| `core_gtest` | All core unit tests |
| `experiment03` | Experiment 03 binary |
| `experiment03_lib` | Experiment 03 library only |
| `experiment04` | Experiment 04 binary |
| `experiment04_lib` | Experiment 04 library only |
| `experiment05` | Experiment 05 binary (thesis primary) |
| `experiment05_lib` | Experiment 05 library only |
| `e05_profile_audit_gtest` | 1099 tests verifying all 157 E05 profiles (140 phase00 wavelet×scale sweep + 16 phase01 + debug) parse + validate |
| `experiment_02` | Experiment 02 binary |
| `trainer_gtest` | Trainer/EpochResult/TrainerConfig tests |
| `profile_audit_gtest` | 25 tests verifying all 5 article profiles parse + validate |
| `nn_progress` | Progress bar library |
| `analysis-cppcheck` | cppcheck static analysis |
| `analysis-clang-tidy` | clang-tidy static analysis |
| `analysis-all` | All static analysis |
| `clean-cache` | Clear ccache |

**Debug build** (for sanitisers or gdb):
```bash
cmake --preset=Clang_20.1.8_x86_64-pc-linux-gnu
cmake --build out/build/Clang_20.1.8_x86_64-pc-linux-gnu --target <target> -j$(nproc)
```

---

## Language & compiler rules

- **C++20** required. `std::span`, `std::ranges`, concepts, designated initialisers all available.
- Compiler: clang preferred (project ships clang preset). `clang-tidy` must pass.
- `-Wall` is on. `-Wno-sign-compare` suppressed. Fix warnings, do not add more suppressions.
- **No raw `new`/`delete`** — use RAII, smart pointers, value types.
- **No naked xtensor includes** in targets that should remain backend-agnostic. Ensure core modules only depend on the `Tensor` interface, not the `XTensorBackend` implementation directly.

---

## Module / Layer contract

Every layer inherits `Module<Backend>`. Required overrides:

```cpp
template <typename Backend>
struct MyLayer : public Module<Backend> {
    using Tensor = typename Module<Backend>::Tensor;

    // REQUIRED
    auto forward(const Tensor& input, bool requires_grad) -> Tensor override;
    auto backward(const Tensor& grad_output) -> Tensor override;

    // If layer has trainable params — return span of raw pointers to member Tensors
    auto params() -> std::span<nn::Tensor*> override;

    // If layer is stateful (SNN) — clear hidden state for next sequence
    void reset_state() override;

    // If layer should be saveable
    auto state_dict() const -> std::map<std::string, nn::Tensor> override;
    void load_state_dict(const std::map<std::string, nn::Tensor>&) override;
};
```

Rules:
- `forward(requires_grad=true)` **must** be called before `backward()` — caches intermediate state.
- `params()` returns raw pointers to **member** tensors (not temporaries). Store params as struct members, not locals.
- `reset_state()` must clear ALL persistent state (e.g., `v_mem`, `adapt_a`, LSTM hidden/cell).
- Stateful layers (SNN, LSTM): call `reset_state()` between independent sequences/batches.

---

## Tensor shape conventions

| Context | Shape | Notes |
|---|---|---|
| Standard 2D | `(rows, cols)` | rows=batch, cols=features |
| SNN time-major | `(T*B, F)` | T time steps, B batch, F features. Row order: t0 rows, then t1 rows, … |
| `T*B % time_steps == 0` | invariant | violated → `std::invalid_argument` |
| 3D data | `(d1, d2*d3)` | stored as 2D; access via `at(d1, d2, d3)` |

Gradient shape always matches forward input shape.

---

## SNN-specific invariants

1. **Time-major layout**: input to `LifBPTTImpl` and `SpikeTimeLossImpl` is `(T*B, F)`, not `(B, T, F)`.
2. **Loss ↔ encoding must match**:
   - Rate-coded → `SpikeCountLoss`
   - Latency-coded → `SpikeTimeLoss`
   - Mixing these reverses gradient direction.
3. **Surrogate arg order**: `LifImpl` and `LifBPTTImpl` constructors take `surrogate_grad` **before** `adapt_decay`/`adapt_coupling`. Wrong type passed → compile error.
4. **SNN lr**: biophysical params (R, C, V_th) need ~10× smaller lr than weights. Use `Adam::attach_with_scales()`. `TrainerConfig::snn_lr_scale = 0.1` documents the intent.
5. **β = exp(−Δt/(R·C))** clamped: R and C are clamped to `1e-6` in forward and grad is zeroed in clamped region. Never let optimizer drive them negative.
6. **readout_mode**: `LifBPTTImpl` with `readout_mode=true` emits `v_mem` directly — no spike/reset. Backward is purely continuous. Don't mix with spike losses.
7. **Experiment04 SNN architecture modes are INPUT TRANSFORMS, not network architecture changes.** `dense`/`conv1d`/`recurrent` in the profile `snn_architectures` list select how the raw signal is pre-processed before entering the shared autoencoder network. All three modes use the same `linear:64:leaky / linear:32:identity` network. `conv1d` = 3-tap smoothing `{0.25, 0.5, 0.25}`; `recurrent` = stand-alone LIF transform on input; `dense` = pass-through.
8. **Experiment04 SNN builder only parses `linear:width[:activation]`.** Entries like `conv1d:64:kernel=3`, `pool1d:4`, `residual` in `encoder_layer_spec`/`decoder_layer_spec` cause `parse_layer_module_spec` to throw at startup. Only `linear` entries are instantiated into the SNN network.

---

## Adding a new layer checklist

1. Header in `include/layers/<category>/MyLayer.hpp`
2. Inherit `Module<Backend>`, implement `forward`, `backward`, `params` (if trainable), `reset_state` (if stateful), `state_dict`/`load_state_dict` (if serialisable)
3. Add to `include/layers/Layers.hpp` convenience alias if needed
4. Add gtest in nearest `tests/` directory, named `mylayer_gtest.cpp`
5. Wire test into `CMakeLists.txt` under `core_gtest` or relevant experiment test target
6. Update `.wiki/Core/Layers.md` with new entry
7. Build: `cmake --build out/build/max-performance --target core_gtest -j$(nproc)`
8. Test: `ctest --test-dir out/build/max-performance -R mylayer --output-on-failure`

---

## Model serialisation

Save/load via `state_dict()` → `NetworkSerializer` or `NnSaver`:

```cpp
// Full model (map<string,Tensor>)
#include "nn/saver/NetworkSerializer.hpp"
NetworkSerializer::save(model.state_dict(), "model.npz");
model.load_state_dict(NetworkSerializer::load("model.npz"));

// Single weight/bias pair (legacy)
#include "nn/saver/NnSaver.hpp"
NnSaver::save("prefix", weights, bias);  // → prefix_weights.npy, prefix_bias.npy
```

---

## Project layout

```
include/          Public headers (backend-agnostic interface)
  layers/
    activations/     ReLU, LeakyReLU, Sigmoid, Tanh
    base/            Module<Backend>
    convolution/     Conv2D, MaxPool2D
    dense/           Linear
    losses/          MSELoss, MAELoss, CrossEntropyLoss, SpikeCountLoss, SpikeTimeLoss
    regularization/  L1/L2 regularizers
    residual/        ResNetBlock
    spiking/         Lif, LifBPTT, ThresholdDependentBatchNorm, PoissonLatentLayer
  optimizers/        Adam, SGD
  statistics/        kfold.hpp (KFold, StratifiedKFold, NestedKFold), metrics
  tensor/            Tensor.hpp, XTensorBackend, OpenCLTensorBackend
  data_loaders/       10.1117/ (audio+EEG), datasets, samplers, sources
  saver/             NnSaver, NetworkSerializer
  wave/              WAV I/O
  wavelet/           Wavelet packet decomposition
  paraconsistent/    Da Costa paraconsistent logic (thesis novel contribution)

src/core/            Implementation + tests
  training/          Trainer.hpp, TrainerConfig.hpp, EpochResult.hpp
  statistics/        kfold.cpp, metrics
  models/autoencoder/ BaseAutoencoder + concrete models

src/experiments/
  00–04/             Independent experiment binaries; each has lib/, tests/, profiles/

results/             Experiment output (JSON, CSV, .npy)
.wiki/               Documentation wiki (keep in sync with code changes)
  graphify-out/      Knowledge graph files
scripts/             Analysis scripts (Python + bash)
cmake/               Modular CMake includes
```

---

## Test conventions

- Framework: **GoogleTest**. File suffix: `*_gtest.cpp`.
- Place tests in `tests/` sibling to the code under test.
- Use `EXPECT_*` over `ASSERT_*` unless subsequent steps would crash.
- SNN tests: always call `layer.reset_state()` between independent forward passes.
- No mocking of internal state — test through the public API only.

---

## Wiki maintenance

When adding/changing any layer, loss, optimizer, or training feature:
1. Update `.wiki/Core/Layers.md` (or relevant Core/ page)
2. Update `.wiki/References.md` if new citations added
3. Update concept page in `.wiki/Concepts/` if theory changed
4. Run graphify if structure changed significantly: check `.opencode/plugins/graphify.js` for invocation

---

## Key file index

| What | Where |
|---|---|
| LIF neuron (single-step) | `include/layers/spiking/Lif.hpp` |
| LIF neuron (full BPTT) | `include/layers/spiking/LifBPTT.hpp` |
| tdBN | `include/layers/spiking/ThresholdDependentBatchNorm.hpp` |
| Poisson VAE latent | `include/layers/spiking/PoissonLatentLayer.hpp` |
| Spike count loss + reg | `include/layers/losses/SpikeCountLoss.hpp` |
| First-spike time loss | `include/layers/losses/SpikeTimeLoss.hpp` |
| Adam optimizer | `include/optimizers/Adam.hpp` |
| SGD optimizer | `include/optimizers/SGD.hpp` |
| Module base | `include/layers/base/Module.hpp` |
| Tensor | `include/tensor/Tensor.hpp` |
| KFold / NestedKFold | `include/statistics/kfold.hpp` |
| Trainer | `src/core/training/Trainer.hpp` |
| TrainerConfig | `src/core/training/TrainerConfig.hpp` |
| EpochResult | `src/core/training/EpochResult.hpp` |
| Surrogate gradients | `include/layers/spiking/ExponentialSurrogate.hpp`, `BoxcarSurrogate.hpp` |
| Paraconsistent logic | `include/paraconsistent/` |
| Wiki | `.wiki/` |
| Graphify output | `.wiki/graphify-out/` |
| CMake presets | `CMakePresets.json` |
| Exp04 dataset loading | `src/experiments/04/lib/src/E04Dataset.cpp` |
| Exp04 encoding transforms | `src/experiments/04/lib/src/E04Encoding.cpp` |
| Exp04 training loop | `src/experiments/04/lib/src/E04Training.cpp` |
| Exp04 output / CSV / DAT writers | `src/experiments/04/lib/src/E04Output.cpp` |
| Exp04 profile parser | `src/experiments/04/lib/include/E04Config.hpp` |
| Exp04 SNN/LSTM builder | `src/experiments/04/lib/include/E04Training.hpp` |
| Exp04 profile audit tests | `src/experiments/04/tests/profile_audit_gtest.cpp` |
| Exp05 config parser | `src/experiments/05/lib/include/E05Config.hpp` |
| Exp05 dataset loader | `src/experiments/05/lib/src/E05Dataset.cpp` |
| Exp05 feature extraction | `src/experiments/05/lib/src/E05FeatureExtraction.cpp` |
| Exp05 paraconsistent ranking | `src/experiments/05/lib/src/E05Paraconsistent.cpp` |
| Exp05 classifiers | `src/experiments/05/lib/src/E05Classifiers.cpp` |
| Exp05 output writers | `src/experiments/05/lib/src/E05Output.cpp` |
| Exp05 profile audit tests | `src/experiments/05/tests/e05_profile_audit_gtest.cpp` |
| Paper CSV aggregator | `scripts/pipeline/build_paper_data.py` |
| Article run script | `scripts/pipeline/run_article_profiles.sh` |

---

## Experiment04 paper pipeline

Full chain from profiles to compiled PDF:

```bash
# 1. Run all article profiles (~2.5 h: LSTM ~10 min + 3×SNN ~45 min each)
cd software/nn
./scripts/pipeline/run_article_profiles.sh
# writes results/article_{lstm_ae,snn_dense,snn_conv1d,snn_recurrent}_comparative_metrics.csv
# writes .../conference71070Guaiaquil/data/article_*_*.dat  (pgfplots DAT files)

# 2. Aggregate into paper_*.csv (called automatically by run_article_profiles.sh)
python3 scripts/pipeline/build_paper_data.py \
  --results-dir results \
  --data-dir .../conference71070Guaiaquil/data \
  --profiles-dir src/experiments/04/profiles

# 3. Compile paper
cd documentation/07-articlesProduced/conference71070Guaiaquil
pdflatex paper.tex && bibtex paper && pdflatex paper.tex && pdflatex paper.tex
```

**Column mapping** (`build_paper_data.py` reads `comparative_metrics.csv`):
- `model == "lstm-ae"` → label `LSTM-AE`
- `model == "snn-ae"` + `architecture == "dense/conv1d/recurrent"` → label `SNN-{arch}`

**Profile guard**: `profile_audit_gtest` (25 tests × 5 profiles) verifies every profile
parses, validates, has `loss=mse`, `seed_deterministic=false`, and consistent sweep arrays.
Run after any profile edit:
```bash
cmake --build out/build/max-performance --target profile_audit_gtest -j$(nproc)
ctest --test-dir out/build/max-performance -R profile_audit --output-on-failure
```
