# copilot-instructions.md — Comprehensive and Extensive Version (Master Guide for Copilot)

**Purpose**
Canonical document guiding automatic code generation (Copilot, scripts, PRs) for the C++ SNN + Autoencoder + EEG+Audio framework. Must be the first reference when generating code, changing APIs, CMake, or experimental pipelines. Update it whenever there are API/ABI changes.

---

## Table of Contents (Quick Navigation)

1. Overview and Objectives
2. Repository Structure
3. Public Contracts (APIs) — Signatures & Invariants
4. Detailed Component Specifications
5. Equations and Implementation: LIF, Surrogate Gradients, Van Rossum, TTFS, BSA
6. Spike-Encoders: Pseudo-Code + C++ Snippets
7. Data Pipeline & Formats (.mat, .npy) + Dataset Schema
8. Preprocessing (Audio + EEG) — Recipes and Recommended Parameters
9. Windowing & Synchronization — Specifications and Reference Code
10. SAE Architecture, Loss, Training Loop, Surrogate Gradient Choices
11. ResNet-SNN Downstream — Usage Patterns
12. Statistical Evaluation & Experimental Tests (Nested CV, Wilcoxon, etc.)
13. Metrics and Analysis Scripts (Python/C++ Snippets)
14. CMake, Build, Targets, Vendoring, and Module Templates
15. Tests (GTest), Coverage, Sanitizers, CI (GitHub Actions) — YAML Example
16. Performance & Profiling — Tips, Microbenchmarks, OpenMP Best Practices
17. Security, Dependencies, Packaging, and License
18. Code Style, Linting, Formatting, PR Reviews, Changelog & Versioning
19. Useful Templates: Layer, Optimizer, DataLoader, MatFile Reader, DataLoader Iterator, Adam Implementation
20. Experiment Orchestration: Configs, Run Scripts, Logs, Reproducibility
21. Benchmarking & Reproducibility Plan (Seed Policy, Deterministic Ops)
22. Automatic PR Checklists and Generated Templates
23. Roadmap and Immediate Tasks (Prioritized)
24. Actions I Can Execute Now (Create Files, PR Skeleton, CI, etc.)

---

# 1 Overview and Objectives

- Primary Language: **C++20**.
- Main Dependencies: **Eigen**, **GoogleTest**, **CMake**. Optional Vendored: `cnpy`, `matio`, `matio-cpp`, `imgui`, `implot`.
- Domain: Spiking Neural Networks (SNN), Sparse Autoencoders, EEG+Audio Synchronization, support for .mat (MATLAB) and experimental pipelines.
- Non-Functional Requirements: Compilable on Linux; Testable; Safe (RAII); Efficient (Eigen/noalias/OpenMP); Auditable (Logs, Metrics); Reproducible.

---

# 2 Repository Structure (Detailed)

```
nn/
├── src/
│   ├── tensor/            # Tensor wrapper (Eigen)
│   ├── layers/            # Module, Linear, LIF, Sequential, activations
│   ├── optimizers/        # Adam, SGD, interfaces
│   ├── dataLoaders/       # Dataset, DataLoader, MatFile
│   ├── experiments/       # train_sae, train_resnet_snn, extract_embeddings
│   ├── initializers/      # xavier, kaimingSNN
│   ├── encoders/          # rate, ttfs, bsa implementations
│   ├── wave/              # processing, resample
│   ├── wavelet/           # wavelet transforms
│   ├── util/              # logging, time, checks
│   └── stats/             # metrics, statistical tests
├── tests/
│   ├── data/              # small .mat test files, synthetic datasets
│   └── unit/              # gtest suites
├── cmake/
│   ├── exec_layers.cmake
│   ├── exec_tensor.cmake
│   ├── Flags.cmake
│   └── Main.cmake
├── configs/               # defaults.yaml, grids
├── scripts/               # preprocessing, run_experiment, analyze_results
├── tools/                 # Octave scripts (inspect_pronounced_pair.m)
├── .github/
│   └── workflows/         # CI workflows YAML
└── docs/                  # design docs, math, equations
```

---

# 3 Public Contracts (APIs) — Signatures & Invariants

> **Rule:** Any public change requires updating this document + CHANGELOG + semantic versioning.

## 3.1 Tensor (src/tensor/Tensor.h)

```cpp
class Tensor {
public:
    Eigen::MatrixXf data;
    Eigen::MatrixXf grad;
    Tensor() = default;
    explicit Tensor(int rows, int cols);
    auto rows() const -> int { return data.rows(); }
    auto cols() const -> int { return data.cols(); }
    auto reshape(int rows, int cols) -> void; // throws runtime_error if incompatible
    auto clone() const -> Tensor;
    auto zero_grad() -> void;
    static auto from_vector(const std::vector<float>& v, int rows, int cols) -> Tensor;
};
```

- **Invariant:** `data.size()` consistent; `grad` compatible; public operations validate shapes and throw `std::runtime_error` on invalid inputs.

## 3.2 Module (src/layers/Module.h)

```cpp
struct Module {
    virtual ~Module() = default;
    virtual Tensor forward(const Tensor& input) = 0;
    virtual Tensor backward(const Tensor& grad_output) = 0;
    virtual void train(bool on) { training_ = on; }
protected:
    bool training_ = true;
};
```

- `Sequential` manages `std::vector<std::shared_ptr<Module>>`.

## 3.3 Dataset / DataLoader (src/dataLoaders/)

- `class Dataset { virtual auto get_item(size_t idx) -> std::tuple<Tensor, Tensor> = 0; virtual auto size() const -> size_t = 0; virtual auto collate(const std::vector<std::tuple<Tensor, Tensor>>& batch) -> std::tuple<Tensor, Tensor>; };`
- `DataLoader` provides C++ iterators: `.begin()`, `.end()`; parameters: `batch_size`, `shuffle`, `seed`.

## 3.4 Optimizer (src/optimizers/Optimizer.h)

```cpp
struct Optimizer {
    virtual ~Optimizer() = default;
    virtual void attach(const std::vector<Tensor*>& params) = 0;
    virtual void zero_grad() = 0;
    virtual void step() = 0;
};
```

- Implement `Adam`, `SGD`. `attach` must store pointers to parameters (Tensor\*).

---

# 4 Detailed Component Specifications

(Includes extended contracts, invariants, expected complexity, required tests.)

## 4.1 Linear Layer

- Signature: `Linear(int in_features, int out_features, bool bias=true)`.
- Forward: `output = input * weight.t() + bias` (batch x features). Use `noalias()` for optimization. Validate shapes.
- Backward: Compute grads for weights and inputs; accumulate gradients in `weights.grad` and `bias.grad`.
- Tests: Forward shape, backward grad numeric check (finite differences) and edge cases (batch=0, in_features mismatch).

## 4.2 LIF / Leaky Layer (SNN)

- Provide discrete-time LIF update:

  - Membrane: `V[t+1] = alpha * V[t] + W*x[t] - S[t]*V_reset`
  - Spike: `S[t] = H(V[t] - V_threshold)` (Heaviside)
  - Surrogate gradient g'(V) used in backprop.

- Parameters: `tau_m`, `V_thr`, `V_reset`, `alpha = exp(-dt / tau_m)`.
- Stateful per timestep. Provide `reset()` method.
- Tests: Single neuron response to step input (expected firing pattern), surrogate gradient sanity check.

## 4.3 MatFile Wrapper

- Methods: `read_first_numeric_variable()`, `read_variable(name)`, `write_variable(name, data)`.
- Validate MAT v5 header, var type (double/float), dims; map to `Eigen::Map` if possible.
- Tests: Read synthetic `tests/data/test_small.mat`.

## 4.4 Encoders (Rate, TTFS, BSA)

- Each encoder implements interface:

```cpp
struct SpikeEncoder {
    virtual ~SpikeEncoder() = default;
    virtual Tensor encode(const Tensor& analog_window) = 0; // returns spikes (time x features) or spike times depending on impl
};
```

- Provide both Python pseudo-implementations and C++ optimized versions in `src/encoders/`.

---

# 5 Equations and Implementation — LIF, Surrogate Gradients, Van Rossum, TTFS, BSA

## 5.1 LIF (Discretized)

- Membrane update discrete:

  - `V[t+1] = decay * V[t] + I[t]` where `decay = exp(-dt / tau_m)` and `I[t] = W * x[t] + b`.

- Spike emission:

  - `S[t] = Theta(V[t] - V_thr)` where `Theta` is the Heaviside function.

- Reset:

  - After spike, `V[t] = V[t] - V_reset` (or `V[t] = V_reset_value` depending on the model).

## 5.2 Surrogate Gradient

- Use surrogate `sigma(x)` approximating derivative of Heaviside.
- Examples:

  - Fast sigmoid surrogate derivative: `sigma'(x) = 1 / (1 + abs(pi * x))^2` (or another form).
  - Arctan surrogate: derivative of `atan(k*x)` approximates spike derivative.

- Implementation tip: Compute surrogate derivative as function of `V - V_thr`, scale by `scale_factor`.

## 5.3 Van Rossum Distance (Reconstruction Loss for Spike Trains)

- For spike trains `s(t)` and `r(t)`, convolve with exponential kernel `h(t) = exp(-t/τ)/τ`. Then `d^2 = ∫ (s*h - r*h)^2 dt`.
- Discrete approximation: Convolve spike trains and compute MSE.
- Use efficient FFT-based convolution or incremental IIR filter `y[t] = y[t-1] * exp(-dt/τ) + s[t]/τ`.

## 5.4 TTFS (Time-to-First-Spike)

- For amplitude `a∈[0,1]`, `t_spike = T_max * (1 - a)`. No spike if `a <= 0` (or set t_spike = T_max).
- Implement representation as `spike_time` array or as one-hot temporal spike train.

## 5.5 BSA / Threshold-Based Encoding

- Maintain potential `p[t] += window[t]`. If `p[t] > threshold`: fire spike, `p[t] -= reset_value`.

---

# 6 Spike-Encoders: Pseudo-Code + C++ Snippets

## 6.1 Poisson Rate (Python Pseudo)

```python
def rate_encode(window, scaling_factor=1.0):
    # window: (time, features) normalized to [0,1]
    spike_prob = window * scaling_factor
    spikes = (np.random.rand(*window.shape) < spike_prob).astype(np.float32)
    return spikes
```

## 6.2 TTFS (C++ Sketch)

```cpp
// input: Eigen::MatrixXf window (features x 1) values in [0,1]
// output: std::vector<int> spike_times (size = features) in [0, T_max) or T_max for no spike.
std::vector<int> ttfs_encode(const Eigen::VectorXf& window, int T_max) {
    std::vector<int> spike_times(window.size(), T_max);
    for (int i=0;i<window.size();++i) {
        float a = window[i];
        if (a > 0.0f) spike_times[i] = static_cast<int>((1.0f - a) * (T_max - 1));
    }
    return spike_times;
}
```

## 6.3 BSA (C++ Sketch)

```cpp
Eigen::MatrixXi bsa_encode(const Eigen::VectorXf& window, float threshold, float reset) {
    Eigen::MatrixXi spikes(window.size(), 1);
    float potential = 0.0f;
    for (int t=0; t<window.size(); ++t) {
        potential += window[t];
        if (potential > threshold) {
            spikes(t,0) = 1;
            potential -= reset;
        }
    }
    return spikes;
}
```

---

# 7 Data Pipeline & Formats (.mat, .npy) + Dataset Schema

## 7.1 Expected Schema for `EEG` and `Audio` (According to Your Original Dataset)

- `EEG`: `N_rows x 24579`

  - 24576 samples (6 channels × 4096 samples = 4 s @ 1024 Hz) + 3 labels (modality, stimulus, artifact).
  - `EEG(row, 1:24576)` → raw EEG samples; `EEG(row, 24577)` = modality; `24578` = stimulus; `24579` = artifact.

- `Audio`: `M_rows x 176402`

  - 176400 samples (4 s @ 44100 Hz) + 2 labels (stimulus, EEG_index).
  - `Audio(row, 1:176400)` → mono audio; `Audio(row, 176401)` = stimulus; `176402` = EEG_index (synchronized EEG row).

## 7.2 MatFile Reader Expectations

- `MatFile::read_first_numeric_variable()` returns `Eigen::MatrixXf` with warning if too large. Provide safety caps: `max_features`, `max_elements`.

## 7.3 DataLoader Contract

- `DataLoader` returns batches `(inputs, targets)` where `inputs` shape = `(batch, features...)`.
- Provide deterministic shuffle via seed, and an optional `sampler` that can stratify by speaker or stimulus.

---

# 8 Preprocessing (Audio + EEG) — Recipes and Recommended Parameters

## 8.1 Audio (Recommendations)

- Resample 44.1kHz → 16kHz for speech features (unless high-frequency content needed). Use polyphase resampling for quality.
- Band-pass 80–7600 Hz (5th-order Butterworth) to remove DC and high-frequency noise.
- Z-score normalization per recording: `(x - mean) / std`.
- Compute envelope: `abs(hilbert(x))` → resample to EEG rate for alignment.

## 8.2 EEG

- Band-pass 1–40 Hz (5th-order Butterworth), linear-phase filters preferred (FIR) or zero-phase via filtfilt.
- Notch at 50/60 Hz + harmonics (IIR notch filter with Q=30).
- Z-score per channel.
- Artifact removal: Simple blink removal by thresholding or advanced ICA. If using ICA, save mixing matrix for reproducibility.
- IMPORTANT: If dataset pre-filtered (documented), avoid reintroducing filter delay; use zero-phase or compensate delays.

---

# 9 Windowing & Synchronization — Specifications and Reference Code

## 9.1 Defaults

- window_sec = 1.5 (recommended), overlap = 0.5 (50%).
- Provide grid search settings as earlier.

## 9.2 Synchronized Window Extraction (C++ Sketch)

- Implement in `src/util/windowing.cpp` using sample counts computed from rates.
- Must check alignment via `Audio(:, end)` index to map EEG row.

---

# 10 SAE Architecture, Loss, Training Loop, Surrogate Gradients

## 10.1 Architecture (Default)

- Input → Linear (1024) → LIF (1024 → 512) → LIF (512 → D embedding) → Decoder symmetric.
- Embedding D = 128 default (configurable).

## 10.2 Loss

- `L_total = L_recon + lambda_sparsity * L_sparsity + lambda_reg * ||weights||^2`
- `L_recon`: Van Rossum distance between original and reconstructed spike trains (preferred) or MSE on low-passed signals.
- `L_sparsity`: Mean firing rate penalty (L1 on firing rates).

## 10.3 Training Loop (C++ Pseudocode)

```cpp
for epoch in range(max_epochs):
    for batch in dataloader:
        optimizer.zero_grad();
        auto spikes = model.forward(batch.inputs);
        auto recon = decoder.forward(spikes);
        float loss = van_rossum_loss(recon, batch.targets) + sparsity_loss(spikes);
        // backward with surrogate gradients
        auto grad = loss.backward();
        model.backward(grad);
        optimizer.step();
    validate();
    if early_stopping_condition: break;
```

## 10.4 Surrogate Gradients (Recommendations)

- Implement multiple choices: `atan`, `fast_sigmoid`, `piecewise_linear`. Use `config` to select.
- Ensure numerical stability; clip gradients where needed.

---

# 11 ResNet-SNN Downstream — Usage Patterns

- Treat embedding vector (D-dim) as static input; replicate across time if SNN temporal layers expect temporal input.
- Replace ReLU with LIF in residual blocks; keep batchnorm optional — for SNN, batchnorm across time can be problematic; use LayerNorm on embeddings.

---

# 12 Statistical Evaluation & Experimental Tests

## 12.1 Protocol

- **Nested Cross-Validation**: Outer splits speakers (e.g., leave-out K speakers), inner loop tune hyperparams.
- Report `mean ± std` for primary metrics across outer folds.
- For pairwise comparisons between pipelines, use **Wilcoxon signed-rank** with corrected p-values (Bonferroni) for multiple comparisons.

## 12.2 Implementation Snippets (Python)

- Provide `analyze_results.py` that loads `results.csv`, computes aggregate stats, runs Wilcoxon, outputs JSON summary and plots (matplotlib).

---

# 13 Metrics and Analysis Scripts (Snippets)

## 13.1 Metric Calc (C++ or Python)

- Top1 accuracy, confusion matrix, per-class precision/recall, AUC (sklearn for Python), silhouette score (sklearn), spike budget average.

## 13.2 Example Python CLI `analyze_results.py`

- Read results, compute mean/std, run Wilcoxon between top configurations, output JSON.

---

# 14 CMake, Build, Vendoring & Templates

## 14.1 Naming

- Library target: `nn_<component>` (e.g., `nn_tensor`, `nn_layers`).
- Test targets: `nn_<component>_gtest`.

## 14.2 Example Module CMake

```cmake
add_library(nn_tensor src/tensor/Tensor.cpp)
target_include_directories(nn_tensor PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(nn_tensor PUBLIC Eigen3::Eigen)
add_executable(nn_tensor_test tests/unit/tensor_test.cpp)
target_link_libraries(nn_tensor_test PRIVATE nn_tensor GTest::gtest_main)
gtest_discover_tests(nn_tensor_test)
```

## 14.3 Vendoring

- Put 3rd-party in `lib/` and create `cmake/Vendor*.cmake` that defines imported targets (Eigen3::Eigen, etc.).

---

# 15 Tests (GTest), Coverage, Sanitizers, CI (GitHub Actions) — YAML Example

## 15.1 CI: `build-and-test.yml` (Skeleton)

```yaml
name: build-and-test
on: [push, pull_request]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install deps
        run: sudo apt-get update && sudo apt-get install -y libeigen3-dev ...
      - name: Configure
        run: cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
      - name: Build
        run: cmake --build build -- -j$(nproc)
      - name: Run tests
        run: ctest --test-dir build --output-on-failure -j4
      - name: clang-format check
        run: clang-format --version && git diff --exit-code
```

## 15.2 Sanitizers Job (ASAN/TSAN) — Nightly or on-demand to save time.

---

# 16 Performance & Profiling — Tips

- Use `matrix.noalias()` and `block` for views.
- Profile with `perf`, `valgrind --tool=callgrind` or `google-perftools`.
- Microbenchmark: Measure `spikes/sec`, `embeddings/sec`, memory use.
- For OpenMP: Use environment var `OMP_NUM_THREADS`, avoid nested parallelism.
- Prefer `Eigen::Map` for memory mapping data without copy.

---

# 17 Security, Dependencies, Packaging, and License

- Explicit dependency list in `cmake/PackageChecking.cmake`.
- License file: Choose (e.g., MIT) — include `LICENSE`.
- Do not allow dynamic download of remote code during builds in CI (security).
- For binary releases, provide SHA256 checksums.

---

# 18 Code Style, Linting, Formatting, PR Reviews, Changelog & Versioning

## 18.1 Style

- `clang-format` standard (add `.clang-format` in the repository).
- `clang-tidy` with checks: modernize, performance, cppcoreguidelines.

## 18.2 PR Template (docs/.github/PULL_REQUEST_TEMPLATE.md)

- Checklist (build, tests, format, docs), description, breaking changes, migration steps.

## 18.3 CHANGELOG.md (Keep a changelog based on semantic versioning)

- Format: `## [Unreleased]` then `### Added/Changed/Fixed/Removed`.

## 18.4 Versioning

- Semantic Versioning (MAJOR.MINOR.PATCH). Breaking changes bump major.

---

# 19 Templates & Snippets (For Copilot to Generate Safely)

## 19.1 New Layer Template (Header + Impl + Test) — As shown earlier.

## 19.2 Adam Optimizer (Sketch)

- Maintains `m`, `v` per-parameter; bias-correction; supported `attach(params)` where `params` are `Tensor*` (weights only); `step()` updates `param->data`.

## 19.3 DataLoader Iterator Sketch

- Internal buffer of indices, shuffle with `std::shuffle(indices.begin(), indices.end(), rng)`, yield batches by slicing indices.

## 19.4 MatFile Read-First-Numeric Var (Sketch)

- Use `matio` to iterate vars, select first numeric, check dims, map to Eigen::MatrixXf.

---

# 20 Experiment Orchestration: Configs, Run Scripts, Logs, Reproducibility

## 20.1 configs/defaults.yaml

```yaml
window_sec: 1.5
overlap: 0.5
encoder: bsa
embedding_dim: 128
optimizer:
  name: adam
  lr: 1e-4
train:
  batch_size: 32
  epochs: 200
seed: 42
```

## 20.2 run_experiment.sh / Python

- Read config, create copy with overrides, set `RANDOM_SEED`, create logdir with timestamp, save hyperparams JSON, checkpoint model every N epochs.

## 20.3 Logging & Artifacts

- Save: `model.pt` (or custom format), `training.log` (loss per epoch), `results.csv` (metrics), `embeddings/*.npy`, `config_used.yaml`, `git_commit.txt`.

---

# 21 Benchmarking & Reproducibility Plan

## 21.1 Seed Policy

- Single `SEED` used to seed: C++ RNGs (`std::mt19937`), Eigen random initializers seeded via `srand`, Python np.random and torch (if used).
- Save `SEED` in `config` and `git_commit` for traceability.

## 21.2 Deterministic Ops

- Avoid non-deterministic multi-threaded ops for experiments requiring exact reproducibility. Use single-threaded or set `OMP_NUM_THREADS=1` and document the environment.

---

# 22 Automatic PR Checklists and Templates

## 22.1 PR Auto-Checklist (Copilot Must Produce and Verify)

- Build passes.
- Tests pass.
- Clang-format applied.
- Clang-tidy no critical error.
- Docs/CHANGELOG updated if public API changed.
- cmake/exec\_\* updated.

## 22.2 PR Description Template (Generate Automatically)

- Summary, files changed, build steps, tests run, CI status, breaking changes, migration notes.

---

# 23 Roadmap & Immediate Tasks (Priority)

1. Create synthetic `.mat` for CI in `tests/data/` and CTest that runs `dataLoader_demo` (High).
2. Add `configs/defaults.yaml` and adapt `run_experiment` to use configs (High).
3. Implement `Adam` and `Linear` with tests and CMake targets (High).
4. Add GitHub Actions `build-and-test` + `clang-format` (High).
5. Add sanitizers job (Medium).
6. Add benchmark script for spikes/second (Medium).
7. Document LIF equations and surrogate gradients in `docs/` (Medium).

---

# 24 Actions I Can Execute Now (Choose One or More)

Respond with the numbers separated by commas (e.g., `1,3`), or say `all` (I will start in prioritized order: 1 → 2 → 3 → 4 → 5 ...):

1. Create `tests/data/test_small.mat` (synthetic) and add CTest + update `cmake/exec_tests.cmake`.
2. Generate `configs/defaults.yaml` and `scripts/run_experiment.py` (skeleton CLI).
3. Implement `src/layers/Linear.{h,cpp}` + `tests/unit/linear_test.cpp` + CMake target.
4. Create GitHub Actions workflow `build-and-test.yml` with build+tests+clang-format.
5. Generate template `src/encoders/PoissonRate.{h,cpp}` and unit test.
6. Create PR skeleton with the above files and ready commit message.
7. Generate document `docs/math_lif.md` with complete equations and discretization (LaTeX).
8. Create `configs/ci_sanitizers.yaml` and CI job skeleton (ASAN/TSAN).
9. Generate `README_ACTION_HISTORY.md` from git log (skeleton script).
10. All (execute 1→9 in order).

---

# 25 Useful Attachments (Model Snippets / Ready Examples)

(Includes small snippets already presented, CMake lines, CI YAML snippet, config sample — all consolidatable into real files if you request.)

---

# 26 Operational Policies and Final Guardrails (Summary)

- Copilot/generator **does not** modify public APIs without version bump and CHANGELOG.
- All automatic generation creates minimal tests.
- Seeds and checkpoints are always saved.
- For compilation changes (CMake), add `exec_*.cmake` and document.
- Do not download dependencies at runtime CI without approval.
