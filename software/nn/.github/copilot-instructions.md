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

- Primary Language: **C++20** (with **C** support for vendored libraries).

- Main Dependencies: **Eigen**, **GoogleTest**, **CMake**, **OpenMP**. Vendored: `cnpy`, `matio`, `matio-cpp`, `imgui`, `implot`, **FFTW3** (with OpenMP + threading), **NFFT3** (via autotools/ExternalProject).

- Domain: Spiking Neural Networks (SNN), Sparse Autoencoders, EEG+Audio Synchronization, support for .mat (MATLAB) and experimental pipelines.

- Non-Functional Requirements: Compilable on Linux (Arch Linux primary target); Testable (GTest); Safe (RAII); Efficient (Eigen/noalias/OpenMP with explicit parallelization); Auditable (Logs, Metrics); Reproducible (deterministic seeding).

- Build Status: **v0.2.0** — Successfully compiles with OpenMP enabled; FFTW3 with threading and NFFT3 fully integrated; all executables and tests building.

- Use the acquisition/protocol defaults documented in project reference materials when experiments refer to human-collected data; include metadata fields in stored records for full traceability.

---

# 2 Repository Structure (Detailed)

```
nn/
├── .clang-format              # Configuration for clang-format, ensuring consistent code style.
├── .clang-tidy                # Configuration for clang-tidy, a C++ static analysis tool.
├── .clangd                    # Configuration for clangd, a language server for C++.
├── CMakeLists.txt             # Main CMake build script for the project.
├── test.mat                   # Example MATLAB data file for testing.
├── TODO.md                    # Markdown file for tracking project tasks and to-dos.
├── utils_test.mat             # Utility test MATLAB data file.
├── utils_test2.mat            # Another utility test MATLAB data file.
├── utils_test3.mat            # Third utility test MATLAB data file.
├── .github/                   # GitHub specific configurations.
│   └── copilot-instructions.md  # Instructions and guidelines for Copilot.
├── .vscode/                   # VSCode editor configurations.
│   ├── c_cpp_properties.json    # C/C++ extension settings for VSCode.
│   ├── launch.json              # Debugging configurations for VSCode.
│   ├── settings.json            # Workspace settings for VSCode.
│   └── tasks.json               # Task configurations for VSCode.
├── build/                     # CMake build directory (ignored by git)
│   ├── _deps/                   # External dependencies built by CMake.
│   ├── bin/                     # Compiled executable binaries.
│   ├── lib/                     # Compiled libraries.
│   └── src/                     # Build artifacts for source files.
├── cmake/                     # CMake modules and scripts for project configuration.
│   ├── Flags.cmake              # Compiler flags (C++20, OpenMP, optimizations, warnings).
│   ├── Policies.cmake           # CMake policies for project consistency.
│   ├── PackageChecking.cmake    # System dependency checks (BLAS, LAPACK, HDF5, OpenGL, etc.).
│   ├── ProjectSetup.cmake       # General project setup (C++ standard, visibility).
│   ├── VendorIncludes.cmake     # Main vendor aggregator (cnpy, imgui, implot, matio, eigen parallel).
│   ├── VendorGTest.cmake        # Google Test (FetchContent).
│   ├── VendorFFTW.cmake         # FFTW3 with `ENABLE_OPENMP=ON` and `ENABLE_THREADS=ON`; finds OpenMP::OpenMP_CXX.
│   ├── VendorNFFT3.cmake        # NFFT3 via ExternalProject; runs `./bootstrap.sh && autoreconf`; configures with `--enable-openmp --enable-shared`; imports as SHARED library with OpenMP::OpenMP_C.
│   ├── VendorMatio.cmake        # Matio (MATLAB .mat I/O) via FetchContent.
│   ├── VendorMatioCppShim.cmake # Matio C++ wrapper shim.
│   └── VendorEigenParallel.cmake # Eigen parallelization (OpenMP flags, BLAS/LAPACK).
├── debug/                     # Debugging related files.
│   └── gdb/                     # GDB debugger configurations.
│       ├── printers.py          # Python scripts for GDB pretty printers.
│       └── __pycache__/         # Python cache directory.
├── lib/                       # Vendored external libraries (source code).
│   ├── cnpy/                    # Source for cnpy library (Numpy file format I/O).
│   ├── imgui/                   # Source for Dear ImGui (immediate mode GUI).
│   ├── implot/                  # Source for ImPlot (plotting extension for Dear ImGui).
│   ├── matio/                   # Source for Matio library (MATLAB .mat file I/O).
│   └── matio-cpp/               # Source for Matio C++ wrapper.
└── src/                       # Source code for the main project.
    ├── core/                    # Core components of the framework.
    │   ├── CMakeLists.txt         # CMake build script for the core library.
    │   ├── NetworkSerializer.hpp  # Header for network serialization utilities.
    │   ├── NnSaver.hpp            # Header for neural network saving functionalities.
    │   ├── dataLoaders/           # Implementations for Dataset, DataLoader, MatFile readers.
    │   ├── initializers/          # Weight initialization strategies (e.g., xavier, kaimingSNN).
    │   ├── layers/                # Neural network layer implementations (e.g., Module, Linear, LIF, Sequential, activations).
    │   ├── linearAlgebra/         # Custom linear algebra utilities and helpers.
    │   ├── optimizers/            # Optimization algorithms (e.g., Adam, SGD, interfaces).
    │   ├── paraconsistent/        # Components related to paraconsistent logic.
    │   ├── statistics/            # Statistical metrics and test implementations.
    │   ├── tensor/                # Custom Tensor wrapper built on Eigen.
    │   ├── tests/                 # Unit tests for core components.
    │   ├── utility/               # General utility functions (e.g., logging, time, checks).
    │   ├── wave/                  # Wave processing and resampling utilities.
    │   └── wavelet/               # Wavelet transform implementations.
    ├── experiments/               # Executables for specific research experiments.
    │   ├── autoEncoderLeakyReLUAndSpikeTest.cpp # Experiment for autoencoder with LeakyReLU and spiking.
    │   ├── autoEncoderLeakyReLUTest.cpp # Experiment for autoencoder with LeakyReLU.
    │   ├── plotSpikingNetwork.cpp # Experiment for plotting spiking network activity.
    │   ├── resnet_demo.cpp        # Demonstration of ResNet integration.
    │   └── 01/                    # Directory for Experiment 01 related files.
    ├── main_app/                  # Main application entry point.
    │   └── main.cpp               # Main source file for the application.
    └── util/                      # General utilities not specific to core components.
        ├── batching.cpp           # Implementation for data batching.
        ├── batching.hpp           # Header for data batching utilities.
        ├── CMakeLists.txt         # CMake build script for the utility library.
        ├── imguiGlfw.cpp          # ImGui and GLFW integration implementation.
        ├── imguiGlfw.hpp          # Header for ImGui and GLFW integration.
        ├── loadingData.cpp        # Implementation for general data loading.
        ├── parallel_batching.hpp  # Header for parallel data batching.
        ├── synthetic_spike_data.cpp # Implementation for generating synthetic spike data.
        ├── synthetic_spike_data.hpp # Header for synthetic spike data generation.
        ├── vectorizationCheck.cpp # Implementation for checking vectorization capabilities.
        ├── vectorizationCheck.hpp # Header for vectorization checks.
        └── tests/                 # Unit tests for utility functions.
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

- The Exponential / SuperSpike surrogate must be implemented and available as a selectable surrogate alongside `atan`, `fast_sigmoid`, and `piecewise_linear`. Provide `src/layers/surrogate.h` with unit tests validating gradient shapes and numerical stability.

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

- Include derivation and Euler discretization in `docs/math_lif.md`. Implement forward Euler time-stepping as baseline and document stability constraints (dt relative to tau).

## 5.2 Surrogate Gradient

- Use surrogate `sigma(x)` approximating derivative of Heaviside.

- Examples:
  - Fast sigmoid surrogate derivative: `sigma'(x) = 1 / (1 + abs(pi * x))^2` (or another form).
  - Arctan surrogate: derivative of `atan(k*x)` approximates spike derivative.
  - Exponential / SuperSpike: provide implementation and config switch.

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

- **`Sxx_EEG.mat`**: `N_rows x 24579`
  - Each row contains 6 concatenated EEG channels plus 3 labels.
  - **EEG Data**: 24,576 samples (6 channels × 4096 samples = 4s @ 1024 Hz).
    - `F3`: Samples 1-4096
    - `F4`: Samples 4097-8192
    - `C3`: Samples 8193-12288
    - `C4`: Samples 12289-16384
    - `P3`: Samples 16385-20480
    - `P4`: Samples 20481-24576

  - **Labels**:
    - **Modality** (column 24577): `1` (Imagined), `2` (Pronounced).
    - **Stimulus** (column 24578): `1-5` (A, E, I, O, U), `6` (Arriba), `7` (Abajo), `8` (Adelante), `9` (Atrás), `10` (Derecha), `11` (Izquierda).
    - **Artifacts** (column 24579): `1` (No artifacts), `2` (Blink present).

- **`Sxx_Audio.mat`**: `M_rows x 176402`
  - Each row contains a mono audio signal plus 2 labels.
  - **Audio Data**: 176,400 samples (4s @ 44.1 kHz).
  - **Labels**:
    - **Stimulus** (column 176401): Uses the same encoding as the EEG stimulus label.
    - **EEG Index** (column 176402): The corresponding row index in the `Sxx_EEG.mat` file for synchronization.

- When using acquisition data following the project's protocol, store EEG recorded at **1000 Hz**, 16-bit quantization, and include sampling metadata. For capture blocks of 5 seconds, store 5000 samples per channel (5 s × 1000 Hz). When mat files contain signals recorded at other sampling rates, include `sampling_rate` and explicit audio↔EEG mapping fields (audio_row_index or timestamp) to guarantee deterministic alignment.

## 7.2 MatFile Reader Expectations

- `MatFile::read_first_numeric_variable()` returns `Eigen::MatrixXf` with warning if too large. Provide safety caps: `max_features`, `max_elements`.

## 7.3 DataLoader Contract

- `DataLoader` returns batches `(inputs, targets)` where `inputs` shape = `(batch, features...)`.
- Provide deterministic shuffle via seed, and an optional `sampler` that can stratify by speaker or stimulus.
- Persist metadata per-record: subject id, start/stop timestamps, sentence id, modality, noise condition. This metadata must travel with batches for analysis and reproducibility.

---

# 8 Preprocessing (Audio + EEG) — Recipes and Recommended Parameters

## 8.1 Audio (Recommendations)

- Band-pass 80–7600 Hz (5th-order Butterworth) to remove DC and high-frequency noise.
- Z-score normalization per recording: `(x - mean) / std`.
- Compute envelope: `abs(hilbert(x))` → resample to EEG rate for alignment.
- Support MFCC and wavelet-packet transforms as optional feature branches; provide parameters in config.

## 8.2 EEG

- Acquisition prefilter (applied at recording) should be wide enough to capture neural energy while respecting Nyquist; for the default acquisition rate **1000 Hz**, use a sensible prefilter (e.g., 1–450 Hz) to avoid aliasing while preserving relevant band content.
- Notch at 50/60 Hz + harmonics (IIR notch filter with Q=30); default mains notch set to 60 Hz (Brazil).
- Z-score per channel.
- Artifact removal: simple threshold-based blink detection as baseline; optional ICA pipeline for advanced removal — when ICA is used, persist mixing matrices for reproducibility.
- When re-filtering previously pre-filtered data, use zero-phase filtering (filtfilt) or compensate for filter delay to avoid timing misalignment.

---

# 9 Windowing & Synchronization — Specifications and Reference Code

## 9.1 Defaults

- `window_sec` = 1.5 (recommended), `overlap` = 0.5 (50%). Provide grid-search options for different windows and overlaps.
- Compute sample counts from configured sampling rates (avoid hard-coded values).

## 9.2 Synchronized Window Extraction (C++ Sketch)

- Implement in `src/util/windowing.cpp` using sample counts computed from rates.
- Use protocol timings to compute offsets: `display_sec = 5`, `cue_sec = 1`, `capture_sec = 5`. With EEG at **1000 Hz**, capture block = `capture_sec * sampling_rate = 5 * 1000 = 5000` samples. Window extraction code must compute indices via `sampling_rate` and durations.
- Validate audio↔EEG mapping using the explicit mapping fields stored in files.

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

- Implement multiple choices: `exponential_super_spike`, `atan`, `fast_sigmoid`, `piecewise_linear`. Select via config.
- Ensure numerical stability; clip gradients where needed; provide unit tests for surrogate gradient behavior.

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
- Ensure result rows include modality and noise-condition metadata for stratified analysis.

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

## 14.3 Vendoring & External Projects

- **FetchContent libraries** (Eigen, FFTW, Matio, ImGui, ImPlot, cnpy, GoogleTest): Downloaded and built as part of CMake configuration. Source placed in `build/_deps/<lib>-src/`, built in `build/_deps/<lib>-build/`, artifacts in `build/lib/` or similar.
- **AutoTools libraries** (NFFT3): Use `ExternalProject_Add()` to download, run `./bootstrap.sh && autoreconf --install --force`, then `./configure --enable-openmp --enable-shared`, `make -j4`, `make install` into `build/_deps/nfft3-install/`.
- **OpenMP integration**: Explicitly call `find_package(OpenMP REQUIRED)` in vendor modules. Both FFTW and NFFT3 configure with `--enable-openmp` and link against `OpenMP::OpenMP_C` / `OpenMP::OpenMP_CXX`.
- **Shared vs. Static**: NFFT3 imports as `SHARED IMPORTED` (`.so` library); FFTW3 primary target is the main library (library type determined by CMake config).
- **Dependency consistency**: NFFT3 depends on FFTW3 and OpenMP; ensure FFTW is configured before NFFT3 in CMake include order.

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

- Add unit test that verifies `capture_samples == sampling_rate * capture_sec`. For default acquisition values (`sampling_rate = 1000`, `capture_sec = 5`), the test asserts `capture_samples == 5000`. This prevents regressions when protocol defaults change.

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

## 18.1 Style & Static Analysis

- **clang-format**: `.clang-format` at repository root; applied on all `.cpp` / `.hpp` files.
- **clang-tidy**: Configured in `.clang-tidy` (canonical location, separate from `.clangd`). Includes checks: `clang-analyzer-*`, `modernize-*`, `performance-*`, `readability-*`, `bugprone-*`. Excludes: `misc-*`, `cppcoreguidelines-non-private-member-variables-in-classes`, `readability-identifier-length`, `readability-magic-numbers`, `bugprone-easily-swappable-parameters`.
- **clangd config** (v0.2.0+): Simplified `.clangd` with only `CompileFlags` (compilation database + include paths) and `Index.Background: true`. Schema warnings for `Threads`, `StorePreamble`, and inline `ClangTidy` removed (moved to `.clang-tidy`); compatible with Arch Linux clangd extension.

## 18.2 Editor Configuration (Arch Linux)

- **clangd binary**: Install via `sudo pacman -S clang clang-tools-extra`.
- **VS Code clangd extension**: Ensure latest version (marketplace: llvm-vs-code-extensions.vscode-clangd); uses `.clangd` + `.clang-tidy` + `compile_commands.json` from `build/`.
- **Build system**: CMake generates `compile_commands.json` in `build/` directory (used by clangd for accurate code intelligence).

## 18.3 PR Template (docs/.github/PULL_REQUEST_TEMPLATE.md)

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

acquisition:
  eeg:
    sampling_rate: 1000 # Hz (default acquisition rate)
    bit_depth: 16 # bits
    prefilter: [1, 450] # Hz bandpass applied at acquisition (<= Nyquist)
    mains_notch: 60 # Hz (Brazil)
protocol:
  display_sec: 5
  cue_sec: 1
  capture_sec: 5
dataset:
  categories: [fonadas, imaginadas, mista]
  noise_conditions: [with_noise, without_noise]
```

- `run_experiment` and downstream code must compute sample counts from `acquisition.eeg.sampling_rate` and durations; for default values above, capture block = `5 * 1000 = 5000` samples.

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
- When ICA artifact removal is used, persist mixing matrices and metadata necessary to reproduce artifact-removal offline.

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

# 23 Useful Attachments (Model Snippets / Ready Examples)

(Includes small snippets already presented, CMake lines, CI YAML snippet, config sample — all consolidatable into real files if you request.)

---

# 24 Operational Policies and Final Guardrails (Summary)

- Copilot/generator **does not** modify public APIs without version bump and CHANGELOG.
- All automatic generation creates minimal tests.
- Seeds and checkpoints are always saved.
- For compilation changes (CMake), add `exec_*.cmake` and document.
- Do not download dependencies at runtime CI without approval.
- Do not introduce non-determinism in experiments without explicit config options.
- Always document the experimental protocol and data handling steps for traceability.
- Always follow the following protocol when change the code:
  1. Identify the component to change.
  2. Split the change into smaller, manageable parts.
  3. For each part, check for compilation breaks and run the tests.
  4. If any tests and compilation fails, debug and fix the issues before proceeding.
  5. Update documentation and comments.
  6. Review the entire change for consistency and correctness.
  7. Update `copilot-instructions.md` if public API changes.
  8. Update `CHANGELOG.md` with semantic versioning.

---

# 25 Configuration and Manifest Files

This section provides references to key configuration files that define the tools and skills available in the development environment.

## 25.1 Tools Manifest

- **File:** `.github/tools_manifest.json`
- **Purpose:** Lists all available command-line tools, their versions, and usage notes. This file is the canonical reference for what tools can be executed in the environment.

## 25.2 Claude Skills

- **File:** `.github/claude-skills.yaml`
- **Purpose:** Defines a set of higher-level "skills" or "recipes" that can be invoked. These skills are composed of one or more command-line tools and are designed to automate common development tasks.
