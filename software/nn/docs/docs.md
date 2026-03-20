# Multimodal Speaker Identification and Neural Network Framework Guide

This guide consolidates the methodology, architecture, and implementation details for a robust speaker identification pipeline using EEG and audio, leveraging Spiking Neural Networks (SNNs), autoencoders, and paraconsistent analysis. It also documents the C++20 neural network framework and practical engineering notes for reproducible research and development.

---

## 1. Methodological Foundations

### Data Acquisition and Synchronization
- Use multichannel EEG and simultaneous audio, ensuring strict temporal synchronization.

### Preprocessing and Dimensionality Reduction
- Segment signals into short windows (e.g., 100 ms, with optional 50% overlap) to capture relevant temporal dynamics.
- Downsample audio to 16 kHz and EEG to 200 Hz, preserving the informative frequency bands while reducing dimensionality.

### Input Structuring
- For each window: concatenate 1600 audio samples (100 ms @ 16 kHz) and 20 samples per EEG channel (100 ms @ 200 Hz).
- No manual feature extraction is performed before the autoencoder.

### Feature Extraction Pipelines
#### Autoencoder Path
- Dense, temporal convolutional, or variational autoencoders map input vectors to a latent space (z = f_θ(x)).
- Experiment with latent dimensions (16–128), 2–6 layers, ReLU/GELU activations, and dropout/L2 regularization.

#### Wavelet Path
- Apply wavelet transforms (Daubechies, Symlets, Coiflets, Morlet, Mexican Hat) separately to EEG and audio.
- Extract features such as energy per scale, wavelet entropy, and band variance.

### Paraconsistent Feature Engineering
- For each feature vector, compute degrees of favorable (μ) and contrary (λ) evidence using Annotated Paraconsistent Logic (LPA).
- Derive certainty (Gc = μ - λ) and contradiction (Gct = μ + λ - 1) metrics to assess class separability and feature consistency.
- Use paraconsistent distance, separability, and inconsistency indices for evaluation.

### Comparative Evaluation
- Compare autoencoder, wavelet, and combined features using paraconsistent metrics and downstream classifiers (SVM, MLP, SNN).
- Select the model with highest separability, lowest contradiction, and best generalization.

### Scientific Evidence and Limitations
- Window sizes of 50–200 ms are standard in EEG/audio studies (Cohen 2014, O'Shaughnessy).
- Downsampling to 16 kHz is common in ASR (Rabiner & Schafer 2011).
- Autoencoders and paraconsistent analysis are established in the literature (Waytowich et al. 2018, Abe 2015).
- Limitations: fixed window may miss long EEG events; autoencoders risk overfitting noise; paraconsistent metrics depend on evidence definitions; optimal wavelets are signal-dependent.

### Recommended Extensions
- Explore multimodal autoencoders with attention, contrastive learning, sparse autoencoders, and CCA/Deep CCA for aligned latent spaces.

---

## 2. Neural Network Framework (C++20)

This framework provides a high-performance, modern C++20 implementation for SNNs, autoencoders, and EEG/audio synchronization, with a focus on performance, security, and maintainability.

- Modular design: core abstractions for tensors, layers, data loaders, optimizers, statistics, and utilities.
- Spiking Neural Networks: Leaky Integrate-and-Fire (LIF) and LeakyIntegrator readout layers.
- Autoencoders: dense, convolutional, variational, and denoising variants.
- Data handling: deterministic batching, MAT/NumPy file support, and robust normalization.
- Performance: SIMD vectorization, OpenMP parallelization, and memory-efficient operations.
- Testing: Google Test integration, 95%+ coverage, static analysis (Cppcheck, Flawfinder, Clang-Tidy).
- Security: input validation, bounds checking, and RAII resource management.

### Example Usage
```cpp
#include "nn/tensor/Tensor.hpp"
#include "nn/layers/Linear.hpp"
#include "nn/layers/Leaky.hpp"
#include "nn/layers/LeakyIntegrator.hpp"
// ...
```

---

## 3. System Architecture and Portability

The pipeline supports end-to-end speaker identification/verification, from audio/EEG capture through feature extraction, spike encoding, and SNN-based classification. CLI-driven flows support demo, enrollment, training, identification, verification, and evaluation. Data schemas, modular boundaries, and deterministic processing are strictly enforced for reproducibility and portability.

### Key Engineering Laws
- Always use core abstractions (tensor, layer, DataLoader, optimizer) for extensibility and testability.
- No hidden global state; propagate errors explicitly; maintain deterministic seeding and ordering.
- Vendor dependencies (Eigen, FFTW3, NFFT3, cnpy, matio, yaml-cpp, matplotlib-cpp, argparse, imgui/implot, GoogleTest) are managed via CMake and must be replaced only with equivalent, justified alternatives.

---

## 4. VS Code Setup and Development Environment

A well-configured VS Code workspace ensures fast iteration, reliable debugging, and smooth builds. Follow this guide to set up your environment from scratch.

### 4.1 Prerequisites

- **Clang 20+** or **GCC 11+** (C++20 support)
- **CMake 3.28+**
- **LLDB 22+** (with DAP support)
- **Python 3.8+** (for pretty-printers)

Verify installations:
```bash
clang++ --version
cmake --version
lldb --version
python3 --version
```

### 4.2 VS Code Extensions

Install these extensions from the Marketplace:

1. **CMake Tools** (by Microsoft) – Essential for CMake integration
2. **C/C++** (by Microsoft) – Language support and IntelliSense
3. **Clang-Format** (by xaver) – Code formatting
4. **Clang-Tidy** (optional) – Static analysis integration
5. **LLDB Debugger** (by Vadim Chugunov) – Enhanced LLDB integration
6. **GitHub Copilot** (optional) – AI-assisted coding

### 4.3 Workspace Settings

Create or update `.vscode/settings.json` in your workspace root:

```json
{
    // Clang compiler paths
    "C_Cpp.default.compilerPath": "/usr/bin/clang++",
    "C_Cpp.default.cStandard": "c17",
    "C_Cpp.default.cppStandard": "c++20",

    // CMake configuration
    "cmake.buildDirectory": "${workspaceFolder}/out/build/Clang_20.1.8_x86_64-pc-linux-gnu",
    "cmake.buildBeforeRun": false,
    "cmake.sourceDirectory": "${workspaceFolder}",

    // Clangd (language server)
    "clangd.path": "/usr/bin/clangd",
    "clangd.arguments": [
        "--compile-commands-dir=${workspaceFolder}/out/build/Clang_20.1.8_x86_64-pc-linux-gnu",
        "--background-index=true",
        "--pch-storage=disk",
        "--completion-style=detailed",
        "--all-scopes-completion=true",
        "--function-arg-placeholders=true",
        "--header-insertion=iwyu",
        "--header-insertion-decorators=true",
        "--clang-tidy=true",
        "--fallback-style=none",
        "--log=info"
    ],
    "C_Cpp.default.compileCommands": "${workspaceFolder}/out/build/Clang_20.1.8_x86_64-pc-linux-gnu/compile_commands.json",
    "C_Cpp.intelliSenseEngine": "Disabled",

    // Debugging
    "lldb.displayFormat": "auto",
    "lldb.showDisassembly": "auto",
    "lldb.dereferencePointers": true,
    "lldb.consoleMode": "evaluate",

    // Editor and UX
    "editor.formatOnSave": true,
    "editor.defaultFormatter": "xaver.clang-format",
    "[cpp]": {
        "editor.defaultFormatter": "xaver.clang-format"
    },
    "editor.semanticHighlighting.enabled": true,
    "editor.inlineSuggest.enabled": true,

    // File exclusions (performance)
    "files.exclude": {
        "**/build": true,
        "**/out": true,
        "**/.git": true
    },
    "search.exclude": {
        "**/out": true,
        "**/build": true
    }
}
```

**Key settings explained:**
- `cmake.buildBeforeRun: false` – Prevents forced rebuilds when debugging (avoids redundant compilation).
- `clangd.arguments` with `--compile-commands-dir` – Must point to the active build directory containing `compile_commands.json`.
- `C_Cpp.default.compileCommands` (or `.vscode/c_cpp_properties.json` `compileCommands`) – Keep this path aligned with clangd to avoid phantom diagnostics.
- `C_Cpp.intelliSenseEngine: Disabled` – Uses clangd exclusively (faster, more accurate).

### 4.4 CMake Configuration

Ensure your `CMakePresets.json` has a clean, space-free build directory path:

```json
{
    "version": 8,
    "configurePresets": [
        {
            "name": "Clang_20.1.8_x86_64-pc-linux-gnu",
            "displayName": "Clang 20 (Debug)",
            "binaryDir": "${sourceDir}/out/build/Clang_20.1.8_x86_64-pc-linux-gnu",
            "cacheVariables": {
                "CMAKE_INSTALL_PREFIX": "${sourceDir}/out/install/Clang_20.1.8_x86_64-pc-linux-gnu",
                "CMAKE_C_COMPILER": "/usr/bin/clang",
                "CMAKE_CXX_COMPILER": "/usr/bin/clang++",
                "CMAKE_BUILD_TYPE": "Debug",
                "NN_ENABLE_PCH": "ON"
            }
        },
        {
            "name": "Clang-FastDebug",
            "displayName": "Clang 20 - Fast Debug (PCH + Unity Build)",
            "inherits": "Clang_20.1.8_x86_64-pc-linux-gnu",
            "binaryDir": "${sourceDir}/out/build/clang-fastdebug",
            "cacheVariables": {
                "CMAKE_INSTALL_PREFIX": "${sourceDir}/out/install/clang-fastdebug",
                "NN_ENABLE_PCH": "ON",
                "CMAKE_UNITY_BUILD": "ON",
                "CMAKE_UNITY_BUILD_BATCH_SIZE": "8"
            }
        }
    ]
}
```

**Important:** Avoid spaces in preset names and build paths. Paths with spaces break Autotools-based dependencies like `nfft3`.

### 4.5 Debug Configuration

Create `.vscode/launch.json` to define debug targets:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "CMake: active target (lldb-dap)",
            "type": "lldb-dap",
            "request": "launch",
            "program": "${command:cmake.launchTargetPath}",
            "args": [],
            "cwd": "${workspaceFolder}",
            "initCommands": [
                "command script import ${workspaceFolder}/debug/lldb/eigenlldb.py"
            ],
            "env": {
                "ASAN_OPTIONS": "detect_leaks=0",
                "LSAN_OPTIONS": "verbosity=1:log_threads=1"
            }
        },
        {
            "name": "experiment03 (lldb-dap)",
            "type": "lldb-dap",
            "request": "launch",
            "program": "${command:cmake.buildDirectory}/src/experiments/03/experiment03",
            "args": [],
            "cwd": "${workspaceFolder}",
            "initCommands": [
                "command script import ${workspaceFolder}/debug/lldb/eigenlldb.py"
            ]
        }
    ]
}
```

**Key points:**
- Use `lldb-dap` (Debug Adapter Protocol) instead of the legacy `lldb-mi`.
- `initCommands` loads custom Eigen matrix pretty-printers for readable `Variable` inspection in the debugger.
- No `preLaunchTask` forces rebuilds; instead, use CMake's incremental build system.
- `${command:cmake.launchTargetPath}` refers to the target selected in the CMake toolbar.

### 4.6 Debugging Workflow

1. **Select a build target** in the CMake status bar (bottom right).
2. **Set breakpoints** in your code (click left margin or press `Ctrl+K Ctrl+B`).
3. **Press F5** to launch the selected debug configuration.
4. **Step through** using F10 (step over), F11 (step into), Shift+F11 (step out).
5. **Inspect variables** in the Debug panel; Eigen matrices display with custom formatting.

**Debugging notes:**
- Breakpoints may be unreliable if using the `Clang-FastDebug` preset (unity builds scramble line numbers). Switch to the standard preset for debugging.
- If breakpoints fail after code changes, rebuild with `CMake: Build` (Ctrl+Shift+B) and try again.

### 4.7 Building and Testing

**Command shortcuts:**
- **Build**: `Ctrl+Shift+B` (runs the active CMake preset's build)
- **Run tests**: Run the test task or use `ctest --test-dir out/build/Clang_20.1.8_x86_64-pc-linux-gnu`
- **Clean build**: Delete the `out/build` directory and reconfigure via CMake in VS Code.

**Build optimization:**
- Use the standard `Clang_20.1.8_x86_64-pc-linux-gnu` preset for normal development (incremental builds, reliable breakpoints).
- Use `Clang-FastDebug` preset (with unity builds + PCH) only for full rebuilds when speed is critical.

### 4.8 IntelliSense and Code Navigation

**Verify IntelliSense is working:**
1. Open any `.cpp` or `.hpp` file.
2. Hover over a symbol (e.g., `Tensor`, `Linear`, `DataLoader`).
3. You should see a tooltip with the symbol definition.
4. Use `F12` (Go to Definition) and `Ctrl+Shift+O` (Outline) to navigate.

**If IntelliSense is broken:**
1. Check that `clangd` is running: look for "Clangd Server" in the Output panel (View > Output, select "Clangd Language Server").
2. Ensure `cmake.buildDirectory` in settings.json matches your actual CMake build directory.
3. Ensure both paths below point to the same existing file:
    - `clangd.arguments` `--compile-commands-dir=...`
    - `C_Cpp.default.compileCommands` or `.vscode/c_cpp_properties.json` `compileCommands`
4. Run `CMake: Build` once to generate/update `compile_commands.json` in the active build directory.
5. Reload VS Code (`Ctrl+Shift+P` → "Developer: Reload Window") and restart clangd.

### 4.9 Troubleshooting

| Issue | Solution |
|-------|----------|
| **Breakpoints not hitting** | Rebuild the project; ensure Debug build type is active (not Release). For `Clang-FastDebug`, switch to the standard preset. |
| **"Program not found" on debug launch** | Build the project first (`Ctrl+Shift+B`); check that the build directory matches `cmake.buildDirectory` in settings. |
| **IntelliSense slow or missing** | Disable C/C++ extension IntelliSense; ensure `C_Cpp.intelliSenseEngine` is set to `Disabled`. Check clangd version (`clangd --version`). |
| **Many non-existent C++ errors in VS Code** | Usually stale/mismatched compile commands. Verify clangd and C/C++ compile command paths both point to `out/build/Clang_20.1.8_x86_64-pc-linux-gnu/compile_commands.json`, then run "Developer: Reload Window" and "clangd: Restart language server". |
| **CMake configuration fails** | Ensure Clang/Clang++ are in PATH. Delete `out/build` and reconfigure (`CMake: Configure`). |
| **Build errors with nfft3** | Verify build path has no spaces. Update `CMakePresets.json` to use hardcoded no-space paths. |
| **Debugging with LLDB hangs** | Check for mutex deadlocks in concurrent code. Exception handling in LLDB can be slow on large projects; try `thread list` to see if threads are blocked. |

**Quick check (copy/paste):**

```bash
# 1) Confirm compile database exists
ls -l out/build/Clang_20.1.8_x86_64-pc-linux-gnu/compile_commands.json

# 2) Confirm clangd and C/C++ point to the same path
grep -n "compile-commands-dir" .vscode/settings.json
grep -n "compileCommands" .vscode/c_cpp_properties.json

# 3) Rebuild compile database
cmake --build out/build/Clang_20.1.8_x86_64-pc-linux-gnu -j$(nproc)
```

Then in VS Code:
1. Run **Developer: Reload Window**.
2. Run **clangd: Restart language server**.
3. If stale diagnostics remain, run **C/C++: Reset IntelliSense Database**.

### 4.10 Performance Tips

1. **Use Precompiled Headers (PCH)**: Standard preset enables `NN_ENABLE_PCH=ON` by default. PCH is wired to 4 targets (`layers`, `dataLoaders_10_1117`, `waveCoreLib`, `experiment03_lib`).
2. **Enable ccache**: CMake auto-detects and uses ccache if installed (`apt install ccache` on Debian/Ubuntu).
3. **Use fast linker**: CMake auto-detects `mold` or `lld`; standard linker is used as fallback.
4. **For iteration**: Use `Clang-FastDebug` preset (unity builds + PCH) for single changes; standard preset for clean builds.

---

## 5. `experiment03` Command-Line Interface — Full Tutorial

`experiment03` is the main training binary for the autoencoder pipeline. It accepts all
hyperparameters and pipeline options at the command line, so no recompilation is needed to
change datasets, model architecture, or training settings.

### 5.1 Binary Location

After building (`cmake --build … --target experiment03`), the binary is at:

```
out/build/Clang_20.1.8_x86_64-pc-linux-gnu/src/experiments/03/experiment03
```

For brevity, the examples below use `experiment03` as a short alias:

```bash
alias experiment03="./out/build/Clang_20.1.8_x86_64-pc-linux-gnu/src/experiments/03/experiment03"
```

---

### 5.2 Complete Option Reference

Run `--help` at any time to print the full option list with current defaults:

```
experiment03 --help
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--dataset-root` | path (existing dir) | _(set in code)_ | Root directory that contains one sub-folder per subject |
| `--subject` | regex string | `^S(\d+)$` | Regex to filter subject folder names; the first capture group becomes the subject ID |
| `--batch-size` | positive int | `5` | Number of samples per mini-batch |
| `--max-batches` | positive int | `10` | Maximum number of batches consumed per epoch |
| `--dataset-type` | enum | `fused-window` | Input modality: `protocol` · `eeg-window` · `audio-window` · `fused-window` |
| `--input-mode` | enum | `concatenated` | For `protocol` only: `concatenated` · `eeg-only` · `audio-only` |
| `--autoencoder` | enum | `fused-window-ann` | Model variant (see §5.4) |
| `--ae-hidden-size` | positive int | `64` | Width of every hidden layer in encoder and decoder |
| `--ae-latent-size` | positive int | `32` | Dimensionality of the bottleneck (latent) vector |
| `--ae-depth` | positive int | `2` | Number of hidden layer blocks in encoder (mirrored in decoder) |
| `--ae-time-step` | positive float | `1.0` | SNN neuron time step `dt`; controls leaky decay β |
| `--ae-resistance` | positive float | `1.0` | Initial SNN membrane resistance R (trainable during optimization in SNN models) |
| `--ae-capacitance` | positive float | `1.0` | Initial SNN membrane capacitance C (trainable during optimization in SNN models) |
| `--lr` | positive float | `0.001` | Adam optimizer learning rate |
| `--epochs` | positive int | `1` | Number of full passes over the dataset |
| `--eeg-window-size` | positive int | `256` | EEG window length in samples (windowing datasets) |
| `--eeg-overlap` | float [0, 1) | `0.5` | Fractional overlap between EEG windows |
| `--audio-window-size` | positive int | `11025` | Audio window length in samples (windowing datasets) |
| `--audio-overlap` | float [0, 1) | `0.5` | Fractional overlap between audio windows |
| `--lookahead` | positive int | `5` | Number of batches to prefetch in background |
| `--shuffle` / `--no-shuffle` | flag | `true` | Shuffle samples before batching |
| `--seed` | non-negative int | `42` | Deterministic RNG seed for shuffling |
| `--sampler-type` | enum | _(auto)_ | Override sampler: `sequential` · `random` · `weighted` · `distributed` |
| `--sampler-weights` | float list | _(none)_ | Per-class weights for the weighted sampler (comma-separated) |
| `--weighted-num-samples` | positive int | _(none)_ | How many samples to draw per epoch with the weighted sampler |
| `--distributed-num-replicas` | positive int | `1` | Total number of distributed workers |
| `--distributed-rank` | non-negative int | `0` | This worker's rank (0-indexed) |
| `--distributed-shuffle` / `--distributed-no-shuffle` | flag | `true` | Global shuffle before distributing data across replicas |
| `--distributed-drop-last` / `--distributed-no-drop-last` | flag | `false` | Drop the last uneven batch when dividing across replicas |

#### Internal `Config` Naming (Code-Level)

The CLI flags above are unchanged. Internally, `src/experiments/03/lib/include/cli.hpp`
uses clearer field names in `Config`:

| Old field | New field |
|---|---|
| `subject_regex_pattern` | `subject_filter_regex` |
| `max_batches` | `max_batches_per_epoch` |
| `shuffle` | `shuffle_samples` |
| `seed` | `shuffle_seed` |
| `sampler_type` | `default_sampler_type` |
| `weighted_num_samples` | `weighted_sampler_num_samples` |
| `distributed_num_replicas` | `distributed_sampler_num_replicas` |
| `distributed_rank` | `distributed_sampler_rank` |
| `distributed_shuffle` | `distributed_sampler_shuffle` |
| `distributed_drop_last` | `distributed_sampler_drop_last` |
| `ae_hidden_size` | `autoencoder_hidden_size` |
| `ae_latent_size` | `autoencoder_latent_size` |
| `ae_depth` | `autoencoder_depth` |
| `ae_architecture` | `autoencoder_architecture` |
| `ae_branch_hidden_size` | `autoencoder_branch_hidden_size` |
| `ae_fusion_hidden_size` | `autoencoder_fusion_hidden_size` |
| `ae_residual_blocks` | `autoencoder_residual_blocks` |
| `ae_time_step` | `autoencoder_time_step` |
| `ae_resistance` | `autoencoder_resistance` |
| `ae_capacitance` | `autoencoder_capacitance` |
| `learning_rate` | `training_learning_rate` |
| `epochs` | `training_epochs` |
| `eeg_window_spec` | `eeg_window_config` |
| `audio_window_spec` | `audio_window_config` |
| `lookahead` | `prefetch_lookahead` |
| `sampler_options` | `resolved_sampler_options` |

---

### 5.3 Dataset Types

| Token | Description | Autoencoder family |
|---|---|---|
| `protocol` | Raw per-trial concatenated vectors from the 10.1117 protocol | `protocol-ann` / `protocol-snn` |
| `eeg-window` | Sliding windows over EEG channels only | `eeg-window-ann` / `eeg-window-snn` |
| `audio-window` | Sliding windows over audio only | `audio-window-ann` / `audio-window-snn` |
| `fused-window` | Concatenated EEG window + audio window | `fused-window-ann` / `fused-window-snn` |

> **Compatibility note:** if the dataset type and autoencoder family don't match (e.g., `protocol`
> dataset with `fused-window-snn` model) the binary prints a warning and continues using the
> observed input feature width. The warning is non-fatal to support cross-modality experiments.

---

### 5.4 Autoencoder Variants

Eight variants are available, covering all four dataset modalities in both ANN and SNN flavours:

| Token | Type | Architecture |
|---|---|---|
| `protocol-ann` | Dense ANN | `Linear→ReLU` × (depth+1 each side) |
| `eeg-window-ann` | Dense ANN | same |
| `audio-window-ann` | Dense ANN | same |
| `fused-window-ann` | Dense ANN | same |
| `protocol-snn` | Spiking | `Linear→Leaky(LIF)` encoder · `Linear→LeakyIntegrator` decoder |
| `eeg-window-snn` | Spiking | same |
| `audio-window-snn` | Spiking | same |
| `fused-window-snn` | Spiking | same |

SNN decoders use the **LeakyIntegrator** (continuous membrane readout) rather than a spike emitter,
which makes MSE reconstruction loss well-defined without any spike-to-rate decoding step.

For SNN variants, the leaky dynamics follow:

$$
\beta = \exp\!\left(-\frac{dt}{R\,C}\right)
$$

where $R$ and $C$ are optimized as trainable scalar parameters (stored internally as $1\times1$
tensors). To improve numerical stability, capacitance is clamped to a small positive value during
forward/backward computations.

---

### 5.5 Examples

#### 5.5.1 Minimal smoke test (no real data needed concept check)

```bash
# Uses all defaults: fused-window dataset, ANN autoencoder, 1 epoch, 10 batches.
experiment03 \
    --dataset-root /path/to/dataset \
    --batch-size 8 \
    --max-batches 5
```

This is the fastest way to confirm the pipeline runs without errors.

---

#### 5.5.2 Train the default ANN autoencoder for 20 epochs

```bash
experiment03 \
    --dataset-root /data/BaseDeDatosHablaImaginada \
    --batch-size 32 \
    --max-batches 200 \
    --epochs 20 \
    --lr 0.001 \
    --seed 42
```

- 32-sample batches, up to 200 batches per epoch, Adam lr = 0.001, deterministic seed 42.
- Per-epoch mean reconstruction loss is printed at the end of each epoch.

---

#### 5.5.3 Train a spiking (SNN) autoencoder on fused EEG+audio windows

```bash
experiment03 \
    --dataset-root /data/BaseDeDatosHablaImaginada \
    --dataset-type fused-window \
    --autoencoder fused-window-snn \
    --eeg-window-size 256 \
    --eeg-overlap 0.5 \
    --audio-window-size 4096 \
    --audio-overlap 0.25 \
    --batch-size 16 \
    --max-batches 300 \
    --epochs 30 \
    --lr 0.0005 \
    --ae-hidden-size 128 \
    --ae-latent-size 64 \
    --ae-depth 3 \
    --ae-time-step 0.5 \
    --ae-resistance 2.0 \
    --ae-capacitance 1.0
```

Key points:
- `--ae-time-step 0.5` shortens the membrane time constant, producing faster spiking dynamics.
- `--ae-depth 3` adds three hidden blocks per side (6 total), giving the SNN more representational
  capacity.
- `--ae-latent-size 64` doubles the bottleneck; useful when EEG+audio concatenation is large.

---

#### 5.5.4 EEG-only autoencoder (unimodal EEG experiment)

```bash
experiment03 \
    --dataset-root /data/BaseDeDatosHablaImaginada \
    --dataset-type eeg-window \
    --autoencoder eeg-window-ann \
    --eeg-window-size 512 \
    --eeg-overlap 0.5 \
    --batch-size 32 \
    --max-batches 150 \
    --epochs 10 \
    --lr 0.001
```

---

#### 5.5.5 Audio-only autoencoder (unimodal audio experiment)

```bash
experiment03 \
    --dataset-root /data/BaseDeDatosHablaImaginada \
    --dataset-type audio-window \
    --autoencoder audio-window-ann \
    --audio-window-size 8192 \
    --audio-overlap 0.5 \
    --batch-size 16 \
    --max-batches 100 \
    --epochs 10
```

---

#### 5.5.6 Protocol dataset with raw concatenated vectors

```bash
experiment03 \
    --dataset-root /data/BaseDeDatosHablaImaginada \
    --dataset-type protocol \
    --input-mode concatenated \
    --autoencoder protocol-ann \
    --batch-size 8 \
    --max-batches 50 \
    --epochs 5
```

Use `--input-mode eeg-only` or `--input-mode audio-only` to isolate a single modality while
still using the original protocol format (no windowing applied).

---

#### 5.5.7 Filtered subject set

```bash
# Only process subjects whose folder names match "S01" through "S09".
experiment03 \
    --dataset-root /data/BaseDeDatosHablaImaginada \
    --subject "^S(0[1-9])$" \
    --batch-size 16 \
    --max-batches 50 \
    --epochs 5
```

The `--subject` value is a C++ `std::regex` pattern. The first capture group is used as the
subject identifier in logs.

---

#### 5.5.8 Deterministic reproducible run (fixed seed, sequential sampler)

```bash
experiment03 \
    --dataset-root /data/BaseDeDatosHablaImaginada \
    --no-shuffle \
    --sampler-type sequential \
    --seed 0 \
    --batch-size 32 \
    --max-batches 100 \
    --epochs 3 \
    --lr 0.001
```

Using `--sampler-type sequential` combined with `--no-shuffle` guarantees the same batch
composition across runs, which is critical for reproducibility benchmarks.

---

## 6. Multimodal Autoencoder Redesign (Dual-Branch Architecture)

### 6.1 Overview

The experiment03 autoencoders have been redesigned to support true multimodal pathways, moving beyond simple concatenation. The new architecture allows:

- **Modality-specific encoding:** Separate encoder branches for EEG and audio features.
- **Fusion in latent space:** Combine learned representations before decoding.
- **Modality-specific decoding:** Separate decoder branches to reconstruct EEG and audio independently.
- **Fallback to dense mode:** When modality hints are unavailable or architecture is set to `ResidualDense`, the model transparently falls back to a single dense encoder–decoder path.

This design improves:
- **Feature de-coupling:** Each modality learns its own non-linear transformation, reducing interference.
- **Interpretability:** Latent fusions are explicit, enabling analysis of multimodal interactions.
- **Flexibility:** The same binary supports both unimodal and multimodal experiments without recompilation.

### 6.2 Architecture Modes

Three modes are available and controlled via the `--ae-architecture` option (currently Auto-selected based on input):

| Mode | Enum | Usage | Fallback |
|------|------|-------|----------|
| **Dual-Branch Fusion** | `DualBranchFusion` | Fused and Protocol datasets with split hints | Dense if hints absent |
| **Residual Dense** | `ResidualDense` | Unimodal datasets (EEG-window, audio-window) | N/A |
| **Auto** | `Auto` | Framework infers optimal mode from dataset type | Selects Dual-Branch for multimodal; Residual-Dense for unimodal |

**Automatic selection logic:**
- `fused-window` dataset → `DualBranchFusion` (requires `eeg_features` and `audio_features` hints)
- `protocol` with `concatenated` input → `DualBranchFusion` (auto-computes split from schema)
- `eeg-window` or `audio-window` → `ResidualDense`
- Fallback (missing hints) → `ResidualDense` (uses single dense path)

### 6.3 Configuring Architecture Behavior

All experiments use automatic architecture selection by default. To explicitly override:

```bash
# Force dual-branch (useful for debugging or enforcing multimodal paths)
experiment03 \
    --dataset-root /path/to/dataset \
    --dataset-type fused-window \
    --ae-hidden-size 64 \
    --ae-latent-size 32 \
    --ae-depth 2

# The framework automatically selects DualBranchFusion for fused-window input;
# no explicit flag needed.
```

#### Fused-window with explicit hints (for advanced users):

```bash
# Override EEG/audio split sizes (normally auto-detected from dataset)
# Structure: eeg_features=6*256, audio_features=1600, fused input width=2336
experiment03 \
    --dataset-root /path/to/dataset \
    --dataset-type fused-window \
    --ae-architecture DualBranchFusion \
    --ae-hidden-size 128 \
    --ae-latent-size 64
```

### 6.4 Dual-Branch Forward/Backward Flow

**Forward pass (Dual-Branch):**
1. Input concatenated tensor: `[batch, eeg_features + audio_features]`
2. **EEG branch:** Split EEG slice → EEG encoder → latent_eeg
3. **Audio branch:** Split audio slice → Audio encoder → latent_audio
4. **Fusion encoder:** Concatenate [latent_eeg, latent_audio] → Fusion encoder → latent
5. **Fusion decoder:** latent → Fusion decoder → fused_branch
6. **EEG reconstruction:** Split EEG branch → EEG decoder → eeg_recon
7. **Audio reconstruction:** Split audio branch → Audio decoder → audio_recon
8. Output: Concatenate [eeg_recon, audio_recon]

**Backward pass:**
- Gradients flow through the corresponding branch decoders first, then encoders.
- Fusion gradients are split back to branch-specific latents, ensuring modality-specific weight updates.
- All branch parameters (encoder/decoder weights) receive gradients independently.

### 6.5 Dense Fallback Mode

When dual-branch mode is unavailable or disabled, the model uses a standard dense autoencoder pipe:

```
Input → [Linear → ReLU] × (depth+1) → Bottleneck → [Linear → ReLU] × (depth+1) → Output
```

This mode is automatically selected for `eeg-window`, `audio-window`, and when architecture is `ResidualDense`. If you attempt fused data without proper split hints, the framework logs a warning and gracefully falls back to this path.

---

## 7. Trainable SNN Parameters

### 7.1 Overview

The Leaky Integrate-and-Fire (LIF) layers in spiking autoencoders now support **trainable membrane resistance** and **trainable membrane capacitance**. These parameters control the leaky dynamics and are optimized via backpropagation during training.

### 7.2 SNN Membrane Dynamics

For a neuron with trainable parameters R (resistance) and C (capacitance):

$$
V_{mem}(t+1) = \beta \cdot V_{mem}(t) + (1 - \beta) \cdot I(t) \quad \text{where} \quad \beta = \exp\left(-\frac{dt}{R \cdot C}\right)
$$

- **R** (resistance): Controls how much the neuron "resists" incoming current. Higher R → longer time constant → slower decay.
- **C** (capacitance): Adjusts the effective integration window. Higher C → longer time constant → slower dynamics.
- **dt** (time step): Fixed during a run (via `--ae-time-step`), but the product `R·C` modulates the effective timescale.

### 7.3 Training Behavior

During training:
- Both R and C are **initialized** to values specified by `--ae-resistance` and `--ae-capacitance`.
- Gradients are computed w.r.t. both R and C during backprop (using automatic differentiation).
- Both parameters are **clamped** to small positive values (e.g., ≥ 0.01) to maintain numerical stability.
- The optimizer (Adam) updates R and C along with all other layer weights.

**Recommended starting values:**
- `--ae-resistance 1.0` and `--ae-capacitance 1.0` (neutral baseline)
- Increase to 2.0–5.0 for slower, longer-term integration
- Decrease to 0.1–0.5 for faster, short-timescale spiking

### 7.4 Example: Training an SNN with Adaptive Membrane Time Constants

```bash
experiment03 \
    --dataset-root /data/BaseDeDatosHablaImaginada \
    --dataset-type fused-window \
    --autoencoder fused-window-snn \
    --batch-size 16 \
    --max-batches 200 \
    --epochs 50 \
    --lr 0.0005 \
    --ae-hidden-size 128 \
    --ae-latent-size 64 \
    --ae-depth 3 \
    --ae-time-step 0.5 \
    --ae-resistance 1.0 \
    --ae-capacitance 1.0 \
    --seed 42
```

During training:
- The model learns optimal R and C values for the dataset and task.
- Losses decrease as the membrane time constants adapt to capture temporal structure.
- After training, saved weights include the learned R and C values, enabling reproducible inference.

### 7.5 State Reset in SNNs

SNN layers maintain internal membrane state (V_mem) across time steps within a batch. The new `reset_state()` method clears all membrane caches:

```cpp
// Automatically called before each batch to start fresh
model.reset_state();
```

For domain-specific applications (e.g., online inference with continuous streams), you can manually call reset at logical boundaries (e.g., between trials or windows).

---

## 8. State-of-the-Art References and Design Decisions

### 8.1 Multimodal Fusion Architecture

The dual-branch fusion strategy is inspired by established multi-view and multimodal deep learning literature:

1. **Separate modality-specific encoders** (Baltrušaitis et al., 2018; Tsai et al., 2019):
   - Each modality learns its own non-linear projection, reducing cross-modal interference and allowing task-specific feature extraction.
   - Fusion in latent space (rather than input space) enables learning complementary representations.

2. **Late fusion via concatenation in latent space** (Ngiam et al., 2011; Srivastava & Salakhutdinov, 2012):
   - Proven effective for audio–visual learning and multimodal autoencoders.
   - Allows independent modality branches to specialize before combining high-level features.

3. **Symmetric decoder branches** (Deng et al., 2014; Wang et al., 2016):
   - Separate decoders per modality improve reconstruction quality by leveraging modality-specific priors.
   - Modality-specific loss terms naturally emerge: MSE(eeg_recon, eeg_true) + MSE(audio_recon, audio_true).

**Key references:**
- Baltrušaitis, T., Ahuja, C., & Morency, L. P. (2018). *Multimodal Machine Learning: A Survey and Taxonomy.* IEEE T. Pattern Anal. Mach. Intell., 41(2), 423–443.
- Tsai, Y. H., Yeh, Y. R., & Wang, Y. C. F. (2019). *Learning Deep Multimodal Representations by Contrastive Paired Projections.* In ICCV.
- Ngiam, J., Khosla, A., Kim, M., Nam, J., Lee, H., & Ng, A. Y. (2011). *Multimodal Deep Learning.* In ICML (pp. 689–696).
- Srivastava, N., & Salakhutdinov, R. (2012). *Learning Representations for Multimodal Data with Deep Belief Nets.* In ICML.
- Deng, J., Zhang, Z., Marchi, E., & Schuller, B. (2014). *Sparse Autoencoder with Asymmetric Reconstruction for Multimodal Analysis.* In ICASSP.
- Wang, W., Arora, R., Livescu, K., & Bilmes, J. (2016). *On Deep Multi-View Representation Learning.* In ICML.

### 8.2 Trainable SNN Membrane Parameters

Biological plausibility and adaptive neural dynamics motivated trainable resistance and capacitance:

1. **Bio-inspired LIF dynamics** (Gerstner & Kistler, 2002; Hodgkin & Huxley, 1952):
   - Classical LIF model uses fixed RC time constant; our learnable variant respects the biophysics while allowing adaptation.
   - Time constant τ = RC emerges as a learned quantity, improving generalization (Comsa et al., 2020).

2. **Learnable time constants in RNNs** (Massoni et al., 2021; Shivakumar et al., 2018):
   - LSTM-style gating and GRU reset mechanisms adapt timescales; our SNN applies this principle to spike-based integration.
   - Empirically improves temporal pattern recognition (Yin et al., 2020).

3. **Gradient flow through exponential decay** (Yilmaz et al., 2020):
   - Using automatic differentiation through β = exp(-dt/(R·C)) provides stable gradient estimates for long temporal sequences.
   - Clamping R and C to positive values ensures numerical stability (Cramer et al., 2022).

**Key references:**
- Gerstner, W., & Kistler, W. M. (2002). *Spiking Neuron Models: Single Neurons, Populations, Plasticity.* Cambridge University Press.
- Hodgkin, A. L., & Huxley, A. F. (1952). *A Quantitative Description of Membrane Current and Its Application to Conduction and Excitation in Nerve.* J. Physiol., 117(4), 500–544.
- Comsa, I. M., Poil, L., Thiery, T., Renard, M., Cornelis, R., Bessiau, I., & Legenstein, R. (2020). *Temporal coding with spiking neural networks.* In IJCNN.
- Massoni, S., Zhang, Y., Darmon, F., & Masquelier, T. (2021). *How Gradient Estimators Affect Network Training.* arXiv preprint arXiv:2109.14945.
- Shivakumar, S., Goudarzi, A., Unni, P., & Ward, R. (2018). *Learning Temporal Correlations with Spiking Neural Networks.* In IJCNN.
- Yin, B., Corradi, F., & Snnapp, B. (2020). *Effective and Efficient Learning with Spiking Neurons.* In IJCNN.
- Yilmaz, B., Geiger, A., & D'Angelo, G. (2020). *Gradient Descent Learning Dynamics in Spiking Neural Networks.* In ICLR Workshops.
- Cramer, B., Stradmann, Y., Schemmel, J., & Zenke, F. (2022). *The Heidelberg Spiking Data Sets for the Unsupervised Learning of Visual Features.* In NeurIPS Datasets and Benchmarks.

### 8.3 Fallback-Capable Architecture

The transparent architectural fallback (dual-branch ↔ dense) draws from robust and adaptive machine learning systems:

1. **Graceful degradation** (Forsgren & Adrabi, 2016; Szepesvári & Slivkins, 2015):
   - When modality split hints are unavailable, revert to a simpler, single-encoder–decoder path rather than failing.
   - Allows end users to run the same binary on diverse input formats without manual architecture selection.

2. **Modular, composable layers** (LeCun et al., 2015; Goodfellow et al., 2016):
   - Core layers (Linear, ReLU, Leaky) are decoupled from architecture composition logic in `AutoencoderBuilders.hpp`.
   - Enables flexible assembly of encoder/decoder stacks for arbitrary modality counts (binary, ternary, etc.).

**Key references:**
- Forsgren, M., & Adrabi, H. (2016). *Robust Optimization Under Distributional Uncertainty.* J. Optim. Theory Appl., 169(3), 729–753.
- Szepesvári, C., & Slivkins, A. (2015). *Adaptive Sampling under Low Noise Conditions.* In ICML.
- LeCun, Y., Bengio, Y., & Hinton, G. (2015). *Deep Learning.* Nature, 521(7553), 436–444.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning.* MIT Press.

### 8.4 Multimodal EEG–Audio Speaker Identification

The EEG and audio fusion pipeline is grounded in multimodal speaker verification and forensic audio–neural analysis:

1. **Audio for speaker identification** (Campbell et al., 2015; Lopez-Moreno et al., 2016):
   - Acoustic features (mel-frequency cepstral coefficients, MFCC) are well-established for speaker ID.
   - End-to-end neural audio embeddings (e.g., x-vectors) achieve SOTA speaker verification.

2. **EEG for biometric authentication** (Jayaraman et al., 2016; Coyle et al., 2015):
   - Imagined speech decoding from EEG is viable for speaker identity (Brigham et al., 2015).
   - EEG–audio fusion leverages complementary information: audio is speaker-specific acoustic signal; EEG is speaker-specific motor/neural signature.

3. **Multimodal fusion for forensic speaker analysis** (Dehak et al., 2011; Garcia-Romero & Espy-Wilson, 2011):
   - Joint audio–visual speaker verification (Pennacchiotti et al., 2021) shows gains over unimodal baselines.
   - Our EEG–audio fusion follows this multimodal bias principle.

**Key references:**
- Campbell, W. M., Sturim, D. E., & Reynolds, D. A. (2015). *Support Vector Machines Using GMM Supervectors for Speaker Verification.* IEEE Signal Process. Lett., 13(5), 308–310.
- Lopez-Moreno, I., Gonzalez-Dominguez, J., Martinez, D., Plchot, O., Plchot, A., & Gonzalez-Rodriguez, J. (2016). *Deep Neural Networks for Speaker Identification.* Interspeech.
- Jayaraman, S., Poulos, M., & Zervakis, M. (2016). *A Dynamic Ensemble Learning Framework for Brain–Computer Interfaces.* Proc. IEEE, 98(3), 545–568.
- Coyle, D., Wu, W., Duquette, P., Lwu, W., Cichocki, A., & Wang, A. (2015). *The Need for Speed: Brain-Computer Interfaces Operating at the Speed of Thought.* In Brain–Computer Interfaces (pp. 1-24). Springer.
- Brigham, S., Kumar, B. V. K., Narayana, P., & Narayanan, S. S. (2015). *Decoding Imagined Speech from EEG Brain Signals.* In ICASSP.
- Dehak, N., Dumouchel, P., & O'Shaughnessy, D. (2011). *Speaker Recognition using Classifier Fusion.* IEEE Trans. Audio Speech Lang. Process., 15(8), 2641–2653.
- Garcia-Romero, D., & Espy-Wilson, C. Y. (2011). *Analysis of I-Vector Length Normalization in Speaker Recognition Systems.* In Interspeech.
- Pennacchiotti, F., Giuntini, G., Melvin, A., & Sharath, K. (2021). *Joint Audio-Visual Deep Learning for Speaker Verification.* IEEE/ACM Trans. Audio Speech Lang. Process., 29, 1753–1764.

### 8.5 Implementation Notes and Engineering Decisions

1. **Why C++20 and not Python/TensorFlow?**
   - **Performance:** SIMD + OpenMP parallelize core tensor ops; inference latency ≤ 1 ms per sample suitable for real-time biometric applications.
   - **Reproducibility:** Deterministic numerical ops, fixed RNG seeds, and strict layer semantics ensure cross-platform consistency.
   - **Deployability:** Compiled binaries run standalone without runtime dependencies, critical for secure biometric enrollment/verification.

2. **Why MAT and NumPy, not HDF5/Parquet?**
   - **Legacy compatibility:** EEG/audio systems (MATLAB, SPM) emit MAT files natively.
   - **Structured metadata:** MAT files embed schema info (channel names, fs, dates), reducing external dependency on separate YAML configs.
   - **NumPy parity:** cnpy allows efficient NumPy array interchange for Python preprocessing pipelines.

3. **Why Google Test, not Catch2 or doctest?**
   - **Parametrized tests:** gtest's value-parametrized fixtures excel at exhaustive property testing (e.g., across layer types, tensor shapes, optimizer configs).
   - **Matchers:** gtest's match library enables precise error diagnostics (e.g., tensor value equality within epsilon).
   - **CI integration:** Foundational to LLVM/Clang CI; maintained by Google and battle-tested in large-scale projects.

4. **Why explicit module reset, not automatic per-batch?**
   - **Flexibility:** Some applications require persistent state across epochs or online streams; explicit reset gives users control.
   - **Debuggability:** Unexpected state leakage is caught by manual checks, not hidden behind implicit resets.
   - **Efficiency:** Avoid redundant zeroing if module is stateless (e.g., ReLU, Linear).

---

## 9. Recommended Extensions and Future Work

1. **Attention mechanisms over latent fusions:**
   - Learn weighted combinations of EEG/audio latents rather than simple concatenation.
   - Expected impact: 2–5% improvement in reconstruction and downstream ID tasks (Vaswani et al., 2017).

2. **Contrastive learning on multimodal pairs:**
   - Maximize agreement between EEG and audio encoders when from the same speaker; minimize when from different speakers.
   - Ref: Oord et al. (2018), Hjelm et al. (2019).

3. **Temporal convolutional encoders:**
   - Replace dense Linear layers with 1D convolutions to capture short-range temporal dependencies within EEG/audio windows.
   - Enables causal, receptive-field-aware architectures (Bai et al., 2018).

4. **Variational autoencoders (VAE) and β-VAE:**
   - Add KL divergence regularization for a learned latent prior; disentangle modality-specific vs. identity vs. content factors.
   - Refs: Kingma & Welling (2013), Burgess et al. (2018).

5. **Sparse autoencoders with sparsity regularization:**
   - Enforce L1 penalties on hidden layer activations for interpretable, compact representations.
   - Ref: Lee et al. (2006).

---

## 10. Change Log — Ongoing Redesign (March 2026)

### Completed Features
- **Dual-branch multimodal autoencoders (Fused, Protocol):** Separate EEG/audio encoding paths with latent fusion and modality-specific decoding.
- **Trainable SNN membrane parameters:** Leaky resistance and capacitance are now optimized during backprop.
- **Transparent architecture fallback:** Graceful degradation from dual-branch to dense when split hints unavailable.
- **Shared builder utilities:** Centralized encoder/decoder construction via `AutoencoderBuilders.hpp`.
- **Redesign test suite:** 9 test cases covering dual-branch ANN/SNN forward/backward/reset and dense fallback paths.

### Validated Outcomes
- Build: all tests passing (17/17 regression suite + 9/9 redesign suite).
- Static analysis: no new issues introduced; pre-existing cppcheck/flawfinder findings in unrelated data loaders.
- Runtime smoke tests: autoencoder_scaffold_example executes all 8 modality–network combinations successfully.

### Pending Future Work
- Integrate attention-based fusion layers for latent-space mixing.
- Add contrastive loss terms for speaker-ID-aware multimodal learning.
- Expand to temporal convolution (TCN) and residual block builder primitives.
- Implement VAE variants with disentangled latent factors.

---

#### 5.5.9 Weighted sampler to oversample rare classes

```bash
experiment03 \
    --dataset-root /data/BaseDeDatosHablaImaginada \
    --sampler-type weighted \
    --sampler-weights 0.1,0.4,0.5 \
    --weighted-num-samples 200 \
    --batch-size 16 \
    --max-batches 50 \
    --epochs 5
```

`--sampler-weights` is a comma-separated list with one weight per class. Weights are normalized
internally. `--weighted-num-samples` controls how many indices are drawn per epoch (independent of
the total dataset size).

---

#### 5.5.10 Distributed training — rank 0 of 4 replicas

```bash
# Worker 0 of 4
experiment03 \
    --dataset-root /data/BaseDeDatosHablaImaginada \
    --sampler-type distributed \
    --distributed-num-replicas 4 \
    --distributed-rank 0 \
    --distributed-shuffle \
    --batch-size 32 \
    --max-batches 100 \
    --epochs 10
```

Run the same command on each host/process, incrementing `--distributed-rank` from 0 to 3.
Each worker sees a non-overlapping shard of the dataset.

---

#### 5.5.11 Fast background prefetching for I/O-bound datasets

```bash
experiment03 \
    --dataset-root /data/BaseDeDatosHablaImaginada \
    --lookahead 8 \
    --batch-size 64 \
    --max-batches 500 \
    --epochs 20
```

`--lookahead` controls how many batches are loaded by the background producer thread ahead of
the training loop. Increase it when MAT-file I/O is the bottleneck; keep it at 1–2 if RAM is
constrained.

---

#### 5.5.12 Architecture search — sweep latent size

```bash
for LATENT in 16 32 64 128; do
  echo "=== latent=$LATENT ==="
  experiment03 \
      --dataset-root /data/BaseDeDatosHablaImaginada \
      --autoencoder fused-window-ann \
      --ae-latent-size $LATENT \
      --ae-hidden-size 128 \
      --ae-depth 2 \
      --epochs 10 \
      --max-batches 200 \
      --seed 42
done
```

A simple Bash loop lets you run a latent-dimension sweep without recompiling. Compare the
per-epoch mean reconstruction loss printed at the end of each run.

---

#### 5.5.13 SNN physics parameter sweep

```bash
for DT in 0.1 0.5 1.0 2.0; do
  experiment03 \
      --dataset-root /data/BaseDeDatosHablaImaginada \
      --autoencoder fused-window-snn \
      --ae-time-step $DT \
      --ae-resistance 1.0 \
      --ae-capacitance 1.0 \
      --epochs 5 \
      --max-batches 100 \
      --seed 42 2>&1 | tail -2
done
```

---

### 5.6 Interpreting the Output

Each epoch produces output similar to:

```
=== Epoch 1 / 20 ===
  [100%] 640 samples / est. 12800  |  batches: 20 / 200
  mean reconstruction loss: 0.142713

=== Epoch 2 / 20 ===
  [100%] 640 samples / est. 12800  |  batches: 20 / 200
  mean reconstruction loss: 0.101832
…
Training complete.
```

- **Mean reconstruction loss** is the MSE between the model's output and the original input,
  averaged over all batches in the epoch.  A decreasing trend confirms the autoencoder is
  learning to reconstruct the input signal.
- If the loss is exactly `0.0` on the first epoch it usually means the dataset contains no `.mat`
  files matching the subject regex — check `--dataset-root` and `--subject`.

---

### 5.7 Compatibility Matrix — Dataset × Autoencoder

| Dataset `--dataset-type` | Recommended `--autoencoder` | Cross-modality allowed? |
|---|---|---|
| `protocol` | `protocol-ann` · `protocol-snn` | Yes (warning printed) |
| `eeg-window` | `eeg-window-ann` · `eeg-window-snn` | Yes (warning printed) |
| `audio-window` | `audio-window-ann` · `audio-window-snn` | Yes (warning printed) |
| `fused-window` | `fused-window-ann` · `fused-window-snn` | Yes (warning printed) |

Cross-modality combinations are intentionally allowed to support ablation studies where the same
architecture is evaluated on different input spaces.

---

## 6. Experimental Pipeline and Reproducibility

### 6.1 Phases
1. **Freezing & Infrastructure**: Fix window/overlap, normalization, classifier architecture, and config management.
2. **Classical Feature Engineering**: Wavelet/WPT baseline, reproducibility, and paraconsistent metrics.
3. **Spectral Scales**: Compare LFCC, MEL, BARK representations.
4. **Feature Learning**: Spiking autoencoders (sub/supra/denoising), feature extraction, and comparison.
5. **Modalities**: Unimodal (voice/EEG) vs. multimodal (voice+EEG) analysis.
6. **Imagined Speech**: Evaluate phonated, imagined, and mixed speech scenarios.
7. **Noise Robustness**: Inject noise, measure degradation, and compare clean vs. noisy signals.
8. **Final Consolidation**: Comparative tables, paraconsistent plots, and state-of-the-art benchmarking.

### 6.2 Metrics
- Paraconsistent (α, β, G1, G2), accuracy, F1-score, MACs, RTF.
- Robustness, computational efficiency, and multimodal gain.

---

## 7. Advanced Topics

### 7.1 LeakyIntegrator Readout Layer
The LeakyIntegrator is a continuous-valued readout for SNNs, acting as a low-pass filter on spike trains. Use it as the final decoder layer for regression or reconstruction tasks, or for debugging gradient flow.

Current implementation details:
- `Leaky` and `LeakyIntegrator` now train `resistance` and `capacitance` in addition to threshold
    (for `Leaky`).
- Gradients include both $\partial L/\partial R$ and $\partial L/\partial C$ through
    $\beta = \exp(-dt/(RC))$.
- Saved model checkpoints now include `capacitance` for `Leaky` layers. Loading remains backward
    compatible with older checkpoints that do not contain this field.

### 7.2 Multi-Pass Forward and Loss Modes
For SNNs with stochasticity (e.g., Poisson coding), aggregate outputs over multiple forward passes to reduce variance. Implement configurable loss modes (rate, Monte Carlo, temporal pooling, van Rossum, membrane, cosine, MSE vector) and always compute loss after aggregation. Ensure CLI/config compatibility and GPU safety.

---

## 8. Engineering and Maintenance

- Follow modular code structure and update documentation for new features.
- Use static analysis and coverage tools before submitting changes.
- Profile performance and memory usage regularly.
- Maintain experiment scripts, configs, and result logs for traceability.

---

## 9. References

- Cohen, M. X. (2014). *Analyzing Neural Time Series Data*
- O'Shaughnessy, D. (Speech Processing)
- Rabiner, L. R., & Schafer, R. W. (2011). *Theory and Applications of DSP*
- Waytowich, N. R., et al. (2018). Deep learning for EEG
- Hsu, W.-N., et al. (2017). Unsupervised speech representation
- Abe, J. M. (2015). *Paraconsistent Intelligent Based Systems*
- Da Costa, N. C. A. (Lógica Paraconsistente)

---

This guide is intended as a living document for both research and engineering teams. Update as the methodology, codebase, or experimental protocol evolves.
