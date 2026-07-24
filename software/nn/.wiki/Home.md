# Neural Network Library (nn)

This is a comprehensive C++20 neural network library designed for research and experimentation with spiking neural networks (SNN), LSTM autoencoders, and multimodal learning on EEG and audio data.

## Quick Start

```cpp
// Minimal example: Create and train a simple autoencoder.
// Include prefix is "core/...", not "nn/..." -- there is no include/nn/ directory.
#include "core/models/autoencoder/AutoencoderBuilders.hpp"
#include "core/training/Trainer.hpp"

nn::models::autoencoder::AutoencoderConfig config{
    .input_features = 128,
    .hidden_size = 64,
    .latent_size = 32,
    .depth = 2
};

auto model = nn::models::autoencoder::builders::create("audio", config);
nn::training::TrainerConfig trainerCfg{
    .epochs = 10,
    .batch_size = 32,
    .learning_rate = 0.001F
};
nn::training::Trainer trainer(*model, trainerCfg);
auto history = trainer.fit_autoencoder(training_data, validation_data);
```

## Project Overview

The `nn` library is organized into several key components:

- **Core Library** (`src/core/`) - Tensor operations, layers, optimizers
- **Data Loaders** (`include/data_loaders/`) - EEG, audio, and multimodal datasets
- **Experiments** (`src/experiments/`) - Research experiments with trained models

## New here? Start with the tutorials

These are step-by-step and every command in them has been run as written:

1. **[Getting Started](./Tutorials/Getting-Started.md)** — build the library, run a test, run a
   real experiment. ~20 minutes, mostly waiting on the compiler.
2. **[Adding a Layer](./Tutorials/Adding-a-Layer.md)** — the layer contract, a worked example
   with a gradient check, and the mistakes that catch everyone.

Then follow one of these reading paths depending on what you came for:

| I want to… | Read, in order |
|---|---|
| **Understand the research** | [Research Context](./Research-Context.md) → [Imagined Speech and EEG](./Concepts/Plain/Imagined-Speech-and-EEG.md) → [Experiment05](./Experiments/Thesis.md) |
| **Understand spiking networks** | [SNN and Surrogate Gradients (plain)](./Concepts/Plain/SNN-and-Surrogate-Gradients.md) → [Membrane Dynamics](./Concepts/Membrane-Dynamics.md) → [Time-Major Layout](./Concepts/Time-Major-Layout.md) |
| **Work on the framework** | [Getting Started](./Tutorials/Getting-Started.md) → [Architecture](./Architecture.md) → [Tensor](./Core/Tensor.md) → [Layers](./Core/Layers.md) |
| **Re-run the experiments** | [Re-run Runbook](./Guides/Re-run-Runbook.md) → [Running Experiment05 Profiles](./Guides/Running-Thesis-Profiles.md) |
| **Know why the code is like this** | [Engineering Fixes Log](./Guides/Engineering-Fixes-Log.md) |

> **"Plain" pages exist for most topics.** They explain the same idea without heavy maths and
> are the better starting point. They are linked in the *Plain* column below — you do not have
> to hunt for them.

---

## Table of Contents

### Concepts — the ideas behind the code

| Topic | Technical | Plain language |
|---|---|---|
| Spiking networks & surrogate gradients | [SNN and Surrogate Gradients](./Concepts/SNN-and-Surrogate-Gradients.md) | [plain](./Concepts/Plain/SNN-and-Surrogate-Gradients.md) |
| LIF membrane dynamics | [Membrane Dynamics](./Concepts/Membrane-Dynamics.md) | [plain](./Concepts/Plain/Membrane-Dynamics.md) |
| Spike encoding (rate vs latency) | [Spike Encoding](./Concepts/Spike-Encoding.md) | [plain](./Concepts/Plain/Spike-Encoding.md) |
| Spike-rate regularization | [Spike Rate Regularization](./Concepts/Spike-Rate-Regularization.md) | [plain](./Concepts/Plain/Spike-Rate-Regularization.md) |
| tdBN for deep SNNs | [Threshold-Dependent Batch Norm](./Concepts/Threshold-Dependent-Batch-Normalization.md) | — |
| `(T*B, F)` tensor convention | [Time-Major Layout](./Concepts/Time-Major-Layout.md) | [plain](./Concepts/Plain/Time-Major-Layout.md) |
| LSTM and BPTT | [LSTM and BPTT](./Concepts/LSTM-and-BPTT.md) | [plain](./Concepts/Plain/LSTM-and-BPTT.md) |
| Autoencoders | [Autoencoders](./Concepts/Autoencoders.md) | [plain](./Concepts/Plain/Autoencoders.md) |
| Residual blocks / skip connections | [Residual Blocks](./Concepts/Residual-Blocks.md) | [plain](./Concepts/Plain/Residual-Blocks.md) |
| Weight initialisation | [Weight Initialisation](./Concepts/Weight-Initialisation.md) | [plain](./Concepts/Plain/Weight-Initialisation.md) |
| Adam optimiser | [Adam Optimiser](./Concepts/Adam-Optimiser.md) | [plain](./Concepts/Plain/Adam-Optimiser.md) |
| Input normalisation | [Data Normalisation](./Concepts/Data-Normalisation.md) | [plain](./Concepts/Plain/Data-Normalisation.md) |
| Cross-validation (incl. nested) | [K-Fold Cross-Validation](./Concepts/K-Fold-Cross-Validation.md) | [plain](./Concepts/Plain/K-Fold-Cross-Validation.md) |
| LFCC speaker features | [LFCC](./Concepts/LFCC.md) | [plain](./Concepts/Plain/LFCC.md) |
| Imagined speech & EEG biometrics | [Imagined Speech and EEG](./Concepts/Imagined-Speech-and-EEG.md) | [plain](./Concepts/Plain/Imagined-Speech-and-EEG.md) |
| `time_steps` vs `delta_t` | [What time_steps Really Means](./Concepts/Time-Steps.md) | — |
| Multi-objective search (NSGA-II) | [Multi-Objective Optimisation](./Concepts/Multi-Objective-Optimisation.md) | — |

### Core modules — the code itself

| Module | Technical | Plain language |
|---|---|---|
| Tensor (start here) | [Tensor](./Core/Tensor.md) | [plain](./Core/Plain/Tensor.md) |
| Layers | [Layers](./Core/Layers.md) | — |
| Optimizers | [Optimizers](./Core/Optimizers.md) | — |
| Training loop | [Training](./Core/Training.md) | — |
| Models (autoencoders) | [Models](./Core/Models.md) | — |
| Data loaders | [DataLoaders](./Core/DataLoaders.md) | — |
| Initializers | [Initializers](./Core/Initializers.md) | — |
| Paraconsistent features | [Paraconsistent](./Core/Paraconsistent.md) | [plain](./Core/Plain/Paraconsistent.md) |
| Wavelet transforms | [Wavelet](./Core/Wavelet.md) | [plain](./Core/Plain/Wavelet.md) |
| Audio / WAV | [Wave](./Core/Wave.md) | [plain](./Core/Plain/Wave.md) |
| Signal windowing | [Windowing](./Core/Windowing.md) | [plain](./Core/Plain/Windowing.md) |
| Statistics & metrics | [Statistics](./Core/Statistics.md) | [plain](./Core/Plain/Statistics.md) |
| Linear algebra | [LinearAlgebra](./Core/LinearAlgebra.md) | — |
| Serialization | [Saver](./Core/Saver.md) | — |
| Device (CPU/OpenCL) | [Device](./Core/Device.md) | — |
| Logging | [Logging](./Core/Logging.md) | — |

### Experiments

Directories, build targets and classes are named after **what each experiment is**, not a
number. The old numeric names still appear inside stored result files (run tags like
`e05_p00_...`), so this is the mapping:

| Name | Was | What it is |
|---|---|---|
| `paraconsistentBaseline` | 00 | Frozen wavelet + paraconsistent baseline |
| `waveletAE` | 02 | Wavelet autoencoder pipeline |
| `autoencoderRunner` | 03 | Autoencoder training runner |
| `guayaquil` | 04 | SNN vs LSTM comparative — the conference paper |
| `thesis` | 05 | **Thesis primary** — biometric authentication via imagined speech |
| `paraconsistentGA` | — | NSGA-II autoencoder architecture search ranked by `d_penalized` (extends thesis phase00) |

Run in order; `thesis` is the primary experiment.

- [ParaconsistentBaseline](./Experiments/ParaconsistentBaseline.md) — wavelet + paraconsistent baseline
- [WaveletAE](./Experiments/WaveletAE.md) — wavelet autoencoder pipeline
- [AutoencoderRunner](./Experiments/AutoencoderRunner.md) — autoencoders (audio, EEG, fused)
- [Guayaquil](./Experiments/Guayaquil.md) — SNN vs LSTM comparative (conference paper)
- [Thesis](./Experiments/Thesis.md) — **thesis primary**: biometric authentication
  of dysphonic speakers via imagined speech
- [paraconsistentGA](./Experiments/ParaconsistentGA.md) — NSGA-II search over autoencoder
  architectures, ranked by paraconsistent `d_penalized` under a latency constraint
  ([design spec](./Experiments/ParaconsistentGA-Design.md))

### Research context

- [Research Context](./Research-Context.md) — thesis overview, goals, datasets, pipeline
- [Notebooks](./Notebooks.md) — Python/Jupyter prototyping

### Guides

**Building and testing**
- [Build System](./Guides/Build-System.md) — CMake presets, and the two `ctest` gotchas that
  silently make test commands match nothing
- [Ground-Truth and Smoke Testing](./Guides/Ground-Truth-and-Smoke-Testing.md) — PyTorch/snnTorch parity
- [Test Quality and Determinism](./Guides/Test-Quality-and-Determinism.md)
- [Static Analysis](./Guides/Static-Analysis.md)

**Running experiments**
- [Re-run Runbook](./Guides/Re-run-Runbook.md) — regenerate every result, in dependency order
- [Running Experiment05 Profiles](./Guides/Running-Thesis-Profiles.md) — the Thesis runner
- [Grid Runbook](./Guides/Grid-Runbook.md) — SNN grid tests

**Performance and debugging**
- [OpenCL Debugging and Performance](./Guides/OpenCL-Debugging-And-Performance.md) — **read
  before touching the OpenCL backend** (contains a memory-corruption hazard warning)
- [Memory Diagnostics](./Guides/Memory-Diagnostics.md) — leak vs. bounded high-water-mark
- [PGO](./Guides/PGO.md) — profile-guided optimization

**Project history and conventions**
- [Engineering Fixes Log](./Guides/Engineering-Fixes-Log.md) — the D1–D6 decision log
- [Naming Conventions](./Development/Naming-Conventions.md)
- [Dual-Agent Consensus](./Development/Dual-Agent-Consensus.md)

### Demos

Runnable examples per subsystem — see [Demos/Overview](./Demos/Overview.md) for the full list.

**C++:** [FFT](./Demos/fft-demo.md) · [Wavelet](./Demos/wavelet-demo.md) ·
[LFCC](./Demos/lfcc-feature-demo.md) · [ResNet classifier](./Demos/resnet-classifier-demo.md) ·
[SNN speaker](./Demos/snn-speaker-demo.md) · [Spike plotter](./Demos/snn-spike-plotter.md) ·
[WPT biometrics](./Demos/wpt-voice-biometrics.md) ·
[Autoencoder LeakyReLU](./Demos/autoencoder-leakyrelu.md)

**Python:** [Multimodal EEG+Audio](./Demos/multimodal-eeg-audio.md) ·
[SNN hyperparameter search](./Demos/snn-hyperparam-search.md) ·
[Voice biometrics](./Demos/voice-biometrics-snn-py.md)

### System architecture

- [Architecture](./Architecture.md) — high-level design and module interactions

## Key Features

1. **Multiple Backend Support**: xtensor (CPU) and OpenCL (GPU) tensor backends
2. **Spiking Neural Networks**: Leaky Integrate-and-Fire neurons with surrogate gradients
3. **LSTM Autoencoders**: Sequence-to-sequence learning for time-series
4. **Multimodal Learning**: Combined EEG and audio processing pipeline
5. **Experiment Tracking**: JSON-based results logging with metrics

## Recent Highlights

- OpenCL tensor backend gained a tuned lhs-transposed matmul path used by
    Linear backward `dL/dW` on GPU.
- See details and benchmark numbers in [Core/Tensor](./Core/Tensor.md#recent-opencl-optimization-2026-05-02).
- OpenCL SNN integration now includes Lif layer forward/backward tests running
    against `OpenCLTensorBackend`, plus a stability fix for default-constructed
    OpenCL tensor host storage.
- See [Core/Layers](./Core/Layers.md) and [Core/Tensor](./Core/Tensor.md).
- `GPUBufferPool` gained a 1 GiB global cache ceiling (previously only
    capped per-bucket, letting cached pinned buffers accumulate unbounded over
    a long run). `run_thesis_profiles.sh`'s per-job memory budget was also bumped
    2048MB → 5120MB after measuring real `thesis` phase00 peaks
    (~4.4GB for voice, ~2.1GB for EEG) — the old default let 4 heavy jobs
    oversubscribe a 17GB box into swap thrashing.
- See [Core/Tensor](./Core/Tensor.md) and
    [Guides/Running Experiment05 Profiles](./Guides/Running-Thesis-Profiles.md).

## Requirements

- C++20 compatible compiler
- CMake 3.16+
- xtensor
- OpenCL (optional, for GPU acceleration)

## Building

```bash
cmake --preset=max-performance
cmake --build --preset=max-performance -j$(nproc)
ctest --test-dir out/build/max-performance --output-on-failure -j4
```

## Knowledge Graph

The wiki includes an integrated knowledge graph powered by graphify:

- **[graphify-out/graph.html](./graphify-out/graph.html)** - Interactive visualization
- **[graphify-out/GRAPH_REPORT.md](./graphify-out/GRAPH_REPORT.md)** - God nodes, communities, surprising connections

### Query the Graph

Start MCP server for agent access:
```bash
python -m graphify.serve .wiki/graphify-out/graph.json
```

Query from agent:
```
/graphify query "tensor operations"
/graphify path "Tensor" "Optimizer"
/graphify explain "Lif"
```

## License

MIT License
