# Neural Network Library (nn)

This is a comprehensive C++20 neural network library designed for research and experimentation with spiking neural networks (SNN), LSTM autoencoders, and multimodal learning on EEG and audio data.

## Quick Start

```cpp
// Minimal example: Create and train a simple autoencoder
#include "nn/models/autoencoder/AutoencoderBuilders.hpp"
#include "nn/training/Trainer.hpp"

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

## Table of Contents

### Core Modules
- [Tensor](./Core/Tensor.md) - Core tensor data structure
- [Layers](./Core/Layers.md) - Neural network layers (dense, convolutional, spiking)
- [Optimizers](./Core/Optimizers.md) - Optimization algorithms (Adam, SGD)
- [DataLoaders](./Core/DataLoaders.md) - Data loading and batching
- [Initializers](./Core/Initializers.md) - Weight initialization strategies
- [LinearAlgebra](./Core/LinearAlgebra.md) - Linear algebra utilities
- [Statistics](./Core/Statistics.md) - Metrics and statistical functions
- [Wave](./Core/Wave.md) - Audio signal processing and WAV loading
- [Wavelet](./Core/Wavelet.md) - Wavelet transforms
- [Paraconsistent](./Core/Paraconsistent.md) - Paraconsistent logic features
- [Saver](./Core/Saver.md) - Model serialization
- [Models](./Core/Models.md) - Autoencoder model implementations
- [Training](./Core/Training.md) - Training loop with progress bars
- [Device](./Core/Device.md) - CPU/OpenCL device abstraction
- [Logging](./Core/Logging.md) - Centralized logging system
- [Windowing](./Core/Windowing.md) - Signal windowing utilities

### Concepts
- [LSTM and BPTT](./Concepts/LSTM-and-BPTT.md) - Long Short-Term Memory and Backpropagation Through Time
- [SNN and Surrogate Gradients](./Concepts/SNN-and-Surrogate-Gradients.md) - Spiking neural networks with surrogate gradient learning
- [Autoencoders](./Concepts/Autoencoders.md) - Autoencoder architectures and training
- [Residual Blocks](./Concepts/Residual-Blocks.md) - Skip connections and ResNet blocks
- [Weight Initialisation](./Concepts/Weight-Initialisation.md) - Xavier, Kaiming initialization
- [Adam Optimiser](./Concepts/Adam-Optimiser.md) - Adaptive moment estimation
- [Data Normalisation](./Concepts/Data-Normalisation.md) - Input normalization techniques
- [K-Fold Cross-Validation](./Concepts/K-Fold-Cross-Validation.md) - Cross-validation and nested k-fold for unbiased evaluation
- [Spike Rate Regularization](./Concepts/Spike-Rate-Regularization.md) - Preventing dead/bursting neurons in SNN autoencoders
- [Spike Encoding](./Concepts/Spike-Encoding.md) - Rate coding vs latency coding and matching loss functions
- [LFCC](./Concepts/LFCC.md) - Linear Frequency Cepstral Coefficients for speaker verification
- [Imagined Speech and EEG](./Concepts/Imagined-Speech-and-EEG.md) - Neuroscience of covert speech and EEG biometrics
- [Time-Major Layout](./Concepts/Time-Major-Layout.md) - `(T*B, F)` tensor convention for SNN layers
- [Membrane Dynamics](./Concepts/Membrane-Dynamics.md) - LIF RC circuit, β = exp(−Δt/(RC)), β clamping

### Experiments
- [Experiment00](./Experiments/Experiment00.md) - Wavelet + paraconsistent baseline (Phase 0)
- [Experiment02](./Experiments/Experiment02.md) - Wavelet autoencoder pipeline
- [Experiment03](./Experiments/Experiment03.md) - Autoencoder experiments (audio, EEG, fused)
- [Experiment04](./Experiments/Experiment04.md) - SNN vs LSTM comparative with FSDD
- [Experiment05](./Experiments/Experiment05.md) - Biometric authentication of dysphonic speakers via imagined speech (thesis primary)

### Research Context
- [Research-Context](./Research-Context.md) - Thesis overview, goals, datasets, and pipeline

### Demos
Runnable examples covering each major subsystem. See [Demos/Overview](./Demos/Overview.md) for the full listing with one-line descriptions.

**C++ Demos:**
- [FFT Demo](./Demos/fft-demo.md) - FFTW3 forward/inverse FFT correctness check
- [Wavelet Demo](./Demos/wavelet-demo.md) - DWT and WPT decomposition visualisation
- [LFCC Feature Demo](./Demos/lfcc-feature-demo.md) - Batch LFCC feature extraction pipeline
- [ResNet Classifier Demo](./Demos/resnet-classifier-demo.md) - Residual MLP classifier on synthetic data
- [SNN Speaker Demo](./Demos/snn-speaker-demo.md) - End-to-end SNN speaker identification CLI
- [SNN Spike Plotter](./Demos/snn-spike-plotter.md) - Real-time ImGui/ImPlot LIF membrane visualiser
- [WPT Voice Biometrics](./Demos/wpt-voice-biometrics.md) - WPT → Poisson → residual SNN biometrics
- [Autoencoder LeakyReLU](./Demos/autoencoder-leakyrelu.md) - Spiking autoencoder BPTT validation

**Python Demos:**
- [Multimodal EEG + Audio](./Demos/multimodal-eeg-audio.md) - EEG/audio fusion with paraconsistent analysis
- [SNN Hyperparameter Search](./Demos/snn-hyperparam-search.md) - 4-stage HyperBand-style search for SNN autoencoder
- [Voice Biometrics SNN (Python)](./Demos/voice-biometrics-snn-py.md) - Python WPT → SNN speaker biometrics CLI

### Plain Language Guides
Accessible explanations without heavy math — good starting point before reading the technical pages.

**Concepts:**
- [Adam Optimiser — Plain](./Concepts/Plain/Adam-Optimiser.md)
- [Autoencoders — Plain](./Concepts/Plain/Autoencoders.md)
- [Data Normalisation — Plain](./Concepts/Plain/Data-Normalisation.md)
- [Imagined Speech and EEG — Plain](./Concepts/Plain/Imagined-Speech-and-EEG.md)
- [K-Fold Cross-Validation — Plain](./Concepts/Plain/K-Fold-Cross-Validation.md)
- [LFCC — Plain](./Concepts/Plain/LFCC.md)
- [LSTM and BPTT — Plain](./Concepts/Plain/LSTM-and-BPTT.md)
- [Residual Blocks — Plain](./Concepts/Plain/Residual-Blocks.md)
- [SNN and Surrogate Gradients — Plain](./Concepts/Plain/SNN-and-Surrogate-Gradients.md)
- [Spike Encoding — Plain](./Concepts/Plain/Spike-Encoding.md)
- [Spike Rate Regularization — Plain](./Concepts/Plain/Spike-Rate-Regularization.md)
- [Weight Initialisation — Plain](./Concepts/Plain/Weight-Initialisation.md)
- [Time-Major Layout — Plain](./Concepts/Plain/Time-Major-Layout.md)
- [Membrane Dynamics — Plain](./Concepts/Plain/Membrane-Dynamics.md)

**Core Modules:**
- [Tensor — Plain](./Core/Plain/Tensor.md)
- [Wave (Audio Processing) — Plain](./Core/Plain/Wave.md)
- [Wavelet Transform — Plain](./Core/Plain/Wavelet.md)
- [Signal Windowing — Plain](./Core/Plain/Windowing.md)
- [Paraconsistent Feature Engineering — Plain](./Core/Plain/Paraconsistent.md)
- [Statistics and Metrics — Plain](./Core/Plain/Statistics.md)

### System Architecture
- [Architecture](./Architecture.md) - High-level system design and module interactions

### Development Guides
- [Build System](./Guides/Build-System.md) - CMake configuration and workflows
- [Grid Runbook](./Guides/Grid-Runbook.md) - Running and analyzing SNN grid tests
- [PGO](./Guides/PGO.md) - Profile-guided optimization workflow
- [Static Analysis](./Guides/Static-Analysis.md) - Code quality tools and policies
- [Test Quality and Determinism](./Guides/Test-Quality-and-Determinism.md) - Deterministic and SOTA-aligned testing criteria, outcomes, and next steps
- [Naming Conventions](./Development/Naming-Conventions.md) - C++ code style guidelines
- [Dual-Agent Consensus](./Development/Dual-Agent-Consensus.md) - Claude Code + OpenCode dual-agent workflow

## Key Features

1. **Multiple Backend Support**: xtensor (CPU) and OpenCL (GPU) tensor backends
2. **Spiking Neural Networks**: Leaky Integrate-and-Fire neurons with surrogate gradients
3. **LSTM Autoencoders**: Sequence-to-sequence learning for time-series
4. **Multimodal Learning**: Combined EEG and audio processing pipeline
5. **Experiment Tracking**: JSON-based results logging with metrics

## Recent Highlights

- OpenCL tensor backend gained a tuned lhs-transposed matmul path used by
    Linear backward `dL/dW` on GPU.
- See details in [Core/Tensor](./Core/Tensor.md) and benchmark evidence in
    [results/opencl_lhs_transposed_benchmark_2026-05-02.md](../results/opencl_lhs_transposed_benchmark_2026-05-02.md).
- OpenCL SNN integration now includes Lif layer forward/backward tests running
    against `OpenCLTensorBackend`, plus a stability fix for default-constructed
    OpenCL tensor host storage.
- See [Core/Layers](./Core/Layers.md) and [Core/Tensor](./Core/Tensor.md).

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
