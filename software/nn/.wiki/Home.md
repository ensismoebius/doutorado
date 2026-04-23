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
- **Data Loaders** (`include/nn/dataLoaders/`) - EEG, audio, and multimodal datasets
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
- [K-Fold Cross-Validation](./Concepts/K-Fold-Cross-Validation.md) - Cross-validation for model selection

### Experiments
- [Experiment03](./Experiments/Experiment03.md) - Autoencoder experiments (audio, EEG, fused)
- [Experiment04](./Experiments/Experiment04.md) - SNN vs LSTM comparative with FSDD

### System Architecture
- [Architecture](./Architecture.md) - High-level system design and module interactions

### Development Guides
- [Build System](./Guides/Build-System.md) - CMake configuration and workflows
- [Grid Runbook](./Guides/Grid-Runbook.md) - Running and analyzing SNN grid tests
- [PGO](./Guides/PGO.md) - Profile-guided optimization workflow
- [Static Analysis](./Guides/Static-Analysis.md) - Code quality tools and policies
- [Naming Conventions](./Development/Naming-Conventions.md) - C++ code style guidelines

## Key Features

1. **Multiple Backend Support**: Eigen (CPU) and OpenCL (GPU) tensor backends
2. **Spiking Neural Networks**: Leaky Integrate-and-Fire neurons with surrogate gradients
3. **LSTM Autoencoders**: Sequence-to-sequence learning for time-series
4. **Multimodal Learning**: Combined EEG and audio processing pipeline
5. **Experiment Tracking**: JSON-based results logging with metrics

## Requirements

- C++20 compatible compiler
- CMake 3.16+
- Eigen 3.4+
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
/graphify explain "Leaky"
```

## License

MIT License
