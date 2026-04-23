# Experiment04: SNN vs LSTM Comparative

Experiment04 implements a comparative study between Spiking Neural Networks (SNNs) and LSTM autoencoders on time-series data, with support for the Free Spoken Digit Dataset (FSDD).

## Theoretical Background

### Sequence-to-Sequence Learning

LSTM autoencoders compress variable-length sequences into fixed-size latent vectors: 

1. **Encoder LSTM**: Processes input sequence, produces final hidden state
2. **Latent Space**: Fixed-dimensional representation of entire sequence
3. **Decoder LSTM**: Reconstructs sequence from latent state

### SNN Surrogate Gradients

Spiking Neural Networks use surrogate gradient methods to approximate the non-differentiable spike function [1]:

$$\frac{\partial S}{\partial V} \approx \frac{\partial \sigma}{\partial V}$$

where $\sigma$ is a smooth approximation (e.g., fast sigmoid).

### Comparative Framework

The experiment compares:
- **LSTM Autoencoder**: Standard recurrent autoencoder with BPTT
- **SNN Autoencoder**: Layer-based spiking autoencoder with Leaky Integrate-and-Fire neurons

## Implementation

### Comparative Configuration

```cpp
// File: src/experiments/04/lib/include/ComparativeConfig.hpp
struct ComparativeConfig
{
    std::string dataset_root = "/home/ensismoebius/Documentos/UNESP/doutorado/databases/fsdDataset";
    std::string results_dir = "results";
    std::string run_tag = "snn_lstm_compare";

    std::uint32_t seed = 42;
    int repeats = 3;

    int window_size = 256;
    int batch_size = 100;
    int max_train_samples = 500;
    int max_val_samples = 100;

    int epochs = 100;
    int early_stop_patience = 20;
    float learning_rate = 1e-3f;
    float anomaly_tau = 0.25f;

    int hidden_size = 64;
    int latent_size = 32;

    // Dataset selection
    std::vector<std::string> datasets = {"fsdd"};
    
    // Encoding methods
    std::vector<std::string> encodings = {"direct", "poisson", "latency"};
    
    // SNN architectures to compare
    std::vector<std::string> snn_architectures = {"dense", "conv1d", "recurrent"};
    
    // Layer configurations
    std::vector<int> layers = {1, 2, 3};
    
    // SNN parameters
    std::vector<float> v_th_values = {0.5f, 1.0f, 1.5f};
    std::vector<float> alpha_values = {0.8f, 0.9f, 0.99f};
};
```

### Profile Configurations

The experiment uses JSON profiles in `src/experiments/04/profiles/`:

| Profile | Description | Key Parameters |
|---------|-------------|---------------|
| `lstm-default.json` | Default LSTM/SNN comparison | window=256, batch=100, epochs=100, hidden=64 |
| `lstm-compare.json` | Comprehensive grid search | 3 repeats, all encodings, layers, thresholds |
| `lstm-deep.json` | Deeper network | hidden=192, layers=2 |
| `lstm-lightweight.json` | Smoke test | window=256, batch=4, epochs=1 |

### Dataset Support

**FSDD (Free Spoken Digit Dataset)**:
- Location: `/home/ensismoebius/Documentos/UNESP/doutorado/databases/fsdDataset`
- Format: `.wav` audio files (16-bit PCM, mono, 8kHz)
- Organization: `{digit}_{speaker}_{index}.wav`
- Samples: ~3,000 recordings (50 digits × 6 speakers)

### WAV Loading

```cpp
// File: src/experiments/04/lib/src/ComparativeDataset.cpp
#include "nn/wave/Wav.h"

// Load FSDD audio files
Wav wav_file;
wav_file.read(file.string());
const auto& raw_data = wav_file.get_data();  // std::vector<double>

// Convert to Tensor
nn::Tensor signal(static_cast<nn::Index>(raw_data.size()), 1);
for (std::size_t i = 0; i < raw_data.size(); ++i)
{
    signal.at(static_cast<nn::Index>(i), 0) = static_cast<float>(raw_data[i]);
}
```

### Progress Tracking

Real-time progress bars during training using `nn::utility::printProgress`:

```cpp
// Inside ComparativeTraining.cpp
printProgress(train_samples.size(),
    1,
    train_samples.size() * cfg.epochs,
    epoch * train_samples.size() + train_samples.size(),
    epoch * train_samples.size() + train_samples.size(),
    false,
    run_id,
    total_runs,
    epoch + 1,
    cfg.epochs,
    train_samples.size(),
    train_samples.size(),
    static_cast<double>(val_mse),
    std::span<nn::Tensor*>{},
    "LSTM");
```

Output (multiline ANSI):
```
LSTM Fold:  [===================>                  ]  50%
Epoch: [===================>                  ]  50% (50/100)
Batch: [========================================] 100% (500/500b, 500/500s)  loss: 1.219896
```

## Usage

```bash
# Default run with FSDD
./experiment04 --comparative --comparative-config lstm-default

# With explicit dataset path
./experiment04 --comparative --comparative-config lstm-default --dataset-root /path/to/fsdDataset

# Lightweight smoke test
./experiment04 --comparative --comparative-config lstm-lightweight
```

### Outputs

Results are written to `results/`:
- `{run_tag}_comparative_metrics.csv` - Full metrics per configuration
- `{run_tag}_publication_table.csv` - Formatted for publication
- `{run_tag}_summary.json` - JSON summary

## Key Differences from Experiment03

| Feature | Experiment03 | Experiment04 |
|---------|-------------|--------------|
| Model | Feedforward AE | LSTM + SNN comparative |
| Input | Fixed-dim vectors | Variable-length sequences |
| Latent | Vector | Final hidden state |
| BPTT | Not used | LSTM uses BPTT |
| Dataset | 10.1117 EEG/Audio | FSDD (spoken digits) |
| Progress | Legacy async | nn::progress (ANSI) |

## Common Pitfalls

1. **Sequence Truncation**: Very long sequences may need truncation

2. **Hidden State Reset**: Ensure proper initialization between sequences

3. **SNN Threshold**: $V_{th}$ affects spiking behavior significantly

4. **FSDD Path**: Profiles must point to the correct dataset location

## See Also

- [LSTM and BPTT](../Concepts/LSTM-and-BPTT.md) - Theory
- [SNN and Surrogate Gradients](../Concepts/SNN-and-Surrogate-Gradients.md)
- [Autoencoders](../Concepts/Autoencoders.md)
- [Wave Processing](../Core/Wave.md)
- [Training](./Core/Training.md) - Progress bars
- [Experiment03](../Experiments/Experiment03.md) - Feedforward autoencoder

## References

[1] S. Hochreiter and J. Schmidhuber, "Long short-term memory," *Neural Computation*, vol. 9, no. 8, pp. 1735–1780, Nov. 1997. [Online]. Available: https://doi.org/10.1162/neco.1997.9.8.1735

[2] A. Graves, "Generating sequences with recurrent neural networks," arXiv preprint arXiv:1308.0850, 2013. [Online]. Available: https://arxiv.org/abs/1308.0850

[3] FSDD Dataset: https://github.com/Jakobovski/Free-Spoken-Digits-Dataset
