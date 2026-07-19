# Models

Model implementations including autoencoders and base classes.

## Theoretical Background

### Autoencoder Architecture

An autoencoder learns to compress data into a lower-dimensional latent space and reconstruct it:

- **Encoder**: $z = f(W_e x + b_e)$ - maps input to latent
- **Decoder**: $\hat{x} = f(W_d z + b_d)$ - maps latent to reconstruction
- **Loss**: $L(x, \hat{x}) = \| x - \hat{x} \|^2$

The latent dimension $z$ controls compression. Undercomplete ($z < x$) forces learning of useful structure [8].

### Multimodal Autoencoders

For EEG + audio fusion:
- Dual encoders produce separate embeddings
- Fusion layer combines before bottleneck
- Joint reconstruction of both modalities

## How It Is Implemented Here

### Base Autoencoder

```cpp
// File: src/core/models/autoencoder/BaseAutoencoder.hpp
template <typename Backend>
class BaseAutoencoder : public Module<Backend>
{
protected:
    Module<Backend>& encoder_;
    Module<Backend>& decoder_;

public:
    auto encode(const Tensor& input) -> Tensor
    {
        return encoder_.forward(input, false);
    }

    auto decode(const Tensor& latent) -> Tensor
    {
        return decoder_.forward(latent, false);
    }

    auto forward(const Tensor& input, bool requires_grad) -> Tensor override
    {
        auto latent = encode(input);
        return decode(latent);
    }
};
```

### Autoencoder Configuration

```cpp
// File: src/core/models/autoencoder/Config.hpp
struct AutoencoderConfig
{
    // Architecture
    int input_features = 128;
    int hidden_size = 64;
    int latent_size = 32;
    int depth = 1;

    // Loss
    std::string loss_type = "mse";

    // For multimodal
    int eeg_features = 0;
    int audio_features = 0;
    int branch_hidden_size = 0;

    // SNN parameters
    float time_step = 1.0f;
    float resistance = 1.0f;
    float capacitance = 1.0f;
};
```

### Autoencoder Builders

```cpp
// File: src/core/models/autoencoder/AutoencoderBuilders.hpp
namespace builders
{
enum class AutoencoderType
{
    AudioWindow,         // Audio only
    EegWindow,         // EEG only
    FusedWindow,       // Multimodal
    AudioWindowSpiking, // Audio SNN
    EegWindowSpiking,   // EEG SNN
    FusedWindowSpiking, // Multimodal SNN
    Protocol,         // Adaptive
    ProtocolSpiking    // Adaptive SNN
};

std::unique_ptr<BaseAutoencoder<Backend>> create(
    AutoencoderType type, 
    const AutoencoderConfig& config);

inline std::unique_ptr<BaseAutoencoder<Backend>> create(
    const std::string& type_str, 
    const AutoencoderConfig& config)
{
    return create(from_string(type_str), config);
}
}
```

## Data Flow

```mermaid
flowchart LR
    subgraph Input
        x[Input<br/>batch×input_dim]
    end

    subgraph Encoder
        enc1[Linear 128→64]
        enc2[Linear 64→32]
        z[Latent<br/>batch×latent_dim]
    end

    subgraph Decoder
        dec1[Linear 32→64]
        dec2[Linear 64→128]
        xhat[Reconstruction<br/>batch×input_dim]
    end

    subgraph Loss
        loss[L2 Loss<br/>||x - xhat||²]
    end

    x --> enc1 --> enc2 --> z
    z --> dec1 --> dec2 --> xhat
    x --> loss
    xhat --> loss
```

## Usage Example

```cpp
// File: src/experiments/03/lib/src/experiment03.cpp
#include "core/models/autoencoder/AutoencoderBuilders.hpp"
#include "core/models/autoencoder/Config.hpp"

// Create audio autoencoder
nn::models::autoencoder::AutoencoderConfig config{
    .input_features = 128,
    .hidden_size = 64,
    .latent_size = 32,
    .depth = 2
};

auto model = nn::models::autoencoder::builders::create(
    "audio", config);

// Training
nn::Tensor output = model->forward(input, true);
nn::Tensor loss = MSE_loss(output, target);
model->backward(loss.grad());

// Encode new data
nn::Tensor latent = model->encode(new_input);

// Decode
nn::Tensor reconstructed = model->decode(latent);
```

## Common Pitfalls

1. **Latent Dimension**: Too small loses information; too large may overfit

2. **Overcomplete Learning**: If latent > input, network may learn identity

3. **Reconstruction Quality**: Low loss doesn't guarantee good features

4. **Modality Mismatch**: Ensure feature dimensions correct for multimodal

## See Also

- [Autoencoders](../Concepts/Autoencoders.md) - Theory
- [Tensor](./Tensor.md) - Data structure
- [Layers](./Layers.md) - Building blocks
- [Experiment03](../Experiments/Experiment03.md) - Usage

## References

[1] P. Vincent et al., "Extracting and composing robust features with denoising autoencoders," in *Proc. 25th Int. Conf. Machine Learning (ICML)*, 2008, pp. 1096–1103. [Online]. Available: https://doi.org/10.1145/1390156.1390294

[2] G. E. Hinton and R. R. Salakhutdinov, "Reducing the dimensionality of data with neural networks," *Science*, vol. 313, no. 5786, pp. 504–507, Jul. 2006. [Online]. Available: https://doi.org/10.1126/science.1127647

> In-text numbers follow the project-wide numbering in [References](../References.md). The entries cited above are reproduced here.

[8] P. Vincent, H. Larochelle, Y. Bengio, and P.-A. Manzagol, "Extracting and composing robust features with denoising autoencoders," in Proc. 25th Int. Conf. Machine Learning (ICML), 2008, pp. 1096–1103. [Online]. Available: https://doi.org/10.1145/1390156.1390294
