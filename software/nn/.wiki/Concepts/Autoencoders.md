# Autoencoders

Autoencoders are neural networks trained to reconstruct their inputs, learning compressed representations (latent codes) in the process.

## Theoretical Background

### Basic Autoencoder

An autoencoder consists of:
- **Encoder**: $z = f(W_e x + b_e)$ - compresses input to latent space
- **Decoder**: $\hat{x} = f(W_d z + b_d)$ - reconstructs from latent

Training minimizes reconstruction loss:
$$L(x, \hat{x}) = \| x - \hat{x} \|^2$$

### Denoising Autoencoder

DAEs corrupt input with noise and learn to reconstruct clean version [8]:
$$\hat{x} = \text{Decoder}(Encoder(\tilde{x}))$$
where $\tilde{x} = x + \text{noise}$

This forces the encoder to learn robust features.

### Variational Autoencoder (VAE)

VAEs learn a probability distribution over latents [9]:
$$q(z|x) = \mathcal{N}(\mu(x), \sigma(x))$$
$$L = \text{Reconstruction} + \text{KL}(q(z|x) || p(z))$$

### Latent Space

The latent dimension controls compression:
- **Overcomplete**: $z > x$ - can learn identity but not useful
- **Undercomplete**: $z < x$ - forced to learn structure (desired)
- **Sparse**: many $z_i \approx 0$ - promotes disentangled representations

## How It Is Implemented Here

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

### Configuration

```cpp
// File: src/core/models/autoencoder/Config.hpp
struct AutoencoderConfig
{
    int input_features = 128;
    int hidden_size = 64;
    int latent_size = 32;
    int depth = 1;
    std::string loss_type = "mse";

    // SNN parameters
    float time_step = 1.0f;
    float resistance = 1.0f;
    float capacitance = 1.0f;
};
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
#include "nn/models/autoencoder/AutoencoderBuilders.hpp"

// Create audio autoencoder
nn::models::autoencoder::AutoencoderConfig cfg{
    .input_features = 128,   // MFCC features
    .hidden_size = 64,
    .latent_size = 32,
    .depth = 2
};

auto model = nn::models::autoencoder::builders::create(
    "audio", cfg);

// Train
nn::training::Trainer trainer(*model, trainerConfig);
auto history = trainer.fit_autoencoder(training_data, validation_data);

// Encode new data
Tensor latent = model->encode(new_input);

// Reconstruct
Tensor reconstructed = model->decode(latent);
```

## Common Pitfalls

1. **Latent Dimension**: Too small loses information; too large may overfit

2. **Overcomplete Learning**: If latent > input, network may learn identity

3. **Reconstruction Quality**: Low loss doesn't guarantee good features; visualize latent space

4. **Modality Mismatch**: For multimodal (EEG+Audio), ensure feature dimensions are correct

## See Also

- [Experiment03](../Experiments/Experiment03.md) - Autoencoder experiments
- [Experiment04](../Experiments/Experiment04.md) - LSTM autoencoder
- [Weight Initialisation](./Weight-Initialisation.md) - Important for training

## References

[1] P. Vincent, H. Larochelle, Y. Bengio, and P.-A. Manzagol, "Extracting and composing robust features with denoising autoencoders," in Proc. 25th Int. Conf. Machine Learning (ICML), 2008, pp. 1096–1103.

[2] P. Vincent, H. Larochelle, I. Guyon, and Y. Bengio, "Stacked denoising autoencoders: Learning useful representations in a deep network with a local denoising criterion," J. Mach. Learn. Res., vol. 11, pp. 3371–3408, 2010.
