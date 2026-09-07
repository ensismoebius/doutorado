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

There is no `BaseAutoencoder` class. The shared plumbing every "two
Sequentials" window autoencoder (audio/EEG, ANN/spiking) inherits is
`EncoderDecoderAutoencoder` — a subclass supplies only the built encoder/decoder
`Sequential`s. The fused and protocol autoencoders deliberately do **not**
use this base (their `encode`/`decode` do real multi-branch work of their own):

```cpp
// File: include/models/autoencoder/EncoderDecoderAutoencoder.hpp
struct EncoderDecoderAutoencoder : Module<nn::Backend>
{
    using Tensor = typename Module<nn::Backend>::Tensor;

    nn::Sequential encoder_;
    nn::Sequential decoder_;

    EncoderDecoderAutoencoder(nn::Sequential encoder, nn::Sequential decoder);

    auto encode(const Tensor& input, bool requires_grad = true) -> Tensor;
    auto decode(const Tensor& latent, bool requires_grad = true) -> Tensor;

    auto forward(const Tensor& input, bool requires_grad = true) -> Tensor override;
    auto backward(const Tensor& grad_output) -> Tensor override;
    auto params() -> std::span<Tensor*> override;
    void reset_state() override;   // no-op for ANN; clears membrane state for SNN
};
```

### Autoencoder Configuration

```cpp
// File: include/models/autoencoder/AutoencoderConfig.hpp
namespace nn::models::autoencoder
{
struct AutoencoderConfig
{
    std::string loss_type = "mse";

    int input_features = 128;
    int hidden_size = 64;
    int latent_size = 32;
    int depth = 1;
    // Declarative layer specs (encoder_layer_spec, decoder_layer_spec, ...)
    // and AutoencoderArchitecture override depth/hidden_size tapering when set.

    int eeg_features = 0;      // multimodal split hints
    int audio_features = 0;
    int branch_hidden_size = 0;
    int fusion_hidden_size = 0;

    // SNN parameters (ignored by ANN models)
    int time_steps = 0;        // 0 = UNSET; SNN builders raise rather than assume 1
    float delta_t = 1.0f;
    float resistance = 1.0f;
    float capacitance = 1.0f;
    float voltage_threshold = 1.0f;
    float firing_rate_reg_lambda = 0.0f;
};
}
```

### Autoencoder Builders

There is no `builders` namespace and no `AutoencoderType` enum/`create()`
factory pair. `include/models/autoencoder/AutoencoderBuilders.hpp` actually
declares free functions (`build_ann_encoder`, `build_ann_decoder`,
`build_snn_encoder`, `build_snn_decoder`) that assemble a `Sequential` from an
`AutoencoderConfig` — used internally by each concrete autoencoder's
constructor, not exposed as a type-selecting factory. The real per-type
dispatch lives in the autoencoderRunner experiment, selecting among the
concrete classes by an actual enum:

```cpp
// File: src/experiments/autoencoderRunner/lib/include/AutoencoderRunnerAutoencoderType.hpp
enum class AutoencoderRunnerAutoencoderType
{
    ProtocolAnn, EegWindowAnn, AudioWindowAnn, FusedWindowAnn,
    ProtocolSnn, EegWindowSnn, AudioWindowSnn, FusedWindowSnn,
};
```

```cpp
// File: src/experiments/autoencoderRunner/lib/src/autoencoderRunner_helpers.cpp
auto build_autoencoder_model(const Config& config, nn::Index input_features)
    -> std::unique_ptr<Module<nn::Backend>>
{
    AutoencoderConfig model_cfg{ /* populated from config fields */ };
    switch (config.autoencoder_type)
    {
        case AutoencoderRunnerAutoencoderType::ProtocolAnn:
            return std::make_unique<ProtocolAutoencoder>(model_cfg);
        case AutoencoderRunnerAutoencoderType::EegWindowAnn:
            return std::make_unique<EegWindowAutoencoder>(model_cfg);
        // ... AudioWindowAnn, FusedWindowAnn, and the four *Snn cases
    }
    throw std::runtime_error("Unsupported autoencoder type");
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
// File: src/experiments/autoencoderRunner/lib/src/autoencoderRunner.cpp
#include "AutoencoderRunnerAutoencoderType.hpp"
#include "models/autoencoder/AutoencoderConfig.hpp"

// autoencoderRunner's Config (JSON profile) selects the concrete class via
// config.autoencoder_type; build_autoencoder_model() does the dispatch —
// see autoencoderRunner_helpers.cpp above.
auto model = build_autoencoder_model(config, input_features);

// Training
nn::Tensor output = model->forward(input, true);
nn::Tensor grad_output = loss.backward(output);
model->backward(grad_output);

// EncoderDecoderAutoencoder-based subclasses (not Fused/Protocol) expose
// encode()/decode() directly:
// nn::Tensor latent = model->encode(new_input);
// nn::Tensor reconstructed = model->decode(latent);
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
- [AutoencoderRunner](../Experiments/AutoencoderRunner.md) - Usage

## References

[1] P. Vincent et al., "Extracting and composing robust features with denoising autoencoders," in *Proc. 25th Int. Conf. Machine Learning (ICML)*, 2008, pp. 1096–1103. [Online]. Available: https://doi.org/10.1145/1390156.1390294

[2] G. E. Hinton and R. R. Salakhutdinov, "Reducing the dimensionality of data with neural networks," *Science*, vol. 313, no. 5786, pp. 504–507, Jul. 2006. [Online]. Available: https://doi.org/10.1126/science.1127647

> In-text numbers follow the project-wide numbering in [References](../References.md). The entries cited above are reproduced here.

[8] P. Vincent, H. Larochelle, Y. Bengio, and P.-A. Manzagol, "Extracting and composing robust features with denoising autoencoders," in Proc. 25th Int. Conf. Machine Learning (ICML), 2008, pp. 1096–1103. [Online]. Available: https://doi.org/10.1145/1390156.1390294
