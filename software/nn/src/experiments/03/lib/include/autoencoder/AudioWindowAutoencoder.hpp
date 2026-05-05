#ifndef EXPERIMENT03_AUDIO_WINDOW_AUTOENCODER_HPP
#define EXPERIMENT03_AUDIO_WINDOW_AUTOENCODER_HPP

#include <vector>

#include "AutoencoderConfig.hpp"
#include "nn/layers/Layers.hpp"
#include "nn/layers/base/Module.hpp"
#include "nn/tensor/Tensor.hpp"

/**
 * @file AudioWindowAutoencoder.hpp
 * @brief ANN autoencoder for audio-window features.
 *
 * Architecture (symmetric):
 *   Encoder: Linear(input_features → hidden) → ReLU → [×depth] → Linear(hidden → latent) → ReLU
 *   Decoder: Linear(latent → hidden) → ReLU → [×depth] → Linear(hidden → input_features)
 */
struct AudioWindowAutoencoder : Module<nn::Backend>
{
    using Tensor = typename Module<nn::Backend>::Tensor;

    nn::Sequential encoder_;
    nn::Sequential decoder_;

    explicit AudioWindowAutoencoder(const AutoencoderConfig& cfg);

    auto encode(const Tensor& input, bool requires_grad = true) -> Tensor;
    auto decode(const Tensor& latent, bool requires_grad = true) -> Tensor;

    auto forward(const Tensor& input, bool requires_grad = true) -> Tensor override;
    auto backward(const Tensor& grad_output) -> Tensor override;

    // Owned concatenation of encoder + decoder parameter pointers.
    std::vector<Tensor*> param_ptrs_;
    auto params() -> std::span<Tensor*> override;
};

#endif // EXPERIMENT03_AUDIO_WINDOW_AUTOENCODER_HPP
