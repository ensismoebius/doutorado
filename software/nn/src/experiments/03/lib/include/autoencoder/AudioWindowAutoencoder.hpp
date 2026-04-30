#ifndef EXPERIMENT03_AUDIO_WINDOW_AUTOENCODER_HPP
#define EXPERIMENT03_AUDIO_WINDOW_AUTOENCODER_HPP

#include <vector>

#include "AutoencoderConfig.hpp"
#include "nn/layers/base/Module.hpp"
#include "nn/layers/Layers.hpp"
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
    Sequential encoder_;
    Sequential decoder_;

    explicit AudioWindowAutoencoder(const AutoencoderConfig& cfg);

    auto encode(const nn::Tensor& input, bool requires_grad = true) -> nn::Tensor;
    auto decode(const nn::Tensor& latent, bool requires_grad = true) -> nn::Tensor;

    auto forward(const nn::Tensor& input, bool requires_grad = true) -> nn::Tensor override;
    auto backward(const nn::Tensor& grad_output) -> nn::Tensor override;

    // Owned concatenation of encoder + decoder parameter pointers.
    std::vector<nn::Tensor*> param_ptrs_;
    auto params() -> std::span<nn::Tensor*> override;
};

#endif // EXPERIMENT03_AUDIO_WINDOW_AUTOENCODER_HPP
