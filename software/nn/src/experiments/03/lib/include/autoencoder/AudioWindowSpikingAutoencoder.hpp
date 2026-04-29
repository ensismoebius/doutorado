#ifndef EXPERIMENT03_AUDIO_WINDOW_SPIKING_AUTOENCODER_HPP
#define EXPERIMENT03_AUDIO_WINDOW_SPIKING_AUTOENCODER_HPP

#include <vector>

#include "AutoencoderConfig.hpp"
#include "nn/layers/base/Module.hpp"
#include "nn/layers/eigen/Layers.hpp"
#include "nn/tensor/Tensor.hpp"

/**
 * @file AudioWindowSpikingAutoencoder.hpp
 * @brief SNN autoencoder for audio-window features.
 *
 * Architecture:
 *   Encoder: Linear(input → hidden) → Leaky → [×depth] → Linear(hidden → latent) → Leaky
 *   Decoder: Linear(latent → hidden) → LeakyIntegrator → [×depth] → Linear(hidden → input)
 */
struct AudioWindowSpikingAutoencoder : Module<nn::Backend>
{
    Sequential encoder_;
    Sequential decoder_;

    explicit AudioWindowSpikingAutoencoder(const AutoencoderConfig& cfg);

    auto encode(const nn::Tensor& input, bool requires_grad = true) -> nn::Tensor;
    auto decode(const nn::Tensor& latent, bool requires_grad = true) -> nn::Tensor;

    auto forward(const nn::Tensor& input, bool requires_grad = true) -> nn::Tensor override;
    auto backward(const nn::Tensor& grad_output) -> nn::Tensor override;

    std::vector<nn::Tensor*> param_ptrs_;
    auto params() -> std::span<nn::Tensor*> override;

    void reset_state() override;
};

#endif // EXPERIMENT03_AUDIO_WINDOW_SPIKING_AUTOENCODER_HPP
