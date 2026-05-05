#ifndef EXPERIMENT03_AUDIO_WINDOW_SPIKING_AUTOENCODER_HPP
#define EXPERIMENT03_AUDIO_WINDOW_SPIKING_AUTOENCODER_HPP

#include <vector>

#include "AutoencoderConfig.hpp"
#include "nn/layers/Layers.hpp"
#include "nn/layers/base/Module.hpp"
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
    using Tensor = typename Module<nn::Backend>::Tensor;

    nn::Sequential encoder_;
    nn::Sequential decoder_;

    explicit AudioWindowSpikingAutoencoder(const AutoencoderConfig& cfg);

    auto encode(const Tensor& input, bool requires_grad = true) -> Tensor;
    auto decode(const Tensor& latent, bool requires_grad = true) -> Tensor;

    auto forward(const Tensor& input, bool requires_grad = true) -> Tensor override;
    auto backward(const Tensor& grad_output) -> Tensor override;

    std::vector<Tensor*> param_ptrs_;
    auto params() -> std::span<Tensor*> override;

    void reset_state() override;
};

#endif // EXPERIMENT03_AUDIO_WINDOW_SPIKING_AUTOENCODER_HPP
