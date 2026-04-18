#ifndef EXPERIMENT03_EEG_WINDOW_SPIKING_AUTOENCODER_HPP
#define EXPERIMENT03_EEG_WINDOW_SPIKING_AUTOENCODER_HPP

#include <vector>

#include "AutoencoderConfig.hpp"
#include "nn/layers/base/Module.hpp"
#include "nn/layers/eigen/Layers.hpp"
#include "nn/tensor/Tensor.hpp"

/**
 * @file EegWindowSpikingAutoencoder.hpp
 * @brief SNN autoencoder for EEG-window features.
 *
 * Architecture:
 *   Encoder: Linear(input → hidden) → Leaky → [×depth] → Linear(hidden → latent) → Leaky
 *   Decoder: Linear(latent → hidden) → LeakyIntegrator → [×depth] → Linear(hidden → input)
 */
struct EegWindowSpikingAutoencoder : Module<nn::EigenTensorBackend>
{
    Sequential encoder_;
    Sequential decoder_;

    explicit EegWindowSpikingAutoencoder(const AutoencoderConfig& cfg);

    auto encode(const nn::Tensor& input, bool requires_grad = true) -> nn::Tensor;
    auto decode(const nn::Tensor& latent, bool requires_grad = true) -> nn::Tensor;

    auto forward(const nn::Tensor& input, bool requires_grad = true) -> nn::Tensor override;
    auto backward(const nn::Tensor& grad_output) -> nn::Tensor override;

    std::vector<nn::Tensor*> param_ptrs_;
    auto params() -> std::span<nn::Tensor*> override;

    void reset_state() override;
};

#endif // EXPERIMENT03_EEG_WINDOW_SPIKING_AUTOENCODER_HPP
