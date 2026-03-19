#ifndef EXPERIMENT03_FUSED_WINDOW_SPIKING_AUTOENCODER_HPP
#define EXPERIMENT03_FUSED_WINDOW_SPIKING_AUTOENCODER_HPP

#include <memory>
#include <vector>

#include "AutoencoderConfig.hpp"
#include "nn/layers/Module.hpp"
#include "nn/layers/Sequential.hpp"
#include "nn/tensor/Tensor.hpp"

/**
 * @file FusedWindowSpikingAutoencoder.hpp
 * @brief SNN autoencoder for fused (EEG + audio) window features.
 *
 * The caller must concatenate EEG and audio tensors along the feature axis
 * before passing them to `forward()`.  `input_features` in `AutoencoderConfig`
 * must equal `eeg_features + audio_features`.
 *
 * Architecture:
 *   Encoder: Linear(input → hidden) → Leaky → [×depth] → Linear(hidden → latent) → Leaky
 *   Decoder: Linear(latent → hidden) → LeakyIntegrator → [×depth] → Linear(hidden → input)
 */
struct FusedWindowSpikingAutoencoder : Module
{
    Sequential encoder_;
    Sequential decoder_;

    explicit FusedWindowSpikingAutoencoder(const AutoencoderConfig& cfg);

    auto encode(const nn::Tensor& input, bool requires_grad = true) -> nn::Tensor;
    auto decode(const nn::Tensor& latent, bool requires_grad = true) -> nn::Tensor;

    auto forward(const nn::Tensor& input, bool requires_grad = true) -> nn::Tensor override;
    auto backward(const nn::Tensor& grad_output) -> nn::Tensor override;

    auto params() -> std::vector<nn::Tensor*> override;

    void reset_state() override;
};

#endif // EXPERIMENT03_FUSED_WINDOW_SPIKING_AUTOENCODER_HPP
