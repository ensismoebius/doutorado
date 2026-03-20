#ifndef EXPERIMENT03_FUSED_WINDOW_SPIKING_AUTOENCODER_HPP
#define EXPERIMENT03_FUSED_WINDOW_SPIKING_AUTOENCODER_HPP

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
    Sequential eeg_encoder_;
    Sequential audio_encoder_;
    Sequential fusion_encoder_;
    Sequential fusion_decoder_;
    Sequential eeg_decoder_;
    Sequential audio_decoder_;
    int eeg_features_ = 0;
    int audio_features_ = 0;

    explicit FusedWindowSpikingAutoencoder(const AutoencoderConfig& cfg);

    auto encode(const nn::Tensor& input, bool requires_grad = true) -> nn::Tensor;
    auto decode(const nn::Tensor& latent, bool requires_grad = true) -> nn::Tensor;

    auto forward(const nn::Tensor& input, bool requires_grad = true) -> nn::Tensor override;
    auto backward(const nn::Tensor& grad_output) -> nn::Tensor override;

    std::vector<nn::Tensor*> param_ptrs_;
    auto params() -> std::span<nn::Tensor*> override;

    void reset_state() override;
};

#endif // EXPERIMENT03_FUSED_WINDOW_SPIKING_AUTOENCODER_HPP
