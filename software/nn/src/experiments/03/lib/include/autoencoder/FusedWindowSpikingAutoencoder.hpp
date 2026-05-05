#ifndef EXPERIMENT03_FUSED_WINDOW_SPIKING_AUTOENCODER_HPP
#define EXPERIMENT03_FUSED_WINDOW_SPIKING_AUTOENCODER_HPP

#include <vector>

#include "AutoencoderConfig.hpp"
#include "nn/layers/Layers.hpp"
#include "nn/layers/base/Module.hpp"
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
struct FusedWindowSpikingAutoencoder : Module<nn::Backend>
{
    using Tensor = typename Module<nn::Backend>::Tensor;

    nn::Sequential eeg_encoder_;
    nn::Sequential audio_encoder_;
    nn::Sequential fusion_encoder_;
    nn::Sequential fusion_decoder_;
    nn::Sequential eeg_decoder_;
    nn::Sequential audio_decoder_;
    int eeg_features_ = 0;
    int audio_features_ = 0;

    explicit FusedWindowSpikingAutoencoder(const AutoencoderConfig& cfg);

    auto encode(const Tensor& input, bool requires_grad = true) -> Tensor;
    auto decode(const Tensor& latent, bool requires_grad = true) -> Tensor;

    auto forward(const Tensor& input, bool requires_grad = true) -> Tensor override;
    auto backward(const Tensor& grad_output) -> Tensor override;

    std::vector<Tensor*> param_ptrs_;
    auto params() -> std::span<Tensor*> override;

    void reset_state() override;
};

#endif // EXPERIMENT03_FUSED_WINDOW_SPIKING_AUTOENCODER_HPP
