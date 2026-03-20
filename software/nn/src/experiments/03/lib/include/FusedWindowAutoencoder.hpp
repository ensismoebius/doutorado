#ifndef EXPERIMENT03_FUSED_WINDOW_AUTOENCODER_HPP
#define EXPERIMENT03_FUSED_WINDOW_AUTOENCODER_HPP

#include <vector>

#include "AutoencoderConfig.hpp"
#include "nn/layers/Module.hpp"
#include "nn/layers/Sequential.hpp"
#include "nn/tensor/Tensor.hpp"

/**
 * @file FusedWindowAutoencoder.hpp
 * @brief ANN autoencoder for fused (EEG + audio) window features.
 *
 * The caller is responsible for concatenating EEG and audio tensors along the
 * feature axis before passing them in.  `input_features` in `AutoencoderConfig`
 * must therefore equal `eeg_features + audio_features`.
 *
 * Architecture (symmetric):
 *   Encoder: Linear(input_features → hidden) → ReLU → [×depth] → Linear(hidden → latent) → ReLU
 *   Decoder: Linear(latent → hidden) → ReLU → [×depth] → Linear(hidden → input_features)
 */
struct FusedWindowAutoencoder : Module
{
    Sequential eeg_encoder_;
    Sequential audio_encoder_;
    Sequential fusion_encoder_;
    Sequential fusion_decoder_;
    Sequential eeg_decoder_;
    Sequential audio_decoder_;
    int eeg_features_ = 0;
    int audio_features_ = 0;

    explicit FusedWindowAutoencoder(const AutoencoderConfig& cfg);

    auto encode(const nn::Tensor& input, bool requires_grad = true) -> nn::Tensor;
    auto decode(const nn::Tensor& latent, bool requires_grad = true) -> nn::Tensor;

    auto forward(const nn::Tensor& input, bool requires_grad = true) -> nn::Tensor override;
    auto backward(const nn::Tensor& grad_output) -> nn::Tensor override;

    auto params() -> std::vector<nn::Tensor*> override;
};

#endif // EXPERIMENT03_FUSED_WINDOW_AUTOENCODER_HPP
