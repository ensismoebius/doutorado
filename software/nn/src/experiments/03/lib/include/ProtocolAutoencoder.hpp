#ifndef EXPERIMENT03_PROTOCOL_AUTOENCODER_HPP
#define EXPERIMENT03_PROTOCOL_AUTOENCODER_HPP

#include <vector>

#include "AutoencoderConfig.hpp"
#include "nn/layers/Module.hpp"
#include "nn/layers/Sequential.hpp"
#include "nn/tensor/Tensor.hpp"

/**
 * @file ProtocolAutoencoder.hpp
 * @brief ANN autoencoder for Protocol101117 (full-trial) features.
 *
 * Architecture (symmetric):
 *   Encoder: Linear(input_features → hidden) → ReLU → [×depth] → Linear(hidden → latent) → ReLU
 *   Decoder: Linear(latent → hidden) → ReLU → [×depth] → Linear(hidden → input_features)
 *
 * Use `encode()` and `decode()` to run each half independently.
 * `forward()` chains both halves (reconstruction).
 */
struct ProtocolAutoencoder : Module
{
    bool use_dual_branch_ = false;

    Sequential encoder_;
    Sequential decoder_;

    Sequential eeg_encoder_;
    Sequential audio_encoder_;
    Sequential fusion_encoder_;
    Sequential fusion_decoder_;
    Sequential eeg_decoder_;
    Sequential audio_decoder_;

    int eeg_features_ = 0;
    int audio_features_ = 0;

    explicit ProtocolAutoencoder(const AutoencoderConfig& cfg);

    /// Run the encoder and return the latent representation.
    auto encode(const nn::Tensor& input, bool requires_grad = true) -> nn::Tensor;

    /// Run the decoder and return the reconstruction.
    auto decode(const nn::Tensor& latent, bool requires_grad = true) -> nn::Tensor;

    auto forward(const nn::Tensor& input, bool requires_grad = true) -> nn::Tensor override;
    auto backward(const nn::Tensor& grad_output) -> nn::Tensor override;

    auto params() -> std::vector<nn::Tensor*> override;
};

#endif // EXPERIMENT03_PROTOCOL_AUTOENCODER_HPP
