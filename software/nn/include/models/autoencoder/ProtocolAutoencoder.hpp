#ifndef NN_MODELS_AUTOENCODER_PROTOCOL_AUTOENCODER_HPP
#define NN_MODELS_AUTOENCODER_PROTOCOL_AUTOENCODER_HPP

#include <map>
#include <string>
#include <vector>

#include "layers/Layers.hpp"
#include "layers/base/Module.hpp"
#include "models/autoencoder/AutoencoderConfig.hpp"
#include "tensor/Tensor.hpp"

namespace nn::models::autoencoder
{

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
struct ProtocolAutoencoder : Module<nn::Backend>
{
    using Tensor = typename Module<nn::Backend>::Tensor;

    bool use_dual_branch_ = false;

    nn::Sequential encoder_;
    nn::Sequential decoder_;

    nn::Sequential eeg_encoder_;
    nn::Sequential audio_encoder_;
    nn::Sequential fusion_encoder_;
    nn::Sequential fusion_decoder_;
    nn::Sequential eeg_decoder_;
    nn::Sequential audio_decoder_;

    int eeg_features_ = 0;
    int audio_features_ = 0;

    explicit ProtocolAutoencoder(const AutoencoderConfig& cfg);

    /// Run the encoder and return the latent representation.
    auto encode(const Tensor& input, bool requires_grad = true) -> Tensor;

    /// Run the decoder and return the reconstruction.
    auto decode(const Tensor& latent, bool requires_grad = true) -> Tensor;

    auto forward(const Tensor& input, bool requires_grad = true) -> Tensor override;
    auto backward(const Tensor& grad_output) -> Tensor override;

    std::vector<Tensor*> param_ptrs_;
    auto params() -> std::span<Tensor*> override;

    /// Merge every sub-Sequential's state_dict, prefixed by its member name
    /// ("encoder.", "decoder.", "eeg_encoder.", ...). Whichever sub-Sequentials this
    /// profile did not build (e.g. eeg_encoder_ when use_dual_branch_ is false) are
    /// empty and contribute nothing — no modality-specific branching needed here.
    auto state_dict() const -> std::map<std::string, Tensor> override;
    void load_state_dict(const std::map<std::string, Tensor>& sd) override;
};
} // namespace nn::models::autoencoder

#endif // NN_MODELS_AUTOENCODER_PROTOCOL_AUTOENCODER_HPP
