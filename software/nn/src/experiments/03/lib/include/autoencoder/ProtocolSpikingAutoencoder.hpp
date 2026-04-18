#ifndef EXPERIMENT03_PROTOCOL_SPIKING_AUTOENCODER_HPP
#define EXPERIMENT03_PROTOCOL_SPIKING_AUTOENCODER_HPP

#include <vector>

#include "AutoencoderConfig.hpp"
#include "nn/layers/base/Module.hpp"
#include "nn/layers/eigen/Layers.hpp"
#include "nn/tensor/Tensor.hpp"

/**
 * @file ProtocolSpikingAutoencoder.hpp
 * @brief Spiking Neural Network (SNN) autoencoder for Protocol101117 features.
 *
 * Architecture:
 *   Encoder: Linear(input → hidden) → Leaky → [×depth] → Linear(hidden → latent) → Leaky
 *   Decoder: Linear(latent → hidden) → LeakyIntegrator → [×depth] → Linear(hidden → input)
 *
 * The encoder emits binary spike tensors; the decoder integrates spikes back to
 * continuous membrane potentials.  Call `reset_state()` between independent
 * sequences/trials to zero all membrane potentials.
 */
struct ProtocolSpikingAutoencoder : Module<nn::EigenTensorBackend>
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

    explicit ProtocolSpikingAutoencoder(const AutoencoderConfig& cfg);

    /// Run the encoder and return spike tensors for the latent layer.
    auto encode(const nn::Tensor& input, bool requires_grad = true) -> nn::Tensor;

    /// Run the decoder and return reconstructed continuous activations.
    auto decode(const nn::Tensor& latent, bool requires_grad = true) -> nn::Tensor;

    auto forward(const nn::Tensor& input, bool requires_grad = true) -> nn::Tensor override;
    auto backward(const nn::Tensor& grad_output) -> nn::Tensor override;

    std::vector<nn::Tensor*> param_ptrs_;
    auto params() -> std::span<nn::Tensor*> override;

    /// Reset all stateful (membrane potential) layers between sequences.
    void reset_state() override;
};

#endif // EXPERIMENT03_PROTOCOL_SPIKING_AUTOENCODER_HPP
