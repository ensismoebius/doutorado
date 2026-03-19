#ifndef EXPERIMENT03_PROTOCOL_SPIKING_AUTOENCODER_HPP
#define EXPERIMENT03_PROTOCOL_SPIKING_AUTOENCODER_HPP

#include <memory>
#include <vector>

#include "AutoencoderConfig.hpp"
#include "nn/layers/Module.hpp"
#include "nn/layers/Sequential.hpp"
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
struct ProtocolSpikingAutoencoder : Module
{
    Sequential encoder_;
    Sequential decoder_;

    explicit ProtocolSpikingAutoencoder(const AutoencoderConfig& cfg);

    /// Run the encoder and return spike tensors for the latent layer.
    auto encode(const nn::Tensor& input, bool requires_grad = true) -> nn::Tensor;

    /// Run the decoder and return reconstructed continuous activations.
    auto decode(const nn::Tensor& latent, bool requires_grad = true) -> nn::Tensor;

    auto forward(const nn::Tensor& input, bool requires_grad = true) -> nn::Tensor override;
    auto backward(const nn::Tensor& grad_output) -> nn::Tensor override;

    auto params() -> std::vector<nn::Tensor*> override;

    /// Reset all stateful (membrane potential) layers between sequences.
    void reset_state() override;
};

#endif // EXPERIMENT03_PROTOCOL_SPIKING_AUTOENCODER_HPP
