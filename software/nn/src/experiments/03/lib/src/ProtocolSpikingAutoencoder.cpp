/**
 * @file src/experiments/03/lib/src/ProtocolSpikingAutoencoder.cpp
 * @brief Spiking autoencoder implementation for protocol-formatted inputs.
 *
 * Builds SNN encoder and decoder stacks using `Leaky` and
 * `LeakyIntegrator` layers and exposes the `ProtocolSpikingAutoencoder`
 * wrapper that conforms to the project's `Module` interface.
 */

#include "ProtocolSpikingAutoencoder.hpp"

#include "AutoencoderBuilders.hpp"

ProtocolSpikingAutoencoder::ProtocolSpikingAutoencoder(const AutoencoderConfig& cfg)
    : encoder_(experiment03::autoencoders::build_snn_encoder(cfg, cfg.input_features, cfg.hidden_size)),
      decoder_(experiment03::autoencoders::build_snn_decoder(cfg, cfg.input_features, cfg.hidden_size))
{
}

auto ProtocolSpikingAutoencoder::encode(const nn::Tensor& input, bool requires_grad) -> nn::Tensor
{
    return encoder_.forward(input, requires_grad);
}

auto ProtocolSpikingAutoencoder::decode(const nn::Tensor& latent, bool requires_grad) -> nn::Tensor
{
    return decoder_.forward(latent, requires_grad);
}

auto ProtocolSpikingAutoencoder::forward(const nn::Tensor& input, bool requires_grad) -> nn::Tensor
{
    return decode(encode(input, requires_grad), requires_grad);
}

auto ProtocolSpikingAutoencoder::backward(const nn::Tensor& grad_output) -> nn::Tensor
{
    nn::Tensor grad = decoder_.backward(grad_output);
    return encoder_.backward(grad);
}

auto ProtocolSpikingAutoencoder::params() -> std::vector<nn::Tensor*>
{
    auto p = encoder_.params();
    auto d = decoder_.params();
    p.insert(p.end(), d.begin(), d.end());
    return p;
}

void ProtocolSpikingAutoencoder::reset_state()
{
    experiment03::autoencoders::reset_sequential_state(encoder_);
    experiment03::autoencoders::reset_sequential_state(decoder_);
}
