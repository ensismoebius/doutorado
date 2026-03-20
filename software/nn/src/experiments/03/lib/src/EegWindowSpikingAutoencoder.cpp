/**
 * @file src/experiments/03/lib/src/EegWindowSpikingAutoencoder.cpp
 * @brief Spiking autoencoder implementation for EEG window inputs.
 */

#include "EegWindowSpikingAutoencoder.hpp"

#include "AutoencoderBuilders.hpp"

EegWindowSpikingAutoencoder::EegWindowSpikingAutoencoder(const AutoencoderConfig& cfg)
    : encoder_(
          experiment03::autoencoders::build_snn_encoder(cfg, cfg.input_features, cfg.hidden_size)),
      decoder_(
          experiment03::autoencoders::build_snn_decoder(cfg, cfg.input_features, cfg.hidden_size))
{
}

auto EegWindowSpikingAutoencoder::encode(const nn::Tensor& input, bool requires_grad) -> nn::Tensor
{
    return encoder_.forward(input, requires_grad);
}

auto EegWindowSpikingAutoencoder::decode(const nn::Tensor& latent, bool requires_grad) -> nn::Tensor
{
    return decoder_.forward(latent, requires_grad);
}

auto EegWindowSpikingAutoencoder::forward(const nn::Tensor& input, bool requires_grad) -> nn::Tensor
{
    return decode(encode(input, requires_grad), requires_grad);
}

auto EegWindowSpikingAutoencoder::backward(const nn::Tensor& grad_output) -> nn::Tensor
{
    nn::Tensor grad = decoder_.backward(grad_output);
    return encoder_.backward(grad);
}

auto EegWindowSpikingAutoencoder::params() -> std::vector<nn::Tensor*>
{
    auto p = encoder_.params();
    auto d = decoder_.params();
    p.insert(p.end(), d.begin(), d.end());
    return p;
}

void EegWindowSpikingAutoencoder::reset_state()
{
    experiment03::autoencoders::reset_sequential_state(encoder_);
    experiment03::autoencoders::reset_sequential_state(decoder_);
}
