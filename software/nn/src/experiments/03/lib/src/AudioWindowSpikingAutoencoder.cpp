/**
 * @file src/experiments/03/lib/src/AudioWindowSpikingAutoencoder.cpp
 * @brief Spiking autoencoder implementation for audio-window inputs.
 */

#include "AudioWindowSpikingAutoencoder.hpp"

#include "AutoencoderBuilders.hpp"

AudioWindowSpikingAutoencoder::AudioWindowSpikingAutoencoder(const AutoencoderConfig& cfg)
    : encoder_(experiment03::autoencoders::build_snn_encoder(
          cfg, cfg.input_features, std::max(cfg.hidden_size, cfg.latent_size * 4))),
      decoder_(experiment03::autoencoders::build_snn_decoder(
          cfg, cfg.input_features, std::max(cfg.hidden_size, cfg.latent_size * 4)))
{
}

auto AudioWindowSpikingAutoencoder::encode(const nn::Tensor& input, bool requires_grad)
    -> nn::Tensor
{
    return encoder_.forward(input, requires_grad);
}

auto AudioWindowSpikingAutoencoder::decode(const nn::Tensor& latent, bool requires_grad)
    -> nn::Tensor
{
    return decoder_.forward(latent, requires_grad);
}

auto AudioWindowSpikingAutoencoder::forward(const nn::Tensor& input, bool requires_grad)
    -> nn::Tensor
{
    return decode(encode(input, requires_grad), requires_grad);
}

auto AudioWindowSpikingAutoencoder::backward(const nn::Tensor& grad_output) -> nn::Tensor
{
    nn::Tensor grad = decoder_.backward(grad_output);
    return encoder_.backward(grad);
}

auto AudioWindowSpikingAutoencoder::params() -> std::vector<nn::Tensor*>
{
    auto p = encoder_.params();
    auto d = decoder_.params();
    p.insert(p.end(), d.begin(), d.end());
    return p;
}

void AudioWindowSpikingAutoencoder::reset_state()
{
    experiment03::autoencoders::reset_sequential_state(encoder_);
    experiment03::autoencoders::reset_sequential_state(decoder_);
}
