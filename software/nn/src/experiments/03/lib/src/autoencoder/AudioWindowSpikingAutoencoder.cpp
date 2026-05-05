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

auto AudioWindowSpikingAutoencoder::encode(const Tensor& input, bool requires_grad)
    -> Tensor
{
    return encoder_.forward(input, requires_grad);
}

auto AudioWindowSpikingAutoencoder::decode(const Tensor& latent, bool requires_grad)
    -> Tensor
{
    return decoder_.forward(latent, requires_grad);
}

auto AudioWindowSpikingAutoencoder::forward(const Tensor& input, bool requires_grad)
    -> Tensor
{
    return decode(encode(input, requires_grad), requires_grad);
}

auto AudioWindowSpikingAutoencoder::backward(const Tensor& grad_output) -> Tensor
{
    Tensor grad = decoder_.backward(grad_output);
    return encoder_.backward(grad);
}

auto AudioWindowSpikingAutoencoder::params() -> std::span<Tensor*>
{
    param_ptrs_.clear();
    auto ep = encoder_.params();
    param_ptrs_.insert(param_ptrs_.end(), ep.begin(), ep.end());
    auto dp = decoder_.params();
    param_ptrs_.insert(param_ptrs_.end(), dp.begin(), dp.end());
    return std::span<Tensor*>{param_ptrs_.data(), param_ptrs_.size()};
}

void AudioWindowSpikingAutoencoder::reset_state()
{
    experiment03::autoencoders::reset_sequential_state(encoder_);
    experiment03::autoencoders::reset_sequential_state(decoder_);
}
