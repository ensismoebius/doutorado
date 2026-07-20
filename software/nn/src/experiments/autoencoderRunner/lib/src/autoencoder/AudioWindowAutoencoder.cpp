/**
 * @file src/experiments/03/lib/src/autoencoder/AudioWindowAutoencoder.cpp
 * @brief ANN implementation of the audio-window autoencoder.
 */

#include "AudioWindowAutoencoder.hpp"

#include "AutoencoderBuilders.hpp"

AudioWindowAutoencoder::AudioWindowAutoencoder(const AutoencoderConfig& cfg)
    : encoder_(experiment03::autoencoders::build_ann_encoder(
          cfg, cfg.input_features, std::max(cfg.hidden_size, cfg.latent_size * 4))),
      decoder_(experiment03::autoencoders::build_ann_decoder(
          cfg, cfg.input_features, std::max(cfg.hidden_size, cfg.latent_size * 4)))
{
}

auto AudioWindowAutoencoder::encode(const Tensor& input, bool requires_grad) -> Tensor
{
    return encoder_.forward(input, requires_grad);
}

auto AudioWindowAutoencoder::decode(const Tensor& latent, bool requires_grad) -> Tensor
{
    return decoder_.forward(latent, requires_grad);
}

auto AudioWindowAutoencoder::forward(const Tensor& input, bool requires_grad) -> Tensor
{
    return decode(encode(input, requires_grad), requires_grad);
}

auto AudioWindowAutoencoder::backward(const Tensor& grad_output) -> Tensor
{
    Tensor grad = decoder_.backward(grad_output);
    return encoder_.backward(grad);
}

auto AudioWindowAutoencoder::params() -> std::span<Tensor*>
{
    param_ptrs_.clear();
    auto ep = encoder_.params();
    param_ptrs_.insert(param_ptrs_.end(), ep.begin(), ep.end());
    auto dp = decoder_.params();
    param_ptrs_.insert(param_ptrs_.end(), dp.begin(), dp.end());
    return std::span<Tensor*>{param_ptrs_.data(), param_ptrs_.size()};
}
