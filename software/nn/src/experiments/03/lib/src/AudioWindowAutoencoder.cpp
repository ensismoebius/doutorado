/**
 * @file src/experiments/03/lib/src/AudioWindowAutoencoder.cpp
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

auto AudioWindowAutoencoder::encode(const nn::Tensor& input, bool requires_grad) -> nn::Tensor
{
    return encoder_.forward(input, requires_grad);
}

auto AudioWindowAutoencoder::decode(const nn::Tensor& latent, bool requires_grad) -> nn::Tensor
{
    return decoder_.forward(latent, requires_grad);
}

auto AudioWindowAutoencoder::forward(const nn::Tensor& input, bool requires_grad) -> nn::Tensor
{
    return decode(encode(input, requires_grad), requires_grad);
}

auto AudioWindowAutoencoder::backward(const nn::Tensor& grad_output) -> nn::Tensor
{
    nn::Tensor grad = decoder_.backward(grad_output);
    return encoder_.backward(grad);
}

auto AudioWindowAutoencoder::params() -> std::vector<nn::Tensor*>
{
    auto p = encoder_.params();
    auto d = decoder_.params();
    p.insert(p.end(), d.begin(), d.end());
    return p;
}
