/**
 * @file src/experiments/03/lib/src/EegWindowAutoencoder.cpp
 * @brief ANN implementation of the EEG window autoencoder.
 */

#include "EegWindowAutoencoder.hpp"

#include "AutoencoderBuilders.hpp"

EegWindowAutoencoder::EegWindowAutoencoder(const AutoencoderConfig& cfg)
    : encoder_(
          experiment03::autoencoders::build_ann_encoder(cfg, cfg.input_features, cfg.hidden_size)),
      decoder_(
          experiment03::autoencoders::build_ann_decoder(cfg, cfg.input_features, cfg.hidden_size))
{
}

auto EegWindowAutoencoder::encode(const nn::Tensor& input, bool requires_grad) -> nn::Tensor
{
    return encoder_.forward(input, requires_grad);
}

auto EegWindowAutoencoder::decode(const nn::Tensor& latent, bool requires_grad) -> nn::Tensor
{
    return decoder_.forward(latent, requires_grad);
}

auto EegWindowAutoencoder::forward(const nn::Tensor& input, bool requires_grad) -> nn::Tensor
{
    return decode(encode(input, requires_grad), requires_grad);
}

auto EegWindowAutoencoder::backward(const nn::Tensor& grad_output) -> nn::Tensor
{
    nn::Tensor grad = decoder_.backward(grad_output);
    return encoder_.backward(grad);
}

auto EegWindowAutoencoder::params() -> std::span<nn::Tensor*>
{
    param_ptrs_.clear();
    auto ep = encoder_.params();
    param_ptrs_.insert(param_ptrs_.end(), ep.begin(), ep.end());
    auto dp = decoder_.params();
    param_ptrs_.insert(param_ptrs_.end(), dp.begin(), dp.end());
    return std::span<nn::Tensor*>{param_ptrs_.data(), param_ptrs_.size()};
}
