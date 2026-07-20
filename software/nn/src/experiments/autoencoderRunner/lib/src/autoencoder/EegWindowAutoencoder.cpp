/**
 * @file src/experiments/autoencoderRunner/lib/src/autoencoder/EegWindowAutoencoder.cpp
 * @brief ANN implementation of the EEG window autoencoder.
 */

#include "EegWindowAutoencoder.hpp"

#include "AutoencoderBuilders.hpp"

EegWindowAutoencoder::EegWindowAutoencoder(const AutoencoderConfig& cfg)
    : encoder_(autoencoderRunner::autoencoders::build_ann_encoder(
          cfg, cfg.input_features, cfg.hidden_size)),
      decoder_(autoencoderRunner::autoencoders::build_ann_decoder(
          cfg, cfg.input_features, cfg.hidden_size))
{
}

auto EegWindowAutoencoder::encode(const Tensor& input, bool requires_grad) -> Tensor
{
    return encoder_.forward(input, requires_grad);
}

auto EegWindowAutoencoder::decode(const Tensor& latent, bool requires_grad) -> Tensor
{
    return decoder_.forward(latent, requires_grad);
}

auto EegWindowAutoencoder::forward(const Tensor& input, bool requires_grad) -> Tensor
{
    return decode(encode(input, requires_grad), requires_grad);
}

auto EegWindowAutoencoder::backward(const Tensor& grad_output) -> Tensor
{
    Tensor grad = decoder_.backward(grad_output);
    return encoder_.backward(grad);
}

auto EegWindowAutoencoder::params() -> std::span<Tensor*>
{
    param_ptrs_.clear();
    auto ep = encoder_.params();
    param_ptrs_.insert(param_ptrs_.end(), ep.begin(), ep.end());
    auto dp = decoder_.params();
    param_ptrs_.insert(param_ptrs_.end(), dp.begin(), dp.end());
    return std::span<Tensor*>{param_ptrs_.data(), param_ptrs_.size()};
}
