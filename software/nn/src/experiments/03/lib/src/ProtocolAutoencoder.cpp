/**
 * @file src/experiments/03/lib/src/ProtocolAutoencoder.cpp
 * @brief Deterministic (ANN) autoencoder implementation for protocol inputs.
 */

#include "ProtocolAutoencoder.hpp"

#include "AutoencoderBuilders.hpp"

ProtocolAutoencoder::ProtocolAutoencoder(const AutoencoderConfig& cfg)
    : encoder_(experiment03::autoencoders::build_ann_encoder(cfg, cfg.input_features, cfg.hidden_size)),
      decoder_(experiment03::autoencoders::build_ann_decoder(cfg, cfg.input_features, cfg.hidden_size))
{
}

auto ProtocolAutoencoder::encode(const nn::Tensor& input, bool requires_grad) -> nn::Tensor
{
    return encoder_.forward(input, requires_grad);
}

auto ProtocolAutoencoder::decode(const nn::Tensor& latent, bool requires_grad) -> nn::Tensor
{
    return decoder_.forward(latent, requires_grad);
}

auto ProtocolAutoencoder::forward(const nn::Tensor& input, bool requires_grad) -> nn::Tensor
{
    return decode(encode(input, requires_grad), requires_grad);
}

auto ProtocolAutoencoder::backward(const nn::Tensor& grad_output) -> nn::Tensor
{
    nn::Tensor grad = decoder_.backward(grad_output);
    return encoder_.backward(grad);
}

auto ProtocolAutoencoder::params() -> std::vector<nn::Tensor*>
{
    auto p = encoder_.params();
    auto d = decoder_.params();
    p.insert(p.end(), d.begin(), d.end());
    return p;
}
