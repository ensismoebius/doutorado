#include "ProtocolAutoencoder.hpp"

#include <memory>
#include <stdexcept>
#include <vector>

#include "nn/layers/Linear.hpp"
#include "nn/layers/ReLU.hpp"

static auto build_encoder(const AutoencoderConfig& cfg) -> Sequential
{
    if (cfg.depth < 1) throw std::invalid_argument("AutoencoderConfig::depth must be >= 1");

    Sequential enc;
    int in_size = cfg.input_features;
    for (int d = 0; d < cfg.depth; ++d)
    {
        enc.add_module(std::make_shared<Linear>(in_size, cfg.hidden_size));
        enc.add_module(std::make_shared<ReLU>());
        in_size = cfg.hidden_size;
    }
    enc.add_module(std::make_shared<Linear>(in_size, cfg.latent_size));
    enc.add_module(std::make_shared<ReLU>());
    return enc;
}

static auto build_decoder(const AutoencoderConfig& cfg) -> Sequential
{
    Sequential dec;
    int in_size = cfg.latent_size;
    for (int d = 0; d < cfg.depth; ++d)
    {
        dec.add_module(std::make_shared<Linear>(in_size, cfg.hidden_size));
        dec.add_module(std::make_shared<ReLU>());
        in_size = cfg.hidden_size;
    }
    dec.add_module(std::make_shared<Linear>(in_size, cfg.input_features));
    return dec;
}

ProtocolAutoencoder::ProtocolAutoencoder(const AutoencoderConfig& cfg)
    : encoder_(build_encoder(cfg)), decoder_(build_decoder(cfg))
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
