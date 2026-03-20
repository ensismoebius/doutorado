/**
 * @file src/experiments/03/lib/src/EegWindowSpikingAutoencoder.cpp
 * @brief Spiking autoencoder implementation for EEG window inputs.
 */

#include "EegWindowSpikingAutoencoder.hpp"

#include <memory>
#include <stdexcept>
#include <vector>

#include "nn/layers/Leaky.hpp"
#include "nn/layers/LeakyIntegrator.hpp"
#include "nn/layers/Linear.hpp"

static auto build_snn_encoder(const AutoencoderConfig& cfg) -> Sequential
{
    if (cfg.depth < 1) throw std::invalid_argument("AutoencoderConfig::depth must be >= 1");

    Sequential enc;
    int in_size = cfg.input_features;
    for (int d = 0; d < cfg.depth; ++d)
    {
        enc.add_module(std::make_shared<Linear>(in_size, cfg.hidden_size));
        enc.add_module(std::make_shared<Leaky>(cfg.time_step, cfg.resistance, cfg.capacitance));
        in_size = cfg.hidden_size;
    }
    enc.add_module(std::make_shared<Linear>(in_size, cfg.latent_size));
    enc.add_module(std::make_shared<Leaky>(cfg.time_step, cfg.resistance, cfg.capacitance));
    return enc;
}

static auto build_snn_decoder(const AutoencoderConfig& cfg) -> Sequential
{
    Sequential dec;
    int in_size = cfg.latent_size;
    for (int d = 0; d < cfg.depth; ++d)
    {
        dec.add_module(std::make_shared<Linear>(in_size, cfg.hidden_size));
        dec.add_module(
            std::make_shared<LeakyIntegrator>(cfg.time_step, cfg.resistance, cfg.capacitance));
        in_size = cfg.hidden_size;
    }
    dec.add_module(std::make_shared<Linear>(in_size, cfg.input_features));
    dec.add_module(
        std::make_shared<LeakyIntegrator>(cfg.time_step, cfg.resistance, cfg.capacitance));
    return dec;
}

EegWindowSpikingAutoencoder::EegWindowSpikingAutoencoder(const AutoencoderConfig& cfg)
    : encoder_(build_snn_encoder(cfg)), decoder_(build_snn_decoder(cfg))
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
    for (auto& layer : encoder_.layers) layer->reset_state();
    for (auto& layer : decoder_.layers) layer->reset_state();
}
