#ifndef EXPERIMENT03_EEG_WINDOW_AUTOENCODER_HPP
#define EXPERIMENT03_EEG_WINDOW_AUTOENCODER_HPP

#include <vector>

#include "AutoencoderConfig.hpp"
#include "layers/Layers.hpp"
#include "layers/base/Module.hpp"
#include "tensor/Tensor.hpp"

/**
 * @file EegWindowAutoencoder.hpp
 * @brief ANN autoencoder for EEG-window features.
 *
 * Architecture (symmetric):
 *   Encoder: Linear(input_features → hidden) → ReLU → [×depth] → Linear(hidden → latent) → ReLU
 *   Decoder: Linear(latent → hidden) → ReLU → [×depth] → Linear(hidden → input_features)
 */
struct EegWindowAutoencoder : Module<nn::Backend>
{
    using Tensor = typename Module<nn::Backend>::Tensor;

    nn::Sequential encoder_;
    nn::Sequential decoder_;

    explicit EegWindowAutoencoder(const AutoencoderConfig& cfg);

    auto encode(const Tensor& input, bool requires_grad = true) -> Tensor;
    auto decode(const Tensor& latent, bool requires_grad = true) -> Tensor;

    auto forward(const Tensor& input, bool requires_grad = true) -> Tensor override;
    auto backward(const Tensor& grad_output) -> Tensor override;

    std::vector<Tensor*> param_ptrs_;
    auto params() -> std::span<Tensor*> override;
};

#endif // EXPERIMENT03_EEG_WINDOW_AUTOENCODER_HPP
