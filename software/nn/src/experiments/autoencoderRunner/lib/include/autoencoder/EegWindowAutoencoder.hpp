#ifndef EXPERIMENT03_EEG_WINDOW_AUTOENCODER_HPP
#define EXPERIMENT03_EEG_WINDOW_AUTOENCODER_HPP

#include <vector>

#include "AutoencoderConfig.hpp"
#include "EncoderDecoderAutoencoder.hpp"
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
struct EegWindowAutoencoder : EncoderDecoderAutoencoder
{
    using Tensor = typename Module<nn::Backend>::Tensor;

    /// Builds the two Sequentials; everything else -- forward, backward,
    /// params, reset_state -- is EncoderDecoderAutoencoder's.
    explicit EegWindowAutoencoder(const AutoencoderConfig& cfg);
};

#endif // EXPERIMENT03_EEG_WINDOW_AUTOENCODER_HPP
