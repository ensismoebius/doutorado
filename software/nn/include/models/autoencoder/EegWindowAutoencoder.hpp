#ifndef NN_MODELS_AUTOENCODER_EEG_WINDOW_AUTOENCODER_HPP
#define NN_MODELS_AUTOENCODER_EEG_WINDOW_AUTOENCODER_HPP

#include <vector>

#include "layers/Layers.hpp"
#include "layers/base/Module.hpp"
#include "models/autoencoder/AutoencoderConfig.hpp"
#include "models/autoencoder/EncoderDecoderAutoencoder.hpp"
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

#endif // NN_MODELS_AUTOENCODER_EEG_WINDOW_AUTOENCODER_HPP
