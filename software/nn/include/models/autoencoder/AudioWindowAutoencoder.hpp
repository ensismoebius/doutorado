#ifndef NN_MODELS_AUTOENCODER_AUDIO_WINDOW_AUTOENCODER_HPP
#define NN_MODELS_AUTOENCODER_AUDIO_WINDOW_AUTOENCODER_HPP

#include <vector>

#include "layers/Layers.hpp"
#include "layers/base/Module.hpp"
#include "models/autoencoder/AutoencoderConfig.hpp"
#include "models/autoencoder/EncoderDecoderAutoencoder.hpp"
#include "tensor/Tensor.hpp"

namespace nn::models::autoencoder
{

/**
 * @file AudioWindowAutoencoder.hpp
 * @brief ANN autoencoder for audio-window features.
 *
 * Architecture (symmetric):
 *   Encoder: Linear(input_features → hidden) → ReLU → [×depth] → Linear(hidden → latent) → ReLU
 *   Decoder: Linear(latent → hidden) → ReLU → [×depth] → Linear(hidden → input_features)
 */
struct AudioWindowAutoencoder : EncoderDecoderAutoencoder
{
    using Tensor = typename Module<nn::Backend>::Tensor;

    /// Builds the two Sequentials; everything else -- forward, backward,
    /// params, reset_state -- is EncoderDecoderAutoencoder's.
    explicit AudioWindowAutoencoder(const AutoencoderConfig& cfg);
};
} // namespace nn::models::autoencoder

#endif // NN_MODELS_AUTOENCODER_AUDIO_WINDOW_AUTOENCODER_HPP
