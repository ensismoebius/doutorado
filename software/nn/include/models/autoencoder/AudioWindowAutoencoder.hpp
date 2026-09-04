#ifndef EXPERIMENT03_AUDIO_WINDOW_AUTOENCODER_HPP
#define EXPERIMENT03_AUDIO_WINDOW_AUTOENCODER_HPP

#include <vector>

#include "AutoencoderConfig.hpp"
#include "EncoderDecoderAutoencoder.hpp"
#include "layers/Layers.hpp"
#include "layers/base/Module.hpp"
#include "tensor/Tensor.hpp"

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

#endif // EXPERIMENT03_AUDIO_WINDOW_AUTOENCODER_HPP
