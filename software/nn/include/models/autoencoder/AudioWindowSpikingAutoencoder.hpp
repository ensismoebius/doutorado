#ifndef NN_MODELS_AUTOENCODER_AUDIO_WINDOW_SPIKING_AUTOENCODER_HPP
#define NN_MODELS_AUTOENCODER_AUDIO_WINDOW_SPIKING_AUTOENCODER_HPP

#include <vector>

#include "layers/Layers.hpp"
#include "layers/base/Module.hpp"
#include "models/autoencoder/AutoencoderConfig.hpp"
#include "models/autoencoder/EncoderDecoderAutoencoder.hpp"
#include "tensor/Tensor.hpp"

namespace nn::models::autoencoder
{

/**
 * @file AudioWindowSpikingAutoencoder.hpp
 * @brief SNN autoencoder for audio-window features.
 *
 * Architecture:
 *   Encoder: Linear(input → hidden) → Lif → [×depth] → Linear(hidden → latent) → Lif
 *   Decoder: Linear(latent → hidden) → LifIntegrator → [×depth] → Linear(hidden → input)
 */
struct AudioWindowSpikingAutoencoder : EncoderDecoderAutoencoder
{
    using Tensor = typename Module<nn::Backend>::Tensor;

    /// Builds the two Sequentials; everything else -- forward, backward,
    /// params, reset_state -- is EncoderDecoderAutoencoder's.
    explicit AudioWindowSpikingAutoencoder(const AutoencoderConfig& cfg);
};
} // namespace nn::models::autoencoder

#endif // NN_MODELS_AUTOENCODER_AUDIO_WINDOW_SPIKING_AUTOENCODER_HPP
