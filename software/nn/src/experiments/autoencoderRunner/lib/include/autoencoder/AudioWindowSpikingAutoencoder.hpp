#ifndef EXPERIMENT03_AUDIO_WINDOW_SPIKING_AUTOENCODER_HPP
#define EXPERIMENT03_AUDIO_WINDOW_SPIKING_AUTOENCODER_HPP

#include <vector>

#include "AutoencoderConfig.hpp"
#include "EncoderDecoderAutoencoder.hpp"
#include "layers/Layers.hpp"
#include "layers/base/Module.hpp"
#include "tensor/Tensor.hpp"

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

#endif // EXPERIMENT03_AUDIO_WINDOW_SPIKING_AUTOENCODER_HPP
