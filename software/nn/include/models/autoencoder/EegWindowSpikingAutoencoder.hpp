#ifndef EXPERIMENT03_EEG_WINDOW_SPIKING_AUTOENCODER_HPP
#define EXPERIMENT03_EEG_WINDOW_SPIKING_AUTOENCODER_HPP

#include <vector>

#include "AutoencoderConfig.hpp"
#include "EncoderDecoderAutoencoder.hpp"
#include "layers/Layers.hpp"
#include "layers/base/Module.hpp"
#include "tensor/Tensor.hpp"

/**
 * @file EegWindowSpikingAutoencoder.hpp
 * @brief SNN autoencoder for EEG-window features.
 *
 * Architecture:
 *   Encoder: Linear(input → hidden) → Lif → [×depth] → Linear(hidden → latent) → Lif
 *   Decoder: Linear(latent → hidden) → LifIntegrator → [×depth] → Linear(hidden → input)
 */
struct EegWindowSpikingAutoencoder : EncoderDecoderAutoencoder
{
    using Tensor = typename Module<nn::Backend>::Tensor;

    /// Builds the two Sequentials; everything else -- forward, backward,
    /// params, reset_state -- is EncoderDecoderAutoencoder's.
    explicit EegWindowSpikingAutoencoder(const AutoencoderConfig& cfg);
};

#endif // EXPERIMENT03_EEG_WINDOW_SPIKING_AUTOENCODER_HPP
