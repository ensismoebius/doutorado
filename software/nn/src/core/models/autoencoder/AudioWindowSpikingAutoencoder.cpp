/**
 * @file src/core/models/autoencoder/AudioWindowSpikingAutoencoder.cpp
 * @brief Spiking autoencoder implementation for audio-window inputs.
 */

#include "models/autoencoder/AudioWindowSpikingAutoencoder.hpp"

#include "models/autoencoder/AutoencoderBuilders.hpp"

namespace nn::models::autoencoder
{

AudioWindowSpikingAutoencoder::AudioWindowSpikingAutoencoder(const AutoencoderConfig& cfg)
    : EncoderDecoderAutoencoder(
          build_snn_encoder(
              cfg, cfg.input_features, std::max(cfg.hidden_size, cfg.latent_size * 4)),
          build_snn_decoder(
              cfg, cfg.input_features, std::max(cfg.hidden_size, cfg.latent_size * 4)))
{
}
} // namespace nn::models::autoencoder
