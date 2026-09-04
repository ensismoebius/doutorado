/**
 * @file src/core/models/autoencoder/AudioWindowAutoencoder.cpp
 * @brief ANN implementation of the audio-window autoencoder.
 */

#include "models/autoencoder/AudioWindowAutoencoder.hpp"

#include "models/autoencoder/AutoencoderBuilders.hpp"

namespace nn::models::autoencoder
{

AudioWindowAutoencoder::AudioWindowAutoencoder(const AutoencoderConfig& cfg)
    : EncoderDecoderAutoencoder(
          build_ann_encoder(
              cfg, cfg.input_features, std::max(cfg.hidden_size, cfg.latent_size * 4)),
          build_ann_decoder(
              cfg, cfg.input_features, std::max(cfg.hidden_size, cfg.latent_size * 4)))
{
}
} // namespace nn::models::autoencoder
