/**
 * @file src/experiments/autoencoderRunner/lib/src/autoencoder/AudioWindowAutoencoder.cpp
 * @brief ANN implementation of the audio-window autoencoder.
 */

#include "AudioWindowAutoencoder.hpp"

#include "AutoencoderBuilders.hpp"

AudioWindowAutoencoder::AudioWindowAutoencoder(const AutoencoderConfig& cfg)
    : EncoderDecoderAutoencoder(
          autoencoderRunner::autoencoders::build_ann_encoder(
              cfg, cfg.input_features, std::max(cfg.hidden_size, cfg.latent_size * 4)),
          autoencoderRunner::autoencoders::build_ann_decoder(
              cfg, cfg.input_features, std::max(cfg.hidden_size, cfg.latent_size * 4)))
{
}
