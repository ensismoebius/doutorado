/**
 * @file src/experiments/autoencoderRunner/lib/src/autoencoder/AudioWindowSpikingAutoencoder.cpp
 * @brief Spiking autoencoder implementation for audio-window inputs.
 */

#include "AudioWindowSpikingAutoencoder.hpp"

#include "AutoencoderBuilders.hpp"

AudioWindowSpikingAutoencoder::AudioWindowSpikingAutoencoder(const AutoencoderConfig& cfg)
    : EncoderDecoderAutoencoder(
          autoencoderRunner::autoencoders::build_snn_encoder(
              cfg, cfg.input_features, std::max(cfg.hidden_size, cfg.latent_size * 4)),
          autoencoderRunner::autoencoders::build_snn_decoder(
              cfg, cfg.input_features, std::max(cfg.hidden_size, cfg.latent_size * 4)))
{
}
