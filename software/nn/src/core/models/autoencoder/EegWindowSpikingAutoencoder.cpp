/**
 * @file src/core/models/autoencoder/EegWindowSpikingAutoencoder.cpp
 * @brief Spiking autoencoder implementation for EEG window inputs.
 */

#include "models/autoencoder/EegWindowSpikingAutoencoder.hpp"

#include "models/autoencoder/AutoencoderBuilders.hpp"

EegWindowSpikingAutoencoder::EegWindowSpikingAutoencoder(const AutoencoderConfig& cfg)
    : EncoderDecoderAutoencoder(autoencoderRunner::autoencoders::build_snn_encoder(
                                    cfg, cfg.input_features, cfg.hidden_size),
          autoencoderRunner::autoencoders::build_snn_decoder(
              cfg, cfg.input_features, cfg.hidden_size))
{
}
