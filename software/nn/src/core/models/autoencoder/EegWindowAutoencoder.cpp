/**
 * @file src/core/models/autoencoder/EegWindowAutoencoder.cpp
 * @brief ANN implementation of the EEG window autoencoder.
 */

#include "models/autoencoder/EegWindowAutoencoder.hpp"

#include "models/autoencoder/AutoencoderBuilders.hpp"

EegWindowAutoencoder::EegWindowAutoencoder(const AutoencoderConfig& cfg)
    : EncoderDecoderAutoencoder(autoencoderRunner::autoencoders::build_ann_encoder(
                                    cfg, cfg.input_features, cfg.hidden_size),
          autoencoderRunner::autoencoders::build_ann_decoder(
              cfg, cfg.input_features, cfg.hidden_size))
{
}
