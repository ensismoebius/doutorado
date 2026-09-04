/**
 * @file src/core/models/autoencoder/EegWindowAutoencoder.cpp
 * @brief ANN implementation of the EEG window autoencoder.
 */

#include "models/autoencoder/EegWindowAutoencoder.hpp"

#include "models/autoencoder/AutoencoderBuilders.hpp"

namespace nn::models::autoencoder
{

EegWindowAutoencoder::EegWindowAutoencoder(const AutoencoderConfig& cfg)
    : EncoderDecoderAutoencoder(build_ann_encoder(cfg, cfg.input_features, cfg.hidden_size),
          build_ann_decoder(cfg, cfg.input_features, cfg.hidden_size))
{
}
} // namespace nn::models::autoencoder
