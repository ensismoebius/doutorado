/**
 * @file src/core/models/autoencoder/EegWindowSpikingAutoencoder.cpp
 * @brief Spiking autoencoder implementation for EEG window inputs.
 */

#include "models/autoencoder/EegWindowSpikingAutoencoder.hpp"

#include "models/autoencoder/AutoencoderBuilders.hpp"

namespace nn::models::autoencoder
{

EegWindowSpikingAutoencoder::EegWindowSpikingAutoencoder(const AutoencoderConfig& cfg)
    : EncoderDecoderAutoencoder(build_snn_encoder(cfg, cfg.input_features, cfg.hidden_size),
          build_snn_decoder(cfg, cfg.input_features, cfg.hidden_size))
{
}
} // namespace nn::models::autoencoder
