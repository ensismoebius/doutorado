/**
 * @file src/experiments/03/lib/src/experiment03_helpers.cpp
 * @brief Implementation of small helper utilities for Experiment03.
 */

#include "experiment03_helpers.hpp"

#include <stdexcept>

#include "AudioWindowAutoencoder.hpp"
#include "AudioWindowSpikingAutoencoder.hpp"
#include "EegWindowAutoencoder.hpp"
#include "EegWindowSpikingAutoencoder.hpp"
#include "FusedWindowAutoencoder.hpp"
#include "FusedWindowSpikingAutoencoder.hpp"
#include "ProtocolAutoencoder.hpp"
#include "ProtocolSpikingAutoencoder.hpp"

namespace experiment03
{

auto to_sqlite_dataset_type(Experiment03DatasetType dataset_type)
    -> nn::dataLoaders::SqliteDatasetType
{
    switch (dataset_type)
    {
        case Experiment03DatasetType::Protocol:
            return nn::dataLoaders::SqliteDatasetType::Protocol;
        case Experiment03DatasetType::EegWindow:
            return nn::dataLoaders::SqliteDatasetType::EegWindow;
        case Experiment03DatasetType::AudioWindow:
            return nn::dataLoaders::SqliteDatasetType::AudioWindow;
        case Experiment03DatasetType::FusedWindow:
            return nn::dataLoaders::SqliteDatasetType::FusedWindow;
    }

    return nn::dataLoaders::SqliteDatasetType::Protocol;
}

auto build_autoencoder_model(const Config& config, nn::Index input_features)
    -> std::unique_ptr<Module>
{
    AutoencoderConfig model_cfg{};
    model_cfg.input_features =
        config.effective_autoencoder_input_features(static_cast<int>(input_features));
    model_cfg.hidden_size = config.autoencoder_hidden_size;
    model_cfg.latent_size = config.autoencoder_latent_size;
    model_cfg.depth = config.autoencoder_depth;
    model_cfg.layer_sizes = config.autoencoder_layer_sizes;
    model_cfg.architecture = config.effective_autoencoder_architecture();
    model_cfg.branch_hidden_size = config.autoencoder_branch_hidden_size;
    model_cfg.fusion_hidden_size = config.autoencoder_fusion_hidden_size;
    model_cfg.residual_blocks = config.autoencoder_residual_blocks;
    model_cfg.time_step = config.autoencoder_time_step;
    model_cfg.resistance = config.autoencoder_resistance;
    model_cfg.capacitance = config.autoencoder_capacitance;
    model_cfg.eeg_features = config.effective_autoencoder_eeg_features();
    model_cfg.audio_features = config.effective_autoencoder_audio_features();

    switch (config.autoencoder_type)
    {
        case Experiment03AutoencoderType::ProtocolAnn:
            return std::make_unique<ProtocolAutoencoder>(model_cfg);
        case Experiment03AutoencoderType::EegWindowAnn:
            return std::make_unique<EegWindowAutoencoder>(model_cfg);
        case Experiment03AutoencoderType::AudioWindowAnn:
            return std::make_unique<AudioWindowAutoencoder>(model_cfg);
        case Experiment03AutoencoderType::FusedWindowAnn:
            return std::make_unique<FusedWindowAutoencoder>(model_cfg);
        case Experiment03AutoencoderType::ProtocolSnn:
            return std::make_unique<ProtocolSpikingAutoencoder>(model_cfg);
        case Experiment03AutoencoderType::EegWindowSnn:
            return std::make_unique<EegWindowSpikingAutoencoder>(model_cfg);
        case Experiment03AutoencoderType::AudioWindowSnn:
            return std::make_unique<AudioWindowSpikingAutoencoder>(model_cfg);
        case Experiment03AutoencoderType::FusedWindowSnn:
            return std::make_unique<FusedWindowSpikingAutoencoder>(model_cfg);
    }

    throw std::runtime_error("Unsupported autoencoder type");
}

} // namespace experiment03
