/**
 * @file src/experiments/autoencoderRunner/lib/src/DatasetBuilder.cpp
 * @brief Implementation of DatasetBuilder for Experiment03.
 */

#include "DatasetBuilder.hpp"

#include <stdexcept>

#include "data_loaders/10.1117/datasets/windowed/AudioWindowDataset.hpp"
#include "data_loaders/10.1117/datasets/windowed/EEGWindowDataset.hpp"
#include "data_loaders/10.1117/datasets/windowed/FusedWindowDataset.hpp"

namespace autoencoderRunner
{
auto DatasetBuilder::build() -> std::shared_ptr<Dataset>
{
    if (!cfg_) throw std::runtime_error("DatasetBuilder: config not set");

    switch (cfg_->dataset_type)
    {
        case Experiment03DatasetType::Protocol:
        {
            auto ds = std::make_shared<Dataset101117>(discovered_);
            ds->set_input_mode(cfg_->dataset_input_mode);
            return ds;
        }
        case Experiment03DatasetType::EegWindow:
            return std::make_shared<EEGWindowDataset>(discovered_, cfg_->window_eeg_config);
        case Experiment03DatasetType::AudioWindow:
            return std::make_shared<AudioWindowDataset>(discovered_, cfg_->window_audio_config);
        case Experiment03DatasetType::FusedWindow:
            return std::make_shared<FusedWindowDataset>(
                discovered_, cfg_->window_eeg_config, cfg_->window_audio_config);
    }

    throw std::runtime_error("Unsupported dataset type");
}

} // namespace autoencoderRunner
