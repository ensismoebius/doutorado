/**
 * @file src/experiments/03/lib/src/DatasetBuilder.cpp
 * @brief Implementation of DatasetBuilder for Experiment03.
 */

#include "DatasetBuilder.hpp"

#include <stdexcept>

#include "nn/dataLoaders/10.1117/windowing/AudioWindowDataset.hpp"
#include "nn/dataLoaders/10.1117/windowing/EEGWindowDataset.hpp"
#include "nn/dataLoaders/10.1117/windowing/FusedWindowDataset.hpp"

namespace experiment03
{
auto DatasetBuilder::build() -> std::shared_ptr<Dataset>
{
    if (!cfg_) throw std::runtime_error("DatasetBuilder: config not set");

    switch (cfg_->dataset_type)
    {
        case Experiment03DatasetType::Protocol:
        {
            auto ds = std::make_shared<Protocol101117Dataset>(discovered_);
            ds->set_input_mode(cfg_->input_mode);
            return ds;
        }
        case Experiment03DatasetType::EegWindow:
            return std::make_shared<EEGWindowDataset>(discovered_, cfg_->eeg_window_config);
        case Experiment03DatasetType::AudioWindow:
            return std::make_shared<AudioWindowDataset>(discovered_, cfg_->audio_window_config);
        case Experiment03DatasetType::FusedWindow:
            return std::make_shared<FusedWindowDataset>(
                discovered_, cfg_->eeg_window_config, cfg_->audio_window_config);
    }

    throw std::runtime_error("Unsupported dataset type");
}

} // namespace experiment03
