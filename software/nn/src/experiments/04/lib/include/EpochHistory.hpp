#pragma once

#include <cstddef>
#include <vector>

namespace comparative_autoencoder_experiment
{

struct EpochHistory
{
    // Epoch-level aggregates
    std::vector<float> epoch_nums;
    std::vector<float> train_losses;
    std::vector<float> val_losses;

    // Batch-level raw data for convergence plots
    std::vector<float> batch_losses;
    std::vector<float> batch_epochs;  // which epoch each batch belongs to
};

} // namespace comparative_autoencoder_experiment
