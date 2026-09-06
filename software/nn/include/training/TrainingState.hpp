#pragma once

/**
 * @file TrainingState.hpp
 * @brief TrainingState struct (extracted from ITrainingCallback.hpp).
 */

#include "core/training/EpochResult.hpp"

namespace nn::training
{

struct TrainingState
{
    int epoch = 0;
    int total_epochs = 0;
    int batch = 0;
    int total_batches = 0;
    float batch_progress = 0.0F;
    float batch_loss = 0.0F;
    const EpochResult* last_epoch_result = nullptr;
};

} // namespace nn::training
