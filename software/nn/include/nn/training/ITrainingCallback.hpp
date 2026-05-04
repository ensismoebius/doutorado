#pragma once

#include <vector>

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

struct ITrainingCallback
{
    virtual void on_train_begin(int /*total_epochs*/) {}
    virtual void on_train_end(const std::vector<EpochResult>& /*history*/) {}
    virtual void on_epoch_begin(const TrainingState& /*state*/) {}
    virtual void on_epoch_end(const TrainingState& /*state*/, const EpochResult& /*result*/) {}
    virtual void on_batch_begin(const TrainingState& /*state*/) {}
    virtual void on_batch_progress(const TrainingState& /*state*/) {}
    virtual void on_batch_end(const TrainingState& /*state*/) {}
    virtual bool should_stop() const
    {
        return false;
    }
    virtual ~ITrainingCallback() = default;
};

} // namespace nn::training
