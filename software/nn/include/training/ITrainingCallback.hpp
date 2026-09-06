#pragma once

#include <vector>

#include "core/training/EpochResult.hpp"
#include "training/TrainingState.hpp"

namespace nn::training
{

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
