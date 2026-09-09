#pragma once

#include <vector>

#include "training/ITrainingCallback.hpp"

namespace guayaquil
{

class BatchLossCollector : public nn::training::ITrainingCallback
{
   public:
    std::vector<float> batch_losses;

    void on_batch_end(const nn::training::TrainingState& state) override
    {
        batch_losses.push_back(state.batch_loss);
    }

    void reset()
    {
        batch_losses.clear();
    }
};

} // namespace guayaquil
