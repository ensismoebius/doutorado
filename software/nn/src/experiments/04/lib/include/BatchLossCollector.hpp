#pragma once

#include <vector>

#include "nn/training/ITrainingCallback.hpp"

namespace comparative_autoencoder_experiment
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

} // namespace comparative_autoencoder_experiment
