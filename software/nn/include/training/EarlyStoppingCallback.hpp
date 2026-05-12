#pragma once

#include <limits>
#include <vector>

#include "core/training/EpochResult.hpp"
#include "training/ITrainingCallback.hpp"

namespace nn::training
{

class EarlyStoppingCallback : public ITrainingCallback
{
public:
    explicit EarlyStoppingCallback(int patience, float min_delta = 1e-8F)
        : patience_(patience), min_delta_(min_delta) {}

    void on_epoch_end(const TrainingState& /*state*/, const EpochResult& result) override
    {
        const float loss = std::isnan(result.val_loss) ? result.train_loss : result.val_loss;

        if (loss < best_ - min_delta_)
        {
            best_       = loss;
            bad_epochs_ = 0;
        }
        else
        {
            ++bad_epochs_;
            if (bad_epochs_ >= patience_)
                stop_ = true;
        }
    }

    bool should_stop() const override { return stop_; }

    float best_val_loss() const { return best_; }

private:
    int   patience_;
    float min_delta_;
    float best_       = std::numeric_limits<float>::infinity();
    int   bad_epochs_ = 0;
    bool  stop_       = false;
};

} // namespace nn::training
