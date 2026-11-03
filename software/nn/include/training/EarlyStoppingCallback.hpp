#pragma once

#include <cmath>
#include <limits>
#include <vector>

#include "core/training/EpochResult.hpp"
#include "training/ITrainingCallback.hpp"

namespace nn::training
{

// Stops training after `patience` epochs in a row without a new best monitored loss (the
// validation loss, or the training loss when there is no validation set).
//
// `arm_below` keeps a network that has not started learning from being stopped. Patience
// counts only once the monitored loss has reached `arm_below` (inclusive); until then no epoch
// is a "bad" one, so training either crosses that level or runs every epoch. Without it a
// network that sits on a plateau for its first epochs (an autoencoder at the error of the
// mean predictor, say) is stopped there, and the result reads as "this model cannot learn"
// when it only had not started: meeting01's pre-validation (GridUnesp job 5314513) stopped 8
// of 11 LSTM-AE search runs on EEG at epochs 8-16 with validation MSE 0.98-1.00, where the
// runs that left the plateau reached 0.58 (meeting01/lib/include/Meeting01Stopping.hpp). The
// default (+inf) arms on the first epoch: the plain patience rule.
class EarlyStoppingCallback : public ITrainingCallback
{
   public:
    explicit EarlyStoppingCallback(int patience,
        float min_delta = 1e-8F,
        float arm_below = std::numeric_limits<float>::infinity())
        : patience_(patience), min_delta_(min_delta), arm_below_(arm_below)
    {
    }

    void on_epoch_end(const TrainingState& state, const EpochResult& result) override
    {
        const float loss = std::isnan(result.val_loss) ? result.train_loss : result.val_loss;

        // Armed before the improvement test, so the epoch that crosses arm_below -- always a
        // new best, since no earlier epoch was that low -- starts the count at zero.
        if (!armed_ && loss <= arm_below_)
        {
            armed_ = true;
            armed_epoch_ = state.epoch;
        }

        if (loss < best_ - min_delta_)
        {
            best_ = loss;
            bad_epochs_ = 0;
        }
        else if (armed_)
        {
            ++bad_epochs_;
            if (bad_epochs_ >= patience_) stop_ = true;
        }
    }

    bool should_stop() const override
    {
        return stop_;
    }

    float best_val_loss() const
    {
        return best_;
    }

    // The epoch whose loss first reached arm_below; 0 while it has not.
    int armed_epoch() const
    {
        return armed_epoch_;
    }

   private:
    int patience_;
    float min_delta_;
    float arm_below_;
    float best_ = std::numeric_limits<float>::infinity();
    int bad_epochs_ = 0;
    bool armed_ = false;
    int armed_epoch_ = 0;
    bool stop_ = false;
};

} // namespace nn::training
