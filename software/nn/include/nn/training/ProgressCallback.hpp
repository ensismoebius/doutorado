#pragma once

#include <cmath>
#include <map>
#include <string>
#include <vector>

#include "core/training/EpochResult.hpp"
#include "nn/progress/ProgressManager.hpp"
#include "nn/training/ITrainingCallback.hpp"

namespace nn::training
{

/**
 * Callback that renders two live progress bars via ProgressManager:
 *   - Epoch bar  — advances once per epoch, displays train/val loss.
 *   - Batch bar  — advances once per batch within the current epoch.
 *
 * Both bars are removed when training ends so the display stays clean.
 * A third "run/fold" bar is typically managed externally by the experiment
 * loop (see RunComparativeExperiment) to show overall repeat progress.
 */
class ProgressCallback : public ITrainingCallback
{
   public:
    explicit ProgressCallback(std::string label) : label_(std::move(label)) {}

    /// Creates epoch and batch progress bars.
    void on_train_begin(int total_epochs) override
    {
        total_epochs_ = total_epochs;
        epoch_bar_id_ = nn::progress::ProgressManager::instance().create_bar(
            label_ + " epoch", static_cast<float>(total_epochs));
        // Batch bar target is updated on the first on_epoch_begin call.
        batch_bar_id_ =
            nn::progress::ProgressManager::instance().create_bar(label_ + " batch", 1.0f);
    }

    /// Captures the per-epoch batch budget so the batch bar has the right target.
    void on_epoch_begin(const TrainingState& state) override
    {
        epoch_batches_ = state.total_batches > 0 ? state.total_batches : 1;
        nn::progress::ProgressManager::instance().set_target(
            batch_bar_id_, static_cast<float>(epoch_batches_));
        // Reset batch bar to zero at the start of each epoch.
        nn::progress::ProgressManager::instance().update_bar(batch_bar_id_, 0.0f);
    }

    /// Advances the batch bar and displays the running loss.
    void on_batch_end(const TrainingState& state) override
    {
        nn::progress::ProgressManager::instance().update_bar(
            batch_bar_id_, static_cast<float>(state.batch), {{"loss", state.batch_loss}});
    }

    /// Advances the epoch bar and displays train/val loss.
    void on_epoch_end(const TrainingState& /*state*/, const EpochResult& result) override
    {
        std::map<std::string, float> metrics;
        metrics["train_loss"] = result.train_loss;
        if (!std::isnan(result.val_loss)) metrics["val_loss"] = result.val_loss;

        nn::progress::ProgressManager::instance().update_bar(
            epoch_bar_id_, static_cast<float>(result.epoch), metrics);
    }

    /// Removes both bars so completed runs don't accumulate on screen.
    void on_train_end(const std::vector<EpochResult>& /*history*/) override
    {
        nn::progress::ProgressManager::instance().remove_bar(batch_bar_id_);
        nn::progress::ProgressManager::instance().remove_bar(epoch_bar_id_);
    }

   private:
    std::string label_;
    uint32_t epoch_bar_id_ = 0;
    uint32_t batch_bar_id_ = 0;
    int total_epochs_ = 0;
    int epoch_batches_ = 1;
};

} // namespace nn::training
