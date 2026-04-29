#pragma once

#include <map>
#include <string>
#include <vector>

#include "core/training/EpochResult.hpp"
#include "nn/progress/ProgressManager.hpp"
#include "nn/training/ITrainingCallback.hpp"

namespace nn::training
{

class ProgressCallback : public ITrainingCallback
{
public:
    explicit ProgressCallback(std::string label) : label_(std::move(label)) {}

    void on_train_begin(int total_epochs) override
    {
        total_epochs_ = total_epochs;
        bar_id_ = nn::progress::ProgressManager::instance().create_bar(
            label_, static_cast<float>(total_epochs));
    }

    void on_epoch_end(const TrainingState& /*state*/, const EpochResult& result) override
    {
        std::map<std::string, float> metrics;
        metrics["train_loss"] = result.train_loss;
        if (!std::isnan(result.val_loss))
            metrics["val_loss"] = result.val_loss;

        nn::progress::ProgressManager::instance().update_bar(
            bar_id_, static_cast<float>(result.epoch), metrics);
    }

    void on_train_end(const std::vector<EpochResult>& /*history*/) override
    {
        nn::progress::ProgressManager::instance().complete_bar(bar_id_);
    }

private:
    std::string label_;
    uint32_t bar_id_   = 0;
    int total_epochs_  = 0;
};

} // namespace nn::training
