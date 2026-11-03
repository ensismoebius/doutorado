#pragma once

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "core/training/EpochResult.hpp"
#include "training/ITrainingCallback.hpp"

namespace nn::training
{

// Hands back the network of the epoch with the lowest validation loss, not the last one.
//
// Early stopping ends a run `patience` epochs AFTER its best epoch, and a run that is not
// stopped ends wherever epoch `epochs` left it; either way the last weights are not the
// ones the validation set chose. This callback copies every trainable tensor when the
// validation loss improves (by more than `min_delta`, the EarlyStoppingCallback criterion)
// and writes the copy back when training ends, so what the caller evaluates is that epoch's
// network (Goodfellow, Bengio & Courville, "Deep Learning", 2016, sec. 7.8, Algorithm 7.1).
//
// Only Model::params() is saved: it is everything the optimizer changes, so anything else
// in the model is the same at every epoch. A run without a validation set throws -- there
// is no best epoch to return, and falling back to the training loss would pick the most
// overfit one.
template <typename Model>
class BestWeightsCallback : public ITrainingCallback
{
   public:
    explicit BestWeightsCallback(Model& model, float min_delta = 1e-8F)
        : model_(model), min_delta_(min_delta)
    {
    }

    void on_epoch_end(const TrainingState& state, const EpochResult& result) override
    {
        if (std::isnan(result.val_loss))
            throw std::logic_error("BestWeightsCallback: epoch " + std::to_string(state.epoch) +
                                   " has no validation loss. It keeps the epoch with the lowest "
                                   "validation loss, so the trainer must be given a validation "
                                   "set (fit_*(train, val)).");
        if (result.val_loss < best_ - min_delta_)
        {
            best_ = result.val_loss;
            best_epoch_ = state.epoch;
            saved_.clear();
            for (const auto* p : model_.params()) saved_.push_back(*p);
        }
    }

    void on_train_end(const std::vector<EpochResult>& /*history*/) override
    {
        if (best_epoch_ == 0)
            throw std::logic_error(
                "BestWeightsCallback: no epoch had a finite validation loss, so "
                "there is no best network to restore. Check the run for a "
                "diverged loss.");
        auto params = model_.params();
        if (params.size() != saved_.size())
            throw std::logic_error("BestWeightsCallback: the model now has " +
                                   std::to_string(params.size()) + " trainable tensors, " +
                                   std::to_string(saved_.size()) + " were saved at epoch " +
                                   std::to_string(best_epoch_) +
                                   ". params() must return the same tensors every call.");
        for (std::size_t i = 0; i < params.size(); ++i) *params[i] = saved_[i];
    }

    // The epoch whose network on_train_end restored (0 before the first epoch ends).
    int best_epoch() const
    {
        return best_epoch_;
    }

   private:
    Model& model_;
    float min_delta_;
    // The element type of params() (a span of Tensor*), whatever the model's backend.
    using ParamTensor = std::remove_cv_t<
        std::remove_pointer_t<typename decltype(std::declval<Model&>().params())::value_type>>;

    float best_ = std::numeric_limits<float>::infinity();
    int best_epoch_ = 0;
    std::vector<ParamTensor> saved_;
};

} // namespace nn::training
