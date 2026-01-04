#pragma once

#include "../tensor/Tensor.hpp"
#include "core/layers/Module.hpp"

// Computes softmax + cross-entropy. Assumes predictions are logits (N x C) and targets are one-hot
// or class indices stored as (N x 1)
class CrossEntropyLoss : public Module
{
   public:
    // store last softmax output for backward
    nn::Tensor last_probs;
    nn::Tensor last_targets; // one-hot targets

    // targets should be passed via set_target as Tensor (N x C) one-hot matrix
    void set_target(const nn::Tensor& target)
    {
        last_targets = target;
    }

    auto forward(const nn::Tensor& input, bool requires_grad = true) -> nn::Tensor override
    {
        // numeric-stable softmax
        const auto& x = input.get_data_ref();
        auto max_per_row = x.rowwise().maxCoeff();
        auto shifted = x.colwise() - max_per_row;
        auto exps = shifted.array().exp();
        auto sums = exps.rowwise().sum();

        nn::Tensor probs(x.rows(), x.cols());
        probs.get_data_ref() = exps.array().colwise() / sums.array();

        if (requires_grad)
        {
            last_probs = probs;
        }

        // compute mean cross-entropy
        auto logp = probs.get_data_ref().array().log();
        float loss =
            -(last_targets.get_data_ref().array() * logp).sum() / static_cast<float>(x.rows());
        nn::Tensor loss_tensor(1, 1);
        loss_tensor.at(0, 0) = loss;
        return loss_tensor;
    }

    auto backward(const nn::Tensor& /*unused*/) -> nn::Tensor override
    {
        // gradient of loss wrt logits: (probs - targets)/N
        nn::Tensor grad(last_probs.rows(), last_probs.cols());
        grad.get_data_ref() = (last_probs.get_data_ref() - last_targets.get_data_ref()) /
                              static_cast<float>(last_probs.rows());
        return grad;
    }
};
