#pragma once

#include <Eigen/Dense>

#include "../tensor/Tensor.hpp"
#include "core/layers/Module.hpp"

// Computes softmax + cross-entropy. Assumes predictions are logits (N x C) and targets are one-hot
// or class indices stored as (N x 1)
class CrossEntropyLoss : public Module
{
   public:
    // store last softmax output for backward
    Eigen::MatrixXf last_probs;
    Eigen::MatrixXf last_targets; // one-hot targets

    // targets should be passed via set_target as Tensor (N x C) one-hot matrix
    void set_target(const Tensor& target)
    {
        last_targets = target.get_data_ref();
    }

    auto forward(const Tensor& logits) -> Tensor override
    {
        // numeric-stable softmax
        Eigen::MatrixXf x = logits.get_data_ref();
        Eigen::VectorXf max_per_row = x.rowwise().maxCoeff();
        Eigen::MatrixXf shifted = x.colwise() - max_per_row;
        Eigen::MatrixXf exps = shifted.array().exp();
        Eigen::VectorXf sums = exps.rowwise().sum();
        last_probs = exps.array().colwise() / sums.array();

        // compute mean cross-entropy
        Eigen::ArrayXf logp = last_probs.array().log();
        float loss = -(last_targets.array() * logp).sum() / static_cast<float>(x.rows());
        return Tensor{Eigen::MatrixXf::Constant(1, 1, loss)};
    }

    auto backward(const Tensor& /*unused*/) -> Tensor override
    {
        // gradient of loss wrt logits: (probs - targets)/N
        Eigen::MatrixXf grad = (last_probs - last_targets) / static_cast<float>(last_probs.rows());
        return Tensor{grad};
    }
};
