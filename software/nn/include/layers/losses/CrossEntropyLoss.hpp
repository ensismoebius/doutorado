#ifndef NN_LAYERS_CROSSENTROPYLOSS_HPP
#define NN_LAYERS_CROSSENTROPYLOSS_HPP

#include <cmath>

#include "layers/base/Module.hpp"

/**
 * @file CrossEntropyLoss.hpp
 * @brief Softmax + cross-entropy loss for multi-class classification.
 *
 * Expected shapes:
 * - `input` (logits): (N x C)
 * - `target` (one-hot): (N x C)
 *
 * Usage pattern:
 * - Call `set_target(one_hot_targets)` before `forward(logits)`.
 * - `forward()` returns a scalar (1x1) mean loss over the batch.
 * - `backward()` ignores its argument and returns `dL/d(logits)` as (N x C).
 *
 * Implementation notes:
 * - This is a pedagogical implementation using explicit loops and a numerically
 *   stable softmax (row-wise max subtraction).
 * - There is no internal checking that targets are valid one-hot vectors.
 */

template <typename Backend>
class CrossEntropyLossImpl : public Module<Backend>
{
    using Tensor = typename Module<Backend>::Tensor;

   public:
    // store last softmax output for backward
    Tensor last_probs;
    Tensor last_targets; // one-hot targets

    // Targets should be passed via set_target as Tensor (N x C) one-hot matrix.
    // (If you store class indices instead, you must convert to one-hot before using
    // this implementation.)
    void set_target(const Tensor& target)
    {
        last_targets = target;
    }

    auto forward(const Tensor& input, bool requires_grad = true) -> Tensor override
    {
        const auto& x = input;

        // Numeric-stable softmax: find max per row
        Tensor max_per_row(x.rows(), 1);
        for (size_t i = 0; i < x.rows(); ++i)
        {
            float max_val = x.at(i, 0);
            for (size_t j = 1; j < x.cols(); ++j)
            {
                max_val = std::max(max_val, x.at(i, j));
            }
            max_per_row.at(i, 0) = max_val;
        }

        // Shift and exponentiate
        Tensor exps(x.rows(), x.cols());
        for (size_t i = 0; i < x.rows(); ++i)
        {
            for (size_t j = 0; j < x.cols(); ++j)
            {
                exps.at(i, j) = std::exp(x.at(i, j) - max_per_row.at(i, 0));
            }
        }

        // Sum per row
        Tensor sums(x.rows(), 1);
        for (size_t i = 0; i < x.rows(); ++i)
        {
            float sum = 0.0f;
            for (size_t j = 0; j < x.cols(); ++j)
            {
                sum += exps.at(i, j);
            }
            sums.at(i, 0) = sum;
        }

        // Normalize: probs[i,j] = exps[i,j] / sums[i]
        Tensor probs(x.rows(), x.cols());
        for (size_t i = 0; i < x.rows(); ++i)
        {
            for (size_t j = 0; j < x.cols(); ++j)
            {
                probs.at(i, j) = exps.at(i, j) / sums.at(i, 0);
            }
        }

        if (requires_grad)
        {
            last_probs = probs;
        }

        // Compute mean cross-entropy: -mean(targets * log(probs))
        // Small epsilon avoids log(0) in edge cases.
        float loss_sum = 0.0f;
        for (size_t i = 0; i < x.rows(); ++i)
        {
            for (size_t j = 0; j < x.cols(); ++j)
            {
                if (last_targets.at(i, j) > 0.0f)
                {
                    loss_sum -= last_targets.at(i, j) * std::log(probs.at(i, j) + 1e-7f);
                }
            }
        }
        float loss = loss_sum / static_cast<float>(x.rows());

        Tensor loss_tensor(1, 1);
        loss_tensor.at(0, 0) = loss;
        return loss_tensor;
    }

    auto backward(const Tensor& /*unused*/) -> Tensor override
    {
        // Gradient of loss wrt logits: (probs - targets) / N
        Tensor grad(last_probs.rows(), last_probs.cols());
        float scale = 1.0f / static_cast<float>(last_probs.rows());
        for (size_t i = 0; i < last_probs.rows(); ++i)
        {
            for (size_t j = 0; j < last_probs.cols(); ++j)
            {
                grad.at(i, j) = (last_probs.at(i, j) - last_targets.at(i, j)) * scale;
            }
        }
        return grad;
    } //
};

#endif // NN_LAYERS_CROSSENTROPYLOSS_HPP
