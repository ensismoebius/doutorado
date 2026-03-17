#ifndef NN_LAYERS_SPIKECOUNTLOSS_HPP
#define NN_LAYERS_SPIKECOUNTLOSS_HPP

#include "nn/layers/Module.hpp"
#include "nn/tensor/Tensor.hpp"

/**
 * @file SpikeCountLoss.hpp
 * @brief Spike-count regression loss (MSE on spike counts).
 *
 * This loss is commonly used in SNN training when the model output is a spike
 * count (or firing-rate proxy) per sample rather than class logits.
 *
 * Expected shapes:
 * - `input`: (N x 1) predicted spike counts (or rates)
 * - `target`: (N x 1) desired spike counts (or rates)
 *
 * Usage pattern:
 * - Call `set_target(target)` before `forward(pred)`.
 * - Call `forward(requires_grad=true)` before `backward()`.
 *
 * Implementation caveat:
 * - `forward()` computes `diff` from `last_input` rather than `input`.
 *   That means if you call `forward(..., requires_grad=false)` in training mode,
 *   `last_input` may refer to a previous batch. This header keeps behavior as-is;
 *   callers should prefer `requires_grad=true` during training.
 */

class SpikeCountLoss : public Module
{
   private:
    nn::Tensor target_;
    nn::Tensor last_input_;
    bool training_ = true;

   public:
    SpikeCountLoss() = default;

    void train(bool on) override
    {
        training_ = on;
    }

    void set_target(const nn::Tensor& t)
    {
        target_ = t;
    }
    auto forward(const nn::Tensor& input, bool requires_grad = true) -> nn::Tensor override
    {
        if (training_ && requires_grad)
        {
            last_input_ = input;
        }
        // pred and target: (n_samples, 1)
        nn::Tensor diff = last_input_;
        diff.subtract_inplace(target_);
        float sum_sq = 0.0f;
        for (size_t i = 0; i < diff.rows(); ++i)
        {
            for (size_t j = 0; j < diff.cols(); ++j)
            {
                float val = diff.at(i, j);
                sum_sq += val * val;
            }
        }
        float loss = sum_sq / static_cast<float>(diff.size());
        nn::Tensor loss_tensor(1, 1);
        loss_tensor.at(0, 0) = loss;
        return loss_tensor;
    }
    auto backward(const nn::Tensor& /*grad_output*/) -> nn::Tensor override
    {
        // Gradient: 2 * (prediction - target) / N
        nn::Tensor grad(last_input_);
        grad.subtract_inplace(target_);
        grad.multiply_scalar_inplace(2.0f / static_cast<float>(last_input_.size()));
        return grad;
    }
};

#endif // NN_LAYERS_SPIKECOUNTLOSS_HPP
