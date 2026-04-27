#ifndef NN_LAYERS_SPIKECOUNTLOSS_HPP
#define NN_LAYERS_SPIKECOUNTLOSS_HPP

#include <algorithm>
#include <cmath>

#include "nn/layers/base/Module.hpp"

/**
 * @file SpikeCountLoss.hpp
 * @brief Spike-count regression loss (MSE on spike counts) with spike-rate regularization.
 *
 * This loss is commonly used in SNN training when the model output is a spike
 * count (or firing-rate proxy) per sample rather than class logits.
 *
 * **Spike-rate regularization** (arXiv:2109.11045):
 * SNN autoencoders suffer from two failure modes that this regularizer prevents:
 * - **Dead neurons**: firing rate < min_rate → neuron never contributes, gradient dies.
 * - **Bursting neurons**: firing rate > max_rate → neuron always fires, loses selectivity.
 * A penalty is added to push the mean firing rate toward [min_rate, max_rate]:
 *   reg = λ * (max(0, min_rate - mean_rate)² + max(0, mean_rate - max_rate)²)
 *
 * Expected shapes:
 * - `input`: (N x F) spike tensor (values 0 or 1); N = samples, F = features.
 * - `target`: (N x F) desired spike tensor.
 *
 * Usage pattern:
 * - Call `set_target(target)` before `forward(pred)`.
 * - Call `forward(requires_grad=true)` before `backward()`.
 * - Set `min_rate`, `max_rate`, `rate_reg_lambda` to enable regularization.
 *
 * Reference: [31] "Training deep spiking auto-encoders without bursting or dying
 * neurons through regularization," arXiv:2109.11045, 2021.
 */
template <typename Backend>
class SpikeCountLossImpl : public Module<Backend>
{
    using Tensor = typename Module<Backend>::Tensor;

   private:
    Tensor target_;
    Tensor last_input_;
    bool training_ = true;
    float last_mean_rate_ = 0.0f;

   public:
    // Spike-rate regularization parameters
    float min_rate = 0.05f;       ///< Lower bound for acceptable mean firing rate (dead-neuron guard)
    float max_rate = 0.80f;       ///< Upper bound for acceptable mean firing rate (burst guard)
    float rate_reg_lambda = 0.0f; ///< Regularization weight (0 = disabled)

    SpikeCountLossImpl() = default;

    void train(bool on) override { training_ = on; }

    void set_target(const Tensor& t) { target_ = t; }

    /// Last computed mean firing rate (useful for logging sparsity metrics).
    float last_mean_rate() const { return last_mean_rate_; }

    auto forward(const Tensor& input, bool requires_grad = true) -> Tensor override
    {
        if (training_ && requires_grad)
        {
            last_input_ = input;
        }

        // MSE on spike counts
        Tensor diff = last_input_;
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
        float mse = sum_sq / static_cast<float>(diff.size());

        // Spike-rate regularization: penalise dead and bursting neurons
        float reg = 0.0f;
        if (rate_reg_lambda > 0.0f && last_input_.size() > 0)
        {
            float spike_sum = 0.0f;
            for (size_t i = 0; i < last_input_.rows(); ++i)
            {
                for (size_t j = 0; j < last_input_.cols(); ++j)
                {
                    spike_sum += last_input_.at(i, j);
                }
            }
            last_mean_rate_ = spike_sum / static_cast<float>(last_input_.size());

            float under = std::max(0.0f, min_rate - last_mean_rate_);
            float over  = std::max(0.0f, last_mean_rate_ - max_rate);
            reg = rate_reg_lambda * (under * under + over * over);
        }

        Tensor loss_tensor(1, 1);
        loss_tensor.at(0, 0) = mse + reg;
        return loss_tensor;
    }

    auto backward(const Tensor& /*grad_output*/) -> Tensor override
    {
        // Base gradient: 2*(prediction - target)/N
        Tensor grad(last_input_);
        grad.subtract_inplace(target_);
        grad.multiply_scalar_inplace(2.0f / static_cast<float>(last_input_.size()));

        // Regularization gradient: pushes mean rate toward [min_rate, max_rate].
        // d_reg/d_spike_ij = 2*lambda*(mean_rate - clamp(mean_rate, min_rate, max_rate)) / N
        if (rate_reg_lambda > 0.0f && last_input_.size() > 0)
        {
            float clamped = std::clamp(last_mean_rate_, min_rate, max_rate);
            float d_reg = 2.0f * rate_reg_lambda * (last_mean_rate_ - clamped)
                          / static_cast<float>(last_input_.size());
            for (size_t i = 0; i < grad.rows(); ++i)
            {
                for (size_t j = 0; j < grad.cols(); ++j)
                {
                    grad.at(i, j) += d_reg;
                }
            }
        }
        return grad;
    }
};

#endif // NN_LAYERS_SPIKECOUNTLOSS_HPP
