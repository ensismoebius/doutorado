#ifndef NN_LAYERS_SPIKETIMELOSS_HPP
#define NN_LAYERS_SPIKETIMELOSS_HPP

#include <limits>

#include "nn/layers/base/Module.hpp"

/**
 * @file SpikeTimeLoss.hpp
 * @brief First-spike time loss for latency-coded SNN autoencoders.
 *
 * In latency coding, input magnitude is encoded as time-to-first-spike: high
 * magnitude → early spike, low magnitude → late or absent spike. This loss
 * computes MSE on the first-spike time of each (batch, feature) pair.
 *
 * **Important:** use this loss ONLY when the model uses latency spike encoding.
 * For rate-coded SNNs use SpikeCountLoss. Mixing encoding and loss types leads
 * to incorrect gradient directions.
 *
 * Expected shapes:
 * - `input`  : (T*B, F) spike tensor (values 0 or 1), time-major layout
 * - `target` : (T*B, F) spike tensor (values 0 or 1), time-major layout
 *   where T = time_steps, B = batch_size, F = features.
 *
 * First-spike extraction:
 * - For each (b, f): first_spike_time = min t such that spike[t,b,f]==1, else T.
 * - Missing spike penalty = T (worst-case distance, not infinity, for numeric safety).
 *
 * Reference: [32] S. Comsa et al., "Spiking autoencoders with temporal coding,"
 * Frontiers in Neuroscience, vol. 15, p. 712667, 2021.
 * Also: [42] H. Yang et al., "Time series forecasting via derivative spike
 * encoding and bespoke loss functions for SNNs," Computers, 2024.
 */
template <typename Backend>
class SpikeTimeLossImpl : public Module<Backend>
{
    using Tensor = typename Module<Backend>::Tensor;

   private:
    Tensor target_;
    Tensor last_input_;
    int time_steps_ = 1;
    bool training_ = true;

    // Extract first-spike time for each (batch, feature).
    // Returns a (B, F) tensor of first-spike times in [0, T].
    // No-spike entries are set to T (the maximum penalty time).
    static Tensor first_spike_times(const Tensor& spikes, int T, int B, int F)
    {
        Tensor times(static_cast<size_t>(B), static_cast<size_t>(F));
        for (int b = 0; b < B; ++b)
        {
            for (int f = 0; f < F; ++f)
            {
                float fst = static_cast<float>(T); // default: no spike → penalty = T
                for (int t = 0; t < T; ++t)
                {
                    if (spikes.at(static_cast<size_t>(t * B + b), static_cast<size_t>(f)) > 0.5f)
                    {
                        fst = static_cast<float>(t);
                        break;
                    }
                }
                times.at(static_cast<size_t>(b), static_cast<size_t>(f)) = fst;
            }
        }
        return times;
    } // LCOV_EXCL_LINE

   public:
    explicit SpikeTimeLossImpl(int time_steps = 1) : time_steps_(time_steps) {}

    void train(bool on) override
    {
        training_ = on;
    }

    void set_target(const Tensor& t)
    {
        target_ = t;
    }

    void set_time_steps(int T)
    {
        time_steps_ = T;
    }

    auto forward(const Tensor& input, bool requires_grad = true) -> Tensor override
    {
        if (training_ && requires_grad)
        {
            last_input_ = input;
        }

        const int total_rows = static_cast<int>(last_input_.rows());
        const int F = static_cast<int>(last_input_.cols());
        const int T = time_steps_;
        const int B = (T > 0) ? (total_rows / T) : total_rows;

        Tensor pred_times = first_spike_times(last_input_, T, B, F);
        Tensor tgt_times = first_spike_times(target_, T, B, F);

        float sum_sq = 0.0f;
        for (size_t b = 0; b < static_cast<size_t>(B); ++b)
        {
            for (size_t f = 0; f < static_cast<size_t>(F); ++f)
            {
                float diff = pred_times.at(b, f) - tgt_times.at(b, f);
                sum_sq += diff * diff;
            }
        }

        float loss = sum_sq / static_cast<float>(B * F);
        Tensor loss_tensor(1, 1);
        loss_tensor.at(0, 0) = loss;
        return loss_tensor;
    }

    auto backward(const Tensor& /*grad_output*/) -> Tensor override
    {
        // Gradient approximation: treat first-spike time as a continuous variable.
        // dL/d_input[t,b,f] = 2*(pred_t - tgt_t)/(B*F) if t == first_spike_time, else 0.
        // This is a straight-through estimator through the argmin over t.
        const int total_rows = static_cast<int>(last_input_.rows());
        const int F = static_cast<int>(last_input_.cols());
        const int T = time_steps_;
        const int B = (T > 0) ? (total_rows / T) : total_rows;

        Tensor pred_times = first_spike_times(last_input_, T, B, F);
        Tensor tgt_times = first_spike_times(target_, T, B, F);

        Tensor grad(static_cast<size_t>(total_rows), static_cast<size_t>(F));
        grad.setZero();

        const float scale = 2.0f / static_cast<float>(B * F);
        for (int b = 0; b < B; ++b)
        {
            for (int f = 0; f < F; ++f)
            {
                float pt = pred_times.at(static_cast<size_t>(b), static_cast<size_t>(f));
                float tt = tgt_times.at(static_cast<size_t>(b), static_cast<size_t>(f));
                float g = scale * (pt - tt);
                int t = static_cast<int>(pt);
                if (t < T) // only assign gradient at the actual spike time
                {
                    grad.at(static_cast<size_t>(t * B + b), static_cast<size_t>(f)) = g;
                }
            }
        }
        return grad;
    }
};

#endif // NN_LAYERS_SPIKETIMELOSS_HPP
