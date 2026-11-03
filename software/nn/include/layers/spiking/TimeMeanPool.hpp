#ifndef TIME_MEAN_POOL_HPP
#define TIME_MEAN_POOL_HPP

#include <stdexcept>
#include <string>

#include "layers/base/Module.hpp"

/**
 * @file TimeMeanPool.hpp
 * @brief Mean over the time axis of a time-major `(T*B, F)` tensor, giving `(B, F)`.
 *
 * Why it exists: a spiking stack emits one row per simulation step. A reconstruction loss
 * scored on every row asks each step, driven by at most one binary spike per neuron, to
 * reproduce the whole analog window on its own. Scoring the MEAN over the T steps asks only
 * that the steps get it right on average (a rate readout). Placed after the latent
 * projection it also lets an analog decoder run once per window instead of T times.
 *
 *   input  (T*B, F), row = t*B + b              output (B, F), row = b
 *   y[b, f]           = (1/T) * sum_t x[t*B + b, f]
 *   dL/dx[t*B + b, f] = dL/dy[b, f] / T          every step gets the same share
 *
 * Contract:
 * - `input.rows() % time_steps == 0`, else `std::invalid_argument` (the same invariant
 *   LifBPTT enforces on the tensor it emits).
 * - `backward()` needs a prior `forward()` (it reads B from it), else `std::logic_error`.
 * - No parameters and no state carried between sequences.
 */
template <typename Backend>
struct TimeMeanPoolImpl : public Module<Backend>
{
    /// Tensor type for the active compute backend.
    using Tensor = typename Module<Backend>::Tensor;

    int time_steps; ///< HOW MANY rows of one sample are averaged (see .wiki/Concepts/Time-Steps.md)

    explicit TimeMeanPoolImpl(int time_steps_) : time_steps(time_steps_)
    {
        if (time_steps < 1)
            throw std::invalid_argument("TimeMeanPool: time_steps must be >= 1 (got " +
                                        std::to_string(time_steps) +
                                        "). Pass the BPTT sequence length of the spiking layers "
                                        "whose output this layer averages.");
    }

    auto forward(const Tensor& input, bool /*requires_grad*/ = true) -> Tensor override
    {
        const int rows = input.rows();
        if (rows % time_steps != 0)
            throw std::invalid_argument("TimeMeanPool: input rows (" + std::to_string(rows) +
                                        ") are not a multiple of time_steps (" +
                                        std::to_string(time_steps) +
                                        "). The input must be time-major (T*B, F).");
        const int batch = rows / time_steps;
        const int features = input.cols();

        Tensor output(batch, features);
        output.setZero();
        for (int t = 0; t < time_steps; ++t)
            for (int b = 0; b < batch; ++b)
                for (int f = 0; f < features; ++f) output.at(b, f) += input.at(t * batch + b, f);
        output.multiply_scalar_inplace(1.0F / static_cast<float>(time_steps));

        last_batch_ = batch;
        return output;
    }

    auto backward(const Tensor& grad_output) -> Tensor override
    {
        if (last_batch_ < 0 || grad_output.rows() != last_batch_)
            throw std::logic_error(
                "TimeMeanPool: backward() got " + std::to_string(grad_output.rows()) +
                " rows but the last forward() produced " + std::to_string(last_batch_) +
                ". Call forward() on the same batch first.");
        const int batch = last_batch_;
        const int features = grad_output.cols();
        const float share = 1.0F / static_cast<float>(time_steps);

        Tensor grad_input(time_steps * batch, features);
        for (int t = 0; t < time_steps; ++t)
            for (int b = 0; b < batch; ++b)
                for (int f = 0; f < features; ++f)
                    grad_input.at(t * batch + b, f) = grad_output.at(b, f) * share;
        return grad_input;
    }

   private:
    int last_batch_ = -1;
};

#endif // TIME_MEAN_POOL_HPP
