#ifndef NN_LAYERS_SPIKING_TDBN_HPP
#define NN_LAYERS_SPIKING_TDBN_HPP

#include <cmath>
#include <stdexcept>

#include "nn/layers/base/Module.hpp"
#include "nn/tensor/Tensor.hpp"

/**
 * @file ThresholdDependentBatchNorm.hpp
 * @brief Threshold-Dependent Batch Normalization (tdBN) for deep SNNs.
 *
 * tdBN normalises the pre-spike membrane potential at each time step and
 * rescales the result by V_th / sqrt(T) so that inputs to the next LIF layer
 * maintain the same expected magnitude regardless of network depth or sequence
 * length. This allows training SNNs with many layers without gradient vanishing
 * or exploding membrane potentials.
 *
 * **Forward pass** (per time-step slice, shape B×F):
 *   x_norm  = (x - mean(x)) / sqrt(var(x) + eps)
 *   x_scaled = gamma * x_norm + beta        (learned affine transform)
 *   x_tdbn  = x_scaled * (V_th / sqrt(T))  (threshold-dependent scaling)
 *
 * **Backward pass:** standard batch-norm gradient through the affine transform,
 * with the tdBN scale factor treated as a constant.
 *
 * **Shape contract:**
 * - `forward(input)` expects input of shape (T*B, F) or (B, F).
 *   When time_steps > 1, input is split into T slices of size (B, F) and each
 *   slice is normalised independently (matching snnTorch's per-step BN).
 *
 * Reference: [33] Y. Zheng et al., "Going deeper with directly-trained larger
 * spiking neural networks," AAAI 2021 (Threshold-Dependent Batch Normalization).
 */
template <typename Backend>
class ThresholdDependentBatchNormImpl : public Module<Backend>
{
    using Tensor = typename Module<Backend>::Tensor;

   public:
    float voltage_threshold = 1.0F; ///< V_th of the downstream LIF layer
    int time_steps = 1;             ///< T: number of time steps in the input sequence
    float eps = 1e-5F;              ///< Numerical stability for variance

    Tensor gamma; ///< Learned scale (F,)
    Tensor beta;  ///< Learned shift (F,)

    // Cached for backward
    Tensor x_norm_cache_;
    Tensor input_cache_;

    std::array<Tensor*, 2> param_ptrs_{{&gamma, &beta}};

    [[nodiscard]] auto params() -> std::span<Tensor*> override
    {
        return std::span<Tensor*>{param_ptrs_.data(), param_ptrs_.size()};
    }

    explicit ThresholdDependentBatchNormImpl(
        size_t num_features, float vth = 1.0F, int T = 1, float eps_ = 1e-5F)
        : voltage_threshold(vth), time_steps(T), eps(eps_)
    {
        gamma = Tensor::ones(1, static_cast<size_t>(num_features));
        beta = Tensor(1, static_cast<size_t>(num_features));
        beta.setZero();
    }

    auto forward(const Tensor& input, bool requires_grad = true) -> Tensor override
    {
        if (requires_grad)
        {
            input_cache_ = input;
        }

        const size_t total_rows = input.rows();
        const size_t F = input.cols();
        const size_t T = static_cast<size_t>(time_steps);
        const size_t B = (T > 0) ? (total_rows / T) : total_rows;

        if (total_rows % T != 0)
        {
            throw std::invalid_argument("tdBN: input rows must be divisible by time_steps");
        }

        const float tdbn_scale = voltage_threshold / std::sqrt(static_cast<float>(T));

        Tensor output(total_rows, F);
        if (requires_grad)
        {
            x_norm_cache_ = Tensor(total_rows, F);
        }

        for (size_t t = 0; t < T; ++t)
        {
            const size_t offset = t * B;

            // Compute per-feature mean and variance over the batch dimension
            for (size_t f = 0; f < F; ++f)
            {
                float sum = 0.0f;
                for (size_t b = 0; b < B; ++b)
                {
                    sum += input.at(offset + b, f);
                }
                const float mean = sum / static_cast<float>(B);

                float var_sum = 0.0f;
                for (size_t b = 0; b < B; ++b)
                {
                    float d = input.at(offset + b, f) - mean;
                    var_sum += d * d;
                }
                const float inv_std = 1.0f / std::sqrt(var_sum / static_cast<float>(B) + eps);
                const float g = gamma.at(0, f);
                const float bi = beta.at(0, f);

                for (size_t b = 0; b < B; ++b)
                {
                    float xn = (input.at(offset + b, f) - mean) * inv_std;
                    if (requires_grad)
                    {
                        x_norm_cache_.at(offset + b, f) = xn;
                    }
                    output.at(offset + b, f) = (g * xn + bi) * tdbn_scale;
                }
            }
        }

        return output;
    }

    auto backward(const Tensor& grad_output) -> Tensor override
    {
        const size_t total_rows = grad_output.rows();
        const size_t F = grad_output.cols();
        const size_t T = static_cast<size_t>(time_steps);
        const size_t B = total_rows / T;

        const float tdbn_scale = voltage_threshold / std::sqrt(static_cast<float>(T));

        Tensor grad_input(total_rows, F);
        Tensor d_gamma(1, F);
        d_gamma.setZero();
        Tensor d_beta(1, F);
        d_beta.setZero();

        for (size_t t = 0; t < T; ++t)
        {
            const size_t offset = t * B;
            const float N = static_cast<float>(B);

            for (size_t f = 0; f < F; ++f)
            {
                // Aggregate dL/d_gamma and dL/d_beta
                for (size_t b = 0; b < B; ++b)
                {
                    float dout = grad_output.at(offset + b, f) * tdbn_scale;
                    d_gamma.at(0, f) += dout * x_norm_cache_.at(offset + b, f);
                    d_beta.at(0, f) += dout;
                }

                // Batch-norm input gradient: standard formula
                // dx = (1/N) * gamma * inv_std * (N*dy - sum(dy) - x_norm*sum(dy*x_norm))
                float sum_dout = 0.0f;
                float sum_dout_xn = 0.0f;
                for (size_t b = 0; b < B; ++b)
                {
                    float dout = grad_output.at(offset + b, f) * tdbn_scale * gamma.at(0, f);
                    float xn = x_norm_cache_.at(offset + b, f);
                    sum_dout += dout;
                    sum_dout_xn += dout * xn;
                }

                // Recompute inv_std from input cache
                float sum_sq = 0.0f;
                float mean_val = 0.0f;
                for (size_t b = 0; b < B; ++b)
                {
                    mean_val += input_cache_.at(offset + b, f);
                }
                mean_val /= N;
                for (size_t b = 0; b < B; ++b)
                {
                    float d = input_cache_.at(offset + b, f) - mean_val;
                    sum_sq += d * d;
                }
                const float inv_std = 1.0f / std::sqrt(sum_sq / N + eps);

                for (size_t b = 0; b < B; ++b)
                {
                    float dout = grad_output.at(offset + b, f) * tdbn_scale * gamma.at(0, f);
                    float xn = x_norm_cache_.at(offset + b, f);
                    float dx = (1.0f / N) * inv_std * (N * dout - sum_dout - xn * sum_dout_xn);
                    grad_input.at(offset + b, f) = dx;
                }
            }
        }

        gamma.set_grad(d_gamma);
        beta.set_grad(d_beta);

        return grad_input;
    }

    auto state_dict() const -> std::map<std::string, Tensor> override
    {
        std::map<std::string, Tensor> d;
        d["gamma"] = gamma;
        d["beta"] = beta;
        return d;
    }

    void load_state_dict(const std::map<std::string, Tensor>& sd) override
    {
        auto it = sd.find("gamma");
        if (it != sd.end()) gamma = it->second;
        it = sd.find("beta");
        if (it != sd.end()) beta = it->second;
    }
};

#endif // NN_LAYERS_SPIKING_TDBN_HPP
