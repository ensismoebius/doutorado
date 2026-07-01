#ifndef NN_LAYERS_SPIKING_TDBN_HPP
#define NN_LAYERS_SPIKING_TDBN_HPP

#include <cmath>
#include <map>
#include <stdexcept>
#include <string>

#include "layers/base/Module.hpp"
#include "tensor/Tensor.hpp"

/**
 * @file ThresholdDependentBatchNorm.hpp
 * @brief Threshold-Dependent Batch Normalization (tdBN) for deep SNNs.
 *
 * Canonical formulation of Zheng et al., "Going Deeper With Directly-Trained
 * Larger Spiking Neural Networks," AAAI 2021 [33]. For the k-th feature/channel:
 *
 *     X̂_k = γ_k · ( α·V_th · (X_k − μ_k) / sqrt(σ_k² + ε) ) + β_k
 *
 * where:
 *   - X_k       pre-synaptic input (membrane current) of feature k,
 *   - μ_k, σ_k² mean and variance of feature k computed JOINTLY over the batch
 *               AND the time dimension (all T·B rows of the time-major input),
 *   - α         hyperparameter (paper default 1) setting the target spread,
 *   - V_th      firing threshold of the downstream LIF neuron,
 *   - γ_k, β_k  learnable scale and shift,
 *   - ε         numerical-stability constant.
 *
 * The normalized pre-activation is therefore distributed as N(0, (α·V_th)²)
 * rather than N(0,1): its standard deviation is tied to the threshold so that,
 * regardless of depth or sequence length T, a consistent fraction of neurons sits
 * near V_th and fires. This is what lets very deep SNNs train without vanishing or
 * exploding membrane potentials (the "balanced threshold-firing" argument of [33]).
 *
 * Important: the scale α·V_th multiplies the *normalized* term only — β is NOT
 * scaled — and statistics pool over batch+time, not per time step. (Earlier
 * revisions of this file used a per-step V_th/√T heuristic that is not in the
 * paper; that has been corrected to the source-of-truth formula above.)
 *
 * **Training vs inference.** Like standard BatchNorm, batch statistics are used
 * during training and exponential running averages (running_mean / running_var)
 * are accumulated and used at inference time (train(false)). This keeps inference
 * deterministic and independent of other samples in the batch.
 *
 * **Shape contract.** forward() expects time-major input of shape (T·B, F) (or
 * (B, F) when T=1). rows must be divisible by time_steps. Normalization pools all
 * rows of a column together, matching the paper's batch+time statistics.
 */
template <typename Backend>
class ThresholdDependentBatchNormImpl : public Module<Backend>
{
    using Tensor = typename Module<Backend>::Tensor;

   public:
    float alpha = 1.0F;             ///< α: target std = α·V_th (paper default 1)
    float voltage_threshold = 1.0F; ///< V_th of the downstream LIF layer
    int time_steps = 1;             ///< T: time steps in the time-major input
    float eps = 1e-5F;              ///< Numerical stability for variance
    float momentum = 0.1F;          ///< EMA rate for running statistics

    Tensor gamma; ///< Learned scale γ (1,F)
    Tensor beta;  ///< Learned shift β (1,F)

    Tensor running_mean; ///< Inference mean buffer (1,F)
    Tensor running_var;  ///< Inference variance buffer (1,F)

    // Cached for backward (populated when forward(requires_grad=true)).
    Tensor x_norm_cache_; ///< normalized values x̂ (T·B, F)
    Tensor inv_std_cache_; ///< per-feature 1/sqrt(var+eps) used in forward (1,F)

    bool training_ = true;

    std::array<Tensor*, 2> param_ptrs_{{&gamma, &beta}};

    [[nodiscard]] auto params() -> std::span<Tensor*> override
    {
        return std::span<Tensor*>{param_ptrs_.data(), param_ptrs_.size()};
    }

    void train(bool on) override { training_ = on; }

    explicit ThresholdDependentBatchNormImpl(size_t num_features, float vth = 1.0F,
        int T = 1, float alpha_ = 1.0F, float eps_ = 1e-5F, float momentum_ = 0.1F)
        : alpha(alpha_), voltage_threshold(vth), time_steps(T), eps(eps_), momentum(momentum_)
    {
        const auto F = static_cast<size_t>(num_features);
        gamma = Tensor::ones(1, F);
        beta = Tensor(1, F);
        beta.setZero();
        running_mean = Tensor(1, F);
        running_mean.setZero();
        running_var = Tensor::ones(1, F);
    }

    auto forward(const Tensor& input, bool requires_grad = true) -> Tensor override
    {
        const size_t total_rows = input.rows();
        const size_t F = input.cols();
        const size_t T = static_cast<size_t>(time_steps);

        if (T == 0 || total_rows % T != 0)
            throw std::invalid_argument("tdBN: input rows must be divisible by time_steps");
        if (total_rows == 0)
            throw std::invalid_argument("tdBN: empty input");

        // N pools batch AND time: statistics are taken over every row of a column.
        const float N = static_cast<float>(total_rows);
        const float scale = alpha * voltage_threshold; // α·V_th

        Tensor output(total_rows, F);
        if (requires_grad)
        {
            x_norm_cache_ = Tensor(total_rows, F);
            inv_std_cache_ = Tensor(1, F);
        }

        for (size_t f = 0; f < F; ++f)
        {
            float use_mean = 0.0f;
            float use_var = 0.0f;

            if (training_)
            {
                // Batch+time statistics for this feature.
                float sum = 0.0f;
                for (size_t r = 0; r < total_rows; ++r) sum += input.at(r, f);
                use_mean = sum / N;

                float var_sum = 0.0f;
                for (size_t r = 0; r < total_rows; ++r)
                {
                    const float d = input.at(r, f) - use_mean;
                    var_sum += d * d;
                }
                use_var = var_sum / N; // biased variance, consistent with normalization

                // Update running statistics for inference (standard BN EMA).
                running_mean.at(0, f) = (1.0f - momentum) * running_mean.at(0, f) + momentum * use_mean;
                running_var.at(0, f) = (1.0f - momentum) * running_var.at(0, f) + momentum * use_var;
            }
            else
            {
                use_mean = running_mean.at(0, f);
                use_var = running_var.at(0, f);
            }

            const float inv_std = 1.0f / std::sqrt(use_var + eps);
            if (requires_grad) inv_std_cache_.at(0, f) = inv_std;

            const float g = gamma.at(0, f);
            const float bi = beta.at(0, f);
            for (size_t r = 0; r < total_rows; ++r)
            {
                const float xn = (input.at(r, f) - use_mean) * inv_std;
                if (requires_grad) x_norm_cache_.at(r, f) = xn;
                output.at(r, f) = g * (scale * xn) + bi; // β is not scaled
            }
        }

        return output;
    }

    auto backward(const Tensor& grad_output) -> Tensor override
    {
        const size_t total_rows = grad_output.rows();
        const size_t F = grad_output.cols();
        const float N = static_cast<float>(total_rows);
        const float scale = alpha * voltage_threshold;

        Tensor grad_input(total_rows, F);
        Tensor d_gamma(1, F);
        d_gamma.setZero();
        Tensor d_beta(1, F);
        d_beta.setZero();

        // out = γ·(scale·x̂) + β. Backward over the N = T·B pooled samples per feature.
        for (size_t f = 0; f < F; ++f)
        {
            const float g = gamma.at(0, f);
            const float inv_std = inv_std_cache_.at(0, f); // training-time stat (backward only runs in training)

            // dL/dγ, dL/dβ and the reductions needed for the BN input gradient.
            float sum_dxn = 0.0f;     // Σ (dL/dx̂)
            float sum_dxn_xn = 0.0f;  // Σ (dL/dx̂)·x̂
            for (size_t r = 0; r < total_rows; ++r)
            {
                const float dout = grad_output.at(r, f);
                const float xn = x_norm_cache_.at(r, f);
                d_beta.at(0, f) += dout;
                d_gamma.at(0, f) += dout * scale * xn;
                const float dxn = dout * g * scale; // dL/dx̂
                sum_dxn += dxn;
                sum_dxn_xn += dxn * xn;
            }

            // Standard batch-norm input gradient:
            // dL/dx = (1/N)·inv_std·(N·dx̂ − Σdx̂ − x̂·Σ(dx̂·x̂))
            for (size_t r = 0; r < total_rows; ++r)
            {
                const float dxn = grad_output.at(r, f) * g * scale;
                const float xn = x_norm_cache_.at(r, f);
                grad_input.at(r, f) = (1.0f / N) * inv_std * (N * dxn - sum_dxn - xn * sum_dxn_xn);
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
        d["running_mean"] = running_mean;
        d["running_var"] = running_var;
        return d;
    }

    void load_state_dict(const std::map<std::string, Tensor>& sd) override
    {
        auto it = sd.find("gamma");
        if (it != sd.end()) gamma = it->second;
        it = sd.find("beta");
        if (it != sd.end()) beta = it->second;
        it = sd.find("running_mean");
        if (it != sd.end()) running_mean = it->second;
        it = sd.find("running_var");
        if (it != sd.end()) running_var = it->second;
    }
};

#endif // NN_LAYERS_SPIKING_TDBN_HPP
