#ifndef NN_OPTIMIZERS_SGD_HPP
#define NN_OPTIMIZERS_SGD_HPP

#include <span>
#include <stdexcept>

#include "optimizers/Optimizer.hpp"

/**
 * @file SGD.hpp
 * @brief Stochastic Gradient Descent with optional momentum (Polyak heavy-ball formulation).
 *
 * Lifecycle:
 * - `attach(params)` allocates velocity buffers matching each parameter shape.
 * - `step(params)` applies an update in-place.
 * - `zero_grad(params)` resets gradients on all parameters.
 *
 * Update rule (Polyak / classical momentum), for parameter i:
 *   lr_i = learning_rate * lr_scales_[i]   (per-group scale; 1.0 when unset)
 *   v_t  = μ * v_{t-1} − lr_i * g_t
 *   θ_t  = θ_{t-1} + v_t
 *
 * This is mathematically equivalent to the Sutskever formulation
 * (v = μ*v + g; θ -= lr*v) when μ=0, and produces the same fixed points.
 * Both forms are standard — see Polyak 1964 and Sutskever et al. ICML 2013.
 *
 * Supports the two base-class knobs (see Optimizer.hpp):
 * - `lr_scales_` via `attach_with_scales()` — per-parameter-group learning rates,
 *   used to give SNN biophysical scalars (R, C, V_th) a smaller lr than weights.
 * - `weight_decay` — decoupled L2 (the SGDW variant of Loshchilov & Hutter, ICLR 2019),
 *   applied only to 2-D weight matrices.
 *
 * Reference: [2] D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization,"
 * ICLR 2015. (SGD as baseline comparator)
 */

struct SGD : public Optimizer
{
    using Tensor = Optimizer::Tensor;

    float learning_rate;
    float momentum;

    std::vector<Tensor> velocity;

    explicit SGD(float lr = 0.01F, float momentum_value = 0.0F)
        : learning_rate(lr), momentum(momentum_value)
    {
        if (lr <= 0.0F)
        {
            throw std::invalid_argument("Learning rate must be positive");
        }
        if (lr > 1e8F)
        {
            throw std::invalid_argument("Learning rate is unreasonably large");
        }
    }

    auto attach(std::span<Tensor*> paramsList) -> void override
    {
        // Stores attached_params_ for the no-arg step()/zero_grad(), and resets
        // lr_scales_ to global lr — attach_with_scales() re-assigns them after
        // calling this.
        Optimizer::attach(paramsList);

        velocity.clear();
        velocity.reserve(paramsList.size()); // Pre-allocate memory
        for (auto* param : paramsList)
        {
            if (param == nullptr)
            {
                throw std::invalid_argument("Cannot attach null parameter to optimizer");
            }
            // Initialize velocity tensor with zeros matching parameter shape
            velocity.emplace_back(param->rows(), param->cols());
            velocity.back().set_zero();
        }
    }

    auto step(std::span<Tensor*> paramsList) -> void override
    {
        for (size_t i = 0; i < paramsList.size(); ++i) [[likely]]
        {
            if (paramsList[i] == nullptr)
            {
                throw std::invalid_argument("Parameter pointer is null");
            }
            auto& param = *paramsList[i];
            // Per-parameter lr: use scale if provided, otherwise 1.0 (global lr).
            const float lr_i = learning_rate * (i < lr_scales_.size() ? lr_scales_[i] : 1.0F);

            Tensor grad = param.grad();

            // Decoupled weight decay (SGDW), applied BEFORE the gradient step: Loshchilov &
            // Hutter (ICLR 2019) define the decay against θ_{t-1}, so decaying the already
            // updated θ_t would leave a systematic second-order error (the same deviation
            // ground truth caught in Adam — see Adam.hpp). Same 2-D-only restriction:
            // biases (N×1) and SNN biophysical scalars (1×1: R, C, V_th) are skipped so
            // tau=R·C and V_th survive. add_inplace/multiply_scalar_inplace mutate storage
            // in place, so unlike Adam this does not drop the gradient buffer.
            if (weight_decay > 0.0F && param.rows() > 1 && param.cols() > 1)
            {
                param.multiply_scalar_inplace(1.0F - (lr_i * weight_decay));
            }

            velocity[i].multiply_scalar_inplace(momentum);
            velocity[i].add_inplace(grad.multiply_scalar(-lr_i));
            param.add_inplace(velocity[i]);
        }
    }

    auto zero_grad(std::span<Tensor*> paramsList) -> void override
    {
        for (auto* param : paramsList) [[likely]]
        {
            if (param == nullptr)
            {
                throw std::invalid_argument("Parameter pointer is null");
            }
            param->zero_grad();
        }
    }
};

#endif // SGD_HPP
