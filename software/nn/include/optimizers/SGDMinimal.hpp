#ifndef NN_OPTIMIZERS_SGD_MINIMAL_HPP
#define NN_OPTIMIZERS_SGD_MINIMAL_HPP

#include <algorithm>
#include <span>
#include <stdexcept>

#include "optimizers/Optimizer.hpp"

/**
 * @file SGDMinimal.hpp
 * @brief Minimal SGD implementation without momentum/state.
 *
 * This is the simplest optimizer in the project:
 *   param = param - lr_i * grad,   lr_i = learning_rate * lr_scales_[i]
 *
 * "Minimal" refers to the absence of optimizer *state* (no momentum, no moments) —
 * it still honors the two stateless knobs on the Optimizer base (`lr_scales_` for
 * per-parameter-group learning rates and `weight_decay` for decoupled L2), because an
 * inherited field that a subclass silently ignores is a correctness trap, not a
 * simplification.
 *
 * Implementation note (Tensor semantics):
 * - In this codebase, `nn::Tensor` behaves like a value type; assigning to `*param`
 *   can replace internal storage and drop the gradient buffer.
 * - To preserve gradients (for debugging or subsequent computations), this optimizer
 *   saves a copy of `grad()` and restores it via `set_grad()` after updating data.
 */

struct SGDMinimal : public Optimizer
{
    using Tensor = Optimizer::Tensor;

    float learning_rate;

    explicit SGDMinimal(float learnningRate = 0.01F) : learning_rate(learnningRate)
    {
        if (learnningRate <= 0.0F)
        {
            throw std::invalid_argument("Learning rate must be positive");
        }
        if (learnningRate > 1e8F)
        {
            throw std::invalid_argument("Learning rate is unreasonably large");
        }
    }

    auto step(std::span<Tensor*> params) -> void override
    {
        for (std::size_t i = 0; i < params.size(); ++i) [[likely]]
        {
            if (params[i] == nullptr)
            {
                throw std::invalid_argument("Parameter pointer is null");
            }
            auto* param = params[i];
            // Per-parameter lr: use scale if provided, otherwise 1.0 (global lr).
            const float lr_i = learning_rate * (i < lr_scales_.size() ? lr_scales_[i] : 1.0F);

            // Read the gradient BEFORE any assignment to *param: assigning replaces the
            // tensor's storage and drops the gradient buffer (see the file header).
            Tensor grad_copy = param->grad();

            // Decoupled weight decay: θ ← θ(1 - lr_i*wd), applied BEFORE the gradient step
            // so it acts on θ_{t-1} as Loshchilov & Hutter (ICLR 2019) define it — the same
            // ordering ground truth pinned for Adam (see Adam.hpp). Same 2-D-only
            // restriction: biases and 1x1 SNN scalars (R, C, V_th) are skipped.
            if (weight_decay > 0.0F && param->rows() > 1 && param->cols() > 1)
            {
                *param = param->multiply_scalar(1.0F - (lr_i * weight_decay));
            }

            // Update: param = param - lr_i * grad (uses backend operations)
            auto update = grad_copy.multiply_scalar(lr_i);
            *param = *param - update;

            // Restore gradients (as they are lost during assignment)
            param->set_grad(grad_copy);
        }
    }

    auto zero_grad(std::span<Tensor*> params) -> void override
    {
        for (auto* param : params) [[likely]]
        {
            if (param == nullptr)
            {
                throw std::invalid_argument("Parameter pointer is null");
            }
            param->zero_grad();
        }
    }
    void attach(std::span<Tensor*> params) override
    {
        // cppcheck-suppress useStlAlgorithm
        const bool has_null = std::any_of(
            params.begin(), params.end(), [](const auto* param) { return param == nullptr; });
        if (has_null)
        {
            throw std::invalid_argument("Cannot attach null parameter to optimizer");
        }
        // SGDMinimal has no per-parameter state of its own, but the base still records
        // `params` (for the no-arg step()/zero_grad()) and resets `lr_scales_`.
        Optimizer::attach(params);
    }
};

#endif // NN_OPTIMIZERS_SGD_MINIMAL_HPP
