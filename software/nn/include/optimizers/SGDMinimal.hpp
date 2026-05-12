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
 *   param = param - lr * grad
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
        for (auto* param : params) [[likely]]
        {
            if (param == nullptr)
            {
                throw std::invalid_argument("Parameter pointer is null");
            }
            // Update: param = param - learning_rate * grad (uses backend operations)
            Tensor grad_copy = param->grad();
            auto update = grad_copy.multiply_scalar(learning_rate);
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
        // SGDMinimal doesn't need to store parameters as step/zero_grad take them as arguments
    }
};

#endif // NN_OPTIMIZERS_SGD_MINIMAL_HPP
