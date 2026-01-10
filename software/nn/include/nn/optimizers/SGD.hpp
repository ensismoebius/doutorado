#ifndef SGD_HPP
#define SGD_HPP

#include <span>
#include <stdexcept>

#include "nn/optimizers/Optimizer.hpp"

/**
 * @file SGD.hpp
 * @brief Stochastic Gradient Descent (with optional momentum).
 *
 * Lifecycle:
 * - `attach(params)` allocates velocity buffers matching each parameter shape.
 * - `step(params)` applies an update in-place.
 * - `zero_grad(params)` resets gradients on all parameters.
 *
 * Important caveat:
 * - A conventional SGD update uses `param.grad()`.
 *   This implementation currently builds its velocity from `param` values
 *   (not `grad`), which is atypical and likely not what you want for training.
 *   Treat this as experimental/placeholder unless verified.
 */

struct SGD : public Optimizer
{
    float learning_rate;
    float momentum;

    std::vector<nn::Tensor> velocity;

    explicit SGD(float lr = 0.01F, float momentum = 0.0F) : learning_rate(lr), momentum(momentum)
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

    auto attach(std::span<nn::Tensor*> paramsList) -> void override
    {
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

    auto step(std::span<nn::Tensor*> paramsList) -> void override
    {
        for (size_t i = 0; i < paramsList.size(); ++i) [[likely]]
        {
            if (paramsList[i] == nullptr)
            {
                throw std::invalid_argument("Parameter pointer is null");
            }
            auto& param = *paramsList[i];
            velocity[i] =
                velocity[i].multiply_scalar(momentum).add(param.multiply_scalar(-learning_rate));
            // Element-wise addition to param data
            for (size_t row = 0; row < param.rows(); ++row)
            {
                for (size_t col = 0; col < param.cols(); ++col)
                {
                    param.at(row, col) = param.at(row, col) + velocity[i].at(row, col);
                }
            }
        }
    }

    auto zero_grad(std::span<nn::Tensor*> paramsList) -> void override
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
