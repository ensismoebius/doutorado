#ifndef SGD_MINIMAL_HPP
#define SGD_MINIMAL_HPP

#include <span>
#include <stdexcept>

#include "Optimizer.hpp"

struct SGDMinimal : public Optimizer
{
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

    auto step(std::span<nn::Tensor*> params) -> void override
    {
        for (auto* param : params) [[likely]]
        {
            if (param == nullptr)
            {
                throw std::invalid_argument("Parameter pointer is null");
            }
            // Update: param = param - learning_rate * grad (uses backend operations)
            nn::Tensor grad_copy = param->grad();
            auto update = grad_copy.multiply_scalar(learning_rate);
            *param = *param - update;

            // Restore gradients (as they are lost during assignment)
            param->set_grad(grad_copy);
        }
    }

    auto zero_grad(std::span<nn::Tensor*> params) -> void override
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
    void attach(std::span<nn::Tensor*> params) override
    {
        for (auto* param : params)
        {
            if (param == nullptr)
            {
                throw std::invalid_argument("Cannot attach null parameter to optimizer");
            }
            // SGDMinimal doesn't need to store parameters as step/zero_grad take them as arguments
        }
    }
};

#endif // SGD_MINIMAL_HPP
